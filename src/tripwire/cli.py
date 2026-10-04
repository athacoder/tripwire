"""Tripwire: a statistical regression gate for LLM systems."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import sys
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from . import __version__, datasets, report, stats, store
from .compare import blocked, compare, matrix, summarise
from .config import Config, load_config
from .gate import gate as run_gate
from .models import Case
from .providers import ProviderError, Request, make_provider
from .runner import Summary, run_suite
from .scorers import score_suite, selfcheck

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect and prepare datasets.")
app.add_typer(dataset_app, name="dataset")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to tripwire.toml.")]
DEFAULT_CONFIG = Path("tripwire.toml")
EXIT = {"REGRESSED": 1, "INCONCLUSIVE": 2, "INVALID": 3}


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False,
) -> None:
    # Reports quote model output and dataset text; a legacy Windows console code page
    # cannot print all of it and would otherwise crash mid-report.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _show(rows: dict[str, Any]) -> None:
    width = max(len(k) for k in rows)
    for key, value in rows.items():
        typer.echo(
            f"{key:<{width}}  {value:.3f}"
            if isinstance(value, float)
            else f"{key:<{width}}  {value}"
        )


@contextmanager
def _errors() -> Iterator[None]:
    """Turn expected failures into one red line and exit code 2, not a traceback."""
    try:
        yield
    except (ProviderError, ValueError, FileNotFoundError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e


def _execute[T](work: Coroutine[Any, Any, T]) -> T:
    with _errors():
        return asyncio.run(work)


async def _ready(cfg: Config, name: str, db: sqlite3.Connection) -> tuple[Summary, list[Case]]:
    """Make sure a suite's samples and scores exist. Free when they already do."""
    s = await run_suite(cfg, name)
    score_suite(cfg, name, s.fingerprint, db)
    return s, datasets.load(cfg.root / cfg.get_suite(name).dataset)


def _summary(s: Summary) -> dict[str, Any]:
    made = sum(s.statuses.values())
    speed = f" ({s.output_tokens / s.model_seconds:.1f} output tok/s)" if s.model_seconds else ""
    return {
        "suite": s.suite,
        "fingerprint": s.fingerprint,
        "dataset": s.dataset_version,
        "samples": f"{s.total} wanted, {s.cached} cached, {made} generated, {s.pending} pending",
        "statuses": dict(s.statuses) or "-",
        "errors": s.errors,
        "tokens": f"{s.prompt_tokens} prompt, {s.output_tokens} output",
        "model time": f"{s.model_seconds:.1f} s{speed}",
    }


@app.command()
def run(
    suite: str,
    limit: Annotated[int | None, typer.Option(help="Only the first N cases.")] = None,
    max_minutes: Annotated[float | None, typer.Option(help="Stop cleanly after this long.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Report what would run, call nothing.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Generate and score every missing sample. Existing samples are never regenerated."""
    cfg = load_config(config)
    s = _execute(run_suite(cfg, suite, limit=limit, max_minutes=max_minutes, dry_run=dry_run))
    rows = _summary(s)
    if dry_run:
        eta = f"{s.projected_minutes:.1f} min" if s.projected_minutes is not None else "unknown"
        rows = {k: rows[k] for k in ("suite", "fingerprint", "dataset", "samples")}
        rows["projected"] = eta + " (from samples already stored)"
    else:
        with _errors():
            rows["scores written"] = score_suite(
                cfg, suite, s.fingerprint, store.connect(cfg.db_path)
            )
    _show(rows)
    raise typer.Exit(1 if s.errors else 0)


@app.command()
def score(suite: str, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Re-score stored samples with the current scorers. Calls no model."""
    cfg = load_config(config)
    db = store.connect(cfg.db_path)
    rows = db.execute(
        "SELECT fingerprint FROM runs WHERE suite=? GROUP BY fingerprint", (suite,)
    ).fetchall()
    with _errors():
        written = sum(score_suite(cfg, suite, r["fingerprint"], db) for r in rows)
    typer.echo(f"{written} scores written across {len(rows)} targets")


@app.command("report")
def report_cmd(suite: str, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Score, interval, slices and worst cases for one suite."""
    cfg = load_config(config)
    db = store.connect(cfg.db_path)
    s, cases = _execute(_ready(cfg, suite, db))
    with _errors():
        summary = summarise(db, cfg.get_suite(suite), cases, s.fingerprint)
    typer.echo(report.single(suite, s.fingerprint, summary))


@app.command("selfcheck")
def selfcheck_cmd(suite: str, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Test the eval itself: reference answers must pass, junk answers must fail."""
    with _errors():
        result = selfcheck(load_config(config), suite)
    ok = result.pop("ok")
    _show(result)
    typer.secho(
        "sound" if ok else "the scorer or the dataset is broken", fg="green" if ok else "red"
    )
    raise typer.Exit(0 if ok else 1)


@app.command("compare")
def compare_cmd(
    base: str,
    head: str,
    out: Annotated[Path | None, typer.Option(help="Also write the report to this file.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Compare two suites that share a dataset: BASE is the reference, HEAD the candidate."""
    cfg = load_config(config)
    db = store.connect(cfg.db_path)

    async def work() -> tuple[dict[str, Any], bool]:
        (b, _), (h, head_cases) = await _ready(cfg, base, db), await _ready(cfg, head, db)
        if b.dataset_version != h.dataset_version:
            raise ValueError("the two suites use different datasets; a paired comparison needs one")
        suite = cfg.get_suite(head)
        result = compare(db, suite, head_cases, b.fingerprint, h.fingerprint)
        return result, blocked(result, suite)

    result, is_blocked = _execute(work())
    text = report.comparison(f"{base} → {head}", result, is_blocked)
    if out:
        out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)
    raise typer.Exit(EXIT.get(result["verdict"], 1) if is_blocked else 0)


@app.command("gate")
def gate_cmd(
    suite: str,
    base: Annotated[str, typer.Option(help="Git ref to compare against.")] = "origin/main",
    head: Annotated[
        str | None, typer.Option(help="Git ref to test (default: working tree).")
    ] = None,
    verify_only: Annotated[
        bool, typer.Option(help="Use committed sample bundles; call no model. For CI.")
    ] = False,
    bundle: Annotated[bool, typer.Option(help="Write sample bundles to commit.")] = False,
    out: Annotated[Path | None, typer.Option(help="Also write the report to this file.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Block a change that makes a suite measurably worse than it is at BASE."""
    cfg = load_config(config)
    result, is_blocked = _execute(
        run_gate(cfg, suite, base, head, verify_only=verify_only, bundle=bundle)
    )
    text = report.comparison(suite, result, is_blocked)
    if out:
        out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)
    raise typer.Exit(EXIT.get(result["verdict"], 1) if is_blocked else 0)


@app.command()
def power(
    suite: str,
    drop: Annotated[float, typer.Option(help="True drop to plan for.")] = 0.03,
    discordant: Annotated[
        float | None, typer.Option(help="Share of cases where two runs disagree.")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """How many cases the gate needs, from this suite's real scores."""
    cfg = load_config(config)
    db = store.connect(cfg.db_path)
    s, cases = _execute(_ready(cfg, suite, db))
    conf = cfg.get_suite(suite)
    scores = matrix(db, conf, cases, s.fingerprint)
    measured = conf.reps > 1 and len(scores) > 1
    if discordant is not None:
        source = "given"
    elif measured:
        discordant = float((scores[:, 0] != scores[:, 1]).mean())
        source = "rerun noise, measured from repetitions"
    else:
        discordant, source = 0.10, "assumed; use a suite with reps = 2 to measure it"
    outcomes = (scores[:, 0] >= 0.5).astype(float)
    typer.echo(f"baseline {outcomes.mean():.3f} · discordant rate {discordant:.3f} ({source})")
    typer.echo(f"margin {conf.margin} · alpha {conf.alpha}")
    typer.echo(
        "A real change flips more cases than rerun noise does. For a realistic plan, pass\n"
        "--discordant with the flip rate of a past comparison: (broke + fixed) / cases.\n"
    )
    typer.echo("Cases needed for 80% power (closed form):")
    for d in (0.02, 0.03, 0.05, 0.10):
        needed = stats.sample_size(d, discordant + d, conf.alpha)  # noise plus the drop's own flips
        typer.echo(f"  drop {d:.2f}: {needed}")
    typer.echo(f"\nSimulated verdicts (true drop {drop} | no drop at all):")
    for n in (100, 250, 500, 700, 1000, 1500):
        with _errors():
            hit, null = (
                stats.simulate(outcomes, effect, n, discordant, conf.alpha, conf.margin)
                for effect in (drop, 0.0)
            )
        typer.echo(
            f"  n={n:<5} REGRESSED {hit['REGRESSED']:.2f} INCONCLUSIVE {hit['INCONCLUSIVE']:.2f}"
            f" | PASS {null['PASS'] + null['IMPROVED']:.2f} false REGRESSED {null['REGRESSED']:.3f}"
        )


@app.command("aa")
def aa_cmd(
    suite: str,
    splits: Annotated[int, typer.Option(help="Random self-comparisons to run.")] = 500,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Compare a suite with itself to measure the gate's real false-alarm rate."""
    cfg = load_config(config)
    db = store.connect(cfg.db_path)
    s, cases = _execute(_ready(cfg, suite, db))
    conf = cfg.get_suite(suite)
    scores = matrix(db, conf, cases, s.fingerprint)
    with _errors():
        result = stats.aa(scores, conf.alpha, conf.margin, splits)
    typer.echo(f"{splits} self-comparisons of {len(scores)} cases; the true difference is zero")
    _show(result)
    limit = 2 * conf.alpha  # either false direction; each one-sided bound allows alpha
    false_alarms = result["REGRESSED"] + result["IMPROVED"]
    ok = false_alarms <= limit
    typer.secho(
        f"false-alarm rate {false_alarms:.3f} (nominal at most {limit:.2f})",
        fg="green" if ok else "red",
    )
    raise typer.Exit(0 if ok else 1)


@app.command()
def bench(
    suite: str,
    n: Annotated[int, typer.Option(help="Cases to time.")] = 20,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Measure throughput on a few cases and project how long each split will take."""
    cfg = load_config(config)
    s = _execute(run_suite(cfg, suite, limit=n))
    db = store.connect(cfg.db_path)
    rows = db.execute(
        "SELECT latency_ms, response FROM samples WHERE fingerprint=?", (s.fingerprint,)
    ).fetchall()
    env = db.execute(
        "SELECT env FROM runs WHERE fingerprint=? ORDER BY started_at DESC", (s.fingerprint,)
    ).fetchone()
    if not rows:
        typer.secho("no samples to measure", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    raw = [json.loads(r["response"]) for r in rows]
    seconds = sum(r["latency_ms"] for r in rows) / len(rows) / 1000

    def rate(count: str, duration: str) -> str:
        nanos = sum(d.get(duration, 0) for d in raw)
        return f"{sum(d.get(count, 0) for d in raw) / nanos * 1e9:.0f} tok/s" if nanos else "n/a"

    result = {
        "samples measured": len(rows),
        "seconds per case": f"{seconds:.2f}",
        "prompt speed": rate("prompt_eval_count", "prompt_eval_duration"),
        "output speed": rate("eval_count", "eval_duration"),
        "runtime": json.loads(env["env"]) if env else "-",
    }
    reps = cfg.get_suite(suite).reps
    for path in sorted((cfg.root / cfg.get_suite(suite).dataset).parent.glob("*.jsonl")):
        cases = len(datasets.load(path))
        result[f"projected {path.stem}"] = f"{cases * reps * seconds / 60:.1f} min ({cases} cases)"
    _show(result)


async def _doctor(cfg: Config) -> bool:
    healthy = True
    for name in cfg.suite:
        target = cfg.target(name)
        if target.kind != "prompt":
            continue
        provider = None
        try:
            provider = make_provider(cfg.provider[target.provider])
            digest = await provider.digest(target.model)
            ping = Request(target.model, "", "ping", num_ctx=target.num_ctx, max_tokens=1)
            await asyncio.wait_for(provider.complete(ping), cfg.timeout)
            info = await provider.info(target.model)
            typer.echo(f"ok    {name}: {target.model} {digest[:12]} {info}")
        except (ProviderError, TimeoutError) as e:
            healthy = False
            typer.secho(f"FAIL  {name}: {e or 'timed out'}", fg=typer.colors.RED)
        finally:
            if provider:
                await provider.aclose()
    return healthy


@app.command()
def doctor(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Check that every suite's model is reachable, present and able to generate."""
    raise typer.Exit(0 if _execute(_doctor(load_config(config))) else 1)


@dataset_app.command("lint")
def lint_cmd(
    path: Path,
    prompts: Annotated[
        Path | None, typer.Option(help="Prompt file or directory to check for leaked cases.")
    ] = None,
    embed: Annotated[bool, typer.Option(help="Also compare embeddings (needs Ollama).")] = False,
    base_url: str = "http://localhost:11434",
) -> None:
    """Check a dataset for duplicates, label problems and leakage."""
    with _errors():
        cases = datasets.load(path)
    files = (
        [] if prompts is None else [prompts] if prompts.is_file() else sorted(prompts.rglob("*"))
    )
    context = "\n".join(f.read_text(encoding="utf-8") for f in files if f.is_file())
    result = datasets.lint(cases, context)
    near = result.pop("near_duplicates")
    ok = result.pop("ok")
    result["near_duplicates (warning)"] = len(near)
    if embed:
        try:
            pairs = datasets.embedding_near_duplicates(cases, base_url)
            result["embedding_near_duplicates (warning)"] = len(pairs)
        except (httpx.HTTPError, KeyError) as e:  # optional check: never block the lint on it
            result["embedding_near_duplicates (warning)"] = f"skipped ({type(e).__name__})"
    _show(result)
    for i, j in near[:5]:
        typer.echo(f"  near: {cases[i].text!r} ~ {cases[j].text!r}")
    typer.secho("clean" if ok else "problems found", fg="green" if ok else "red")
    raise typer.Exit(0 if ok else 1)


@dataset_app.command("split")
def split_cmd(
    src: Path,
    out: Path,
    size: Annotated[list[str], typer.Option(help="name=count; repeat for each split.")],
    seed: int = 0,
) -> None:
    """Draw disjoint, stratified random splits from a dataset."""
    try:
        sizes = {name: int(count) for name, count in (item.split("=") for item in size)}
    except ValueError as e:
        raise typer.BadParameter("each --size must look like name=count") from e
    for name, cases in datasets.split(datasets.load(src), sizes, seed).items():
        datasets.save(out / f"{name}.jsonl", cases)
        typer.echo(f"{name}: {len(cases)} cases, version {datasets.version(cases)}")
