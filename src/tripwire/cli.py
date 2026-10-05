"""Tripwire: a statistical regression gate for LLM systems."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import sys
import time
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from . import __version__, datasets, generate, report, stats, store
from .compare import blocked, compare, matrix, summarise
from .config import Config, load_config
from .gate import gate as run_gate
from .judge import Judge, against_rule, calibration, judge_suite, probe
from .models import Case
from .providers import ProviderError, Request, make_provider
from .runner import Summary, now, run_suite
from .scorers import JUDGE, score_suite, selfcheck

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect and prepare datasets.")
judge_app = typer.Typer(no_args_is_help=True, help="Run an LLM judge and measure its accuracy.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(judge_app, name="judge")

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


def _load(path: Path) -> Config:
    """The configuration; a missing or invalid file is reported, not thrown at the user."""
    with _errors():
        return load_config(path)


async def _ready(cfg: Config, name: str, db: sqlite3.Connection) -> tuple[Summary, list[Case]]:
    """Make sure a suite's samples and scores exist. Free when they already do."""
    s = await run_suite(cfg, name)
    score_suite(cfg, name, s.fingerprint, db)
    if JUDGE in cfg.get_suite(name).scorers:
        counts = await judge_suite(cfg, name, s.fingerprint, db)
        if counts["judged"] or counts["failed"]:
            typer.echo(
                f"judge: {counts['judged']} new verdicts, {counts['failed']} failed", err=True
            )
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
    cfg = _load(config)
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
    cfg = _load(config)
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
    cfg = _load(config)
    db = store.connect(cfg.db_path)
    s, cases = _execute(_ready(cfg, suite, db))
    with _errors():
        summary = summarise(db, cfg.get_suite(suite), cases, s.fingerprint)
    typer.echo(report.single(suite, s.fingerprint, summary))


@app.command("selfcheck")
def selfcheck_cmd(suite: str, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Test the eval itself: reference answers must pass, junk answers must fail."""
    with _errors():
        result = selfcheck(_load(config), suite)
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
    cfg = _load(config)
    db = store.connect(cfg.db_path)

    async def work() -> tuple[dict[str, Any], bool]:
        (b, _), (h, head_cases) = await _ready(cfg, base, db), await _ready(cfg, head, db)
        if b.dataset_version != h.dataset_version:
            raise ValueError("the two suites use different datasets; a paired comparison needs one")
        suite = cfg.get_suite(head)
        result = compare(db, suite, head_cases, b.fingerprint, h.fingerprint)
        return result, blocked(result, suite)

    result, is_blocked = _execute(work())
    text = report.comparison(f"{base} → {head}", result, is_blocked, cfg.tracelens_url)
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
    cfg = _load(config)
    result, is_blocked = _execute(
        run_gate(cfg, suite, base, head, verify_only=verify_only, bundle=bundle)
    )
    text = report.comparison(suite, result, is_blocked, cfg.tracelens_url)
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
    first_stage: Annotated[
        int | None, typer.Option(help="Also simulate a two-stage gate with this first stage.")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """How many cases the gate needs, from this suite's real scores."""
    cfg = _load(config)
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
    if first_stage:
        with _errors():
            _two_stage_report(outcomes, drop, discordant, conf.alpha, conf.margin, first_stage)


@app.command("aa")
def aa_cmd(
    suite: str,
    splits: Annotated[int, typer.Option(help="Random self-comparisons to run.")] = 500,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Compare a suite with itself to measure the gate's real false-alarm rate."""
    cfg = _load(config)
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
    cfg = _load(config)
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
    # Each distinct model once: loading a model per suite would only thrash the GPU.
    models: dict[tuple[str, str, int], list[str]] = {}
    for name in cfg.suite:
        target = cfg.target(name)
        if target.kind == "prompt":
            models.setdefault((target.provider, target.model, target.num_ctx), []).append(name)
    if cfg.judge:
        key = (cfg.judge.provider, cfg.judge.model, cfg.judge.num_ctx)
        models.setdefault(key, []).append("the judge")
    for (provider_name, model, num_ctx), users in models.items():
        provider = None
        try:
            provider = make_provider(cfg.provider[provider_name])
            digest = await provider.digest(model)
            ping = Request(model, "", "ping", num_ctx=num_ctx, max_tokens=1)
            await asyncio.wait_for(provider.complete(ping), cfg.timeout)
            info = await provider.info(model)
            typer.echo(f"ok    {model} {digest[:12]} {info}  used by: {', '.join(users)}")
        except (ProviderError, TimeoutError, KeyError) as e:
            healthy = False
            typer.secho(f"FAIL  {model}: {e or 'timed out'}", fg=typer.colors.RED)
        finally:
            if provider:
                await provider.aclose()
    return healthy


@app.command()
def doctor(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Check that every suite's model is reachable, present and able to generate."""
    raise typer.Exit(0 if _execute(_doctor(_load(config))) else 1)


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
    files = (
        [] if prompts is None else [prompts] if prompts.is_file() else sorted(prompts.rglob("*"))
    )
    files = [f for f in files if f.is_file()]
    with _errors():
        cases = datasets.load(path)
        if prompts is not None and not files:
            # A mistyped path must not pass as "nothing leaked into the prompts".
            raise ValueError(f"no prompt files found at {prompts}")
    context = "\n".join(f.read_text(encoding="utf-8") for f in files)
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
    with _errors():
        splits = datasets.split(datasets.load(src), sizes, seed)
    for name, cases in splits.items():
        datasets.save(out / f"{name}.jsonl", cases)
        typer.echo(f"{name}: {len(cases)} cases, version {datasets.version(cases)}")


@judge_app.command("run")
def judge_run(
    suite: str,
    max_minutes: Annotated[float | None, typer.Option(help="Stop cleanly after this long.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Judge every stored sample that has no verdict yet. Slow, and resumable."""
    cfg = _load(config)
    db = store.connect(cfg.db_path)

    async def work() -> dict[str, int]:
        s = await run_suite(cfg, suite)
        score_suite(cfg, suite, s.fingerprint, db)
        return await judge_suite(cfg, suite, s.fingerprint, db, max_minutes=max_minutes)

    _show(_execute(work()))


def _agreement_rows(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("n", 0) < 2:
        return {"items": result.get("n", 0), "note": "too few items to measure agreement"}
    rows = {
        "items": result["n"],
        "agreement": result["agreement"],
        "kappa": f"{result['kappa']:.3f} [{result['kappa_lo']:.3f}, {result['kappa_hi']:.3f}]",
        "passes a true pass": result["sensitivity"],
        "fails a true fail": result["specificity"],
    }
    return {k: ("n/a" if v is None else v) for k, v in rows.items()}


@judge_app.command("probe")
def judge_probe(
    suite: str,
    probes: Annotated[Path, typer.Option(help="JSONL of constructed answers with known verdicts.")],
    alt_rubric: Annotated[
        Path | None, typer.Option(help="A reworded rubric, to measure sensitivity to phrasing.")
    ] = None,
    kind: Annotated[list[str] | None, typer.Option(help="Only these probe kinds.")] = None,
    misses: Annotated[bool, typer.Option(help="Show each probe the judge got wrong.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Test the judge on answers built to be right or wrong in known ways."""
    cfg = _load(config)
    with _errors():
        items = [
            json.loads(line) for line in probes.read_text(encoding="utf-8").split("\n") if line
        ]
    items = [item for item in items if not kind or item["kind"] in kind]
    result = _execute(probe(cfg, suite, items))
    for miss in result["misses"] if misses else []:
        typer.echo(f"MISS {miss['criterion']}/{miss['kind']} (should be {miss['truth']}):")
        typer.echo(f"  answer:    {miss['answer'][:200]}")
        typer.echo(f"  evidence:  {str(miss.get('evidence', ''))[:200]}")
        typer.echo(f"  reasoning: {str(miss.get('reasoning', ''))[:200]}")
    ok = result["failed"] == 0
    for criterion, found in result["criteria"].items():
        typer.secho(f"criterion: {criterion}", bold=True)
        _show({**_agreement_rows(found), **{f"  {k}": v for k, v in found["by_kind"].items()}})
        ok = ok and min(found["by_kind"].values()) >= 0.8
    typer.echo(f"calls without a usable verdict: {result['failed']}")
    if alt_rubric:
        reworded = _execute(probe(cfg, suite, items, str(alt_rubric.resolve())))
        for criterion, verdicts in result["verdicts"].items():
            other = reworded["verdicts"].get(criterion, [])
            if len(other) == len(verdicts):
                flips = sum(a != b for a, b in zip(verdicts, other, strict=True)) / len(verdicts)
                typer.echo(f"{criterion}: {flips:.1%} of verdicts change under the reworded rubric")
    typer.secho(
        "the judge handles every probe kind" if ok else "the judge fails at least one probe kind",
        fg="green" if ok else "red",
    )
    raise typer.Exit(0 if ok else 1)


@judge_app.command("calibrate")
def judge_calibrate(
    suite: str,
    rule: Annotated[
        str | None, typer.Option(help="A rule-based metric to compare with, e.g. fact.pass.")
    ] = None,
    criterion: Annotated[
        str, typer.Option(help="The judge criterion the rule matches.")
    ] = "correct",
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Compare the judge's stored verdicts with human labels, and optionally with a rule."""
    cfg = _load(config)
    db = store.connect(cfg.db_path)
    s, _ = _execute(_ready(cfg, suite, db))
    with _errors():
        found = calibration(cfg, suite, s.fingerprint, db)
    for name, result in found.items():
        typer.secho(f"criterion: {name} (judge mean {result['judge_mean']})", bold=True)
        rows = _agreement_rows(result)
        if "human_estimate" in result:
            lo, hi = result["human_lo"], result["human_hi"]
            rows["human-equivalent score"] = f"{result['human_estimate']:.3f} [{lo:.3f}, {hi:.3f}]"
        typer.echo("against human labels:")
        _show(rows)
    if rule:
        with _errors():
            versus = against_rule(db, cfg, suite, s.fingerprint, criterion, rule)
        typer.secho(f"judge '{criterion}' against rule {rule}:", bold=True)
        _show(_agreement_rows(versus))
        typer.echo(f"disagreements: {len(versus.get('disagreements', []))}")


@app.command()
def label(
    suite: str,
    n: Annotated[int, typer.Option(help="How many answers to label.")] = 120,
    labeller: Annotated[str, typer.Option(help="Your name, stored with each label.")] = "me",
    seed: int = 0,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Label stored answers by hand, blind: no model name and no judge verdict are shown."""
    import random

    cfg = _load(config)
    db = store.connect(cfg.db_path)
    with _errors():
        judge = Judge(cfg)
    s = _execute(run_suite(cfg, suite))
    cases = {c.hash: c for c in datasets.load(cfg.root / cfg.get_suite(suite).dataset)}
    rows = db.execute(
        "SELECT case_hash, rep, output FROM samples WHERE fingerprint=? AND status='ok' "
        "AND rep<? ORDER BY case_hash, rep",
        (s.fingerprint, cfg.get_suite(suite).reps),
    ).fetchall()
    rows = [r for r in rows if r["case_hash"] in cases]
    random.Random(seed).shuffle(rows)  # a random subset, so corrected estimates stay valid
    done = {
        tuple(r)
        for r in db.execute(
            "SELECT case_hash, rep, criterion FROM human_labels WHERE fingerprint=? AND labeller=?",
            (s.fingerprint, labeller),
        )
    }
    for index, row in enumerate(rows[:n], 1):
        case = cases[row["case_hash"]]
        pending = [c for c in judge.criteria if (row["case_hash"], row["rep"], c) not in done]
        if not pending:
            continue
        typer.secho(f"--- answer {index} of {min(n, len(rows))} ---", bold=True)
        for key, value in case.input.items():
            typer.echo(f"{key.upper()}:")
            typer.echo(str(value))
        reference = (
            case.expected.get("reference") if isinstance(case.expected, dict) else case.expected
        )
        typer.echo(f"REFERENCE: {reference}")
        typer.secho(f"ANSWER: {row['output'].strip()}", fg="cyan")
        for criterion in pending:
            typer.echo(f"{criterion}: {judge.criteria[criterion]}")
            choice = typer.prompt("  [y]es / [n]o / [s]kip / [q]uit").strip().lower()[:1]
            if choice == "q":
                raise typer.Exit()
            if choice in ("y", "n"):
                entry = {
                    "fingerprint": s.fingerprint,
                    "case_hash": row["case_hash"],
                    "rep": row["rep"],
                    "criterion": criterion,
                    "value": int(choice == "y"),
                    "labeller": labeller,
                    "created_at": now(),
                }
                store.insert(db, "human_labels", [entry], "OR REPLACE")
    typer.echo("done. Run `tripwire judge calibrate` to compare the judge with your labels.")


@dataset_app.command("perturb")
def perturb_cmd(
    src: Path,
    out: Path,
    kind: Annotated[
        list[str] | None,
        typer.Option(help="typo, lowercase, distractor, shuffle_lines; repeatable."),
    ] = None,
    field: Annotated[
        str | None, typer.Option(help="Input field to change (default: the last).")
    ] = None,
    paraphrase_model: Annotated[
        str | None, typer.Option(help="Also add model-written paraphrases, using this model.")
    ] = None,
    provider: str = "local",
    seed: int = 0,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Write the cases plus variants whose answer must not change: a robustness set."""
    with _errors():
        cases = datasets.load(src)
        result = generate.perturb(cases, kind or ["typo", "lowercase", "distractor"], field, seed)
    if paraphrase_model:
        backend = make_provider(_load(config).provider[provider])

        async def reword() -> list[Case]:
            try:
                return await generate.paraphrase(backend, paraphrase_model, cases, field)
            finally:
                await backend.aclose()

        result += _execute(reword())
    datasets.save(out, result)
    counts: dict[str, int] = {}
    for case in result:
        counts[case.tags[-1]] = counts.get(case.tags[-1], 0) + 1
    _show({**counts, "written to": out})


@dataset_app.command("gen")
def gen_cmd(
    dataset: Path,
    seed_file: Path,
    out: Path,
    model: Annotated[str, typer.Option(help="Model that drafts the cases.")],
    verifier: Annotated[
        str | None, typer.Option(help="Model that screens them (default: same).")
    ] = None,
    n: Annotated[int, typer.Option(help="Cases to ask for.")] = 10,
    examples: Annotated[int, typer.Option(help="Existing cases shown as the format.")] = 5,
    provider: str = "local",
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Draft candidate cases from seed material. They still need `dataset review`."""
    cfg = _load(config)
    with _errors():
        existing = datasets.load(dataset)
        queued = datasets.load(out) if out.exists() else []
        seed_text = seed_file.read_text(encoding="utf-8")
    backend = make_provider(cfg.provider[provider])

    async def work() -> tuple[list[Case], dict[str, int]]:
        try:
            return await generate.draft(
                backend,
                model,
                verifier or model,
                existing[:examples],
                seed_text,
                n,
                existing + queued,
            )
        finally:
            await backend.aclose()

    kept, dropped = _execute(work())
    datasets.save(out, queued + kept)
    _show({"drafted and kept": len(kept), **{f"dropped: {k}": v for k, v in dropped.items()}})
    typer.echo(f"review them with: tripwire dataset review {out} --into {dataset}")


@dataset_app.command("import-tracelens")
def import_tracelens_cmd(
    out: Path,
    url: Annotated[
        str, typer.Option(help="Address of the TraceLens API.")
    ] = "http://localhost:8000",
    limit: Annotated[int, typer.Option(help="Most recent traces to look at.")] = 200,
) -> None:
    """Turn failures diagnosed by TraceLens into candidate regression cases."""
    queued = datasets.load(out) if out.exists() else []
    known = {c.provenance.get("trace_id") for c in queued}
    found = _execute(generate.import_tracelens(url, limit))
    fresh = [c for c in found if c.provenance["trace_id"] not in known]
    datasets.save(out, queued + fresh)
    typer.echo(f"{len(found)} diagnosed failures, {len(fresh)} new candidates written to {out}")
    typer.echo("each needs an expected answer: tripwire dataset review ... --into <dataset>")


@dataset_app.command("review")
def review_cmd(
    candidates: Path,
    into: Annotated[Path, typer.Option(help="Dataset file that approved cases are added to.")],
    reviewer: Annotated[str, typer.Option(help="Your name, stored with each approval.")] = "me",
) -> None:
    """Approve, correct or reject candidate cases. Only approved ones enter the dataset."""
    with _errors():
        queue = datasets.load(candidates)
        dataset = datasets.load(into) if into.exists() else []
    known = {c.hash for c in dataset}
    added = 0
    for index, candidate in enumerate(queue):
        if candidate.provenance.get("status") != "pending":
            continue
        typer.secho(
            f"--- candidate {index + 1} of {len(queue)} ({candidate.source}) ---", bold=True
        )
        for key, value in candidate.input.items():
            typer.echo(f"{key.upper()}: {value}")
        typer.echo(f"TAGS: {', '.join(candidate.tags)}")
        for key, value in candidate.provenance.items():
            if key != "status" and value is not None:
                typer.echo(f"{key}: {value}")
        missing = candidate.expected is None
        typer.secho(
            "EXPECTED: (none yet, you must supply it)"
            if missing
            else f"EXPECTED: {candidate.expected}",
            fg="cyan",
        )
        choice = typer.prompt("  [a]pprove / [e]dit expected / [r]eject / [s]kip / [q]uit")
        choice = choice.strip().lower()[:1]
        if choice == "q":
            break
        if choice not in ("a", "e", "r"):
            continue
        status = "rejected"
        if choice != "r":
            expected = None
            if choice == "e" or missing:
                expected = generate.parse_expected(typer.prompt("  expected answer"))
            case = generate.approve(candidate, reviewer, expected)
            status = "approved"
            if case.hash in known:
                typer.echo("  already in the dataset; not added again")
            else:
                dataset.append(case)
                known.add(case.hash)
                added += 1
        queue[index] = Case(
            input=candidate.input,
            expected=candidate.expected,
            tags=candidate.tags,
            source=candidate.source,
            provenance={**candidate.provenance, "status": status},
        )
        datasets.save(candidates, queue)  # after every decision, so quitting loses nothing
        datasets.save(into, dataset)
    left = sum(c.provenance.get("status") == "pending" for c in queue)
    typer.echo(f"{added} cases added to {into}; {left} candidates still pending")


def _two_stage_report(
    outcomes: Any, drop: float, discordant: float, alpha: float, margin: float, first: int
) -> None:
    total = len(outcomes)
    typer.echo(f"\nTwo-stage gate: {first} cases first, all {total} only if undecided")
    for label, effect in ((f"true drop {drop}", drop), ("no drop at all", 0.0)):
        one = stats.simulate(outcomes, effect, total, discordant, alpha, margin)
        two = stats.simulate(outcomes, effect, total, discordant, alpha, margin, first=first)
        saved = 1 - two["cases"] / total
        typer.echo(
            f"  {label}: decided at stage one {two['early']:.0%} of the time, "
            f"{two['cases']:.0f} cases on average ({saved:.0%} fewer)"
        )
        typer.echo(
            f"    REGRESSED {two['REGRESSED']:.3f} (single stage {one['REGRESSED']:.3f}) · "
            f"INCONCLUSIVE {two['INCONCLUSIVE']:.3f} (single stage {one['INCONCLUSIVE']:.3f})"
        )


@app.command()
def queue(
    suites: Annotated[
        list[str] | None, typer.Argument(help="Suites to run (default: all).")
    ] = None,
    match: Annotated[
        str | None, typer.Option(help="Only suites whose name fits this glob.")
    ] = None,
    max_minutes: Annotated[float | None, typer.Option(help="Total time budget.")] = None,
    judge: Annotated[bool, typer.Option(help="Also judge suites with a judged metric.")] = True,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Run several suites back to back, one model at a time, within one time budget."""
    cfg = _load(config)
    names = [n for n in (suites or list(cfg.suite)) if not match or fnmatch(n, match)]
    with _errors():
        if not names:
            raise ValueError("no suites to run")
        targets = {n: cfg.target(n) for n in names}
    # Group by model: switching models means unloading one and loading another.
    names.sort(key=lambda n: (targets[n].provider, targets[n].model, targets[n].num_ctx))
    db = store.connect(cfg.db_path)
    started = time.monotonic()

    def left() -> float | None:
        if max_minutes is None:
            return None
        return max(0.0, max_minutes - (time.monotonic() - started) / 60)

    async def work() -> list[Summary]:
        plan = [await run_suite(cfg, n, dry_run=True) for n in names]
        for s in plan:
            speed = db.execute(
                "SELECT avg(latency_ms) FROM samples WHERE model=?", (targets[s.suite].model,)
            ).fetchone()[0]
            eta = f"about {s.pending * speed / 60_000:.0f} min" if speed else "time unknown"
            typer.echo(
                f"{s.suite:<32} {targets[s.suite].model:<14} {s.pending:>6} to generate, {eta}"
            )
        done = []
        for name in names:

            def show(s: Summary) -> None:
                made = s.total - s.cached - s.pending
                if made and made % 100 == 0:
                    typer.echo(f"  {s.suite}: {made} of {s.total - s.cached}")

            s = await run_suite(cfg, name, max_minutes=left(), progress=show)
            score_suite(cfg, name, s.fingerprint, db)
            typer.echo(
                f"{name}: {sum(s.statuses.values())} generated, {s.cached} cached, "
                f"{s.errors} errors, {s.pending} pending"
            )
            done.append(s)
        for s in done if judge else []:
            if JUDGE in cfg.get_suite(s.suite).scorers:
                counts = await judge_suite(cfg, s.suite, s.fingerprint, db, max_minutes=left())
                typer.echo(f"{s.suite}: judge {counts}")
                s.pending += counts["pending"]
        return done

    results = _execute(work())
    minutes = (time.monotonic() - started) / 60
    unfinished = sum(s.pending for s in results)
    errors = sum(s.errors for s in results)
    typer.echo(f"finished in {minutes:.1f} min; {unfinished} still pending, {errors} errors")
    raise typer.Exit(1 if unfinished or errors else 0)


@app.command()
def usage(
    days: Annotated[int, typer.Option(help="How far back to look.")] = 14,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Model calls, tokens and model time by day, plus what the cache saved."""
    cfg = _load(config)
    db = store.connect(cfg.db_path)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = db.execute(
        "SELECT substr(created_at, 1, 10) AS day, model, count(*) AS calls, "
        "sum(prompt_tokens) AS prompt, sum(output_tokens) AS output, "
        "sum(latency_ms) / 3600000.0 AS hours FROM samples WHERE created_at >= ? "
        "GROUP BY 1, 2 ORDER BY 1, 2",
        (since,),
    ).fetchall()
    typer.echo(
        f"{'day':<12}{'model':<16}{'calls':>8}{'prompt tok':>13}{'output tok':>12}{'hours':>8}"
    )
    for r in rows:
        typer.echo(
            f"{r['day']:<12}{r['model']:<16}{r['calls']:>8}{r['prompt']:>13}{r['output']:>12}"
            f"{r['hours']:>8.2f}"
        )
    typer.echo(
        f"{'total':<28}{sum(r['calls'] for r in rows):>8}{sum(r['prompt'] for r in rows):>13}"
        f"{sum(r['output'] for r in rows):>12}{sum(r['hours'] for r in rows):>8.2f}"
    )
    verdicts = [
        json.loads(r["detail"] or "{}")
        for r in db.execute("SELECT detail FROM scores WHERE scorer='judge'")
    ]
    judged_hours = sum(v.get("ms") or 0 for v in verdicts) / 3_600_000
    typer.echo(f"judge: {len(verdicts)} verdicts stored, {judged_hours:.2f} hours of model time")
    counts = [
        json.loads(r["env"]).get("samples", {})
        for r in db.execute("SELECT env FROM runs WHERE started_at >= ?", (since,))
    ]
    cached = sum(c.get("cached", 0) for c in counts)
    generated = sum(c.get("generated", 0) for c in counts)
    if cached + generated:
        share = cached / (cached + generated)
        typer.echo(
            f"cache: {cached} of {cached + generated} requested samples were already stored "
            f"({share:.0%} of calls avoided)"
        )
