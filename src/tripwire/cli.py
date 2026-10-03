"""Tripwire: a statistical regression gate for LLM systems."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer

from . import __version__, datasets, store
from .config import Config, load_config
from .providers import ProviderError, Request, make_provider
from .runner import Summary, run_suite

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect and prepare datasets.")
app.add_typer(dataset_app, name="dataset")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to tripwire.toml.")]
DEFAULT_CONFIG = Path("tripwire.toml")


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False,
) -> None:
    pass


def _show(rows: dict[str, Any]) -> None:
    width = max(len(k) for k in rows)
    for key, value in rows.items():
        typer.echo(f"{key:<{width}}  {value}")


def _execute[T](work: Coroutine[Any, Any, T]) -> T:
    try:
        return asyncio.run(work)
    except (ProviderError, ValueError, KeyError, FileNotFoundError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e


def _summary(s: Summary) -> dict[str, Any]:
    made = sum(s.statuses.values())
    return {
        "suite": s.suite,
        "fingerprint": s.fingerprint,
        "dataset": s.dataset_version,
        "samples": f"{s.total} wanted, {s.cached} cached, {made} generated, {s.pending} pending",
        "statuses": dict(s.statuses) or "-",
        "errors": s.errors,
        "tokens": f"{s.prompt_tokens} prompt, {s.output_tokens} output",
        "model time": f"{s.model_seconds:.1f} s"
        + (f" ({s.output_tokens / s.model_seconds:.1f} output tok/s)" if s.model_seconds else ""),
    }


@app.command()
def run(
    suite: str,
    limit: Annotated[int | None, typer.Option(help="Only the first N cases.")] = None,
    max_minutes: Annotated[float | None, typer.Option(help="Stop cleanly after this long.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Report what would run, call nothing.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Generate every missing sample for a suite. Existing samples are never regenerated."""
    s = _execute(
        run_suite(load_config(config), suite, limit=limit, max_minutes=max_minutes, dry_run=dry_run)
    )
    rows = _summary(s)
    if dry_run:
        eta = f"{s.projected_minutes:.1f} min" if s.projected_minutes is not None else "unknown"
        rows = {k: rows[k] for k in ("suite", "fingerprint", "dataset", "samples")}
        rows["projected"] = eta + " (from samples already stored)"
    _show(rows)
    raise typer.Exit(1 if s.errors else 0)


@app.command()
def bench(
    suite: str,
    n: Annotated[int, typer.Option(help="Cases to time.")] = 20,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Measure throughput on a few cases and project how long each split will take."""
    cfg = load_config(config)
    s = _execute(run_suite(cfg, suite, limit=n))
    db = store.connect(cfg.root / cfg.db)
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

    report = {
        "samples measured": len(rows),
        "seconds per case": f"{seconds:.2f}",
        "prompt speed": rate("prompt_eval_count", "prompt_eval_duration"),
        "output speed": rate("eval_count", "eval_duration"),
        "runtime": json.loads(env["env"]) if env else "-",
    }
    reps = cfg.suite[suite].reps
    for path in sorted((cfg.root / cfg.suite[suite].dataset).parent.glob("*.jsonl")):
        cases = len(datasets.load(path))
        report[f"projected {path.stem}"] = f"{cases * reps * seconds / 60:.1f} min ({cases} cases)"
    _show(report)


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
    cases = datasets.load(path)
    files = (
        [] if prompts is None else [prompts] if prompts.is_file() else sorted(prompts.rglob("*"))
    )
    context = "\n".join(f.read_text(encoding="utf-8") for f in files if f.is_file())
    report = datasets.lint(cases, context)
    near = report.pop("near_duplicates")
    ok = report.pop("ok")
    report["near_duplicates (warning)"] = len(near)
    if embed:
        pairs = datasets.embedding_near_duplicates(cases, base_url)
        report["embedding_near_duplicates (warning)"] = len(pairs)
    _show(report)
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
    sizes = {name: int(count) for name, count in (item.split("=") for item in size)}
    for name, cases in datasets.split(datasets.load(src), sizes, seed).items():
        datasets.save(out / f"{name}.jsonl", cases)
        typer.echo(f"{name}: {len(cases)} cases, version {datasets.version(cases)}")
