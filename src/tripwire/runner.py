"""Execute a suite: produce every missing sample, and nothing that already exists."""

from __future__ import annotations

import asyncio
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from . import __version__, datasets, store
from .config import Config
from .models import Case
from .providers import Provider, ProviderError, make_provider
from .targets import Target, seed_for


@dataclass
class Summary:
    suite: str
    fingerprint: str
    dataset_version: str
    total: int = 0
    cached: int = 0
    pending: int = 0  # not attempted yet (dry run, or the time budget ran out)
    errors: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    prompt_tokens: int = 0
    output_tokens: int = 0
    model_seconds: float = 0.0
    projected_minutes: float | None = None


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_sha(root: Path) -> str | None:
    try:
        out = [
            subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout.strip()
            for cmd in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"])
        ]
    except (OSError, subprocess.CalledProcessError):
        return None  # not a git checkout, or git is missing: the run is still valid
    return out[0] + ("-dirty" if out[1] else "")


def _provider_kind(cfg: Config, name: str) -> str:
    tcfg = cfg.target(name)
    if tcfg.kind != "prompt":
        return ""
    if tcfg.provider not in cfg.provider:
        raise ValueError(f"target uses unknown provider {tcfg.provider!r}")
    return cfg.provider[tcfg.provider].kind


def offline_target(cfg: Config, name: str) -> Target:
    """A suite's target without contacting any backend: enough for its static key."""
    return Target(cfg.target(name), cfg.root, Provider(), _provider_kind(cfg, name))


async def resolve(cfg: Config, name: str, provider: Provider | None = None) -> Target:
    """Build a suite's target, asking the backend for the model digest it needs."""
    tcfg, kind = cfg.target(name), _provider_kind(cfg, name)
    provider = provider or (make_provider(cfg.provider[tcfg.provider]) if kind else Provider())
    try:
        return Target(tcfg, cfg.root, provider, kind, await provider.digest(tcfg.model))
    except BaseException:
        await provider.aclose()
        raise


async def run_suite(
    cfg: Config,
    name: str,
    *,
    limit: int | None = None,
    max_minutes: float | None = None,
    dry_run: bool = False,
    provider: Provider | None = None,
) -> Summary:
    suite = cfg.get_suite(name)
    cases = datasets.load(cfg.root / suite.dataset)[:limit]
    target = await resolve(cfg, name, provider)
    db = store.connect(cfg.db_path)
    try:
        target.guard(cases)
        fp, model = target.fingerprint, target.cfg.model
        have = {
            tuple(r)
            for r in db.execute("SELECT case_hash, rep FROM samples WHERE fingerprint=?", (fp,))
        }
        todo = [(c, rep) for c in cases for rep in range(suite.reps) if (c.hash, rep) not in have]
        total = len(cases) * suite.reps
        s = Summary(name, fp, datasets.version(cases), total, total - len(todo), len(todo))
        if dry_run:
            mean = db.execute(
                "SELECT avg(latency_ms) FROM samples WHERE fingerprint=?", (fp,)
            ).fetchone()[0]
            s.projected_minutes = mean * len(todo) / 60_000 if mean else None
            return s

        # A run row still marked "running" belongs to a process that was killed.
        db.execute(
            "UPDATE runs SET status='aborted' WHERE status='running' AND fingerprint=?", (fp,)
        )
        store.insert(
            db, "targets", [{"fingerprint": fp, "spec": json.dumps(target.spec)}], "OR IGNORE"
        )
        store.insert(
            db,
            "cases",
            [
                {
                    "case_hash": c.hash,
                    "input": json.dumps(c.input),
                    "expected": json.dumps(c.expected),
                    "tags": json.dumps(c.tags),
                    "source": c.source,
                }
                for c in cases
            ],
            "OR REPLACE",  # tags may have changed; identity has not
        )
        store.insert(
            db,
            "dataset_cases",
            [{"dataset_version": s.dataset_version, "case_hash": c.hash} for c in cases],
            "OR IGNORE",
        )
        run_id = uuid4().hex[:12]
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tripwire": __version__,
            "provider": target.spec["provider_kind"] or target.cfg.kind,
        }
        store.insert(
            db,
            "runs",
            [
                {
                    "run_id": run_id,
                    "suite": name,
                    "dataset_version": s.dataset_version,
                    "fingerprint": fp,
                    "reps": suite.reps,
                    "git_sha": _git_sha(cfg.root),
                    "env": json.dumps(env),
                    "started_at": now(),
                    "status": "running",
                }
            ],
        )

        deadline = math.inf if max_minutes is None else time.monotonic() + max_minutes * 60
        gate = asyncio.Semaphore(cfg.concurrency)

        async def one(case: Case, rep: int) -> None:
            async with gate:
                if time.monotonic() >= deadline:
                    return
                key = {"fingerprint": fp, "case_hash": case.hash, "rep": rep}
                for attempt in range(1, cfg.retries + 2):
                    start = time.perf_counter()
                    try:
                        resp = await asyncio.wait_for(target.run(case, rep), cfg.timeout)
                        break
                    except Exception as e:
                        if getattr(e, "retryable", False) and attempt <= cfg.retries:
                            delay = min(30.0, cfg.backoff * 2 ** (attempt - 1))
                            await asyncio.sleep(delay * random.uniform(0.5, 1.5))
                            continue
                        # A failed request is not a wrong answer: it never becomes a sample.
                        # That includes a malformed reply or a crash inside a python target;
                        # one bad case must not take the whole run down.
                        kind = (
                            "timeout"
                            if isinstance(e, TimeoutError)
                            else getattr(e, "kind", "target_error")
                        )
                        error = {
                            **key,
                            "attempts": attempt,
                            "kind": kind,
                            "message": f"{type(e).__name__}: {e}"[:500],
                            "created_at": now(),
                        }
                        store.insert(db, "errors", [error])
                        s.errors += 1
                        s.pending -= 1
                        return
                wall_ms = (time.perf_counter() - start) * 1000  # final attempt only
                if target.cfg.kind == "prompt" and not resp.model.startswith(model):
                    raise ProviderError(
                        f"asked for {model!r}, served by {resp.model!r}", kind="model_mismatch"
                    )
                status = {"length": "truncated", "refusal": "refusal"}.get(resp.stop, "ok")
                latency = resp.latency_ms if resp.latency_ms is not None else wall_ms
                sample = {
                    **key,
                    "status": status,
                    "output": resp.text,
                    "model": resp.model,
                    "seed": seed_for(case.hash, rep),
                    "prompt_tokens": resp.prompt_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_ms": latency,
                    "attempts": attempt,
                    "response": json.dumps(resp.raw),
                    "created_at": now(),
                }
                store.insert(db, "samples", [sample])
                s.statuses[status] += 1
                s.prompt_tokens += resp.prompt_tokens
                s.output_tokens += resp.output_tokens
                s.model_seconds += latency / 1000
                s.pending -= 1

        status = "aborted"
        tasks = [asyncio.ensure_future(one(c, rep)) for c, rep in todo]
        try:
            await asyncio.gather(*tasks)
            if await target.provider.digest(model) != target.spec["digest"]:
                raise ProviderError("model digest changed during the run", kind="model_mismatch")
            env.update(await target.provider.info(model))
            status = "partial" if s.pending or s.errors else "complete"
        finally:
            # On a fatal error, stop the siblings before the database handle goes away.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            db.execute(
                "UPDATE runs SET finished_at=?, status=?, env=? WHERE run_id=?",
                (now(), status, json.dumps(env), run_id),
            )
            db.commit()
        return s
    finally:
        db.close()
        await target.aclose()
