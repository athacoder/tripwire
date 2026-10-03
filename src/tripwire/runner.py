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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_sha(root: Path) -> str | None:
    try:
        out = [
            subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout.strip()
            for cmd in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"])
        ]
    except (OSError, subprocess.CalledProcessError):
        return None
    return out[0] + ("-dirty" if out[1] else "")


async def run_suite(
    cfg: Config,
    name: str,
    *,
    limit: int | None = None,
    max_minutes: float | None = None,
    dry_run: bool = False,
    provider: Provider | None = None,
) -> Summary:
    suite, tcfg = cfg.suite[name], cfg.target(name)
    cases = datasets.load(cfg.root / suite.dataset)[:limit]
    is_prompt = tcfg.kind == "prompt"
    pcfg = cfg.provider[tcfg.provider]
    provider = provider or (make_provider(pcfg) if is_prompt else Provider())
    target = None
    try:
        digest = await provider.digest(tcfg.model)
        target = Target(tcfg, cfg.root, provider, pcfg.kind if is_prompt else "", digest)
        target.guard(cases)
        fp = target.fingerprint

        db = store.connect(cfg.root / cfg.db)
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
                    "started_at": _now(),
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
                    except (ProviderError, TimeoutError) as e:
                        if getattr(e, "retryable", False) and attempt <= cfg.retries:
                            delay = min(30.0, cfg.backoff * 2 ** (attempt - 1))
                            await asyncio.sleep(delay * random.uniform(0.5, 1.5))
                            continue
                        # A failed request is not a wrong answer: it never becomes a sample.
                        error = {
                            **key,
                            "attempts": attempt,
                            "kind": getattr(e, "kind", "timeout"),
                            "message": str(e)[:500],
                            "created_at": _now(),
                        }
                        store.insert(db, "errors", [error])
                        s.errors += 1
                        s.pending -= 1
                        return
                wall_ms = (time.perf_counter() - start) * 1000  # final attempt only
                if is_prompt and not resp.model.startswith(tcfg.model):
                    raise ProviderError(
                        f"asked for {tcfg.model!r}, served by {resp.model!r}", kind="model_mismatch"
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
                    "created_at": _now(),
                }
                store.insert(db, "samples", [sample])
                s.statuses[status] += 1
                s.prompt_tokens += resp.prompt_tokens
                s.output_tokens += resp.output_tokens
                s.model_seconds += latency / 1000
                s.pending -= 1

        status = "aborted"
        try:
            await asyncio.gather(*(one(c, rep) for c, rep in todo))
            if await provider.digest(tcfg.model) != digest:
                raise ProviderError("model digest changed during the run", kind="model_mismatch")
            env.update({"provider": pcfg.kind if is_prompt else tcfg.kind})
            env.update(await provider.info(tcfg.model))
            status = "partial" if s.pending or s.errors else "complete"
        finally:
            db.execute(
                "UPDATE runs SET finished_at=?, status=?, env=? WHERE run_id=?",
                (_now(), status, json.dumps(env), run_id),
            )
            db.commit()
            db.close()
        return s
    finally:
        await provider.aclose()
        if target:
            await target.aclose()
