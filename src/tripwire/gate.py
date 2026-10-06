"""The gate: does the working tree (or a ref) regress against a base git ref?

Samples can travel as bundles, small files committed next to the code. A machine with the
model generates them; CI, which has no GPU, re-checks the statistics from the bundles and
never calls a model.

The same comparison answers two more questions: which commit first made a suite worse
(bisect), and whether the model's behaviour moved while the project stood still (canary).
"""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from . import __version__, datasets, store
from .compare import blocked, compare
from .config import Config, load_config
from .judge import judge_suite
from .models import Case
from .providers import Provider
from .runner import now, offline_target, run_suite
from .scorers import JUDGE, score_suite

SAMPLE_COLUMNS = (
    "case_hash, rep, status, output, model, seed, prompt_tokens, output_tokens, "
    "latency_ms, attempts, created_at"
)


def config_at(cfg: Config, ref: str, into: Path) -> Config:
    """The project as it was at a git ref, unpacked into a scratch directory."""
    if ref.startswith("-"):  # git would read it as an option, e.g. --output=<any file>
        raise ValueError(f"not a git ref: {ref!r}")
    done = subprocess.run(
        ["git", "archive", "--format=tar", ref], cwd=cfg.root, capture_output=True, check=False
    )
    if done.returncode:
        raise ValueError(f"cannot read git ref {ref!r}: {done.stderr.decode().strip()}")
    into.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as archive:
        archive.extractall(into, filter="data")
    at = load_config(into / "tripwire.toml")
    at.db = str(cfg.db_path)  # old files, live database
    return at


def _bundle_dir(cfg: Config, dataset_version: str) -> Path:
    return cfg.root / cfg.bundles / dataset_version


def export_bundle(
    db: sqlite3.Connection,
    cfg: Config,
    static_key: str,
    fingerprint: str,
    cases: list[Case],
    reps: int,
) -> Path:
    """Write one target's samples for one dataset version as a deterministic gzip file."""
    version = datasets.version(cases)
    wanted = {c.hash for c in cases}
    rows = db.execute(
        f"SELECT {SAMPLE_COLUMNS} FROM samples WHERE fingerprint=? AND rep<? "
        "ORDER BY case_hash, rep",
        (fingerprint, reps),  # only what the suite uses: bundles are committed, keep them small
    )
    spec = db.execute("SELECT spec FROM targets WHERE fingerprint=?", (fingerprint,)).fetchone()
    header = {
        "fingerprint": fingerprint,
        "static_key": static_key,
        "dataset_version": version,
        "spec": json.loads(spec["spec"]),
        "tripwire": __version__,
    }
    # Judge verdicts cost model time and CI cannot reproduce them, so they travel too.
    # Rule-based scores are left out: CI recomputes those from the outputs.
    verdicts = db.execute(
        "SELECT case_hash, rep, scorer, version, metric, value, detail FROM scores "
        "WHERE fingerprint=? AND scorer=? AND rep<? ORDER BY case_hash, rep, metric, version",
        (fingerprint, JUDGE, reps),
    )
    lines = [header, *(dict(r) for r in (*rows, *verdicts) if r["case_hash"] in wanted)]
    path = _bundle_dir(cfg, version) / f"{static_key}.{fingerprint}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n"
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
        out.write(body.encode())  # mtime=0: same samples, same bytes, no spurious diffs
    return path


def import_bundle(
    db: sqlite3.Connection, cfg: Config, static_key: str, dataset_version: str
) -> str | None:
    """Load the bundle matching a target into the database; returns its fingerprint."""
    found = sorted(_bundle_dir(cfg, dataset_version).glob(f"{static_key}.*.jsonl.gz"))
    if not found:
        return None
    with gzip.open(found[-1], "rt", encoding="utf-8") as lines:
        header, *rows = (json.loads(line) for line in lines)
    samples = [r for r in rows if "scorer" not in r]
    verdicts = [r for r in rows if "scorer" in r]
    fp = header["fingerprint"]
    store.insert(
        db, "targets", [{"fingerprint": fp, "spec": json.dumps(header["spec"])}], "OR IGNORE"
    )
    store.insert(
        db, "samples", [{"fingerprint": fp, "response": "{}", **s} for s in samples], "OR IGNORE"
    )
    store.insert(db, "scores", [{"fingerprint": fp, **v} for v in verdicts], "OR IGNORE")
    return fp


async def gate(
    cfg: Config,
    name: str,
    base_ref: str,
    head_ref: str | None = None,
    *,
    verify_only: bool = False,
    bundle: bool = False,
    provider: Provider | None = None,
) -> tuple[dict[str, Any], bool]:
    """Returns the comparison result and whether it should block a merge."""
    with TemporaryDirectory() as tmp:
        head_cfg = config_at(cfg, head_ref, Path(tmp) / "head") if head_ref else cfg
        base_cfg = config_at(cfg, base_ref, Path(tmp) / "base")
        suite = head_cfg.get_suite(name)
        if name not in base_cfg.suite:
            return {"verdict": "UNCHANGED", "reason": f"suite {name!r} is new at head."}, False

        # The base *target* runs on the head *dataset* and is scored by the head scorers:
        # the old prompt on today's cases, judged by today's rules.
        dataset = (head_cfg.root / suite.dataset).resolve()
        base_cfg.suite[name] = suite.model_copy(
            update={"target": base_cfg.suite[name].target, "dataset": str(dataset)}
        )
        cases = datasets.load(dataset)
        version = datasets.version(cases)
        sides = (base_cfg, head_cfg)
        keys = [offline_target(c, name).static_key for c in sides]
        if keys[0] == keys[1]:
            return {"verdict": "UNCHANGED"}, False  # nothing that affects output changed

        # A gate that may escalate looks first at a fixed, stratified subset, and at the
        # whole set only if that cannot decide. Each look gets half the error budget, so
        # looking twice cannot inflate the false-alarm rate (a union bound).
        staged = suite.on_inconclusive == "escalate" and suite.first_stage < len(cases)
        first = (
            datasets.split(cases, {"first": suite.first_stage}, seed=0)["first"] if staged else []
        )
        stages = [(first, suite.alpha / 2), (cases, suite.alpha / 2)] if staged else [(cases, None)]

        db = store.connect(cfg.db_path)
        try:
            fingerprints: list[str] = []
            if verify_only:
                for side, key in zip(sides, keys, strict=True):
                    fp = import_bundle(db, cfg, key, version)
                    if fp is None:
                        which = "base" if side is base_cfg else "head"
                        reason = (
                            f"no sample bundle for the {which} target (key {key}, dataset "
                            f"{version}). Run `tripwire gate {name} --base {base_ref} --bundle` "
                            f"on a machine with the model and commit the {cfg.bundles}/ directory."
                        )
                        return {"verdict": "INVALID", "reason": reason}, True
                    score_suite(side, name, fp, db)
                    fingerprints.append(fp)

            for number, (subset, alpha) in enumerate(stages, 1):
                if not verify_only:
                    fingerprints = []
                    wanted = {c.hash for c in subset}
                    for side, key in zip(sides, keys, strict=True):
                        run = await run_suite(side, name, only=wanted, provider=provider)
                        fp = run.fingerprint
                        score_suite(side, name, fp, db)
                        if JUDGE in suite.scorers:  # both sides are judged by the head's rubric
                            await judge_suite(head_cfg, name, fp, db)
                        if bundle:
                            export_bundle(db, cfg, key, fp, cases, suite.reps)
                        fingerprints.append(fp)
                result = compare(db, suite, subset, fingerprints[0], fingerprints[1], alpha=alpha)
                result["stage"], result["stages"] = number, len(stages)
                if result["verdict"] != "INCONCLUSIVE":
                    break  # decided (or invalid): the remaining cases are never run

            is_blocked = blocked(result, suite)
            store.insert(
                db,
                "comparisons",
                [
                    {
                        "suite": name,
                        "base": fingerprints[0],
                        "head": fingerprints[1],
                        "verdict": result["verdict"],
                        "result": json.dumps(result),
                        "created_at": now(),
                    }
                ],
            )
            return result, is_blocked
        finally:
            db.close()


def _git(cfg: Config, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cfg.root, capture_output=True, text=True, check=False)
    if done.returncode:
        raise ValueError(f"git {args[0]} failed: {done.stderr.strip()}")
    return done.stdout.strip()


async def bisect(
    cfg: Config,
    name: str,
    good: str,
    bad: str = "HEAD",
    provider: Provider | None = None,
    report: Callable[[str], None] = lambda line: None,
) -> dict[str, Any]:
    """Find the first commit after `good` at which the gate blocks the suite.

    Each probe is an ordinary gate run of one commit against `good`, read from git, so the
    working tree is never touched. A commit the gate cannot decide (INCONCLUSIVE or
    INVALID) cannot split the range, and its neighbours are tried instead. The sample
    cache bounds the work: a commit that leaves the target alone costs nothing.
    """
    if good.startswith("-") or bad.startswith("-"):
        raise ValueError("not a git ref")
    commits = _git(cfg, "rev-list", "--first-parent", "--reverse", f"{good}..{bad}").split()
    if not commits:
        raise ValueError(f"there are no commits after {good} on the way to {bad}")
    seen: dict[int, bool | None] = {}

    async def blocked_at(i: int) -> bool | None:
        if i not in seen:
            result, is_blocked = await gate(cfg, name, good, commits[i], provider=provider)
            decided = result["verdict"] not in ("INCONCLUSIVE", "INVALID")
            seen[i] = is_blocked if decided else None
            subject = _git(cfg, "log", "-1", "--format=%s", commits[i])
            report(f"{commits[i][:10]}  {result['verdict']:<12}  {subject}")
        return seen[i]

    if not await blocked_at(len(commits) - 1):
        raise ValueError(f"{bad} is not blocked against {good}, so there is nothing to find")
    low, high = -1, len(commits) - 1  # commits[low] passes (-1 is `good`); commits[high] is blocked
    while high - low > 1:
        middle = (low + high) // 2
        for probe in sorted(range(low + 1, high), key=lambda i: abs(i - middle)):
            state = await blocked_at(probe)
            if state is not None:
                break
        else:
            break  # everything in between is undecided: the range cannot get narrower
        low, high = (low, probe) if state else (probe, high)
    return {
        "first_bad": commits[high] if high - low == 1 else None,
        "candidates": commits[low + 1 : high + 1],
        "tested": len(seen),
        "commits": len(commits),
    }


async def canary(
    cfg: Config, name: str, size: int = 150, pin: bool = False, provider: Provider | None = None
) -> tuple[dict[str, Any], bool]:
    """Has the model's behaviour moved while nothing in the project changed?

    Reruns a fixed subset of the cases under a fresh salt, so nothing comes from the
    cache, and compares the answers with a pinned reference run of the same target. Seeds
    come from the cases, so an unchanged runtime gives the same answers again. A
    difference means something underneath moved: the runtime's version, the model behind
    a tag, a driver.

    The reference is pinned by fingerprint the first time, not looked up again: a model
    pulled anew has a new digest, and the point is to compare against the old one.
    """
    suite = cfg.get_suite(name)
    cases = datasets.load(cfg.root / suite.dataset)
    subset = datasets.split(cases, {"canary": min(size, len(cases))}, seed=0)["canary"]
    wanted, key = {c.hash for c in subset}, f"canary:{name}"
    db = store.connect(cfg.db_path)

    async def run(variant: str = "") -> str:
        fingerprint = (
            await run_suite(cfg, name, only=wanted, provider=provider, variant=variant)
        ).fingerprint
        score_suite(cfg, name, fingerprint, db)
        if JUDGE in suite.scorers:
            await judge_suite(cfg, name, fingerprint, db)
        return fingerprint

    def runtime(fingerprint: str) -> dict[str, Any]:
        """What generated a fingerprint's samples: the environment of its first run."""
        row = db.execute(
            "SELECT env FROM runs WHERE fingerprint=? ORDER BY started_at", (fingerprint,)
        ).fetchone()
        return json.loads(row["env"]) if row else {}

    def record(base: str, head: str, verdict: str, result: dict[str, Any], at: str) -> None:
        row = {"suite": key, "base": base, "head": head, "verdict": verdict}
        store.insert(db, "comparisons", [{**row, "result": json.dumps(result), "created_at": at}])

    try:
        pinned = db.execute(
            "SELECT base, created_at FROM comparisons WHERE suite=? AND verdict='PINNED' "
            "ORDER BY id DESC",
            (key,),
        ).fetchone()
        if pinned is None or pin:
            reference, since = await run(), now()
            record(reference, reference, "PINNED", {"env": runtime(reference)}, since)
        else:
            reference, since = pinned["base"], pinned["created_at"]
        # The salt has to be new every time, or a second canary in the same second would
        # read the first one's samples back and report that nothing had changed.
        fresh = await run(f"canary {now()} {uuid4().hex[:6]}")
        result = compare(db, suite, subset, reference, fresh)
        answers = [
            {
                r["case_hash"]: r["output"]
                for r in db.execute(
                    "SELECT case_hash, output FROM samples WHERE fingerprint=? AND rep=0", (fp,)
                )
            }
            for fp in (reference, fresh)
        ]
        result.update(
            subset=len(subset),
            identical=sum(h in answers[0] and answers[0][h] == answers[1].get(h) for h in wanted),
            pinned_at=since,
            env={"reference": runtime(reference), "now": runtime(fresh)},
        )
        record(reference, fresh, result["verdict"], result, now())
        return result, blocked(result, suite)
    finally:
        db.close()
