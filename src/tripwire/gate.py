"""The gate: does the working tree (or a ref) regress against a base git ref?

Samples can travel as bundles, small files committed next to the code. A machine with the
model generates them; CI, which has no GPU, re-checks the statistics from the bundles and
never calls a model.
"""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import subprocess
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from . import __version__, datasets, store
from .compare import blocked, compare
from .config import Config, load_config
from .models import Case
from .providers import Provider
from .runner import now, offline_target, run_suite
from .scorers import score_suite

SAMPLE_COLUMNS = (
    "case_hash, rep, status, output, model, seed, prompt_tokens, output_tokens, "
    "latency_ms, attempts, created_at"
)


def config_at(cfg: Config, ref: str, into: Path) -> Config:
    """The project as it was at a git ref, unpacked into a scratch directory."""
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
    lines = [header, *(dict(r) for r in rows if r["case_hash"] in wanted)]
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
        header, *samples = (json.loads(line) for line in lines)
    fp = header["fingerprint"]
    store.insert(
        db, "targets", [{"fingerprint": fp, "spec": json.dumps(header["spec"])}], "OR IGNORE"
    )
    store.insert(
        db, "samples", [{"fingerprint": fp, "response": "{}", **s} for s in samples], "OR IGNORE"
    )
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

        db = store.connect(cfg.db_path)
        try:
            fingerprints = []
            for side, key in zip(sides, keys, strict=True):
                if verify_only:
                    fp = import_bundle(db, cfg, key, version)
                    if fp is None:
                        which = "base" if side is base_cfg else "head"
                        reason = (
                            f"no sample bundle for the {which} target (key {key}, dataset "
                            f"{version}). Run `tripwire gate {name} --base {base_ref} --bundle` "
                            f"on a machine with the model and commit the {cfg.bundles}/ directory."
                        )
                        return {"verdict": "INVALID", "reason": reason}, True
                else:
                    fp = (await run_suite(side, name, provider=provider)).fingerprint
                    if bundle:
                        export_bundle(db, cfg, key, fp, cases, suite.reps)
                score_suite(side, name, fp, db)
                fingerprints.append(fp)

            result = compare(db, suite, cases, fingerprints[0], fingerprints[1])
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
