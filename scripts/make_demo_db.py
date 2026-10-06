"""Pack what a public copy of the dashboard needs into one small file.

    python scripts/make_demo_db.py          # writes demo/tripwire-demo.db.gz

Copies the runs of the benchmark's suites (experiments/zoo.toml) out of the local
database and leaves the rest behind: other suites, raw provider responses, error
messages, human labels, and everything in a run's environment except the runtime's
version. What remains is public data (Banking77), synthetic data, and model output.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

from tripwire import datasets, store
from tripwire.config import load_config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "tripwire-demo.db.gz"
KEPT_ENV = ("ollama", "gpu_share", "samples")


def main() -> None:
    cfg = load_config(ROOT / "experiments" / "zoo.toml")
    source = sqlite3.connect(cfg.db_path.resolve().as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.db"
        demo = store.connect(path)
        demo.execute("PRAGMA journal_mode=DELETE")  # one plain file, no side files
        counts = {}

        def copy(table: str, rows: list[dict]) -> None:
            store.insert(demo, table, rows, "OR IGNORE")
            counts[table] = counts.get(table, 0) + len(rows)

        for name, suite in cfg.suite.items():
            cases = datasets.load(cfg.root / suite.dataset)
            hashes = {c.hash for c in cases}
            runs = source.execute(
                "SELECT * FROM runs WHERE suite=? OR suite LIKE ?", (name, f"{name}@%")
            ).fetchall()
            for run in runs:
                env = json.loads(run["env"])
                kept = {k: env[k] for k in KEPT_ENV if k in env}
                copy("runs", [{**dict(run), "env": json.dumps(kept)}])
            for fingerprint in {r["fingerprint"] for r in runs}:
                copy(
                    "targets",
                    [
                        dict(r)
                        for r in source.execute(
                            "SELECT * FROM targets WHERE fingerprint=?", (fingerprint,)
                        )
                    ],
                )
                samples = source.execute(
                    "SELECT * FROM samples WHERE fingerprint=?", (fingerprint,)
                )
                copy(
                    "samples",
                    [{**dict(r), "response": "{}"} for r in samples if r["case_hash"] in hashes],
                )
                scores = source.execute("SELECT * FROM scores WHERE fingerprint=?", (fingerprint,))
                copy("scores", [dict(r) for r in scores if r["case_hash"] in hashes])
            stored = source.execute("SELECT * FROM cases").fetchall()
            copy("cases", [dict(r) for r in stored if r["case_hash"] in hashes])
            drift = source.execute("SELECT * FROM comparisons WHERE suite=?", (f"canary:{name}",))
            copy("comparisons", [dict(r) for r in drift])
        demo.commit()
        demo.execute("VACUUM")
        demo.close()
        OUT.parent.mkdir(exist_ok=True)
        with OUT.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
            packed.write(path.read_bytes())  # mtime=0: the same data gives the same bytes
        size = path.stat().st_size
    print(f"{counts}")
    print(
        f"{size / 1e6:.1f} MB, packed to {OUT.stat().st_size / 1e6:.1f} MB at {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
