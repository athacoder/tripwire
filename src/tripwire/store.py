"""SQLite storage. Samples are keyed by what produced them, not by which run asked."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Append-only: a database at user_version N has had the first N entries applied.
MIGRATIONS = [
    """
    CREATE TABLE cases(
        case_hash TEXT PRIMARY KEY, input TEXT NOT NULL, expected TEXT,
        tags TEXT NOT NULL, source TEXT NOT NULL);
    CREATE TABLE dataset_cases(
        dataset_version TEXT NOT NULL, case_hash TEXT NOT NULL,
        PRIMARY KEY(dataset_version, case_hash));
    CREATE TABLE targets(fingerprint TEXT PRIMARY KEY, spec TEXT NOT NULL);
    CREATE TABLE samples(
        fingerprint TEXT NOT NULL, case_hash TEXT NOT NULL, rep INTEGER NOT NULL,
        status TEXT NOT NULL, output TEXT NOT NULL, model TEXT NOT NULL, seed INTEGER NOT NULL,
        prompt_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
        latency_ms REAL NOT NULL, attempts INTEGER NOT NULL,
        response TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY(fingerprint, case_hash, rep));
    CREATE TABLE errors(
        id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL, case_hash TEXT NOT NULL,
        rep INTEGER NOT NULL, attempts INTEGER NOT NULL, kind TEXT NOT NULL,
        message TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE runs(
        run_id TEXT PRIMARY KEY, suite TEXT NOT NULL, dataset_version TEXT NOT NULL,
        fingerprint TEXT NOT NULL, reps INTEGER NOT NULL, git_sha TEXT, env TEXT NOT NULL,
        started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL);
    """,
    # Scores carry the scorer's version: improving a scorer adds rows, it never rewrites history.
    """
    CREATE TABLE scores(
        fingerprint TEXT NOT NULL, case_hash TEXT NOT NULL, rep INTEGER NOT NULL,
        scorer TEXT NOT NULL, version TEXT NOT NULL, metric TEXT NOT NULL,
        value REAL NOT NULL, detail TEXT,
        PRIMARY KEY(fingerprint, case_hash, rep, scorer, version, metric));
    CREATE TABLE comparisons(
        id INTEGER PRIMARY KEY, suite TEXT NOT NULL, base TEXT NOT NULL, head TEXT NOT NULL,
        verdict TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
    """,
]


def connect(path: Path | str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    done = db.execute("PRAGMA user_version").fetchone()[0]
    for number, script in enumerate(MIGRATIONS[done:], done + 1):
        db.executescript(script)
        db.execute(f"PRAGMA user_version={number}")
    return db


def insert(db: sqlite3.Connection, table: str, rows: list[dict[str, Any]], mode: str = "") -> None:
    """Insert rows in one transaction. `mode` is "", "OR IGNORE" or "OR REPLACE"."""
    if not rows:
        return
    columns = ",".join(rows[0])
    marks = ",".join("?" * len(rows[0]))
    db.executemany(
        f"INSERT {mode} INTO {table}({columns}) VALUES({marks})", [tuple(r.values()) for r in rows]
    )
    db.commit()
