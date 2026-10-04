"""Deterministic scorers, and the checks that a scorer deserves to be trusted.

A scorer maps (output, expected) to named metrics in [0, 1]. Each has a version: bump it
when its behaviour changes, so old and new scores are never compared with each other.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
from typing import Any

from . import datasets, store
from .config import Config, SuiteCfg
from .models import Case

Scorer = Callable[[str, Any, dict[str, Any]], dict[str, float]]


def normalise(text: Any) -> str:
    """Forgive formatting, not content: case, whitespace, code fences, quotes, end marks."""
    t = re.sub(r"^```\w*|```$", "", str(text).strip()).strip().lower()
    t = re.sub(r"[.!?,;:]+$", "", t.strip("\"'` "))
    return " ".join(t.split())


def _parse_json(output: str) -> Any:
    """The JSON in a reply, tolerating code fences and prose around it."""
    text = re.sub(r"^```\w*|```$", "", output.strip()).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.DOTALL)[:1]):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def _number(value: Any) -> float | None:
    found = re.findall(r"-?\d[\d,]*\.?\d*(?:[eE]-?\d+)?", str(value))
    try:
        return float(found[-1].replace(",", "")) if found else None  # the last number is the answer
    except ValueError:
        return None


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        number = _number(got)
        return number is not None and math.isclose(number, want, rel_tol=1e-6, abs_tol=1e-9)
    return normalise(got) == normalise(want)


def exact(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    """Match after normalisation. If the answer is wrapped in a sentence, it still passes
    when exactly one known label appears in it and that label is the expected one."""
    out, want = normalise(output), normalise(expected)
    if out == want:
        return {"pass": 1.0}
    found = {
        label for label in ctx["labels"] if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", out)
    }
    found = {a for a in found if not any(a != b and a in b for b in found)}  # keep the longest
    return {"pass": float(found == {want})}


def contains(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    wanted = expected if isinstance(expected, list) else [expected]
    return {"pass": float(all(normalise(w) in normalise(output) for w in wanted))}


def regex(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    return {"pass": float(re.search(str(expected), output) is not None)}


def numeric(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    return {"pass": float(_same(output, float(expected)))}


def json_valid(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    return {"pass": float(isinstance(_parse_json(output), (dict, list)))}


def field_f1(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    """Per-field agreement between a JSON reply and the expected record."""
    got = _parse_json(output)
    valid = isinstance(got, dict)
    got = got if valid else {}
    hits = sum(key in got and _same(got[key], value) for key, value in expected.items())
    precision = hits / len(got) if got else 0.0
    recall = hits / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {
        "valid": float(valid),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pass": float(recall == 1.0),
    }


def token_f1(output: str, expected: Any, ctx: dict[str, Any]) -> dict[str, float]:
    got, want = Counter(normalise(output).split()), Counter(normalise(expected).split())
    hits = sum((got & want).values())
    if not hits:
        return {"f1": 0.0}
    precision, recall = hits / sum(got.values()), hits / sum(want.values())
    return {"f1": 2 * precision * recall / (precision + recall)}


SCORERS: dict[str, tuple[Scorer, str]] = {
    "exact": (exact, "1"),
    "contains": (contains, "1"),
    "regex": (regex, "1"),
    "numeric": (numeric, "1"),
    "json_valid": (json_valid, "1"),
    "field_f1": (field_f1, "1"),
    "token_f1": (token_f1, "1"),
}


def scorer(name: str) -> tuple[Scorer, str]:
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r} (known: {', '.join(SCORERS)})")
    return SCORERS[name]


def primary(suite: SuiteCfg) -> tuple[str, str, str]:
    """(scorer, version, metric) of the metric that decides the verdict."""
    name, _, metric = suite.primary_metric.partition(".")
    if name not in suite.scorers:
        raise ValueError(
            f"primary_metric {suite.primary_metric!r} is not among the suite's scorers"
        )
    return name, scorer(name)[1], metric


def context(cases: list[Case]) -> dict[str, Any]:
    return {"labels": sorted({normalise(c.expected) for c in cases if isinstance(c.expected, str)})}


def score_suite(cfg: Config, name: str, fingerprint: str, db: sqlite3.Connection) -> int:
    """Score every stored sample that the current scorer versions have not seen yet.

    Makes no model calls, so changing a scorer and re-scoring is free. Truncated samples
    are left unscored: a cut-off answer is not evidence about quality.
    """
    suite = cfg.get_suite(name)
    cases = {c.hash: c for c in datasets.load(cfg.root / suite.dataset)}
    ctx, rows = context(list(cases.values())), []
    for scorer_name in suite.scorers:
        fn, version = scorer(scorer_name)
        todo = db.execute(
            "SELECT case_hash, rep, output, status FROM samples s WHERE fingerprint=? "
            "AND status != 'truncated' AND NOT EXISTS (SELECT 1 FROM scores x WHERE "
            "x.fingerprint=s.fingerprint AND x.case_hash=s.case_hash AND x.rep=s.rep "
            "AND x.scorer=? AND x.version=?)",
            (fingerprint, scorer_name, version),
        ).fetchall()
        for row in todo:
            if row["case_hash"] not in cases:
                continue
            # A refusal is a graded outcome: score it as an empty answer.
            output = "" if row["status"] == "refusal" else row["output"]
            try:
                metrics = fn(output, cases[row["case_hash"]].expected, ctx)
            except Exception as e:  # a scorer bug must name itself, not look like a model failure
                raise ValueError(
                    f"scorer {scorer_name!r} failed on case {row['case_hash']}: {e!r}"
                ) from e
            rows += [
                {
                    "fingerprint": fingerprint,
                    "case_hash": row["case_hash"],
                    "rep": row["rep"],
                    "scorer": scorer_name,
                    "version": version,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            ]
    store.insert(db, "scores", rows, "OR REPLACE")
    return len(rows)


def selfcheck(cfg: Config, name: str) -> dict[str, Any]:
    """Test the eval before trusting it: reference answers must pass, junk must fail."""
    suite = cfg.get_suite(name)
    cases = datasets.load(cfg.root / suite.dataset)
    scorer_name, _, metric = primary(suite)
    fn, ctx = scorer(scorer_name)[0], context(cases)

    def mean(answer: Callable[[Case], str]) -> float:
        return sum(fn(answer(c), c.expected, ctx)[metric] for c in cases) / len(cases)

    def reference(c: Case) -> str:
        return c.expected if isinstance(c.expected, str) else json.dumps(c.expected)

    labels = Counter(c.expected for c in cases if isinstance(c.expected, str))
    majority = labels.most_common(1)[0] if labels else ("", 0)
    chance = max(majority[1] / len(cases), 0.05)
    report: dict[str, Any] = {
        "metric": suite.primary_metric,
        "oracle (reference answers)": mean(reference),
        "null: empty answer": mean(lambda c: ""),
        "null: constant 'n/a'": mean(lambda c: "n/a"),
        "null: majority answer": mean(lambda c: str(majority[0])),
        "chance ceiling": chance,
    }
    nulls = [v for k, v in report.items() if k.startswith("null")]
    report["ok"] = report["oracle (reference answers)"] >= 0.999 and max(nulls) <= chance + 1e-9
    return report
