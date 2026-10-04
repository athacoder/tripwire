"""Turn stored scores into a single-run summary or a paired comparison with a verdict."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

import numpy as np

from . import stats
from .config import SuiteCfg
from .models import Case
from .scorers import primary

MIN_SLICE = 30  # below this a slice says "insufficient data", never "fine"


def case_scores(
    db: sqlite3.Connection, suite: SuiteCfg, fingerprint: str
) -> dict[str, list[float]]:
    """Primary-metric scores per case, in repetition order, at the current scorer version."""
    scorer, version, metric = primary(suite)
    out: dict[str, list[float]] = defaultdict(list)
    query = (
        "SELECT case_hash, value FROM scores WHERE fingerprint=? AND scorer=? AND version=? "
        "AND metric=? AND rep<? ORDER BY rep"
    )
    for case_hash, value in db.execute(query, (fingerprint, scorer, version, metric, suite.reps)):
        out[case_hash].append(value)
    return out


def _samples(db: sqlite3.Connection, fingerprint: str, reps: int) -> dict[str, list[sqlite3.Row]]:
    out: dict[str, list[sqlite3.Row]] = defaultdict(list)
    query = (
        "SELECT case_hash, status, output, output_tokens, latency_ms FROM samples "
        "WHERE fingerprint=? AND rep<? ORDER BY rep"
    )
    for row in db.execute(query, (fingerprint, reps)):
        out[row["case_hash"]].append(row)
    return out


def _brief(case: Case) -> tuple[str, Any]:
    """What a reader needs from a case: its last input field (the question, when a long
    context comes first) and the reference answer."""
    shown = str(list(case.input.values())[-1]) if case.input else ""
    expected = case.expected
    if isinstance(expected, dict) and "reference" in expected:
        expected = expected["reference"]
    return shown[:160], expected


def _slices(cases: list[Case]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, case in enumerate(cases):
        for tag in case.tags:
            groups[tag].append(i)
    return groups


def summarise(
    db: sqlite3.Connection, suite: SuiteCfg, cases: list[Case], fingerprint: str
) -> dict[str, Any]:
    """One configuration on its own: the score with an interval, slices, cost, worst cases."""
    scores, samples = case_scores(db, suite, fingerprint), _samples(db, fingerprint, suite.reps)
    scored = [c for c in cases if c.hash in scores]
    if not scored:
        raise ValueError("no scored samples for this suite yet; run `tripwire run` first")
    values = np.array([np.mean(scores[c.hash]) for c in scored])
    mean, lo, hi = stats.mean_ci(values, 1 - 2 * suite.alpha)
    rows = [r for c in cases for r in samples.get(c.hash, [])]
    latency = np.array([r["latency_ms"] for r in rows])
    out: dict[str, Any] = {
        "metric": suite.primary_metric,
        "cases": len(cases),
        "scored": len(scored),
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "statuses": {
            s: sum(r["status"] == s for r in rows) for s in ("ok", "truncated", "refusal")
        },
        "output_tokens": int(sum(r["output_tokens"] for r in rows)),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "saturated": mean > 0.95,  # too close to the ceiling to tell configurations apart
        "slices": [
            {"tag": tag, "n": len(idx), "mean": float(values[idx].mean())}
            for tag, idx in sorted(_slices(scored).items())
            if len(idx) >= MIN_SLICE
        ],
        "lowest": [
            {
                "text": _brief(c)[0],
                "expected": _brief(c)[1],
                "output": samples[c.hash][0]["output"].strip()[:160],
                "score": float(v),
            }
            for v, c in sorted(zip(values, scored, strict=True), key=lambda pair: pair[0])[:10]
            if v < 1
        ],
    }
    full = [scores[c.hash] for c in scored if len(scores[c.hash]) == suite.reps]
    if suite.reps > 1 and len(full) > 1:
        out["variance"] = stats.variance(np.array(full))
    return out


def compare(
    db: sqlite3.Connection,
    suite: SuiteCfg,
    cases: list[Case],
    base_fp: str,
    head_fp: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison of two fingerprints on the same cases."""
    base_scores, head_scores = (case_scores(db, suite, fp) for fp in (base_fp, head_fp))
    base_samples, head_samples = (_samples(db, fp, suite.reps) for fp in (base_fp, head_fp))
    both = [c for c in cases if c.hash in base_scores and c.hash in head_scores]
    limits = suite.guardrails
    head_rows = [r for c in cases for r in head_samples.get(c.hash, [])]
    truncation = sum(r["status"] == "truncated" for r in head_rows) / max(1, len(head_rows))
    result: dict[str, Any] = {
        "metric": suite.primary_metric,
        "base_fingerprint": base_fp,
        "head_fingerprint": head_fp,
        "alpha": suite.alpha,
        "margin": suite.margin,
        "cases": len(cases),
        "paired": len(both),
    }

    # Too many cases missing on either side means the plumbing failed, not the model.
    unpaired = 1 - len(both) / max(1, len(cases))
    if len(both) < 2 or unpaired > limits.get("error_rate_max", 0.02):
        result["verdict"] = "INVALID"
        result["reason"] = (
            f"only {len(both)} of {len(cases)} cases have a score on both sides "
            f"({truncation:.1%} of candidate answers were cut off)"
        )
        return result

    base = np.array([np.mean(base_scores[c.hash]) for c in both])
    head = np.array([np.mean(head_scores[c.hash]) for c in both])
    r = stats.paired(base, head, suite.alpha, seed=seed)
    result.update(
        verdict=stats.verdict(r.lo, r.hi, suite.margin),
        base=r.base,
        head=r.head,
        delta=r.delta,
        lo=r.lo,
        hi=r.hi,
        p=r.p,
        mcnemar_p=r.mcnemar_p,
    )

    # Slices: an overall pass can hide one group that collapsed.
    found: list[dict[str, Any]] = []
    for tag, idx in sorted(_slices(both).items()):
        if len(idx) >= MIN_SLICE:
            s = stats.paired(base[idx], head[idx], suite.alpha, n_boot=2_000, seed=seed)
            found.append({"tag": tag, "n": s.n, "delta": s.delta, "lo": s.lo, "hi": s.hi, "p": s.p})
    for item, adjusted in zip(found, stats.adjust([s["p"] for s in found]), strict=True):
        item["p_adjusted"] = adjusted
        item["regressed"] = adjusted < suite.alpha and item["delta"] < 0
    result["slices"] = sorted(found, key=lambda s: s["delta"])
    result["small_slices"] = sum(len(i) < MIN_SLICE for i in _slices(both).values())

    # Guardrails: secondary limits. Ratios get an interval, widened for how many are checked.
    def per_case(samples: dict[str, list[sqlite3.Row]], column: str) -> np.ndarray:
        return np.array([np.mean([row[column] for row in samples[c.hash]]) for c in both])

    def p95(x: np.ndarray, axis: int | None = None) -> Any:
        return np.percentile(x, 95, axis=axis)

    ratios: dict[str, tuple[str, str, Any]] = {
        "output_tokens_ratio_max": ("output tokens", "output_tokens", np.sum),
        "latency_p95_ratio_max": ("latency p95", "latency_ms", p95),
    }
    active = [k for k in ratios if k in limits]
    guardrails = []
    for key in active:
        label, column, stat = ratios[key]
        value, lo, hi = stats.ratio_ci(
            per_case(base_samples, column),
            per_case(head_samples, column),
            stat,
            level=1 - 2 * suite.alpha / len(active),
            seed=seed,
        )
        # Fail only when the whole interval is over the limit; a noisy overshoot is a warning.
        guardrails.append(
            {
                "name": label,
                "value": value,
                "lo": lo,
                "hi": hi,
                "limit": limits[key],
                "ok": not lo > limits[key],
            }
        )
    if "truncation_rate_max" in limits:
        limit = limits["truncation_rate_max"]
        guardrails.append(
            {
                "name": "truncation rate",
                "value": truncation,
                "limit": limit,
                "ok": truncation <= limit,
            }
        )
    result["guardrails"] = guardrails

    # Flips: what a developer actually reads.
    d = head - base

    def detail(i: int) -> dict[str, Any]:
        c = both[i]
        return {
            "text": _brief(c)[0],
            "expected": _brief(c)[1],
            "base_output": base_samples[c.hash][0]["output"].strip()[:120],
            "head_output": head_samples[c.hash][0]["output"].strip()[:120],
            "delta": float(d[i]),
        }

    order = np.argsort(d, kind="stable")
    result["flips"] = {
        "broke": int((d < 0).sum()),
        "fixed": int((d > 0).sum()),
        "flaky": int(((base > 0) & (base < 1)).sum()),
        "stable_pass": int(((d == 0) & (base == 1)).sum()),
        "stable_fail": int(((d == 0) & (base == 0)).sum()),
        "broke_examples": [detail(i) for i in order[:10] if d[i] < 0],
        "fixed_examples": [detail(i) for i in order[::-1][:5] if d[i] > 0],
    }
    return result


def blocked(result: dict[str, Any], suite: SuiteCfg) -> bool:
    """Whether this comparison should stop a merge."""
    verdict = result["verdict"]
    return (
        verdict in ("REGRESSED", "INVALID")
        or (verdict == "INCONCLUSIVE" and suite.on_inconclusive == "fail")
        or any(not g["ok"] for g in result.get("guardrails", []))
        or any(s["regressed"] for s in result.get("slices", []))
    )


def matrix(
    db: sqlite3.Connection, suite: SuiteCfg, cases: list[Case], fingerprint: str
) -> np.ndarray:
    """Scores as a (cases x repetitions) array, keeping only cases with every repetition."""
    scores = case_scores(db, suite, fingerprint)
    full = [scores[c.hash] for c in cases if len(scores.get(c.hash, [])) == suite.reps]
    return np.array(full, dtype=float).reshape(len(full), suite.reps)
