"""Paired statistics for two configurations run on the same cases.

Pure functions over arrays: no I/O, no database. Every array holds one number per case
(the mean over that case's repetitions). The case is the unit of resampling, because
repetitions of one case are correlated and must not be counted as independent evidence.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as st

Array = Any  # anything numpy can turn into a float array


def mean_ci(
    x: Array, level: float = 0.90, n_boot: int = 10_000, seed: int = 0, fast: bool = False
) -> tuple[float, float, float]:
    """Mean with a two-sided interval. BCa bootstrap; `fast` uses the normal approximation."""
    x = np.asarray(x, dtype=float)
    mean = float(x.mean()) if len(x) else math.nan
    if len(x) < 2 or np.all(x == x[0]):
        return mean, mean, mean  # no spread to resample; BCa would return NaN here
    if fast:
        half = st.norm.ppf(0.5 + level / 2) * x.std(ddof=1) / math.sqrt(len(x))
        return mean, mean - half, mean + half
    for method in ("BCa", "percentile"):  # BCa can degenerate on near-constant data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = st.bootstrap(
                (x,),
                np.mean,
                confidence_level=level,
                n_resamples=n_boot,
                method=method,
                rng=np.random.default_rng(seed),
            ).confidence_interval
        if math.isfinite(ci.low) and math.isfinite(ci.high):
            return mean, float(ci.low), float(ci.high)
    return mean, mean, mean


@dataclass
class Paired:
    n: int
    base: float
    head: float
    delta: float
    lo: float  # one-sided (1 - alpha) lower bound on delta
    hi: float  # one-sided (1 - alpha) upper bound on delta
    p: float  # two-sided sign-flip permutation p-value
    mcnemar_p: float | None  # exact McNemar, only when every score is 0 or 1


def paired(
    base: Array,
    head: Array,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
    fast: bool = False,
) -> Paired:
    base, head = np.asarray(base, dtype=float), np.asarray(head, dtype=float)
    d = head - base
    # A two-sided 1-2a interval has a one-sided 1-a bound at each end.
    delta, lo, hi = mean_ci(d, 1 - 2 * alpha, n_boot, seed, fast)
    if not d.any():
        p = 1.0
    elif fast:
        p = float(2 * st.norm.sf(abs(delta) / (d.std(ddof=1) / math.sqrt(len(d)))))
    else:
        # With one sample, "samples" permutations flip the sign of each difference.
        p = float(
            st.permutation_test(
                (d,),
                np.mean,
                permutation_type="samples",
                n_resamples=n_boot - 1,
                vectorized=True,
                rng=np.random.default_rng(seed),
            ).pvalue
        )
    mcnemar = None
    if np.isin(base, (0, 1)).all() and np.isin(head, (0, 1)).all():
        broke, fixed = int((d < 0).sum()), int((d > 0).sum())
        mcnemar = float(st.binomtest(broke, broke + fixed).pvalue) if broke + fixed else 1.0
    return Paired(len(d), float(base.mean()), float(head.mean()), delta, lo, hi, p, mcnemar)


def verdict(lo: float, hi: float, margin: float) -> str:
    """Non-inferiority: the candidate must show it is not worse by more than `margin`."""
    if lo > 0:
        return "IMPROVED"
    if lo > -margin:
        return "PASS"
    return "REGRESSED" if hi < 0 else "INCONCLUSIVE"


def adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg: controls the share of false alarms among flagged slices."""
    return [float(p) for p in st.false_discovery_control(p_values)] if p_values else []


def ratio_ci(
    base: Array,
    head: Array,
    stat: Callable[..., Any] = np.sum,
    level: float = 0.90,
    n_boot: int = 2_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """stat(head) / stat(base) with a percentile bootstrap over cases, kept paired."""
    base, head = np.asarray(base, dtype=float), np.asarray(head, dtype=float)
    if not len(base) or not stat(base):
        return math.nan, math.nan, math.nan
    picks = np.random.default_rng(seed).integers(0, len(base), (n_boot, len(base)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = stat(head[picks], axis=1) / stat(base[picks], axis=1)
    lo, hi = np.nanquantile(ratios, [(1 - level) / 2, (1 + level) / 2])
    return float(stat(head) / stat(base)), float(lo), float(hi)


def variance(scores: Array) -> dict[str, float]:
    """Split the variance of a (cases x repetitions) score matrix.

    Var(mean) = between / n + within / (n * R): if `between` dominates, more repetitions
    buy nothing and only more cases tighten the estimate.
    """
    scores = np.asarray(scores, dtype=float)
    reps = scores.shape[1]
    within = float(scores.var(axis=1, ddof=1).mean()) if reps > 1 else 0.0
    between = max(0.0, float(scores.mean(axis=1).var(ddof=1)) - within / reps)
    total = between + within
    return {"between": between, "within": within, "within_share": within / total if total else 0.0}


def sample_size(drop: float, discordant: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Cases needed to detect a true `drop`.

    `discordant` is the total share of cases on which the two runs disagree: rerun noise
    plus the cases the drop itself flips.
    """
    if not 0 < drop**2 < discordant:
        raise ValueError("need 0 < drop^2 < discordant rate")
    z_a, z_b = st.norm.ppf(1 - alpha), st.norm.ppf(power)
    n = (z_a * math.sqrt(discordant) + z_b * math.sqrt(discordant - drop**2)) ** 2 / drop**2
    return math.ceil(n)


def simulate(
    base: Array,
    effect: float,
    n: int,
    discordant: float,
    alpha: float = 0.05,
    margin: float = 0.03,
    sims: int = 1_000,
    seed: int = 0,
) -> dict[str, float]:
    """How often each verdict comes up when the true drop is `effect`.

    Resamples real pass/fail outcomes, then flips them: symmetric noise at the given
    discordant rate, plus extra pass-to-fail flips worth `effect`. Uses the normal
    approximation so thousands of simulated gates run in seconds.
    """
    base = np.asarray(base, dtype=float)
    rate = float(base.mean())
    if not 0 < rate < 1:
        raise ValueError("simulation needs a baseline with both passes and failures")
    rng = np.random.default_rng(seed)
    down = min(1.0, (discordant / 2 + effect) / rate)  # P(pass -> fail)
    up = min(1.0, (discordant / 2) / (1 - rate))  # P(fail -> pass)
    counts: Counter[str] = Counter()
    for _ in range(sims):
        b = rng.choice(base, n)
        u = rng.random(n)
        h = np.where(b == 1, u >= down, u < up).astype(float)
        r = paired(b, h, alpha, fast=True)
        counts[verdict(r.lo, r.hi, margin)] += 1
    return {name: counts[name] / sims for name in ("IMPROVED", "PASS", "INCONCLUSIVE", "REGRESSED")}


def aa(
    scores: Array,
    alpha: float = 0.05,
    margin: float = 0.03,
    splits: int = 500,
    n_boot: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Compare a configuration with itself to measure the gate's real false-alarm rate.

    Each split deals every case's repetitions at random into an A half and a B half. The
    true difference is exactly zero, so any REGRESSED or IMPROVED verdict is a false alarm.
    """
    scores = np.asarray(scores, dtype=float)
    half = scores.shape[1] // 2
    if half < 1:
        raise ValueError("an A/A test needs at least two repetitions per case")
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    for i in range(splits):
        dealt = rng.permuted(scores, axis=1)
        a, b = dealt[:, :half].mean(axis=1), dealt[:, half : 2 * half].mean(axis=1)
        # Same interval the gate uses; the p-value is not needed for a verdict.
        _, lo, hi = mean_ci(b - a, 1 - 2 * alpha, n_boot, seed + i)
        counts[verdict(lo, hi, margin)] += 1
    out = {
        name: counts[name] / splits for name in ("IMPROVED", "PASS", "INCONCLUSIVE", "REGRESSED")
    }
    out["discordant"] = float((scores[:, 0] != scores[:, 1]).mean())
    return out


def kappa(a: Array, b: Array) -> float:
    """Cohen's kappa for two binary raters: agreement beyond what chance would give."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = float((a == b).mean())
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - chance) / (1 - chance) if chance < 1 else 1.0


def kappa_ci(
    a: Array, b: Array, level: float = 0.90, n_boot: int = 2_000, seed: int = 0
) -> tuple[float, float, float]:
    """Kappa with a percentile bootstrap interval over the rated items."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    draws = [kappa(a[i], b[i]) for i in picks]
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return kappa(a, b), float(lo), float(hi)


def ppi(
    judge_all: Array, judge_labelled: Array, human_labelled: Array, level: float = 0.90
) -> tuple[float, float, float]:
    """Prediction-powered estimate of what humans would have scored on everything.

    The judge's mean over all samples, corrected by the judge's average error on the
    few samples a human also labelled. Valid for the human metric even when the judge
    is biased, as long as the labelled samples are a random subset.
    """
    judge_all = np.asarray(judge_all, dtype=float)
    gap = np.asarray(human_labelled, dtype=float) - np.asarray(judge_labelled, dtype=float)
    estimate = float(judge_all.mean() + gap.mean())
    spread = judge_all.var(ddof=1) / len(judge_all) + gap.var(ddof=1) / len(gap)
    half = float(st.norm.ppf(0.5 + level / 2) * math.sqrt(spread))
    return estimate, estimate - half, estimate + half
