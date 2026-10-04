import numpy as np
import pytest

from tripwire import stats


def outcomes(n, rate, seed):
    return (np.random.default_rng(seed).random(n) < rate).astype(float)


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.01, 0.05, "IMPROVED"),
        (-0.01, 0.04, "PASS"),
        (-0.02, -0.005, "PASS"),  # a real but tolerated drop
        (-0.06, -0.01, "REGRESSED"),
        (-0.08, -0.04, "REGRESSED"),
        (-0.05, 0.02, "INCONCLUSIVE"),
    ],
)
def test_verdict_partitions_every_interval(lo, hi, expected):
    assert stats.verdict(lo, hi, margin=0.03) == expected


def test_identical_runs_pass_with_a_zero_interval():
    x = outcomes(200, 0.7, 1)
    r = stats.paired(x, x)
    assert (r.delta, r.lo, r.hi, r.p, r.mcnemar_p) == (0, 0, 0, 1, 1)
    assert stats.verdict(r.lo, r.hi, 0.03) == "PASS"


def test_mcnemar_matches_the_textbook_value():
    base = np.array([1.0] * 10 + [0.0] * 2 + [1.0] * 88)
    head = np.array([0.0] * 10 + [1.0] * 2 + [1.0] * 88)  # 10 broke, 2 fixed
    r = stats.paired(base, head)
    assert r.mcnemar_p == pytest.approx(0.0386, abs=1e-4)
    assert r.delta == pytest.approx(-0.08)
    assert stats.paired(base * 0.5, head).mcnemar_p is None  # not binary


def test_delta_is_antisymmetric_and_shift_invariant():
    a, b = outcomes(300, 0.7, 2), outcomes(300, 0.6, 3)
    forward, backward = stats.paired(a, b), stats.paired(b, a)
    assert forward.delta == pytest.approx(-backward.delta)
    assert forward.lo == pytest.approx(-backward.hi, abs=0.01)
    assert stats.paired(a + 1, b + 1).delta == pytest.approx(forward.delta)
    order = np.random.default_rng(0).permutation(300)
    assert stats.paired(a[order], b[order]).delta == pytest.approx(forward.delta)


def test_fast_interval_agrees_with_the_bootstrap():
    d = outcomes(500, 0.6, 4) - outcomes(500, 0.65, 5)
    slow, fast = stats.mean_ci(d), stats.mean_ci(d, fast=True)
    assert slow[0] == fast[0]
    assert slow[1] == pytest.approx(fast[1], abs=0.01) and slow[2] == pytest.approx(
        fast[2], abs=0.01
    )


def test_interval_covers_the_truth_at_the_stated_rate():
    """A 90% interval must contain the true difference about 90% of the time."""
    rng, true_delta, covered, trials = np.random.default_rng(6), -0.05, 0, 300
    for i in range(trials):
        base = (rng.random(300) < 0.70).astype(float)
        head = (rng.random(300) < 0.70 + true_delta).astype(float)
        _, lo, hi = stats.mean_ci(head - base, 0.90, n_boot=1000, seed=i)
        covered += lo <= true_delta <= hi
    assert 0.85 <= covered / trials <= 0.95


def test_the_gate_rarely_cries_wolf_and_catches_large_drops():
    base = outcomes(2000, 0.6, 7)
    null = stats.simulate(base, effect=0.0, n=700, discordant=0.10)
    assert null["REGRESSED"] <= 0.08 and null["IMPROVED"] <= 0.08  # nominal 0.05 each
    big = stats.simulate(base, effect=0.10, n=700, discordant=0.10)
    assert big["REGRESSED"] >= 0.99


def test_closed_form_sample_size_matches_simulation():
    assert [stats.sample_size(d, 0.10) for d in (0.05, 0.03, 0.02)] == [246, 685, 1544]
    # The simulation takes rerun noise (0.10); the formula takes the total, noise plus the drop.
    n = stats.sample_size(0.05, 0.15)
    hit = stats.simulate(outcomes(2000, 0.6, 8), effect=0.05, n=n, discordant=0.10, sims=2000)
    assert 0.72 <= hit["REGRESSED"] <= 0.88  # designed for 80% power
    with pytest.raises(ValueError):
        stats.sample_size(0.5, 0.10)


def test_false_discovery_adjustment():
    assert stats.adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert stats.adjust([]) == []


def test_ratio_interval():
    base = np.arange(1.0, 101.0)
    assert stats.ratio_ci(base, base * 2) == pytest.approx((2, 2, 2))
    value, lo, hi = stats.ratio_ci(base, base + np.random.default_rng(9).normal(10, 5, 100))
    assert lo < value < hi and 1.1 < value < 1.3
    assert np.isnan(stats.ratio_ci(np.zeros(5), np.ones(5))[0])  # nothing to divide by


def test_variance_split_says_where_the_noise_is():
    stable = np.repeat(outcomes(200, 0.6, 10)[:, None], 3, axis=1)  # reps always agree
    assert stats.variance(stable)["within_share"] == 0
    noisy = (np.random.default_rng(11).random((200, 3)) < 0.6).astype(float)  # reps independent
    assert stats.variance(noisy)["within_share"] > 0.8


def test_aa_false_alarm_rate_is_near_nominal():
    rng = np.random.default_rng(12)
    chance = rng.random(400)[:, None]  # each case has its own pass probability
    scores = (rng.random((400, 2)) < chance).astype(float)
    result = stats.aa(scores, splits=200, n_boot=500)
    assert result["REGRESSED"] <= 0.10 and result["IMPROVED"] <= 0.10
    assert 0.2 < result["discordant"] < 0.45
    with pytest.raises(ValueError):
        stats.aa(scores[:, :1])
