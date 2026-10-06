import sys
from pathlib import Path

import numpy as np
import pytest

from tripwire.config import load_config

EXPERIMENTS = Path(__file__).parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import proof  # noqa: E402
import zoo  # noqa: E402


def outcomes(drop=0.0, noise=0.05, n=1500, rate=0.6):
    """A baseline, and a candidate that flips `noise` of the cases each way plus `drop` more
    from pass to fail: its effect on these cases is exactly -drop."""
    base = np.zeros(n)
    base[: round(n * rate)] = 1
    head = base.copy()
    head[: round(n * (noise + drop))] = 0
    head[n - round(n * noise) :] = 1
    return base, head


@pytest.fixture(autouse=True)
def fewer_resamples(monkeypatch):
    """The real analysis uses 2,000 bootstrap resamples per gate; these checks need far fewer."""
    monkeypatch.setattr(proof, "N_BOOT", 300)


def test_the_committed_zoo_is_what_the_generator_writes(tmp_path):
    """Stored samples are keyed by these prompts: an edit to zoo.py must be regenerated."""
    zoo.write(tmp_path)
    fresh = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*.*")}
    committed = EXPERIMENTS / "zoo"
    kept = {f"zoo/{p.name}": p.read_bytes() for p in committed.iterdir()}
    kept["zoo.toml"] = (EXPERIMENTS / "zoo.toml").read_bytes()
    assert fresh == kept


def test_every_variant_has_a_suite_and_a_distinct_target():
    cfg = load_config(EXPERIMENTS / "zoo.toml")
    names = [v.name for task in zoo.tasks() for v in task.variants]
    assert len(names) == len(set(names)) and set(names) == set(cfg.suite)
    for task in zoo.tasks():
        assert task.metrics[0] == cfg.get_suite(task.baseline).primary_metric
        specs = [cfg.target(v.name).model_dump_json() for v in task.variants]
        assert len(set(specs)) == len(specs)  # two variants never share a target file


def test_a_large_drop_is_beyond_the_margin_and_the_gate_says_so():
    base, head = outcomes(drop=0.15)
    assert proof.truth(base, head)["kind"] == proof.BEYOND
    assert proof.truth(base, outcomes(drop=0.01)[1])["kind"] == proof.INSIDE
    result = proof.replay(base, head, 250, 700, draws=40)
    assert result["full"]["REGRESSED"] > 0.9 and result["naive"]["700:margin"] > 0.9
    assert result["cases"] < 700  # some gates stopped at the first look


def test_no_drop_is_rarely_called_a_regression():
    base, head = outcomes(drop=0.0)
    assert proof.truth(base, head)["kind"] == proof.NO_DROP
    result = proof.replay(base, head, 250, 700, draws=150)
    assert result["full"]["REGRESSED"] <= 0.12 and result["staged"]["REGRESSED"] <= 0.12
    assert result["naive"]["700:any"] > 0.3  # a bare comparison blocks on noise


def test_identical_scores_always_pass():
    base, _ = outcomes()
    assert proof.truth(base, base)["kind"] == proof.IDENTICAL
    result = proof.replay(base, base, 250, 700, draws=10)
    assert result["first"]["PASS"] == result["staged"]["PASS"] == 1 and result["early"] == 1


def test_too_many_unscored_answers_is_invalid_not_a_verdict():
    base, head = outcomes()
    head[::10] = np.nan
    assert proof.truth(base, head)["kind"] == proof.INVALID
    assert proof.truth(base, np.full(len(base), np.nan))["kind"] == proof.INVALID
    assert proof.replay(base, head, 250, 700, draws=10)["full"]["INVALID"] == 1


def test_a_slice_that_really_fell_is_named():
    base, head = outcomes(drop=0.0)
    tags = {"first": np.arange(len(base)) < 300, "rest": np.arange(len(base)) >= 300}
    found = proof.slice_alarms(base, head, tags, 700, draws=20)
    assert found["worst"][0] == "first" and found["worst"][1] < 0  # where the passes were lost
    assert found["any"] > 0.5 and found["alone"] <= found["any"]


def test_the_expected_curve_matches_the_simulation():
    base, head = outcomes(drop=0.04, noise=0.06)
    t = proof.truth(base, head)
    simulated = proof.replay(base, head, 250, 700, draws=150)["full"]["REGRESSED"]
    assert abs(proof.expected(t["delta"], t["changed"], 700) - simulated) < 0.1
