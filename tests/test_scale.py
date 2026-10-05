import asyncio
import json

import numpy as np
from test_gate import repo, run_gate, set_target  # noqa: F401  (repo is a fixture)
from typer.testing import CliRunner

from tripwire import datasets, stats, store
from tripwire.cli import app
from tripwire.config import load_config
from tripwire.runner import run_suite


def outcomes(n, rate, seed):
    return (np.random.default_rng(seed).random(n) < rate).astype(float)


def test_two_stage_stops_early_only_when_the_first_look_decides():
    base = outcomes(700, 0.7, 1)
    assert stats.two_stage(base, base, first=250) == ("PASS", 250)  # identical: decided at once
    assert stats.two_stage(base, np.zeros(700), first=250) == ("REGRESSED", 250)
    # Balanced noise in the first stage only: 20 passes fail and 20 fails pass. The
    # difference is zero, but 250 cases cannot rule out a 3-point drop; 700 can.
    noisy = base.copy()
    passes, fails = np.flatnonzero(base[:250] == 1)[:20], np.flatnonzero(base[:250] == 0)[:20]
    noisy[passes], noisy[fails] = 0, 1
    assert stats.two_stage(base, noisy, first=250) == ("PASS", 700)
    # Nothing to escalate to when the first stage already covers every case.
    assert stats.two_stage(base[:100], base[:100], first=250) == ("PASS", 100)


def test_looking_twice_does_not_inflate_false_alarms():
    """Each look spends half of alpha, so the two-stage gate stays within alpha overall."""
    base = outcomes(2000, 0.6, 2)
    null = stats.simulate(base, 0.0, 700, discordant=0.10, sims=2000, first=250)
    assert null["REGRESSED"] <= 0.05 and null["IMPROVED"] <= 0.05
    assert 250 <= null["cases"] <= 700 and 0 <= null["early"] <= 1
    single = stats.simulate(base, 0.0, 700, discordant=0.10, sims=2000)
    assert "cases" not in single


def test_a_large_drop_is_caught_at_the_first_stage():
    big = stats.simulate(outcomes(2000, 0.6, 3), 0.20, 700, discordant=0.10, first=250)
    assert big["REGRESSED"] >= 0.99 and big["early"] >= 0.95 and big["cases"] < 300


def escalating(cfg, first=10):
    cfg.suite["demo"].on_inconclusive = "escalate"
    cfg.suite["demo"].first_stage = first
    return cfg


def head_samples(cfg, fingerprint):
    db = store.connect(cfg.db_path)
    return db.execute(
        "SELECT count(*) FROM samples WHERE fingerprint=?", (fingerprint,)
    ).fetchone()[0]


def test_an_escalating_gate_runs_only_the_first_stage_when_that_decides(repo):  # noqa: F811
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.0"\n')
    result, is_blocked = run_gate(escalating(repo))
    assert (result["verdict"], result["stage"], result["stages"]) == ("REGRESSED", 1, 2)
    assert result["cases"] == 10 and result["alpha"] == 0.025 and is_blocked
    assert head_samples(repo, result["head_fingerprint"]) == 20  # 10 cases x 2 reps, not 40 x 2
    assert head_samples(repo, result["base_fingerprint"]) == 20


def test_an_escalating_gate_runs_everything_when_the_first_stage_cannot_decide(repo):  # noqa: F811
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.74"\n')
    result, is_blocked = run_gate(escalating(repo))
    assert (result["stage"], result["cases"]) == (2, 40)
    assert head_samples(repo, result["head_fingerprint"]) == 80
    assert result["verdict"] == "INCONCLUSIVE" and is_blocked  # still undecided: still blocks


def test_the_first_stage_is_a_fixed_stratified_subset(repo):  # noqa: F811
    cases = datasets.load(repo.root / "data.jsonl")
    first = datasets.split(cases, {"first": 12}, seed=0)["first"]
    again = datasets.split(cases[::-1], {"first": 12}, seed=0)["first"]
    assert [c.hash for c in first] == [c.hash for c in again]
    per_kind = [sum(c.tags[0] == f"kind:{k}" for c in first) for k in range(4)]
    assert per_kind == [3, 3, 3, 3]  # 12 cases over four equal strata


def test_a_staged_gate_verifies_from_bundles(repo):  # noqa: F811
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.0"\n')
    local, _ = run_gate(escalating(repo), bundle=True)
    repo.db_path.unlink()
    verified, is_blocked = run_gate(repo, verify_only=True)
    assert (verified["verdict"], verified["stage"], verified["delta"]) == (
        local["verdict"],
        1,
        local["delta"],
    )
    assert is_blocked


def test_a_subset_run_keeps_the_dataset_identity_and_reports_progress(project):
    cases = datasets.load(project.root / "data.jsonl")
    seen = []
    wanted = {c.hash for c in cases[:5]}
    s = asyncio.run(
        run_suite(project, "demo", only=wanted, progress=lambda s: seen.append(s.pending))
    )
    assert s.total == 10 and s.dataset_version == datasets.version(cases)
    assert seen == list(range(9, -1, -1))  # one call per finished sample
    db = store.connect(project.db_path)
    stored = {r[0] for r in db.execute("SELECT DISTINCT case_hash FROM samples")}
    assert stored == wanted
    env = json.loads(db.execute("SELECT env FROM runs").fetchone()[0])
    assert env["samples"] == {"cached": 0, "generated": 10, "errors": 0}


def two_suites(project):
    (project.root / "other.toml").write_text('provider = "mock"\nmodel = "mock:0.5"\n')
    text = (project.root / "tripwire.toml").read_text()
    extra = '\n[suite.zoo-b]\ndataset = "data.jsonl"\ntarget = "other.toml"\n'
    extra += '\n[suite.zoo-a]\ndataset = "data.jsonl"\ntarget = "target.toml"\n'
    (project.root / "tripwire.toml").write_text(text + extra)
    return load_config(project.root / "tripwire.toml")


def test_queue_runs_suites_grouped_by_model_and_is_resumable(project):
    cfg = two_suites(project)
    cli, config = CliRunner(), ["--config", str(cfg.root / "tripwire.toml")]
    first = cli.invoke(app, ["queue", *config])
    assert first.exit_code == 0, first.output
    lines = [line.split(":")[0] for line in first.output.splitlines() if " generated, " in line]
    assert lines == ["zoo-b", "demo", "zoo-a"]  # mock:0.5 first, then both mock:0.8 suites together
    assert "zoo-a: 0 generated, 40 cached" in first.output  # same target and cases as demo, rep 0

    again = cli.invoke(app, ["queue", *config])
    assert again.exit_code == 0 and "demo: 0 generated, 80 cached" in again.output

    only = cli.invoke(app, ["queue", "--match", "zoo-*", *config])
    assert "demo" not in only.output and "zoo-b" in only.output
    assert cli.invoke(app, ["queue", "--match", "nothing-*", *config]).exit_code == 2


def test_queue_respects_the_time_budget(project):
    cfg = two_suites(project)
    cli, config = CliRunner(), ["--config", str(cfg.root / "tripwire.toml")]
    stopped = cli.invoke(app, ["queue", "--max-minutes", "0", *config])
    assert stopped.exit_code == 1 and "160 still pending" in stopped.output
    assert cli.invoke(app, ["queue", *config]).exit_code == 0  # picks up where it stopped


def test_usage_reports_calls_and_what_the_cache_saved(project):
    cli, config = CliRunner(), ["--config", str(project.root / "tripwire.toml")]
    cli.invoke(app, ["run", "demo", *config])
    cli.invoke(app, ["run", "demo", *config])
    shown = cli.invoke(app, ["usage", *config])
    assert shown.exit_code == 0 and "mock:0.8" in shown.output
    assert "80 of 160 requested samples were already stored (50% of calls avoided)" in shown.output


def test_power_reports_the_two_stage_gate(project):
    cli, config = CliRunner(), ["--config", str(project.root / "tripwire.toml")]
    shown = cli.invoke(app, ["power", "demo", "--first-stage", "10", *config])
    assert shown.exit_code == 0 and "Two-stage gate: 10 cases first, all 40" in shown.output
