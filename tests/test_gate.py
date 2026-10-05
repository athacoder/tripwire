import asyncio
import subprocess

import pytest
from typer.testing import CliRunner

from tripwire import store
from tripwire.cli import app
from tripwire.compare import blocked, compare
from tripwire.config import SuiteCfg
from tripwire.gate import gate
from tripwire.models import Case


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(project):
    """The mock project as a git repository with one commit."""
    (project.root / ".gitignore").write_text("*.db*\nbundles/\n")
    git(project.root, "init", "-q", "-b", "main")
    git(project.root, "config", "user.email", "test@example.com")
    git(project.root, "config", "user.name", "Test")
    git(project.root, "add", "-A")
    git(project.root, "commit", "-q", "-m", "baseline")
    return project


def set_target(cfg, text):
    (cfg.root / "target.toml").write_text(text)


def run_gate(cfg, **kwargs):
    return asyncio.run(gate(cfg, "demo", "HEAD", **kwargs))


def samples(cfg):
    return store.connect(cfg.db_path).execute("SELECT count(*) FROM samples").fetchone()[0]


def test_an_unchanged_target_passes_without_generating_anything(repo):
    result, is_blocked = run_gate(repo)
    assert result["verdict"] == "UNCHANGED" and not is_blocked
    assert not repo.db_path.exists()


def test_a_large_drop_is_blocked(repo):
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.4"\n')
    result, is_blocked = run_gate(repo)
    assert result["verdict"] == "REGRESSED" and is_blocked
    assert result["delta"] < -0.2 and result["hi"] < 0 and result["paired"] == 40
    assert result["flips"]["broke"] > result["flips"]["fixed"]


def test_a_change_with_identical_outputs_passes(repo):
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.8"\nsalt = "new"\n')  # new fingerprint
    result, is_blocked = run_gate(repo)
    assert result["verdict"] == "PASS" and not is_blocked
    assert (result["delta"], result["lo"], result["hi"]) == (0, 0, 0)


def test_too_little_data_is_inconclusive_and_blocks_only_when_configured(repo):
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.74"\n')
    result, is_blocked = run_gate(repo)
    assert result["verdict"] == "INCONCLUSIVE" and is_blocked
    assert result["lo"] <= -0.03 and result["hi"] >= 0
    repo.suite["demo"].on_inconclusive = "warn"
    assert not run_gate(repo)[1]


def test_a_broken_candidate_is_invalid_not_regressed(repo):
    (repo.root / "app.py").write_text("def answer(case):\n    raise RuntimeError('down')\n")
    set_target(repo, 'kind = "python"\nentry = "app:answer"\n')
    result, is_blocked = run_gate(repo)
    assert result["verdict"] == "INVALID" and is_blocked and "0 of 40" in result["reason"]


def test_python_targets_are_loaded_separately_for_base_and_head(repo):
    code = "def answer(case):\n    return 'answer ' + case['text'].split()[-1]\n"
    (repo.root / "app.py").write_text(code)
    set_target(repo, 'kind = "python"\nentry = "app:answer"\nwatch = ["app.py"]\n')
    git(repo.root, "commit", "-qam", "use the app")
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-qm", "add the app")
    (repo.root / "app.py").write_text("def answer(case):\n    return 'broken'\n")
    result, _ = run_gate(repo)
    assert (result["base"], result["head"], result["verdict"]) == (1.0, 0.0, "REGRESSED")


def test_two_committed_refs_can_be_compared(repo):
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.4"\n')
    git(repo.root, "commit", "-qam", "weaker model")
    result, _ = asyncio.run(gate(repo, "demo", "HEAD~1", "HEAD"))
    assert result["verdict"] == "REGRESSED"
    with pytest.raises(ValueError, match="cannot read git ref"):
        asyncio.run(gate(repo, "demo", "no-such-ref"))
    # A "ref" that git would read as an option must never reach it.
    planted = repo.root / "planted.tar"
    with pytest.raises(ValueError, match="not a git ref"):
        asyncio.run(gate(repo, "demo", f"--output={planted}"))
    assert not planted.exists()


def test_bundles_let_ci_reach_the_same_verdict_without_a_model(repo, monkeypatch):
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.4"\n')
    local, _ = run_gate(repo, bundle=True)
    bundles = sorted((repo.root / "bundles").rglob("*.jsonl.gz"))
    assert len(bundles) == 2
    before = [b.read_bytes() for b in bundles]

    repo.db_path.unlink()  # CI starts with no database and no model

    async def no_model(*args, **kwargs):
        raise AssertionError("verify-only must not generate samples")

    monkeypatch.setattr("tripwire.gate.run_suite", no_model)
    verified, is_blocked = run_gate(repo, verify_only=True)
    assert is_blocked and verified["verdict"] == local["verdict"]
    assert verified["delta"] == local["delta"] and samples(repo) == 160

    run_gate(repo, verify_only=True)
    assert [b.read_bytes() for b in bundles] == before  # verification never rewrites bundles


def test_a_missing_bundle_blocks_with_instructions(repo):
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.4"\n')
    result, is_blocked = run_gate(repo, verify_only=True)
    assert result["verdict"] == "INVALID" and is_blocked
    assert "--bundle" in result["reason"]


def test_cli_exit_codes_and_report_file(repo):
    cli, config = CliRunner(), ["--config", str(repo.root / "tripwire.toml")]
    assert cli.invoke(app, ["gate", "demo", "--base", "HEAD", *config]).exit_code == 0
    set_target(repo, 'provider = "mock"\nmodel = "mock:0.4"\n')
    out = repo.root / "gate.md"
    done = cli.invoke(app, ["gate", "demo", "--base", "HEAD", "--out", str(out), *config])
    assert done.exit_code == 1 and "REGRESSED" in out.read_text(encoding="utf-8")
    assert "This blocks the merge" in done.output and "Broke" in done.output
    assert (
        cli.invoke(app, ["gate", "demo", "--verify-only", "--base", "HEAD", *config]).exit_code == 3
    )
    assert cli.invoke(app, ["report", "demo", *config]).exit_code == 0
    assert cli.invoke(app, ["selfcheck", "demo", *config]).exit_code == 0


def seed(db, fingerprint, scores, tokens):
    """Write samples and scores for one fingerprint directly: 60 cases in two tagged halves."""
    for i, value in enumerate(scores):
        key = {"fingerprint": fingerprint, "case_hash": CASES[i].hash, "rep": 0}
        sample = {
            **key, "status": "ok", "output": "x", "model": "m", "seed": 0, "prompt_tokens": 1,
            "output_tokens": tokens, "latency_ms": 100.0, "attempts": 1, "response": "{}",
            "created_at": "t",
        }  # fmt: skip
        store.insert(db, "samples", [sample])
        score = {**key, "scorer": "exact", "version": "1", "metric": "pass", "value": value}
        store.insert(db, "scores", [score])


CASES = [
    Case(input={"text": f"case {i}"}, expected="x", tags=["half:a" if i < 30 else "half:b"])
    for i in range(60)
]


def test_a_guardrail_blocks_even_when_quality_holds(tmp_path):
    db = store.connect(tmp_path / "t.db")
    seed(db, "base", [1.0] * 60, tokens=10)
    seed(db, "head", [1.0] * 60, tokens=20)
    suite = SuiteCfg(dataset="d", target="t", guardrails={"output_tokens_ratio_max": 1.25})
    result = compare(db, suite, CASES, "base", "head")
    assert result["verdict"] == "PASS" and blocked(result, suite)
    assert result["guardrails"][0]["value"] == 2 and not result["guardrails"][0]["ok"]


def test_a_collapsed_slice_is_flagged(tmp_path):
    db = store.connect(tmp_path / "t.db")
    seed(db, "base", [1.0] * 60, tokens=10)
    seed(db, "head", [0.0] * 30 + [1.0] * 30, tokens=10)  # half "a" breaks completely
    suite = SuiteCfg(dataset="d", target="t")
    result = compare(db, suite, CASES, "base", "head")
    flagged = {s["tag"]: s["regressed"] for s in result["slices"]}
    assert flagged == {"half:a": True, "half:b": False}
    assert result["flips"]["broke"] == 30 and len(result["flips"]["broke_examples"]) == 10
