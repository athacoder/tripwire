import asyncio
import subprocess

import pytest
from typer.testing import CliRunner

from tripwire import report, store
from tripwire.cli import app
from tripwire.compare import blocked, compare
from tripwire.config import SuiteCfg
from tripwire.gate import bisect, canary, gate
from tripwire.models import Case
from tripwire.providers import Mock, Response


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
    assert result["flips"]["broke"] == 30 and len(result["flips"]["broke_examples"]) == 30
    # The terminal report lists the first ten; a page has room for all of them.
    assert "(top 10 of 30)" in report.comparison("demo", result)
    page = report.to_html(report.comparison("demo", result, shown=(50, 50)), "demo")
    assert "(top 30 of 30)" in page and page.count("<tr>") > 30 and "<table>" in page


def test_a_page_never_runs_what_a_model_wrote():
    said = '<script>alert(1)</script> [x](http://a"onmouseover="alert(2)) | `code`'
    flips = {
        "broke": 1, "fixed": 0, "flaky": 0, "stable_pass": 0, "stable_fail": 0,
        "fixed_examples": [],
        "broke_examples": [
            {"text": "q", "expected": "a", "base_output": "a", "head_output": said, "delta": -1.0}
        ],
    }  # fmt: skip
    result = {
        "verdict": "REGRESSED", "metric": "exact.pass", "alpha": 0.05, "margin": 0.03,
        "base": 1.0, "head": 0.0, "delta": -1.0, "lo": -1.0, "hi": -1.0, "p": 0.01,
        "paired": 1, "cases": 1, "slices": [], "guardrails": [], "flips": flips,
    }  # fmt: skip
    page = report.to_html(report.comparison("demo", result), "a <title>")
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert 'onmouseover="' not in page and "<title>a &lt;title&gt;</title>" in page
    last_row = page.rsplit("<tr>", 1)[1]
    assert "<code>code</code>" in last_row and last_row.count("<td>") == 4  # the pipe is text


def commit(root, path, text, message):
    (root / path).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return done.stdout.strip()


WEAK = 'provider = "mock"\nmodel = "mock:0.4"\n'


def test_bisect_finds_the_commit_that_broke_the_suite(repo):
    good = commit(repo.root, "notes.md", "start", "notes")
    shas = [
        commit(repo.root, "notes.md", "more", "unrelated change"),
        commit(
            repo.root, "target.toml", 'provider = "mock"\nmodel = "mock:0.8"\nsalt = "a"\n', "x"
        ),
        commit(repo.root, "target.toml", WEAK, "weaker model"),  # the one to find
        commit(repo.root, "notes.md", "even more", "unrelated again"),
        commit(repo.root, "target.toml", WEAK + 'salt = "b"\n', "same answers"),
    ]
    lines = []
    found = asyncio.run(bisect(repo, "demo", good, report=lines.append))
    assert found["first_bad"] == shas[2] and found["candidates"] == [shas[2]]
    assert found["commits"] == 5 and found["tested"] < 5 and len(lines) == found["tested"]
    assert any("REGRESSED" in line and "weaker model" in line for line in lines)
    assert (repo.root / "notes.md").read_text() == "even more"  # the working tree was left alone

    with pytest.raises(ValueError, match="nothing to find"):
        asyncio.run(bisect(repo, "demo", good, shas[1]))
    cli, config = CliRunner(), ["--config", str(repo.root / "tripwire.toml")]
    shown = cli.invoke(app, ["bisect", "demo", "--good", good, *config])
    assert shown.exit_code == 0 and f"first blocked commit: {shas[2]}" in shown.output


def test_bisect_says_so_when_the_gate_cannot_decide_a_commit(repo):
    good = commit(repo.root, "notes.md", "start", "notes")
    unclear = commit(repo.root, "target.toml", 'provider = "mock"\nmodel = "mock:0.74"\n', "?")
    bad = commit(repo.root, "target.toml", WEAK, "weaker model")
    found = asyncio.run(bisect(repo, "demo", good))
    assert found["first_bad"] is None and found["candidates"] == [unclear, bad]


class Wrong(Mock):
    """The same model name, different behaviour: what a changed runtime looks like."""

    async def complete(self, r):
        return Response("wrong", r.model)


class Repulled(Mock):
    async def digest(self, model):
        return "a new digest"


def test_the_canary_compares_a_fresh_run_with_a_pinned_one(project):
    def rows():
        db = store.connect(project.db_path)
        found = db.execute("SELECT base, head, verdict FROM comparisons ORDER BY id").fetchall()
        return [tuple(r) for r in found]

    result, is_blocked = asyncio.run(canary(project, "demo", size=12))
    assert (result["identical"], result["subset"], result["verdict"]) == (12, 12, "PASS")
    assert not is_blocked and samples(project) == 48  # 12 cases x 2 reps, pinned and fresh
    (pinned, _, first), (base, fresh, verdict) = rows()
    assert (first, verdict) == ("PINNED", "PASS") and base == pinned != fresh

    # Something underneath changed: same configuration, different answers.
    result, is_blocked = asyncio.run(canary(project, "demo", size=12, provider=Wrong()))
    assert result["verdict"] == "REGRESSED" and is_blocked and result["identical"] < 12
    assert rows()[-1][0] == pinned  # still measured against the first reference

    # A model pulled again has a new digest; the reference must stay the old one.
    asyncio.run(canary(project, "demo", size=12, provider=Repulled()))
    assert rows()[-1][0] == pinned and rows()[-1][2] == "PASS"
    asyncio.run(canary(project, "demo", size=12, pin=True, provider=Repulled()))
    assert rows()[-2][2] == "PINNED" and rows()[-1][0] == rows()[-2][0] != pinned

    cli, config = CliRunner(), ["--config", str(project.root / "tripwire.toml")]
    out = project.root / "canary.html"
    shown = cli.invoke(app, ["canary", "demo", "--cases", "12", "--out", str(out), *config])
    assert shown.exit_code == 0 and "12 of 12 answers are identical" in shown.output
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_swapping_the_sides_of_a_comparison_swaps_the_scores(project):
    (project.root / "weak.toml").write_text(WEAK)
    with (project.root / "tripwire.toml").open("a") as f:
        f.write('[suite.once]\ndataset = "data.jsonl"\ntarget = "weak.toml"\n')  # 1 repetition
    cli, config = CliRunner(), ["--config", str(project.root / "tripwire.toml")]

    def scores(base, head):
        shown = cli.invoke(app, ["compare", base, head, *config]).output
        row = next(line for line in shown.splitlines() if line.startswith("| exact.pass"))
        return [cell.strip() for cell in row.split("|")][2:4]

    scores("demo", "once")  # generate; then make the second repetition differ from the first
    db = store.connect(project.db_path)
    db.execute("UPDATE scores SET value = 0 WHERE rep = 1")
    db.commit()
    assert scores("demo", "once") == scores("once", "demo")[::-1]
