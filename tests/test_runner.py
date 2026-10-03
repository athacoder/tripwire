import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from tripwire import store
from tripwire.cli import app
from tripwire.config import TargetCfg
from tripwire.models import Case
from tripwire.providers import HttpProvider, Mock, Provider, ProviderError, Response
from tripwire.runner import run_suite
from tripwire.targets import Target, seed_for

TOTAL = 80  # 40 cases x 2 repetitions in the `project` fixture


class Scripted(Mock):
    """A mock that fails its first `failures` calls, then behaves; it counts every call."""

    def __init__(self, failures=0, error=None, reply=None, delay=0.0, digests=None):
        self.calls, self.failures, self.error = 0, failures, error
        self.reply, self.delay, self.digests = reply, delay, iter(digests or [])

    async def complete(self, r):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.calls <= self.failures:
            raise self.error
        return self.reply or await super().complete(r)

    async def digest(self, model):
        return next(self.digests, "mock")


def run(cfg, provider=None, **kwargs):
    return asyncio.run(run_suite(cfg, "demo", provider=provider or Scripted(), **kwargs))


def rows(cfg, table):
    return store.connect(cfg.root / cfg.db).execute(f"SELECT * FROM {table}").fetchall()


def test_a_second_run_calls_nothing(project):
    first, second = Scripted(), Scripted()
    assert run(project, first).statuses["ok"] == TOTAL and first.calls == TOTAL
    summary = run(project, second)
    assert second.calls == 0 and summary.cached == TOTAL and summary.pending == 0
    assert [r["status"] for r in rows(project, "runs")] == ["complete", "complete"]


def test_resume_fills_only_the_missing_slots(project):
    run(project, limit=10)
    provider = Scripted()
    summary = run(project, provider)
    assert provider.calls == TOTAL - 20 and summary.cached == 20
    assert len(rows(project, "samples")) == TOTAL


def test_samples_are_reproducible_from_the_seed(project):
    run(project)
    before = {
        (r["case_hash"], r["rep"]): (r["output"], r["seed"]) for r in rows(project, "samples")
    }
    db = store.connect(project.root / project.db)
    db.execute("DELETE FROM samples")
    db.commit()
    run(project)
    after = {(r["case_hash"], r["rep"]): (r["output"], r["seed"]) for r in rows(project, "samples")}
    assert before == after
    case_hash = next(iter(before))[0]
    assert seed_for(case_hash, 0) == seed_for(case_hash, 0) != seed_for(case_hash, 1)


def test_a_retryable_error_is_retried_and_counted(project):
    provider = Scripted(failures=2, error=ProviderError("HTTP 429", retryable=True))
    summary = run(project, provider, limit=1)
    assert (
        summary.errors == 0 and provider.calls == 4
    )  # 3 attempts for the first slot, 1 for the next
    assert sorted(r["attempts"] for r in rows(project, "samples")) == [1, 3]


@pytest.mark.parametrize(("retryable", "attempts"), [(True, 4), (False, 1)])
def test_a_failed_request_never_becomes_a_sample(project, retryable, attempts):
    provider = Scripted(failures=10**6, error=ProviderError("boom", retryable=retryable))
    summary = run(project, provider, limit=1)
    assert summary.errors == 2 and not rows(project, "samples")
    assert {r["attempts"] for r in rows(project, "errors")} == {attempts}
    assert rows(project, "runs")[0]["status"] == "partial"


def test_a_hung_request_times_out_into_errors(project):
    project.timeout = 0.05
    summary = run(project, Scripted(delay=5), limit=1)
    assert summary.errors == 2 and {r["kind"] for r in rows(project, "errors")} == {"timeout"}
    assert not rows(project, "samples")


def test_a_cut_off_answer_is_stored_as_truncated(project):
    summary = run(project, Scripted(reply=Response("par", "mock:0.8", stop="length")), limit=1)
    assert summary.statuses == {"truncated": 2}


def test_an_overlong_prompt_is_rejected_before_any_call(project):
    (project.root / "target.toml").write_text(
        'provider = "mock"\nmodel = "mock:0.8"\nnum_ctx = 16\n'
    )
    provider = Scripted()
    with pytest.raises(ValueError, match="num_ctx"):
        run(project, provider)
    assert provider.calls == 0


def test_a_digest_change_during_the_run_aborts_it(project):
    with pytest.raises(ProviderError, match="digest changed"):
        run(project, Scripted(digests=["before", "after"]))
    assert rows(project, "runs")[0]["status"] == "aborted"


def test_a_different_served_model_aborts_the_run(project):
    with pytest.raises(ProviderError, match="served by"):
        run(project, Scripted(reply=Response("x", "some-other-model")))


def test_the_time_budget_stops_cleanly_and_the_run_resumes(project):
    assert run(project, max_minutes=0).pending == TOTAL
    assert rows(project, "runs")[0]["status"] == "partial" and not rows(project, "samples")
    assert run(project).pending == 0


def test_dry_run_calls_nothing(project):
    provider = Scripted()
    summary = run(project, provider, dry_run=True)
    assert provider.calls == 0 and summary.pending == TOTAL and summary.projected_minutes is None


def fingerprint(root, digest="d", **overrides):
    cfg = TargetCfg(model="m", system="system.md", watch=["app/*.py"], **overrides)
    return Target(cfg, root, Provider(), "ollama", digest).fingerprint


def test_fingerprint_tracks_everything_that_changes_the_output(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "system.md").write_text("Answer briefly.\n")
    (tmp_path / "app" / "logic.py").write_bytes(b"x = 1\n")
    base = fingerprint(tmp_path)
    assert base == fingerprint(tmp_path)
    assert base != fingerprint(tmp_path, digest="other")
    assert base != fingerprint(tmp_path, salt="fresh")
    assert base != fingerprint(tmp_path, temperature=0.2)

    (tmp_path / "app" / "logic.py").write_bytes(b"x = 1\r\n")  # a Windows checkout
    assert base == fingerprint(tmp_path)
    (tmp_path / "app" / "logic.py").write_bytes(b"x = 2\n")
    assert base != fingerprint(tmp_path)
    (tmp_path / "app" / "logic.py").write_bytes(b"x = 1\n")
    (tmp_path / "system.md").write_text("Answer  briefly.\n")  # whitespace can change behaviour
    assert base != fingerprint(tmp_path)


def test_python_and_http_targets(tmp_path):
    (tmp_path / "myapp.py").write_text("def answer(case):\n    return case['text'].upper()\n")
    case = Case(input={"text": "hi"}, expected="HI")
    python = Target(TargetCfg(kind="python", entry="myapp:answer"), tmp_path, Provider())
    assert asyncio.run(python.run(case, 0)).text == "HI"

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"output": f"{body['input']['text']}:{body['seed']}"})

    http = Target(TargetCfg(kind="http", url="http://svc/predict"), tmp_path, Provider())
    http.http = HttpProvider("", transport=httpx.MockTransport(handler))
    assert asyncio.run(http.run(case, 0)).text == f"hi:{seed_for(case.hash, 0)}"


def test_cli_runs_a_suite_end_to_end(project):
    cli = CliRunner()
    assert cli.invoke(app, ["--version"]).exit_code == 0
    config = ["--config", str(project.root / "tripwire.toml")]
    first = cli.invoke(app, ["run", "demo", *config])
    assert first.exit_code == 0 and "80 generated" in first.output
    assert "80 cached, 0 generated" in cli.invoke(app, ["run", "demo", *config]).output
    assert cli.invoke(app, ["run", "nope", *config]).exit_code == 2
    assert cli.invoke(app, ["dataset", "lint", str(project.root / "data.jsonl")]).exit_code == 0


def test_a_crashing_target_is_recorded_and_the_run_continues(project):
    (project.root / "flaky_app.py").write_text(
        "def answer(case):\n"
        "    if case['text'].endswith('3'):\n"
        "        raise RuntimeError('bug in the app')\n"
        "    return 'fine'\n"
    )
    (project.root / "target.toml").write_text('kind = "python"\nentry = "flaky_app:answer"\n')
    summary = asyncio.run(run_suite(project, "demo"))
    assert summary.errors == 8 and summary.statuses["ok"] == TOTAL - 8  # cases 3, 13, 23, 33
    assert {r["kind"] for r in rows(project, "errors")} == {"target_error"}
    assert "bug in the app" in rows(project, "errors")[0]["message"]


def test_a_malformed_reply_is_an_error_not_a_crash(project):
    class Broken(Scripted):
        async def complete(self, r):
            return {}["message"]  # what a reply missing its fields does to the parser

    summary = run(project, Broken(), limit=1)
    assert summary.errors == 2 and not rows(project, "samples")


def test_a_fatal_error_stops_concurrent_work_cleanly(project):
    project.concurrency = 8
    with pytest.raises(ProviderError, match="served by"):
        run(project, Scripted(reply=Response("x", "some-other-model"), delay=0.01))
    assert rows(project, "runs")[0]["status"] == "aborted"


def test_a_killed_run_is_marked_aborted_by_the_next_one(project):
    run(project, limit=1)
    db = store.connect(project.db_path)
    db.execute("UPDATE runs SET status='running'")
    db.commit()
    run(project)
    assert [r["status"] for r in rows(project, "runs")] == ["aborted", "complete"]


def test_a_template_that_does_not_match_the_cases_is_explained(project):
    (project.root / "user.md").write_text("Question: {question}")
    (project.root / "target.toml").write_text(
        'provider = "mock"\nmodel = "mock:0.8"\ntemplate = "user.md"\n'
    )
    with pytest.raises(ValueError, match="cannot be filled"):
        run(project)


def test_an_unknown_suite_lists_the_known_ones(project):
    with pytest.raises(ValueError, match="known: demo"):
        asyncio.run(run_suite(project, "nope"))
