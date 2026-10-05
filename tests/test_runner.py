import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from tripwire import datasets, store
from tripwire.cli import app
from tripwire.config import TargetCfg, load_config
from tripwire.models import Case
from tripwire.providers import HttpProvider, Mock, Provider, ProviderError, Response
from tripwire.runner import offline_target, run_suite
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


def test_a_case_listed_twice_is_generated_once(project):
    cases = datasets.load(project.root / "data.jsonl")
    datasets.save(project.root / "data.jsonl", [*cases[:5], cases[0]])
    provider = Scripted()
    summary = run(project, provider)
    assert provider.calls == 10 and summary.total == 10 and summary.errors == 0


def test_the_server_identifies_a_backend_only_when_no_digest_can(project):
    """Two OpenAI-compatible servers offering one model name must not share samples, and a
    move from one to the other is a change the gate has to see. Ollama models carry a
    digest, so there the address stays out: another machine, same target."""
    text = (project.root / "tripwire.toml").read_text()
    for name, kind, url in (
        ("a", "openai_compat", "http://localhost:1234/v1"),
        ("b", "openai_compat", "https://api.example.com/v1"),
        ("c", "ollama", "http://localhost:11434"),
        ("d", "ollama", "http://another-machine:11434"),
    ):
        text += f'[provider.{name}]\nkind = "{kind}"\nbase_url = "{url}"\n'
        text += f'[suite.{name}]\ndataset = "data.jsonl"\ntarget = "{name}.toml"\n'
        (project.root / f"{name}.toml").write_text(f'provider = "{name}"\nmodel = "llama3"\n')
    (project.root / "tripwire.toml").write_text(text)
    cfg = load_config(project.root / "tripwire.toml")
    key = {name: offline_target(cfg, name).static_key for name in "abcd"}
    assert key["a"] != key["b"] and key["c"] == key["d"] and key["a"] != key["c"]


def test_a_misspelt_setting_is_rejected_not_ignored(project):
    path = project.root / "tripwire.toml"
    text = path.read_text()
    for bad, message in (
        ("concurency = 4\n" + text, "concurency"),
        (text + "[suite.demo.guardrails]\nlatency_p95_ratio = 1.5\n", "unknown guardrail"),
        (text + "alpha = 0.7\n", "alpha"),
    ):
        path.write_text(bad)
        with pytest.raises(ValueError, match=message):
            load_config(path)
    path.write_text(text + "[suite.demo.guardrails]\nlatency_p95_ratio_max = 1.5\n")
    assert load_config(path).suite["demo"].guardrails == {"latency_p95_ratio_max": 1.5}


def test_a_missing_or_broken_config_is_one_line_not_a_traceback(project):
    cli = CliRunner()
    missing = cli.invoke(app, ["run", "demo", "--config", str(project.root / "nope.toml")])
    assert missing.exit_code == 2 and "error:" in missing.output
    (project.root / "tripwire.toml").write_text("not = [valid")
    broken = cli.invoke(app, ["run", "demo", "--config", str(project.root / "tripwire.toml")])
    assert broken.exit_code == 2 and "error:" in broken.output
    lint = ["dataset", "lint", str(project.root / "data.jsonl"), "--prompts"]
    assert cli.invoke(app, [*lint, str(project.root / "no_such_folder")]).exit_code == 2
