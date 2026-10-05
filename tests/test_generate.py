import asyncio
import json

import httpx
import pytest
from conftest import make_cases
from test_gate import git, repo, run_gate, set_target  # noqa: F401  (repo is a fixture)
from typer.testing import CliRunner

from tripwire import datasets, generate, report, store
from tripwire.cli import app
from tripwire.compare import summarise
from tripwire.models import Case
from tripwire.providers import Mock, ProviderError, Response
from tripwire.runner import run_suite
from tripwire.scorers import score_suite


def texts(cases):
    return [c.input["text"] for c in cases]


def test_perturb_keeps_originals_and_links_each_variant_to_its_parent():
    cases = [
        Case(input={"text": f"Please advise what should happen {i}"}, expected="x")
        for i in range(5)
    ]
    out = generate.perturb(cases, ["typo", "lowercase", "distractor"])
    originals, variants = out[:5], out[5:]
    assert [c.hash for c in originals] == [c.hash for c in cases]  # so cached samples are reused
    assert all(c.tags[-1] == "perturb:none" for c in originals) and len(variants) == 15
    parents = {c.hash: c for c in cases}
    for v in variants:
        parent = parents[v.provenance["parent"]]
        assert v.hash != parent.hash and v.expected == parent.expected and v.source == "perturbed"
        assert v.tags[-1] == f"perturb:{v.provenance['perturb']}"


def test_perturbations_do_what_they_say_and_are_reproducible():
    cases = [Case(input={"text": "Please advise what Should happen today"}, expected="x")]
    typo, lower, extra = generate.perturb(cases, ["typo", "lowercase", "distractor"])[1:]
    original = cases[0].input["text"]
    assert sorted(typo.input["text"]) == sorted(original) and typo.input["text"] != original
    assert lower.input["text"] == original.lower()
    assert extra.input["text"].startswith(original + " ") and len(extra.input["text"]) > len(
        original
    )
    again = generate.perturb(cases, ["typo", "lowercase", "distractor"])[1:]
    assert texts(again) == [typo.input["text"], lower.input["text"], extra.input["text"]]
    assert texts(generate.perturb(cases, ["typo"], seed=1)[1:]) != [typo.input["text"]]


def test_a_variant_identical_to_its_parent_is_dropped_and_bad_requests_are_rejected():
    already = [Case(input={"text": "all lower case already"}, expected="x")]
    assert len(generate.perturb(already, ["lowercase"])) == 1
    with pytest.raises(ValueError, match="unknown perturbation"):
        generate.perturb(already, ["explode"])
    with pytest.raises(ValueError, match="no input field"):
        generate.perturb(already, ["typo"], field="question")


def test_shuffle_lines_changes_only_the_order_of_a_chosen_field():
    context = "Policies\n\nRefunds: 30 days.\nShipping: 5 days.\nWarranty: 12 months."
    case = Case(input={"context": context, "question": "How long is the warranty?"}, expected="12")
    shuffled = generate.perturb([case], ["shuffle_lines"], field="context")[1]
    before, after = context.split("\n"), shuffled.input["context"].split("\n")
    assert after[:2] == before[:2] and sorted(after) == sorted(before) and after != before
    assert shuffled.input["question"] == case.input["question"]


class Scripted(Mock):
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    async def complete(self, r):
        self.requests.append(r)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Response(json.dumps(reply), r.model)


def audit(ambiguous=False, wrong=False):
    return {"ambiguous": ambiguous, "answer_wrong": wrong, "reason": "r"}


def test_drafted_cases_are_screened_before_they_become_candidates():
    existing = [
        Case(input={"text": "my card has not arrived"}, expected="card_arrival"),
        Case(input={"text": "the machine kept my card"}, expected="card_swallowed"),
    ]
    drafts = [
        {"input": {"text": "my card has not arrived"}, "expected": "card_arrival"},  # duplicate
        {
            "input": {"text": "My card has not arrived!"},
            "expected": "card_arrival",
        },  # near duplicate
        {"input": {"text": "where is the card I ordered last week"}, "expected": "card_arrival"},
        {"input": {"text": "an unclear message about things"}, "expected": "card_arrival"},
        {"input": {"text": "the cash point swallowed my bank card"}, "expected": "card_arrival"},
        {"input": {"text": "nobody audited this one at all"}, "expected": "card_swallowed"},
    ]
    backend = Scripted(
        [
            {"cases": drafts},
            audit(),
            audit(ambiguous=True),
            audit(wrong=True),
            ProviderError("down"),
        ]
    )
    kept, dropped = asyncio.run(
        generate.draft(backend, "drafter", "checker", existing, "seed text", 6, existing)
    )
    assert texts(kept) == ["where is the card I ordered last week"]
    assert dropped == {"duplicate": 1, "near_duplicate": 1, "flagged": 2, "unverified": 1}
    case = kept[0]
    assert case.source == "llm-drafted" and case.provenance["status"] == "pending"
    assert (
        case.provenance["drafted_by"] == "drafter" and case.provenance["verified_by"] == "checker"
    )

    ask, check = backend.requests[0], backend.requests[1]
    schema = ask.format["properties"]["cases"]["items"]["properties"]
    assert schema["expected"]["enum"] == ["card_arrival", "card_swallowed"]  # only real labels
    assert schema["input"]["required"] == ["text"] and "seed text" in ask.user
    assert (ask.model, check.model) == ("drafter", "checker")


def test_a_drafting_reply_that_is_not_cases_is_an_error():
    backend = Scripted([{"something": "else"}])
    with pytest.raises(ProviderError, match="usable cases"):
        asyncio.run(generate.draft(backend, "m", "m", make_cases(2), "seed", 3, []))


def test_a_model_is_never_scored_against_cases_it_drafted(project):
    cases = make_cases(4)
    drafted = Case(input={"text": "drafted"}, expected="x", provenance={"drafted_by": "mock:0.8"})
    datasets.save(project.root / "data.jsonl", [*cases, drafted])
    with pytest.raises(ValueError, match="drafted by mock:0.8"):
        asyncio.run(run_suite(project, "demo"))


# Shapes recorded from a running TraceLens 0.1.0.
SUMMARIES = [
    {"trace_id": "t1", "pipeline": "rag", "root_cause_stage": "retrieval"},
    {"trace_id": "t2", "pipeline": "rag", "root_cause_stage": None},  # healthy
    {"trace_id": "t3", "pipeline": "rag", "root_cause_stage": "tool"},
]
TRACE = {
    "spans": [
        {
            "name": "preprocess",
            "inputs": {"user_input": "How long does a refund take?"},
            "outputs": {},
        },
        {"name": "llm", "inputs": {"prompt": "..."}, "outputs": {"answer": "68 business days."}},
    ]
}
ROOT_CAUSE = {
    "summary": "fallback summary",
    "likely_root_cause": {
        "summary": "retriever returned refund-2019 instead of refund-2026",
        "candidates": [{"category": "retrieval_failure"}],
    },
}


def tracelens(request):
    path = request.url.path
    if path == "/api/v1/traces":
        offset = int(request.url.params["offset"])
        page = {"items": SUMMARIES[offset : offset + 2], "has_more": offset + 2 < len(SUMMARIES)}
        return httpx.Response(200, json=page)
    return httpx.Response(200, json=ROOT_CAUSE if path.endswith("/root-cause") else TRACE)


def test_tracelens_failures_become_candidates_without_an_expected_answer():
    transport = httpx.MockTransport(tracelens)
    found = asyncio.run(generate.import_tracelens("http://tl", transport=transport))
    assert [c.provenance["trace_id"] for c in found] == ["t1", "t3"]  # paged, healthy skipped
    case = found[0]
    assert case.input == {"user_input": "How long does a refund take?"} and case.expected is None
    assert case.tags == ["stage:retrieval", "failure:retrieval_failure", "pipeline:rag"]
    assert case.source == "production-failure" and case.provenance["status"] == "pending"
    assert "refund-2019" in case.provenance["root_cause"]
    assert "68 business days" in case.provenance["observed_output"]


def test_review_adds_only_approved_cases_and_remembers_every_decision(tmp_path):
    candidates, dataset = tmp_path / "candidates.jsonl", tmp_path / "data.jsonl"
    pending = {"status": "pending"}
    datasets.save(
        candidates,
        [
            Case(input={"text": "no answer yet"}, source="production-failure", provenance=pending),
            Case(
                input={"text": "good draft"}, expected="a", source="llm-drafted", provenance=pending
            ),
            Case(
                input={"text": "bad draft"}, expected="a", source="llm-drafted", provenance=pending
            ),
            Case(
                input={"text": "wrong label"},
                expected="a",
                source="llm-drafted",
                provenance=pending,
            ),
            Case(
                input={"text": "for later"}, expected="a", source="llm-drafted", provenance=pending
            ),
        ],
    )
    datasets.save(dataset, [Case(input={"text": "good draft"}, expected="a")])
    cli = CliRunner()
    command = ["dataset", "review", str(candidates), "--into", str(dataset), "--reviewer", "ana"]
    # approve (must supply the answer), approve a duplicate, reject, edit, then quit
    done = cli.invoke(app, command, input="a\n5 business days\na\nr\ne\nb\nq\n")
    assert (
        done.exit_code == 0
        and "2 cases added" in done.output
        and "1 candidates still pending" in done.output
    )

    stored = datasets.load(dataset)
    assert [(c.input["text"], c.expected) for c in stored] == [
        ("good draft", "a"),
        ("no answer yet", "5 business days"),
        ("wrong label", "b"),
    ]
    assert (
        stored[1].provenance == {"reviewed_by": "ana"} and stored[1].source == "production-failure"
    )
    statuses = [c.provenance["status"] for c in datasets.load(candidates)]
    assert statuses == ["approved", "approved", "rejected", "approved", "pending"]
    again = cli.invoke(app, command, input="s\n")
    assert "0 cases added" in again.output and len(datasets.load(dataset)) == 3


def test_reviewer_input_is_read_as_json_only_when_it_is_json():
    assert generate.parse_expected(' {"value": 3} ') == {"value": 3}
    assert generate.parse_expected("42") == 42
    assert generate.parse_expected("5 business days") == "5 business days"
    assert generate.parse_expected("card_arrival") == "card_arrival"


def test_reports_pair_each_perturbed_case_with_its_original(project):
    cases = generate.perturb(make_cases(40), ["typo", "distractor"])
    datasets.save(project.root / "data.jsonl", cases)
    s = asyncio.run(run_suite(project, "demo"))
    db = store.connect(project.db_path)
    score_suite(project, "demo", s.fingerprint, db)
    summary = summarise(db, project.suite["demo"], cases, s.fingerprint)
    rows = {r["kind"]: r for r in summary["robustness"]}
    assert set(rows) == {"typo", "distractor"} and rows["typo"]["n"] == 40
    assert all(r["lo"] <= r["delta"] <= r["hi"] and 0 <= r["changed"] <= 1 for r in rows.values())
    assert "### Robustness" in report.single("demo", s.fingerprint, summary)
    plain = summarise(db, project.suite["demo"], cases[:40], s.fingerprint)
    assert plain["robustness"] == []  # originals only: nothing to pair


def test_a_trace_id_from_the_target_reaches_the_report(repo):  # noqa: F811
    code = (
        "def answer(case):\n    n = case['text'].split()[-1]\n"
        "    return {{'output': {}, 'trace_id': 'trace-' + n}}\n"
    )
    (repo.root / "app.py").write_text(code.format("'answer ' + n"))
    set_target(repo, 'kind = "python"\nentry = "app:answer"\nwatch = ["app.py"]\n')
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-qm", "an app that reports trace ids")
    (repo.root / "app.py").write_text(code.format("'broken'"))
    result, _ = run_gate(repo)
    example = result["flips"]["broke_examples"][0]
    assert result["verdict"] == "REGRESSED" and example["trace_id"].startswith("trace-")
    text = report.comparison("demo", result, True, "http://localhost:3000/")
    assert f"http://localhost:3000/traces/{example['trace_id']}" in text
    assert "| trace |" in text and "trace" not in report.comparison(
        "demo", {"verdict": "UNCHANGED"}
    )
