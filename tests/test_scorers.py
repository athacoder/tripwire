import asyncio
import json

import pytest

from tripwire import scorers, store
from tripwire.runner import run_suite
from tripwire.scorers import score_suite, selfcheck

LABELS = {
    "labels": ["card_arrival", "card_linking", "top_up", "pending_top_up", "reverted_card_payment"]
}


@pytest.mark.parametrize(
    ("output", "expected", "passed"),
    [
        ("card_arrival\n", "card_arrival", 1),
        ("  Card_Arrival. ", "card_arrival", 1),
        ("```\ncard_arrival\n```", "card_arrival", 1),
        ('"card_arrival"', "card_arrival", 1),
        ("The intent is card_arrival.", "card_arrival", 1),
        ("reverted_card_payment", "reverted_card_payment?", 1),  # the label's own "?" is optional
        ("card_arrival or card_linking", "card_arrival", 0),  # hedging between two labels
        ("pending_top_up", "top_up", 0),  # a longer label is not the shorter one
        ("card_linking", "card_arrival", 0),
        ("", "card_arrival", 0),
        ("I don't know", "card_arrival", 0),
    ],
)
def test_exact_forgives_formatting_but_not_content(output, expected, passed):
    assert scorers.exact(output, expected, LABELS) == {"pass": passed}


def test_numeric_reads_the_number_out_of_a_sentence():
    assert scorers.numeric("The total is 1,234.50 USD", 1234.5, {})["pass"] == 1
    assert scorers.numeric("4", "4.0", {})["pass"] == 1
    assert scorers.numeric("about 5", 4, {})["pass"] == 0
    assert scorers.numeric("no number here", 4, {})["pass"] == 0


def test_contains_and_regex():
    assert scorers.contains("Refunds take 14 days, by card.", ["14 days", "card"], {})["pass"] == 1
    assert scorers.contains("Refunds take 14 days.", ["14 days", "card"], {})["pass"] == 0
    assert scorers.regex("order #A-1042 shipped", r"#A-\d{4}", {})["pass"] == 1


def test_field_f1_scores_each_field_and_tolerates_wrapping():
    expected = {"vendor": "Acme Ltd", "total": 1234.5, "currency": "USD"}
    perfect = (
        'Here you go:\n```json\n{"vendor": "acme ltd", "total": "1,234.50", "currency": "USD"}\n```'
    )
    assert scorers.field_f1(perfect, expected, {}) == {
        "valid": 1, "precision": 1, "recall": 1, "f1": 1, "pass": 1,
    }  # fmt: skip
    partial = scorers.field_f1('{"vendor": "Acme Ltd", "total": 99, "extra": 1}', expected, {})
    assert partial["recall"] == pytest.approx(1 / 3) and partial["precision"] == pytest.approx(
        1 / 3
    )
    assert partial["pass"] == 0
    broken = scorers.field_f1("not json at all", expected, {})
    assert broken == {"valid": 0, "precision": 0, "recall": 0, "f1": 0, "pass": 0}
    assert scorers.json_valid("[1, 2]", None, {})["pass"] == 1


def test_token_f1():
    assert scorers.token_f1("the cat sat", "the cat sat", {})["f1"] == 1
    assert scorers.token_f1("dog", "the cat sat", {})["f1"] == 0
    assert 0 < scorers.token_f1("the cat ran away", "the cat sat", {})["f1"] < 1


def count(cfg, where="1"):
    db = store.connect(cfg.db_path)
    return db.execute(f"SELECT count(*) FROM scores WHERE {where}").fetchone()[0]


def test_scoring_is_idempotent_and_versioned(project, monkeypatch):
    s = asyncio.run(run_suite(project, "demo"))
    db = store.connect(project.db_path)
    assert score_suite(project, "demo", s.fingerprint, db) == 80
    assert score_suite(project, "demo", s.fingerprint, db) == 0  # nothing new to score

    # A changed scorer adds rows under its new version and leaves the old ones alone.
    monkeypatch.setitem(scorers.SCORERS, "exact", (lambda o, e, c: {"pass": 1.0}, "2"))
    assert score_suite(project, "demo", s.fingerprint, db) == 80
    assert count(project, "version='1'") == 80 and count(project, "version='2' AND value=1") == 80


def test_truncated_answers_are_not_scored_and_refusals_score_zero(project):
    s = asyncio.run(run_suite(project, "demo"))
    db = store.connect(project.db_path)
    db.execute("UPDATE samples SET status='truncated' WHERE rep=0")
    db.execute("UPDATE samples SET status='refusal' WHERE rep=1")
    db.commit()
    assert score_suite(project, "demo", s.fingerprint, db) == 40
    assert count(project, "value > 0") == 0


def test_an_unknown_scorer_or_metric_is_rejected(project):
    project.suite["demo"].scorers = ["nope"]
    project.suite["demo"].primary_metric = "nope.pass"
    with pytest.raises(ValueError, match="unknown scorer"):
        scorers.primary(project.suite["demo"])
    project.suite["demo"].scorers = ["contains"]
    with pytest.raises(ValueError, match="not among"):
        scorers.primary(project.suite["demo"])


def test_selfcheck_accepts_a_sound_eval_and_rejects_a_lenient_scorer(project, monkeypatch):
    report = selfcheck(project, "demo")
    assert report["ok"] and report["oracle (reference answers)"] == 1
    monkeypatch.setitem(scorers.SCORERS, "exact", (lambda o, e, c: {"pass": 1.0}, "x"))
    assert not selfcheck(project, "demo")["ok"]  # an empty answer must not pass


def test_selfcheck_handles_structured_answers(project):
    record = {"vendor": "Acme", "total": 10.0}
    lines = [json.dumps({"input": {"text": f"invoice {i}"}, "expected": record}) for i in range(5)]
    (project.root / "data.jsonl").write_text("\n".join(lines) + "\n")
    project.suite["demo"].scorers = ["field_f1"]
    project.suite["demo"].primary_metric = "field_f1.f1"
    assert selfcheck(project, "demo")["ok"]
