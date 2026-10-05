from collections import Counter

import pytest
from conftest import make_cases

from tripwire import datasets
from tripwire.models import Case


def test_hash_ignores_key_order_whitespace_and_tags():
    a = Case(input={"text": "where is  my card", "lang": "en"}, expected="card_arrival")
    b = Case(
        input={"lang": "en", "text": " where is my card\n"}, expected="card_arrival", tags=["x"]
    )
    assert a.hash == b.hash


def test_hash_depends_on_input_and_expected():
    base = Case(input={"text": "hello"}, expected="a")
    assert base.hash != Case(input={"text": "hello!"}, expected="a").hash
    assert base.hash != Case(input={"text": "hello"}, expected="b").hash


def test_version_is_order_independent_and_changes_with_any_case():
    cases = make_cases(10)
    assert datasets.version(cases) == datasets.version(cases[::-1])
    changed = [*cases[:-1], Case(input={"text": "different"}, expected="x")]
    assert datasets.version(cases) != datasets.version(changed)


def test_save_and_load_round_trip(tmp_path):
    cases = make_cases(5)
    # Unicode line separators inside a text are not line ends of the file.
    cases.append(Case(input={"text": "first second\u0085third"}, expected="x"))
    datasets.save(tmp_path / "d.jsonl", cases)
    loaded = datasets.load(tmp_path / "d.jsonl")
    assert [c.hash for c in loaded] == [c.hash for c in cases]
    assert loaded[-1].input == cases[-1].input


def test_lint_passes_a_clean_dataset():
    report = datasets.lint(make_cases(), context="an unrelated system prompt")
    assert report["ok"] and report["labels"] == 40 and report["majority_baseline"] == 1 / 40


def test_lint_catches_planted_problems():
    cases = make_cases(6)
    cases.append(cases[0].model_copy())  # exact duplicate
    cases.append(Case(input={"text": "question number 1"}, expected="another label"))  # conflict
    # Same words as case 2 but different punctuation: a near duplicate, not a conflict.
    cases.append(Case(input={"text": "Question, number 2?"}, expected="answer 2b"))
    cases.append(Case(input={"text": ""}, expected="nothing"))  # empty input
    cases.append(Case(input={"text": "the refund is the answer here"}, expected="refund"))
    report = datasets.lint(cases, context="Example:\nQuestion   number 3\n")
    assert report["duplicates"] == 1
    assert report["conflicting_labels"] == 1
    assert len(report["near_duplicates"]) >= 1
    assert report["empty"] == 1
    assert report["leaked_into_prompt"] == 1
    assert report["answer_in_input"] == 1
    assert not report["ok"]


def test_lint_reports_majority_baseline_for_an_imbalanced_set():
    cases = [
        Case(input={"text": f"message {i}"}, expected="a" if i < 9 else "b") for i in range(10)
    ]
    assert datasets.lint(cases)["majority_baseline"] == 0.9


def test_split_is_disjoint_reproducible_and_order_independent():
    cases = make_cases(40)
    sizes = {"dev": 8, "gate": 16}
    first = datasets.split(cases, sizes, seed=1)
    assert [len(first[k]) for k in sizes] == [8, 16]
    assert not {c.hash for c in first["dev"]} & {c.hash for c in first["gate"]}
    again = datasets.split(cases[::-1], sizes, seed=1)
    assert [c.hash for c in first["gate"]] == [c.hash for c in again["gate"]]
    other = datasets.split(cases, sizes, seed=2)
    assert [c.hash for c in first["gate"]] != [c.hash for c in other["gate"]]


def test_split_is_stratified_by_first_tag():
    parts = datasets.split(make_cases(400), {"a": 100, "b": 200}, seed=3)
    for name, per_stratum in (("a", 25), ("b", 50)):
        counts = Counter(c.tags[0] for c in parts[name])
        assert len(counts) == 4 and all(abs(n - per_stratum) <= 1 for n in counts.values())


def test_split_refuses_to_oversample():
    with pytest.raises(ValueError):
        datasets.split(make_cases(5), {"a": 6})


def test_a_bad_line_is_reported_with_its_number(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"input": {"text": "ok"}, "expected": "a"}\n{"expected": "no input"}\n')
    with pytest.raises(ValueError, match=r"d\.jsonl:2"):
        datasets.load(path)
