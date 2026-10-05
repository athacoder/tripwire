"""Ways to grow a dataset beyond what was first written by hand.

- perturb: variants of existing cases whose answer must not change (typos, casing, an
  irrelevant sentence, shuffled context, a paraphrase). No new labels are needed.
- draft: a model proposes new cases from seed material; a second call screens them.
- import_tracelens: production failures diagnosed by TraceLens become candidate cases.

Drafted and imported cases are only candidates. Nothing enters a dataset until a person
has approved it (`tripwire dataset review`): a model's guess at the right answer is not
ground truth.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from .datasets import _tokens
from .models import Case, sha
from .providers import HttpProvider, Provider, ProviderError, Request

DISTRACTORS = [
    "By the way, the weather has been lovely this week.",
    "Also, my neighbour just got a new puppy.",
    "Sorry for the long message, it has been a busy day.",
    "I am writing this from the train, in case that matters.",
    "Thanks in advance, and have a great weekend.",
]
PERTURBATIONS = ("typo", "lowercase", "distractor", "shuffle_lines")


def _typo(text: str, rng: random.Random) -> str:
    """Swap two neighbouring letters inside up to two longer words."""
    words = text.split(" ")
    long_words = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    for i in rng.sample(long_words, min(2, len(long_words))):
        k = rng.randrange(1, len(words[i]) - 2)
        w = words[i]
        words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2 :]
    return " ".join(words)


def _shuffle_lines(text: str, rng: random.Random) -> str:
    """Reorder the lines after the first: the facts stay, their order changes."""
    head, *rest = text.split("\n")
    body = [line for line in rest if line.strip()]
    if len(body) < 2:
        return text
    shuffled = body[:]
    while shuffled == body:
        rng.shuffle(shuffled)
    blank = [""] if rest and not rest[0].strip() else []
    return "\n".join([head, *blank, *shuffled])


def _apply(kind: str, text: str, rng: random.Random) -> str:
    if kind == "typo":
        return _typo(text, rng)
    if kind == "lowercase":
        return text.lower()
    if kind == "distractor":
        return f"{text} {rng.choice(DISTRACTORS)}"
    if kind == "shuffle_lines":
        return _shuffle_lines(text, rng)
    raise ValueError(f"unknown perturbation {kind!r} (known: {', '.join(PERTURBATIONS)})")


def variant(case: Case, field: str, text: str, kind: str, **provenance: Any) -> Case:
    """A new case that differs from its parent in one input field and nothing else."""
    return Case(
        input={**case.input, field: text},
        expected=case.expected,
        tags=[*case.tags, f"perturb:{kind}"],
        source="perturbed",
        provenance={"parent": case.hash, "perturb": kind, **provenance},
    )


def perturb(
    cases: list[Case], kinds: list[str], field: str | None = None, seed: int = 0
) -> list[Case]:
    """The originals (tagged perturb:none) followed by one variant per case and kind.

    Keeping the originals in the same file puts both on one report, so a slice by
    perturbation shows directly how much each one costs. Variants identical to their
    parent are dropped.
    """
    out = [
        Case(
            input=c.input,
            expected=c.expected,
            tags=[*c.tags, "perturb:none"],
            source=c.source,
            provenance=c.provenance,
        )
        for c in cases
    ]
    for kind in kinds:
        for case in cases:
            key = field or list(case.input)[-1]
            if key not in case.input:
                raise ValueError(f"case has no input field {key!r} to perturb")
            rng = random.Random(f"{seed}:{kind}:{case.hash}")  # stable per case, any order
            text = _apply(kind, str(case.input[key]), rng)
            if text != case.input[key]:
                out.append(variant(case, key, text, kind))
    return out


async def paraphrase(
    provider: Provider, model: str, cases: list[Case], field: str | None = None
) -> list[Case]:
    """Model-written rewordings. A paraphrase can change the meaning: hand-check a sample."""
    system = (
        "Rewrite the text so that it means exactly the same thing in different words. Keep "
        "every fact, name and number. Reply with the rewritten text only."
    )
    out = []
    for case in cases:
        key = field or list(case.input)[-1]
        request = Request(model, system, str(case.input[key]), temperature=0.7, max_tokens=400)
        try:
            text = (await provider.complete(request)).text.strip().strip('"')
        except ProviderError:
            continue
        if text and text != case.input[key]:
            out.append(variant(case, key, text, "paraphrase", paraphrased_by=model))
    return out


def _schema(values: list[Any]) -> dict[str, Any]:
    """A JSON schema that the given example values all fit."""
    kinds = {type(v) for v in values}
    if kinds == {str}:
        return {"type": "string"}
    if kinds == {bool}:
        return {"type": "boolean"}
    if kinds <= {int, float}:
        return {"type": "number"}
    if kinds == {dict} and len({tuple(sorted(v)) for v in values}) == 1:
        keys = sorted(values[0])
        return {
            "type": "object",
            "properties": {k: _schema([v[k] for v in values]) for k in keys},
            "required": keys,
        }
    return {}  # mixed or unknown: leave it unconstrained


async def draft(
    provider: Provider,
    model: str,
    verifier: str,
    examples: list[Case],
    seed_text: str,
    n: int,
    existing: list[Case],
) -> tuple[list[Case], dict[str, int]]:
    """Ask a model for new cases in the style of `examples`, then screen them.

    Returns the candidates that survive, and how many were dropped for each reason.
    """
    labels = sorted({c.expected for c in existing if isinstance(c.expected, str)})
    expected_schema = _schema([c.expected for c in examples])
    if expected_schema == {"type": "string"} and 1 < len(labels) <= 200:
        expected_schema["enum"] = labels  # a classification task: only real labels
    case_schema = {
        "type": "object",
        "properties": {"input": _schema([c.input for c in examples]), "expected": expected_schema},
        "required": ["input", "expected"],
    }
    schema = {
        "type": "object",
        "properties": {"cases": {"type": "array", "items": case_schema}},
        "required": ["cases"],
    }
    shown = "\n".join(json.dumps({"input": c.input, "expected": c.expected}) for c in examples)
    request = Request(
        model,
        "You write test cases for evaluating an AI system. Each case has an input and the "
        "correct expected answer. Write cases that are clear, realistic, different from each "
        "other and from the examples, and whose expected answer is unambiguously right.",
        f"Seed material:\n{seed_text}\n\nExamples of the format:\n{shown}\n\n"
        f"Write {n} new cases based on the seed material. Reply with JSON.",
        temperature=0.8,
        num_ctx=8192,
        max_tokens=4000,
        format=schema,
    )
    reply = await provider.complete(request)
    try:
        drafts = [
            Case(input=d["input"], expected=d["expected"]) for d in json.loads(reply.text)["cases"]
        ]
    except (ValueError, KeyError, TypeError) as e:
        raise ProviderError(f"the drafting model did not return usable cases: {e!r}") from e

    dropped = {"duplicate": 0, "near_duplicate": 0, "flagged": 0, "unverified": 0}
    seen = {c.hash for c in existing}
    seen_tokens = [_tokens(c.text) for c in existing]
    verdict_schema = {
        "type": "object",
        "properties": {
            "ambiguous": {"type": "boolean"},
            "answer_wrong": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["ambiguous", "answer_wrong", "reason"],
    }
    kept = []
    for case in drafts:
        tokens = _tokens(case.text)
        if case.hash in seen:
            dropped["duplicate"] += 1
            continue
        if any(t | tokens and len(t & tokens) / len(t | tokens) >= 0.9 for t in seen_tokens):
            dropped["near_duplicate"] += 1
            continue
        check = Request(
            verifier,
            "You audit one test case. Be conservative: flag only when reasonably confident.",
            f"Seed material:\n{seed_text}\n\nTest case:\n"
            f"{json.dumps({'input': case.input, 'expected': case.expected})}\n\n"
            "ambiguous: could two careful experts disagree on the correct answer?\n"
            "answer_wrong: does the expected answer look wrong or arguable?\nReply with JSON.",
            num_ctx=8192,
            max_tokens=300,
            format=verdict_schema,
        )
        try:
            audit = json.loads((await provider.complete(check)).text)
            flagged = bool(audit["ambiguous"] or audit["answer_wrong"])
        except (ProviderError, ValueError, KeyError, TypeError):
            dropped["unverified"] += 1  # no audit, no candidate: screening is not optional
            continue
        if flagged:
            dropped["flagged"] += 1
            continue
        seen.add(case.hash)
        seen_tokens.append(tokens)
        provenance = {
            "status": "pending",
            "drafted_by": model,
            "verified_by": verifier,
            "seed": sha(seed_text)[:8],
        }
        kept.append(
            Case(
                input=case.input,
                expected=case.expected,
                tags=["origin:drafted"],
                source="llm-drafted",
                provenance=provenance,
            )
        )
    return kept, dropped


async def import_tracelens(url: str, limit: int = 200, transport: Any = None) -> list[Case]:
    """Turn traces that TraceLens diagnosed as failures into candidate regression cases.

    The case input is what the pipeline was given (the first span's inputs). The expected
    answer is left empty on purpose: the trace shows what the pipeline did wrong, not what
    would have been right, so a reviewer supplies it.
    """
    api = HttpProvider(url.rstrip("/"), transport=transport)
    try:
        summaries: list[dict[str, Any]] = []
        while len(summaries) < limit:
            page = await api.call("GET", f"/api/v1/traces?limit=200&offset={len(summaries)}")
            summaries += page["items"]
            if not page.get("has_more"):
                break
        out = []
        for item in summaries[:limit]:
            if not item.get("root_cause_stage"):
                continue  # healthy, or never analysed
            trace = await api.call("GET", f"/api/v1/traces/{item['trace_id']}")
            report = await api.call("GET", f"/api/v1/traces/{item['trace_id']}/root-cause")
            spans = trace.get("spans") or []
            first = next((s.get("inputs") for s in spans if s.get("inputs")), None)
            if not isinstance(first, dict):
                continue
            cause = report.get("likely_root_cause") or {}
            category = next((c.get("category") for c in cause.get("candidates") or []), None)
            observed = next((s["outputs"] for s in reversed(spans) if s.get("outputs")), None)
            tags = [
                f"stage:{item['root_cause_stage']}",
                f"failure:{category or 'unknown'}",
                f"pipeline:{item.get('pipeline') or 'unknown'}",
            ]
            provenance = {
                "status": "pending",
                "trace_id": item["trace_id"],
                "root_cause": cause.get("summary") or report.get("summary"),
                "observed_output": json.dumps(observed)[:400] if observed is not None else None,
            }
            out.append(
                Case(input=first, tags=tags, source="production-failure", provenance=provenance)
            )
        return out
    finally:
        await api.aclose()


def parse_expected(text: str) -> Any:
    """What a reviewer typed: JSON when it parses as an object, list or number, else text."""
    text = text.strip()
    if re.match(r"^[\[{]|^-?\d", text):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def approve(candidate: Case, reviewer: str, expected: Any = None) -> Case:
    """The dataset entry for an approved candidate: reviewed, and no longer pending."""
    provenance = {k: v for k, v in candidate.provenance.items() if k != "status"}
    return Case(
        input=candidate.input,
        expected=candidate.expected if expected is None else expected,
        tags=candidate.tags,
        source=candidate.source,
        provenance={**provenance, "reviewed_by": reviewer},
    )
