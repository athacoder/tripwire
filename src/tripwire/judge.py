"""An LLM judge for answers no rule can score, and the tools to find out how far to trust it.

The judge grades one criterion per call and must quote its evidence before giving a
verdict. Its scores are stored like any scorer's, under a version derived from the model
name and the exact prompt, so changing the rubric never mixes old verdicts with new ones.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
import tomllib
from collections import defaultdict
from typing import Any

import numpy as np

from . import datasets, stats, store
from .config import Config
from .models import Case, sha
from .providers import Provider, ProviderError, Request, context_needed, make_provider

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["evidence", "reasoning", "verdict"],
}


class Judge:
    def __init__(self, cfg: Config, rubric: str | None = None):
        if cfg.judge is None:
            raise ValueError("no [judge] section in the configuration")
        self.cfg, self.conf = cfg, cfg.judge
        self.system = (cfg.root / self.conf.system).read_text(encoding="utf-8")
        self.template = (cfg.root / self.conf.template).read_text(encoding="utf-8")
        rubric_path = cfg.root / (rubric or self.conf.rubric)
        raw = tomllib.loads(rubric_path.read_text(encoding="utf-8"))["criteria"]
        # A criterion is a question, or a table {question, per_sentence, claim_pattern}.
        # With per_sentence the answer is split up and every sentence must pass on its
        # own: a small judge asked about a whole answer checks the first sentence and
        # stops. claim_pattern is a regex; sentences that do not match it are taken to
        # make no claim and pass without a call (greetings, filler, "not covered").
        tables = {k: v for k, v in raw.items() if isinstance(v, dict)}
        self.criteria: dict[str, str] = {
            name: tables[name]["question"] if name in tables else entry
            for name, entry in raw.items()
        }
        self.per_sentence = {name for name, table in tables.items() if table.get("per_sentence")}
        self.claim_pattern = {
            name: re.compile(table["claim_pattern"])
            for name, table in tables.items()
            if "claim_pattern" in table
        }
        # Model name plus every word the judge reads. The digest is recorded per score
        # instead, so CI can compute versions without a model server.
        self.versions = {
            name: sha(json.dumps([self.conf.model, self.system, self.template, raw[name]]))[:8]
            for name, question in self.criteria.items()
        }

    def provider(self) -> Provider:
        return make_provider(self.cfg.provider[self.conf.provider])

    def request(self, case: Case, answer: str, criterion: str) -> Request:
        expected = case.expected
        reference = expected.get("reference", "") if isinstance(expected, dict) else str(expected)
        # The answer is untrusted text: it must not be able to close its own delimiter.
        fields = {
            **case.input,
            "reference": reference,
            "answer": answer.replace("</answer>", ""),
            "criterion": self.criteria[criterion],
        }
        try:
            user = self.template.format_map(fields)
        except KeyError as e:
            raise ValueError(f"judge template needs field {e} that the case does not have") from e
        return Request(
            model=self.conf.model,
            system=self.system,
            user=user,
            temperature=0.0,  # a verdict should not depend on a dice roll
            seed=0,
            num_ctx=self.conf.num_ctx,
            max_tokens=self.conf.max_tokens,
            format=VERDICT_SCHEMA,
        )

    async def grade(
        self, provider: Provider, case: Case, answer: str, criterion: str
    ) -> tuple[float, dict[str, Any]] | None:
        """One verdict, or None when the judge could not produce a usable one."""
        if not answer.strip():
            return 0.0, {"rule": "empty answer"}  # no call needed, and never a pass
        if criterion not in self.per_sentence:
            return await self._ask(provider, case, answer, criterion)
        parts = []
        pattern = self.claim_pattern.get(criterion)
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
            if pattern and not pattern.search(sentence):
                continue  # states nothing checkable: cannot fail a criterion about claims
            graded = await self._ask(provider, case, sentence, criterion)
            if graded is None:
                return None
            parts.append({"sentence": sentence, "verdict": graded[0], **graded[1]})
        if not parts:
            return 1.0, {"rule": "no sentence makes a claim"}
        failing = [p for p in parts if not p["verdict"]]
        shown = failing[0] if failing else parts[0]
        return float(not failing), {
            "evidence": shown["sentence"],
            "reasoning": shown.get("reasoning", ""),
            "sentences": [[p["sentence"], p["verdict"]] for p in parts],
        }

    async def _ask(
        self, provider: Provider, case: Case, answer: str, criterion: str
    ) -> tuple[float, dict[str, Any]] | None:
        for attempt in range(self.cfg.retries + 1):
            try:
                # The same hard limit the runner uses: a stalled server must not hang a
                # judging pass that may be running unattended overnight.
                request = self.request(case, answer, criterion)
                reply = await asyncio.wait_for(provider.complete(request), self.cfg.timeout)
                parsed = json.loads(reply.text)
                detail = {
                    "evidence": parsed.get("evidence", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "tokens": reply.output_tokens,
                    "ms": reply.latency_ms,
                }
                if parsed["verdict"] not in ("yes", "no"):
                    return None
                return float(parsed["verdict"] == "yes"), detail
            except ProviderError as e:
                if not e.retryable or attempt == self.cfg.retries:
                    return None
            except (ValueError, KeyError, TypeError, TimeoutError):
                return None  # unusable or no reply: leave the sample unscored, never guess
        return None


async def judge_suite(
    cfg: Config,
    name: str,
    fingerprint: str,
    db: sqlite3.Connection,
    provider: Provider | None = None,
    max_minutes: float | None = None,
) -> dict[str, int]:
    """Judge every stored sample of a suite that has no verdict yet. Resumable."""
    suite, judge = cfg.get_suite(name), Judge(cfg)
    cases = {c.hash: c for c in datasets.load(cfg.root / suite.dataset)}
    samples = db.execute(
        "SELECT case_hash, rep, output, status FROM samples WHERE fingerprint=? "
        "AND status != 'truncated' AND rep<? ORDER BY case_hash, rep",
        (fingerprint, suite.reps),
    ).fetchall()
    done = {
        tuple(r)
        for r in db.execute(
            "SELECT case_hash, rep, metric, version FROM scores WHERE fingerprint=? "
            "AND scorer='judge'",
            (fingerprint,),
        )
    }
    todo = [
        (row, criterion)
        for criterion in judge.criteria  # one criterion at a time keeps the prompt prefix warm
        for row in samples
        if row["case_hash"] in cases
        and (row["case_hash"], row["rep"], criterion, judge.versions[criterion]) not in done
    ]
    counts = {"judged": 0, "failed": 0, "pending": len(todo)}
    if not todo:
        return counts
    # The check the runner makes for a target. A judge prompt that overflows would be cut
    # without an error, and the verdict would then be about part of the documents.
    need = max(
        context_needed(judge.request(cases[row["case_hash"]], row["output"], criterion))
        for row, criterion in todo
    )
    if need > judge.conf.num_ctx:
        raise ValueError(
            f"longest judge prompt needs about {need} tokens but the judge's num_ctx is "
            f"{judge.conf.num_ctx}"
        )
    provider = provider or judge.provider()
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60
    try:
        digest = await provider.digest(judge.conf.model)
        for row, criterion in todo:
            if deadline is not None and time.monotonic() >= deadline:
                break
            answer = "" if row["status"] == "refusal" else row["output"]
            graded = await judge.grade(provider, cases[row["case_hash"]], answer, criterion)
            counts["pending"] -= 1
            if graded is None:
                counts["failed"] += 1
                continue
            value, detail = graded
            score = {
                "fingerprint": fingerprint,
                "case_hash": row["case_hash"],
                "rep": row["rep"],
                "scorer": "judge",
                "version": judge.versions[criterion],
                "metric": criterion,
                "value": value,
                "detail": json.dumps({**detail, "digest": digest[:12]}),
            }
            store.insert(db, "scores", [score], "OR REPLACE")
            counts["judged"] += 1
    finally:
        await provider.aclose()
    return counts


def agreement(judge: Any, truth: Any, seed: int = 0) -> dict[str, Any]:
    """How well the judge's verdicts match a trusted set of verdicts for the same items."""
    j, t = np.asarray(judge, dtype=float), np.asarray(truth, dtype=float)
    if len(j) < 2:
        return {"n": len(j)}
    kappa, lo, hi = stats.kappa_ci(j, t, seed=seed)
    positives, negatives = t == 1, t == 0
    return {
        "n": len(j),
        "agreement": float((j == t).mean()),
        "kappa": kappa,
        "kappa_lo": lo,
        "kappa_hi": hi,
        # Of the answers that truly pass, how many the judge passes; and the same for fails.
        "sensitivity": float(j[positives].mean()) if positives.any() else None,
        "specificity": float(1 - j[negatives].mean()) if negatives.any() else None,
    }


async def probe(
    cfg: Config, name: str, probes: list[dict[str, Any]], rubric: str | None = None
) -> dict[str, Any]:
    """Run the judge on constructed answers whose correct verdicts are known.

    Each probe is a case plus an answer built to be right or wrong in a specific way
    (a wrong fact, an invented claim, padding, an injected instruction). Nothing is
    stored: probes test the judge, they are not results.
    """
    judge = Judge(cfg, rubric)
    provider = judge.provider()
    verdicts: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    misses: list[dict[str, Any]] = []
    failed = 0
    try:
        for item in probes:
            case = Case(input=item["input"], expected=item["expected"])
            for criterion, truth in item["truth"].items():
                if truth is None or criterion not in judge.criteria:
                    continue
                graded = await judge.grade(provider, case, item["answer"], criterion)
                if graded is None:
                    failed += 1
                else:
                    verdicts[criterion].append((item["kind"], graded[0], float(truth)))
                    if graded[0] != truth:
                        miss = {"criterion": criterion, "kind": item["kind"], "truth": truth}
                        misses.append({**miss, "answer": item["answer"], **graded[1]})
    finally:
        await provider.aclose()
    report: dict[str, Any] = {"failed": failed, "criteria": {}, "verdicts": {}, "misses": misses}
    for criterion, rows in verdicts.items():
        kinds: dict[str, list[bool]] = defaultdict(list)
        for kind, got, truth in rows:
            kinds[kind].append(got == truth)
        report["criteria"][criterion] = {
            **agreement([r[1] for r in rows], [r[2] for r in rows]),
            "by_kind": {k: sum(v) / len(v) for k, v in sorted(kinds.items())},
        }
        report["verdicts"][criterion] = [r[1] for r in rows]
    return report


def calibration(cfg: Config, name: str, fingerprint: str, db: sqlite3.Connection) -> dict[str, Any]:
    """Compare stored judge verdicts with human labels, per criterion."""
    suite, judge = cfg.get_suite(name), Judge(cfg)
    out: dict[str, Any] = {}
    for criterion, version in judge.versions.items():
        verdicts = {
            (r["case_hash"], r["rep"]): r["value"]
            for r in db.execute(
                "SELECT case_hash, rep, value FROM scores WHERE fingerprint=? AND scorer='judge' "
                "AND metric=? AND version=? AND rep<?",
                (fingerprint, criterion, version, suite.reps),
            )
        }
        human = {
            (r["case_hash"], r["rep"]): r["value"]
            for r in db.execute(
                "SELECT case_hash, rep, value FROM human_labels "
                "WHERE fingerprint=? AND criterion=?",
                (fingerprint, criterion),
            )
        }
        shared = sorted(set(verdicts) & set(human))
        result = agreement([verdicts[k] for k in shared], [human[k] for k in shared])
        result["judge_mean"] = float(np.mean(list(verdicts.values()))) if verdicts else None
        if len(shared) >= 10:
            estimate, lo, hi = stats.ppi(
                list(verdicts.values()),
                [verdicts[k] for k in shared],
                [human[k] for k in shared],
            )
            result.update(human_estimate=estimate, human_lo=lo, human_hi=hi)
        out[criterion] = result
    return out


def against_rule(
    db: sqlite3.Connection, cfg: Config, name: str, fingerprint: str, criterion: str, rule: str
) -> dict[str, Any]:
    """Agreement between the judge and a rule-based scorer that measures the same thing."""
    from .scorers import scorer as lookup

    suite, judge = cfg.get_suite(name), Judge(cfg)
    rule_scorer, _, rule_metric = rule.partition(".")

    def scores(scorer: str, version: str, metric: str) -> dict[tuple[str, int], float]:
        query = (
            "SELECT case_hash, rep, value FROM scores WHERE fingerprint=? AND scorer=? "
            "AND version=? AND metric=? AND rep<?"
        )
        rows = db.execute(query, (fingerprint, scorer, version, metric, suite.reps))
        return {(r["case_hash"], r["rep"]): r["value"] for r in rows}

    judged = scores("judge", judge.versions[criterion], criterion)
    ruled = scores(rule_scorer, lookup(rule_scorer)[1], rule_metric)
    shared = sorted(set(judged) & set(ruled))
    result = agreement([judged[k] for k in shared], [ruled[k] for k in shared])
    result["disagreements"] = [k[0] for k in shared if judged[k] != ruled[k]]
    return result
