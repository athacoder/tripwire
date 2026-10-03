"""Load, version, lint and split datasets. A dataset is a JSONL file of cases."""

from __future__ import annotations

import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .models import Case, sha


def load(path: Path) -> list[Case]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(Case.model_validate_json(line))
            except ValueError as e:  # say which line, the parser alone does not
                raise ValueError(f"{path}:{number}: not a valid case: {e}") from e
    return cases


def save(path: Path, cases: list[Case]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(c.model_dump_json(exclude_defaults=True) for c in cases)
    path.write_text(body + "\n", encoding="utf-8", newline="\n")  # same bytes on every OS


def version(cases: list[Case]) -> str:
    """Hash of the sorted case hashes: order-independent, changes if any case changes."""
    return sha("".join(sorted(c.hash for c in cases)))


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower()))


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def lint(cases: list[Case], context: str = "", near: float = 0.9) -> dict[str, Any]:
    """Check a dataset before trusting it. `context` is the prompt text the model will see."""
    by_hash = Counter(c.hash for c in cases)
    by_text: dict[str, set[str]] = defaultdict(set)
    for c in cases:
        by_text[_squash(c.text)].add(str(c.expected))

    tokens = [_tokens(c.text) for c in cases]
    # O(n^2) pair scan: fine to a few thousand cases, switch to MinHash beyond that.
    near_dups = [
        (i, j)
        for i, j in combinations(range(len(cases)), 2)
        if cases[i].hash != cases[j].hash
        and tokens[i] | tokens[j]
        and len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j]) >= near
    ]

    labels = Counter(c.expected for c in cases if isinstance(c.expected, str))
    lengths = sorted(len(c.text) for c in cases)
    ctx = _squash(context)
    report: dict[str, Any] = {
        "cases": len(cases),
        "version": version(cases),
        "duplicates": sum(n - 1 for n in by_hash.values()),
        "conflicting_labels": sum(len(v) > 1 for v in by_text.values()),
        "near_duplicates": near_dups,
        "empty": sum(not c.text.strip() or c.expected in (None, "") for c in cases),
        "leaked_into_prompt": sum(bool(c.text.strip()) and _squash(c.text) in ctx for c in cases),
        "answer_in_input": sum(
            isinstance(c.expected, str)
            and len(c.expected) > 3
            and c.expected.lower() in c.text.lower()
            for c in cases
        ),
        "labels": len(labels),
        "majority_baseline": max(labels.values()) / len(cases) if labels else None,
        "length_min_median_max": (lengths[0], int(statistics.median(lengths)), lengths[-1])
        if lengths
        else None,
    }
    blocking = (
        "duplicates",
        "conflicting_labels",
        "empty",
        "leaked_into_prompt",
        "answer_in_input",
    )
    report["ok"] = bool(cases) and not any(report[k] for k in blocking)
    return report


def embedding_near_duplicates(
    cases: list[Case], base_url: str, model: str = "nomic-embed-text", threshold: float = 0.95
) -> list[tuple[int, int]]:
    """Paraphrase-level duplicates that token overlap misses. Needs a running Ollama."""
    import httpx
    import numpy as np

    payload = {"model": model, "input": [c.text for c in cases]}
    reply = httpx.post(f"{base_url}/api/embed", json=payload, timeout=600).raise_for_status()
    vectors = np.array(reply.json()["embeddings"])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rows, cols = np.where(np.triu(vectors @ vectors.T, 1) >= threshold)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def split(cases: list[Case], sizes: dict[str, int], seed: int = 0) -> dict[str, list[Case]]:
    """Disjoint random splits, each stratified by the first tag.

    Every case gets a position in [0, 1) spread evenly within its stratum, so any
    contiguous slice of the global ordering holds each stratum in proportion.
    """
    if sum(sizes.values()) > len(cases):
        raise ValueError(f"asked for {sum(sizes.values())} cases, only {len(cases)} available")
    rng = random.Random(seed)
    strata: dict[str, list[Case]] = defaultdict(list)
    for c in sorted(cases, key=lambda c: c.hash):  # input order must not matter
        strata[c.tags[0] if c.tags else ""].append(c)
    ranked: list[tuple[float, Case]] = []
    for group in strata.values():
        rng.shuffle(group)
        ranked += [((i + rng.random()) / len(group), c) for i, c in enumerate(group)]
    ranked.sort(key=lambda pair: pair[0])
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = [c for _, c in ranked[start : start + n]]
        start += n
    return out


def manifest(name: str, splits: dict[str, list[Case]], canary: str) -> dict[str, Any]:
    """Summary committed next to the data. The canary string marks the files as eval data."""

    def tag(c: Case, prefix: str) -> str:
        return next((t for t in c.tags if t.startswith(prefix)), "")

    return {
        "name": name,
        "canary": canary,
        "splits": {
            split_name: {
                "version": version(cases),
                "cases": len(cases),
                "sources": dict(Counter(c.source for c in cases)),
                "groups": dict(sorted(Counter(tag(c, "group:") for c in cases).items())),
            }
            for split_name, cases in splits.items()
        },
    }
