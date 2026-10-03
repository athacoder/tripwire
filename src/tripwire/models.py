"""A test case and the content hash that identifies it."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field


def sha(text: str) -> str:
    """First 64 bits of SHA-256: collision-safe far beyond any dataset we will hold."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def canonical(obj: Any) -> str:
    """JSON with sorted keys and collapsed whitespace, so equal content hashes equally."""

    def norm(x: Any) -> Any:
        if isinstance(x, str):
            return " ".join(x.split())
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if isinstance(x, list):
            return [norm(v) for v in x]
        return x

    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Case(BaseModel):
    input: dict[str, Any]
    expected: Any = None
    tags: list[str] = Field(default_factory=list)
    source: str = "human"
    provenance: dict[str, Any] = Field(default_factory=dict)

    @cached_property
    def hash(self) -> str:
        """Identity is input plus expected. Tags are metadata: retagging keeps cached samples."""
        return sha(canonical({"input": self.input, "expected": self.expected}))

    @cached_property
    def text(self) -> str:
        return " ".join(str(v) for v in self.input.values())
