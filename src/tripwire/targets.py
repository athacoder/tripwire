"""The system under test, and the fingerprint that identifies its samples."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from .config import TargetCfg
from .models import Case, sha
from .providers import HttpProvider, Provider, Request, Response


def seed_for(case_hash: str, rep: int) -> int:
    """Each (case, repetition) gets its own fixed seed: reps differ, reruns do not."""
    return int(sha(f"{case_hash}:{rep}")[:8], 16)


def _read(root: Path, rel: str | None) -> str:
    return (root / rel).read_text(encoding="utf-8") if rel else ""


class Target:
    def __init__(
        self,
        cfg: TargetCfg,
        root: Path,
        provider: Provider,
        provider_kind: str = "",
        digest: str = "",
    ):
        self.cfg, self.provider = cfg, provider
        self.system = _read(root, cfg.system)
        self.template = _read(root, cfg.template) or "{text}"
        self.http = HttpProvider("") if cfg.kind == "http" else None
        self.fn: Any = None
        if cfg.kind == "python" and cfg.entry:
            sys.path.insert(0, str(root))
            module, _, name = cfg.entry.partition(":")
            self.fn = getattr(importlib.import_module(module), name)

        files = sorted({p for glob in cfg.watch for p in root.glob(glob) if p.is_file()})
        # Everything that can change the output. Prompt text is hashed exactly as read
        # (newlines normalised so Windows and Linux checkouts agree); whitespace is not
        # collapsed, because reformatting a prompt can change what the model says.
        self.spec: dict[str, Any] = {
            **cfg.model_dump(exclude={"system", "template", "watch", "provider"}),
            "provider_kind": provider_kind,
            "digest": digest,
            "system": self.system,
            "template": self.template,
            "watched": {
                p.relative_to(root).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for p in files
            },
        }
        self.fingerprint = sha(json.dumps(self.spec, sort_keys=True))

    def request(self, case: Case, seed: int) -> Request:
        c = self.cfg
        return Request(
            model=c.model,
            system=self.system,
            user=self.template.format_map(case.input),
            temperature=c.temperature,
            seed=seed,
            num_ctx=c.num_ctx,
            max_tokens=c.max_tokens,
            format=c.format,
            expected=case.expected,
        )

    def guard(self, cases: list[Case]) -> None:
        """Refuse to start if a prompt could overflow the context window.

        Ollama drops the overflow and answers anyway, so this has to be caught up front.
        """
        if self.cfg.kind != "prompt" or not cases:
            return
        longest = max(len(self.system) + len(self.template.format_map(c.input)) for c in cases)
        need = longest // 3 + self.cfg.max_tokens  # 3 chars per token overestimates English
        if need > self.cfg.num_ctx:
            raise ValueError(
                f"longest prompt needs about {need} tokens but num_ctx is {self.cfg.num_ctx}"
            )

    async def run(self, case: Case, rep: int) -> Response:
        seed = seed_for(case.hash, rep)
        if self.cfg.kind == "prompt":
            return await self.provider.complete(self.request(case, seed))
        if self.http:
            body = {"input": case.input, "seed": seed}
            out = (await self.http.call("POST", self.cfg.url or "", body))["output"]
        elif inspect.iscoroutinefunction(self.fn):
            out = await self.fn(case.input)
        else:
            out = await asyncio.to_thread(self.fn, case.input)
        return Response(text=str(out), model=self.cfg.entry or self.cfg.url or "")

    async def aclose(self) -> None:
        if self.http:
            await self.http.aclose()
