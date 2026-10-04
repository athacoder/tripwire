"""The system under test, and the fingerprint that identifies its samples."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
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
        if cfg.kind == "python":
            if not cfg.entry or ":" not in cfg.entry:
                raise ValueError('a python target needs entry = "module:function"')
            module, _, name = cfg.entry.partition(":")
            source = root / (module.replace(".", "/") + ".py")
            # Load by file path under a root-specific name: the gate loads the same module
            # from two checkouts, and a plain import would hand back the first one twice.
            unique = f"_tripwire_{sha(str(root))}_{module.replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(unique, source)
            if spec is None or spec.loader is None or not source.is_file():
                raise ValueError(f"python target not found: {source}")
            loaded = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(root))  # so the module's own imports resolve
            spec.loader.exec_module(loaded)
            self.fn = getattr(loaded, name)
        if cfg.kind == "http" and not cfg.url:
            raise ValueError("an http target needs a url")

        files = sorted({p for glob in cfg.watch for p in root.glob(glob) if p.is_file()})
        # Everything that can change the output. Prompt text is hashed exactly as read
        # (newlines normalised so Windows and Linux checkouts agree); whitespace is not
        # collapsed, because reformatting a prompt can change what the model says.
        static: dict[str, Any] = {
            **cfg.model_dump(exclude={"system", "template", "watch", "provider"}),
            "provider_kind": provider_kind,
            "system": self.system,
            "template": self.template,
            "watched": {
                p.relative_to(root).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for p in files
            },
        }
        # The static key needs no running backend, so CI can compute it; the fingerprint
        # adds the model digest, which only the machine holding the model knows.
        self.static_key = sha(json.dumps(static, sort_keys=True))
        self.spec: dict[str, Any] = {**static, "digest": digest}
        self.fingerprint = sha(json.dumps(self.spec, sort_keys=True))

    def render(self, case: Case) -> str:
        try:
            return self.template.format_map(case.input)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(
                f"user template cannot be filled from case input {sorted(case.input)}: {e!r}. "
                "Literal braces in a template must be doubled."
            ) from e

    def request(self, case: Case, seed: int) -> Request:
        c = self.cfg
        return Request(
            model=c.model,
            system=self.system,
            user=self.render(case),
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
        longest = max(len(self.system) + len(self.render(c)) for c in cases)
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
        await self.provider.aclose()
        if self.http:
            await self.http.aclose()
