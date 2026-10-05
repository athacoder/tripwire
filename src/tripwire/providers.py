"""Model backends. Each turns a Request into a Response; nothing else knows which is in use."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import ProviderCfg
from .models import sha


@dataclass
class Request:
    model: str
    system: str
    user: str
    temperature: float = 0.0
    seed: int = 0
    num_ctx: int = 4096
    max_tokens: int = 64
    format: dict[str, Any] | None = None
    expected: Any = None  # read only by the mock provider


@dataclass
class Response:
    text: str
    model: str
    stop: str = "stop"  # "stop", "length" or "refusal"
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float | None = None  # None means "use the caller's wall clock"
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, kind: str = "request"):
        super().__init__(message)
        self.retryable, self.kind = retryable, kind


def _messages(r: Request) -> list[dict[str, str]]:
    system = [{"role": "system", "content": r.system}] if r.system else []
    return [*system, {"role": "user", "content": r.user}]


class Provider:
    async def complete(self, r: Request) -> Response:
        raise NotImplementedError

    async def digest(self, model: str) -> str:
        """Immutable identity of the model behind a name, when the backend exposes one."""
        return ""

    async def info(self, model: str) -> dict[str, Any]:
        return {}

    async def aclose(self) -> None:
        return None


class HttpProvider(Provider):
    def __init__(self, base_url: str, key: str | None = None, transport: Any = None):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        # No client-side timeout: the runner enforces one wall-clock limit per case.
        self.client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=None, transport=transport
        )

    async def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            reply = await self.client.request(method, path, json=body)
        except httpx.TransportError as e:
            where = f"{self.client.base_url}{path}"
            raise ProviderError(
                f"cannot reach {where} ({type(e).__name__}). Is the server running?",
                retryable=True,
            ) from e
        if reply.status_code >= 400:
            retryable = reply.status_code in (408, 429) or reply.status_code >= 500
            raise ProviderError(f"HTTP {reply.status_code}: {reply.text[:200]}", retryable)
        try:
            return reply.json()
        except ValueError as e:  # a proxy error page, or a truncated body
            raise ProviderError(f"reply is not JSON: {reply.text[:200]!r}", kind="malformed") from e

    async def aclose(self) -> None:
        await self.client.aclose()


class Ollama(HttpProvider):
    async def complete(self, r: Request) -> Response:
        body: dict[str, Any] = {
            "model": r.model,
            "messages": _messages(r),
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": r.temperature,
                "seed": r.seed,
                "num_ctx": r.num_ctx,
                "num_predict": r.max_tokens,
            },
        }
        if r.format:
            body["format"] = r.format
        d = await self.call("POST", "/api/chat", body)
        return Response(
            text=d["message"]["content"],
            model=d["model"],
            stop="length" if d.get("done_reason") == "length" else "stop",
            prompt_tokens=d.get("prompt_eval_count", 0),
            output_tokens=d.get("eval_count", 0),
            latency_ms=(d.get("total_duration", 0) - d.get("load_duration", 0)) / 1e6,
            raw=d,
        )

    async def digest(self, model: str) -> str:
        tags = await self.call("GET", "/api/tags")
        # Ollama treats a bare name as ":latest", so accept either spelling.
        found = [m["digest"] for m in tags["models"] if m["name"] in (model, f"{model}:latest")]
        if not found:
            raise ProviderError(f"model {model!r} is not pulled; run: ollama pull {model}")
        return found[0]

    async def info(self, model: str) -> dict[str, Any]:
        version = (await self.call("GET", "/api/version"))["version"]
        loaded = [m for m in (await self.call("GET", "/api/ps"))["models"] if m["name"] == model]
        gpu = round(loaded[0]["size_vram"] / loaded[0]["size"], 2) if loaded else None
        return {"ollama": version, "gpu_share": gpu}


class OpenAICompat(HttpProvider):
    """Any server speaking the OpenAI chat API: LM Studio, llama.cpp, vLLM, hosted APIs."""

    async def complete(self, r: Request) -> Response:
        body: dict[str, Any] = {
            "model": r.model,
            "messages": _messages(r),
            "temperature": r.temperature,
            "seed": r.seed,
            "max_tokens": r.max_tokens,
        }
        if r.format:
            schema = {"name": "output", "schema": r.format}
            body["response_format"] = {"type": "json_schema", "json_schema": schema}
        d = await self.call("POST", "/chat/completions", body)
        choice, usage = d["choices"][0], d.get("usage") or {}
        stops = {"length": "length", "content_filter": "refusal"}
        return Response(
            text=choice["message"].get("content") or "",
            model=d.get("model", r.model),
            stop=stops.get(choice.get("finish_reason"), "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=d,
        )


def _unit(text: str) -> float:
    return int(sha(text)[:8], 16) / 2**32


class Mock(Provider):
    """Deterministic stand-in for tests and simulations.

    The model name carries the pass rate ("mock:0.8"). Each case has a latent difficulty
    shared by every mock model, so two mock models give paired, correlated results: cases
    well below the rate always pass, cases well above always fail, and the 20% band around
    it is flaky.
    """

    async def complete(self, r: Request) -> Response:
        rate = float(r.model.partition(":")[2] or 0.8)
        chance = min(1.0, max(0.0, (rate - _unit(r.user)) / 0.2 + 0.5))
        passed = _unit(f"{r.model}|{r.user}|{r.seed}") < chance
        text = str(r.expected) if passed else "wrong"
        if r.format:  # asked for a structured verdict: play the judge
            verdict = "yes" if passed else "no"
            text = json.dumps({"evidence": "mock", "reasoning": "mock", "verdict": verdict})
        return Response(text=text, model=r.model, latency_ms=0.0)

    async def digest(self, model: str) -> str:
        return "mock"


def make_provider(cfg: ProviderCfg, transport: Any = None) -> Provider:
    if cfg.kind == "mock":
        return Mock()
    key = None
    if cfg.api_key_env:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise ProviderError(f"environment variable {cfg.api_key_env} is not set")
    cls = Ollama if cfg.kind == "ollama" else OpenAICompat
    return cls(cfg.base_url, key, transport)
