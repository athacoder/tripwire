"""Typed view of tripwire.toml and the target files it points at."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProviderCfg(BaseModel):
    kind: Literal["ollama", "openai_compat", "mock"]
    base_url: str = "http://localhost:11434"
    api_key_env: str | None = None  # name of the environment variable, never the key itself


class TargetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo here would silently change the fingerprint

    kind: Literal["prompt", "python", "http"] = "prompt"
    provider: str = "local"
    model: str = ""
    system: str | None = None  # path to the system prompt, relative to the config file
    template: str | None = None  # path to the user template, filled from the case input
    entry: str | None = None  # "module:function" for kind = "python"
    url: str | None = None  # endpoint for kind = "http"
    watch: list[str] = Field(default_factory=list)  # globs whose contents affect the output
    temperature: float = 0.7
    num_ctx: int = 4096
    max_tokens: int = 64
    format: dict[str, Any] | None = None  # JSON schema the output must follow
    salt: str = ""  # change to force fresh samples


class SuiteCfg(BaseModel):
    dataset: str
    target: str
    reps: int = 1


def _default_providers() -> dict[str, ProviderCfg]:
    return {"local": ProviderCfg(kind="ollama"), "mock": ProviderCfg(kind="mock")}


class Config(BaseModel):
    root: Path
    db: str = "tripwire.db"
    concurrency: int = 1  # local models share one GPU; raise it for hosted providers
    timeout: float = 120.0  # hard wall-clock limit per case, in seconds
    retries: int = 3
    backoff: float = 1.0  # base delay in seconds, doubled per attempt
    provider: dict[str, ProviderCfg] = Field(default_factory=_default_providers)
    suite: dict[str, SuiteCfg] = Field(default_factory=dict)

    def target(self, suite: str) -> TargetCfg:
        path = self.root / self.suite[suite].target
        return TargetCfg(**tomllib.loads(path.read_text(encoding="utf-8")))


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data["provider"] = {**_default_providers(), **data.get("provider", {})}
    return Config(root=path.resolve().parent, **data)
