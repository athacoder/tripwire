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
    model_config = ConfigDict(extra="forbid")

    dataset: str
    target: str
    reps: int = 1
    scorers: list[str] = Field(default_factory=lambda: ["exact"])
    primary_metric: str = "exact.pass"  # "<scorer>.<metric>"; this one decides the verdict
    alpha: float = 0.05  # one-sided error rate of each interval bound
    margin: float = 0.03  # largest drop in the primary metric that is still acceptable
    # What an undecided gate does: block, let it through with a warning, or "escalate":
    # look at `first_stage` cases first and run the rest only if those cannot decide.
    on_inconclusive: Literal["fail", "warn", "escalate"] = "fail"
    first_stage: int = 250
    # Limits on the candidate: output_tokens_ratio_max, latency_p95_ratio_max,
    # truncation_rate_max, error_rate_max.
    guardrails: dict[str, float] = Field(default_factory=dict)
    # Filled in when the config is loaded: judge criterion -> version of its prompt.
    judge_versions: dict[str, str] = Field(default_factory=dict)


class JudgeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "local"
    model: str = "llama3.1:8b"
    system: str  # path to the judge's instructions
    template: (
        str  # path to the user template: case input fields, {reference}, {answer}, {criterion}
    )
    rubric: str  # path to a TOML file with a [criteria] table: name -> question
    num_ctx: int = 2048
    max_tokens: int = 300


def _default_providers() -> dict[str, ProviderCfg]:
    return {"local": ProviderCfg(kind="ollama"), "mock": ProviderCfg(kind="mock")}


class Config(BaseModel):
    root: Path
    db: str = "tripwire.db"
    bundles: str = "bundles"
    concurrency: int = 1  # local models share one GPU; raise it for hosted providers
    timeout: float = 120.0  # hard wall-clock limit per case, in seconds
    retries: int = 3
    backoff: float = 1.0  # base delay in seconds, doubled per attempt
    provider: dict[str, ProviderCfg] = Field(default_factory=_default_providers)
    judge: JudgeCfg | None = None
    tracelens_url: str | None = None  # dashboard address, for trace links in reports
    suite: dict[str, SuiteCfg] = Field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.root / self.db  # an absolute `db` wins, as pathlib joins go

    def get_suite(self, name: str) -> SuiteCfg:
        if name not in self.suite:
            known = ", ".join(sorted(self.suite)) or "none defined"
            raise ValueError(f"unknown suite {name!r} (known: {known})")
        return self.suite[name]

    def target(self, suite: str) -> TargetCfg:
        path = self.root / self.get_suite(suite).target
        return TargetCfg(**tomllib.loads(path.read_text(encoding="utf-8")))


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data["provider"] = {**_default_providers(), **data.get("provider", {})}
    cfg = Config(root=path.resolve().parent, **data)
    if cfg.judge:
        from .judge import Judge  # judge needs Config; import late to avoid a cycle

        versions = Judge(cfg).versions
        for suite in cfg.suite.values():
            suite.judge_versions = versions
    return cfg
