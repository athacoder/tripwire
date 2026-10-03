import asyncio
import json

import httpx
import pytest

from tripwire.config import ProviderCfg
from tripwire.providers import Mock, Ollama, OpenAICompat, ProviderError, Request, make_provider

# Shape recorded from Ollama 0.35.0 (/api/chat, stream off).
OLLAMA_REPLY = {
    "model": "gemma3:4b",
    "message": {"role": "assistant", "content": "Paris"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 3_500_000_000,
    "load_duration": 3_000_000_000,
    "prompt_eval_count": 25,
    "prompt_eval_duration": 400_000_000,
    "eval_count": 2,
    "eval_duration": 50_000_000,
}
REQUEST = Request("gemma3:4b", "be brief", "capital of France?", 0.7, 42, 2048, 8)


def serve(handler):
    return httpx.MockTransport(handler)


def test_ollama_sends_sampling_options_and_parses_the_reply():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OLLAMA_REPLY)

    reply = asyncio.run(Ollama("http://x", transport=serve(handler)).complete(REQUEST))
    assert seen["options"] == {"temperature": 0.7, "seed": 42, "num_ctx": 2048, "num_predict": 8}
    assert seen["stream"] is False and seen["messages"][0]["role"] == "system"
    assert (reply.text, reply.stop, reply.prompt_tokens, reply.output_tokens) == (
        "Paris",
        "stop",
        25,
        2,
    )
    assert reply.latency_ms == 500  # load time is excluded


def test_ollama_reports_a_cut_off_answer():
    cut = {**OLLAMA_REPLY, "done_reason": "length"}
    provider = Ollama("http://x", transport=serve(lambda r: httpx.Response(200, json=cut)))
    assert asyncio.run(provider.complete(REQUEST)).stop == "length"


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (503, True), (400, False), (404, False)]
)
def test_http_errors_are_classified(status, retryable):
    provider = Ollama("http://x", transport=serve(lambda r: httpx.Response(status, text="no")))
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.complete(REQUEST))
    assert caught.value.retryable is retryable


def test_connection_failures_are_retryable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as caught:
        asyncio.run(Ollama("http://x", transport=serve(handler)).complete(REQUEST))
    assert caught.value.retryable


def test_ollama_digest_requires_the_model_to_be_pulled():
    tags = {"models": [{"name": "gemma3:4b", "digest": "abc123"}]}
    provider = Ollama("http://x", transport=serve(lambda r: httpx.Response(200, json=tags)))
    assert asyncio.run(provider.digest("gemma3:4b")) == "abc123"
    with pytest.raises(ProviderError, match="ollama pull"):
        asyncio.run(provider.digest("missing:1b"))


def test_openai_compat_parses_usage_and_stop_reason(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen.update(json.loads(request.content))
        usage = {"prompt_tokens": 17, "completion_tokens": 8}
        choice = {"message": {"content": "Par"}, "finish_reason": "length"}
        return httpx.Response(200, json={"model": "m", "choices": [choice], "usage": usage})

    monkeypatch.setenv("TEST_KEY", "secret")
    cfg = ProviderCfg(kind="openai_compat", base_url="http://x/v1", api_key_env="TEST_KEY")
    provider = make_provider(cfg, serve(handler))
    assert isinstance(provider, OpenAICompat)
    reply = asyncio.run(provider.complete(REQUEST))
    assert seen["auth"] == "Bearer secret" and seen["seed"] == 42 and seen["max_tokens"] == 8
    assert (reply.text, reply.stop, reply.prompt_tokens, reply.output_tokens) == (
        "Par",
        "length",
        17,
        8,
    )
    assert reply.latency_ms is None  # the runner falls back to its own clock


def test_a_named_key_must_be_set(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ProviderError, match="TEST_KEY"):
        make_provider(ProviderCfg(kind="openai_compat", api_key_env="TEST_KEY"))


def test_mock_is_deterministic_and_hits_its_pass_rate():
    def passed(model, i, seed=0):
        request = Request(model, "", f"case {i}", seed=seed, expected="right")
        return asyncio.run(Mock().complete(request)).text == "right"

    assert [passed("mock:0.8", i) for i in range(50)] == [passed("mock:0.8", i) for i in range(50)]
    rate = sum(passed("mock:0.8", i) for i in range(2000)) / 2000
    assert abs(rate - 0.8) < 0.03
    assert sum(passed("mock:0.6", i) for i in range(2000)) / 2000 < rate - 0.1


def test_a_reply_that_is_not_json_is_a_provider_error():
    provider = Ollama("http://x", transport=serve(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(ProviderError, match="not JSON") as caught:
        asyncio.run(provider.complete(REQUEST))
    assert not caught.value.retryable


def test_a_bare_model_name_matches_the_latest_tag():
    tags = {"models": [{"name": "nomic-embed-text:latest", "digest": "abc"}]}
    provider = Ollama("http://x", transport=serve(lambda r: httpx.Response(200, json=tags)))
    assert asyncio.run(provider.digest("nomic-embed-text")) == "abc"
