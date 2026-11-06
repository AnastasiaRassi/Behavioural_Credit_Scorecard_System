"""Checks for the LLM client: provider order, OpenAI settings and error mapping.

OpenAI is replaced by a fake client, so nothing here touches the network. conftest
stubs `llm_client.chat` for every test; the real one is captured at import.
"""
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.ai import llm_client, local_llm

REAL_CHAT = llm_client.chat
MESSAGES = [{"role": "system", "content": "facts"}, {"role": "user", "content": "hi"}]
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _response(text="Hello.", finish_reason="stop", usage=(120, 30)):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None,
    )


class FakeClient:
    """Stands in for openai.OpenAI; returns or raises what it was given."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(llm_client, "openai_api_key", lambda: "sk-test")
    monkeypatch.setattr(llm_client, "openai_model", lambda: "gpt-4o-mini")


def _use(monkeypatch, outcome):
    client = FakeClient(outcome)
    monkeypatch.setattr(llm_client, "_client", lambda: client)
    return client


def test_a_reply_that_stopped_normally_is_returned_with_its_token_counts(monkeypatch, with_key):
    client = _use(monkeypatch, _response())

    call = REAL_CHAT(MESSAGES)

    assert (call.text, call.failure, call.model) == ("Hello.", None, "gpt-4o-mini")
    assert (call.prompt_tokens, call.completion_tokens) == (120, 30)
    assert call.latency_ms >= 0
    assert client.requests[0]["temperature"] == 0.0
    assert client.requests[0]["max_tokens"] == 300
    assert client.requests[0]["model"] == "gpt-4o-mini"


def test_a_cut_off_reply_is_rejected_as_truncated(monkeypatch, with_key):
    _use(monkeypatch, _response(text="Half an ans", finish_reason="length"))

    call = REAL_CHAT(MESSAGES)

    assert (call.text, call.failure) == (None, "llm_truncated")
    assert call.completion_tokens == 30


def test_an_empty_reply_is_a_failure(monkeypatch, with_key):
    _use(monkeypatch, _response(text="  "))

    assert REAL_CHAT(MESSAGES).failure == "llm_empty"


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (openai.APITimeoutError(request=REQUEST), "llm_timeout"),
        (openai.APIConnectionError(request=REQUEST), "llm_unreachable"),
        (openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None), "llm_auth"),
        (openai.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), "llm_rate_limited"),
        (openai.InternalServerError("boom", response=httpx.Response(500, request=REQUEST), body=None), "llm_http_error"),
        (RuntimeError("something else"), "llm_request_failed"),
    ],
)
def test_each_error_maps_to_its_own_failure_and_never_raises(monkeypatch, with_key, error, failure):
    _use(monkeypatch, error)

    call = REAL_CHAT(MESSAGES)

    assert (call.text, call.failure) == (None, failure)
    assert call.model == "gpt-4o-mini"


def test_the_client_is_built_with_a_four_second_timeout_and_no_retries(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    llm_client._client.cache_clear()
    monkeypatch.setattr(llm_client, "openai_api_key", lambda: "sk-test")
    monkeypatch.setattr(openai, "OpenAI", Recorder)
    try:
        llm_client._client()
    finally:
        llm_client._client.cache_clear()

    assert (seen["timeout"], seen["max_retries"], seen["api_key"]) == (4.0, 0, "sk-test")


def test_without_a_key_the_local_model_answers(monkeypatch):
    monkeypatch.setattr(llm_client, "openai_api_key", lambda: None)
    local = local_llm.LlmCall("From Ollama.", "qwen2.5:3b", 5)
    monkeypatch.setattr(local_llm, "chat", lambda messages: local)

    assert REAL_CHAT(MESSAGES) is local


def test_without_a_key_and_without_ollama_there_is_no_text(monkeypatch):
    monkeypatch.setattr(llm_client, "openai_api_key", lambda: None)
    down = local_llm.LlmCall(None, "qwen2.5:3b", 5, failure="llm_unreachable")
    monkeypatch.setattr(local_llm, "chat", lambda messages: down)

    call = REAL_CHAT(MESSAGES)

    assert (call.text, call.failure) == (None, "llm_unreachable")


def test_openai_is_not_called_when_there_is_no_key(monkeypatch):
    monkeypatch.setattr(llm_client, "openai_api_key", lambda: None)
    monkeypatch.setattr(local_llm, "chat", lambda messages: local_llm.LlmCall(None, "m", 0, failure="llm_unreachable"))
    monkeypatch.setattr(llm_client, "_client", lambda: pytest.fail("OpenAI must not be used without a key"))

    REAL_CHAT(MESSAGES)
