"""A free, open-source LLM served locally by Ollama. No API key, no data leaves the machine.

This is the assistant's second choice: `llm_client.chat` uses it when OPENAI_API_KEY is unset.

Every failure yields a call with no text rather than raising, so a caller always has
its template to fall back on: the assistant must keep answering when Ollama is not
running. The call also carries what the chat log needs: model, latency, token counts
and, on failure, why.

Setup, once:  ollama pull qwen2.5:3b   (OLLAMA_MODEL in .env picks another model)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import requests

from app.core.config import ollama_model, ollama_url

logger = logging.getLogger(__name__)

# A 3B model on CPU takes a few seconds to load and around 15 tokens a second after.
TIMEOUT_SECONDS = 60.0
MAX_TOKENS = 350
# Low, so the model restates the facts instead of embellishing them.
TEMPERATURE = 0.2
CONTEXT_TOKENS = 4096


@dataclass(frozen=True)
class LlmCall:
    """One request to the model. `text` is None exactly when `failure` is set."""

    text: str | None
    model: str
    latency_ms: int
    failure: str | None = None  # llm_timeout, llm_unreachable, llm_model_missing, ...
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def chat(messages: Sequence[Mapping[str, str]], timeout: float = TIMEOUT_SECONDS) -> LlmCall:
    """The model's reply to `messages`, or a call with no text if it could not be produced."""
    model = ollama_model()
    started = time.perf_counter()

    def failed(reason: str, body: Mapping | None = None) -> LlmCall:
        return _call(None, model, started, reason, body)

    try:
        response = requests.post(
            f"{ollama_url()}/api/chat",
            json={
                "model": model,
                "messages": list(messages),
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "num_ctx": CONTEXT_TOKENS,
                },
            },
            timeout=timeout,
        )
    # Most specific first: a connect timeout is both a Timeout and a ConnectionError.
    except requests.Timeout:
        logger.warning("local LLM: no reply within %ss; using template", timeout)
        return failed("llm_timeout")
    except requests.ConnectionError:
        logger.warning("local LLM: Ollama not reachable at %s; start it with: ollama serve", ollama_url())
        return failed("llm_unreachable")
    except requests.RequestException as error:
        logger.warning("local LLM: request failed (%s); using template", error)
        return failed("llm_request_failed")

    if response.status_code == 404:
        logger.warning("local LLM: model %s not installed; run: ollama pull %s", model, model)
        return failed("llm_model_missing")
    if not response.ok:
        logger.warning("local LLM: Ollama returned %s; using template", response.status_code)
        return failed("llm_http_error")

    try:
        body = response.json()
    except ValueError:
        logger.warning("local LLM: reply was not JSON; using template")
        return failed("llm_bad_response")

    # "length" means the reply was cut off at MAX_TOKENS: half an answer is worse than the template.
    if body.get("done_reason", "stop") != "stop":
        logger.warning("local LLM: stopped for %s; using template", body.get("done_reason"))
        return failed("llm_truncated", body)

    text = ((body.get("message") or {}).get("content") or "").strip()
    if not text:
        return failed("llm_empty", body)
    return _call(text, model, started, None, body)


def _call(
    text: str | None, model: str, started: float, failure: str | None, body: Mapping | None
) -> LlmCall:
    # Wall clock, so a timeout is recorded at the wait the cardholder actually had.
    # Token counts are Ollama's own and absent when it never answered.
    body = body or {}
    return LlmCall(
        text=text,
        model=model,
        latency_ms=round((time.perf_counter() - started) * 1000),
        failure=failure,
        prompt_tokens=body.get("prompt_eval_count"),
        completion_tokens=body.get("eval_count"),
    )
