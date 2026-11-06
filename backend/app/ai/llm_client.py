"""The assistant's LLM: OpenAI when a key is set, otherwise the local Ollama model.

`chat()` has the contract of `local_llm.chat()`: it returns an `LlmCall` and never
raises, so a caller always has its template to fall back on. With no key and no
Ollama running there is simply no text, and the template answers.

Data leaves the machine when OPENAI_API_KEY is set. What is sent is the fact sheet
(`assistant.fact_sheet`): the risk band, the reasons, the account's figures and six
months of history, never the probability or any protected attribute.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Mapping, Sequence

import openai

from app.ai import local_llm
from app.ai.local_llm import LlmCall
from app.core.config import openai_api_key, openai_model

logger = logging.getLogger(__name__)

# The timeout is the circuit breaker: past 4 s the template answers instead. Retries
# are off because each one would add another full wait for the cardholder.
TIMEOUT_SECONDS = 4.0
MAX_TOKENS = 300
# Zero, so the model restates the facts instead of embellishing them.
TEMPERATURE = 0.0


def chat(messages: Sequence[Mapping[str, str]]) -> LlmCall:
    """The model's reply to `messages`, or a call with no text if it could not be produced."""
    if openai_api_key() is None:
        return local_llm.chat(messages)
    return _openai_chat(messages)


@lru_cache(maxsize=1)
def _client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=openai_api_key(), timeout=TIMEOUT_SECONDS, max_retries=0
    )


def _openai_chat(messages: Sequence[Mapping[str, str]]) -> LlmCall:
    model = openai_model()
    started = time.perf_counter()

    def finished(
        text: str | None, failure: str | None, usage=None
    ) -> LlmCall:
        # Wall clock, so a timeout is recorded at the wait the cardholder had.
        return LlmCall(
            text=text,
            model=model,
            latency_ms=round((time.perf_counter() - started) * 1000),
            failure=failure,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    try:
        response = _client().chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    # Most specific first: a timeout is also a connection error, and the auth and
    # rate-limit errors are also status errors.
    except openai.APITimeoutError:
        logger.warning("llm: no reply within %ss; using template", TIMEOUT_SECONDS)
        return finished(None, "llm_timeout")
    except openai.APIConnectionError:
        logger.warning("llm: OpenAI not reachable; using template")
        return finished(None, "llm_unreachable")
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        logger.warning("llm: OpenAI rejected the API key; using template")
        return finished(None, "llm_auth")
    except openai.RateLimitError:
        logger.warning("llm: OpenAI rate limit or quota hit; using template")
        return finished(None, "llm_rate_limited")
    except openai.OpenAIError as error:
        logger.warning("llm: OpenAI request failed (%s); using template", error)
        return finished(None, "llm_http_error")
    except Exception:
        # The contract is never to raise, whatever the SDK or the network does.
        logger.exception("llm: unexpected failure; using template")
        return finished(None, "llm_request_failed")

    usage = getattr(response, "usage", None)
    if not response.choices:
        return finished(None, "llm_empty", usage)
    choice = response.choices[0]

    # Anything but "stop" (length, content_filter) is a cut-off or withheld answer:
    # half an answer is worse than the template.
    if choice.finish_reason != "stop":
        logger.warning("llm: stopped for %s; using template", choice.finish_reason)
        return finished(None, "llm_truncated", usage)

    text = (choice.message.content or "").strip()
    if not text:
        return finished(None, "llm_empty", usage)
    return finished(text, None, usage)
