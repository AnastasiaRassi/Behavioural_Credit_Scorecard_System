"""A free, open-source LLM served locally by Ollama. No API key, no data leaves the machine.

Every failure returns None rather than raising, so a caller always has its template
to fall back on: the assistant must keep answering when Ollama is not running.

Setup, once:  ollama pull qwen2.5:3b   (OLLAMA_MODEL in .env picks another model)
"""
from __future__ import annotations

import logging
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


def chat(messages: Sequence[Mapping[str, str]], timeout: float = TIMEOUT_SECONDS) -> str | None:
    """The model's reply to `messages`, or None if it could not be produced."""
    try:
        response = requests.post(
            f"{ollama_url()}/api/chat",
            json={
                "model": ollama_model(),
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
        return None
    except requests.ConnectionError:
        logger.warning("local LLM: Ollama not reachable at %s; start it with: ollama serve", ollama_url())
        return None
    except requests.RequestException as error:
        logger.warning("local LLM: request failed (%s); using template", error)
        return None

    if response.status_code == 404:
        logger.warning("local LLM: model %s not installed; run: ollama pull %s", ollama_model(), ollama_model())
        return None
    if not response.ok:
        logger.warning("local LLM: Ollama returned %s; using template", response.status_code)
        return None

    try:
        body = response.json()
    except ValueError:
        logger.warning("local LLM: reply was not JSON; using template")
        return None

    # "length" means the reply was cut off at MAX_TOKENS: half an answer is worse than the template.
    if body.get("done_reason", "stop") != "stop":
        logger.warning("local LLM: stopped for %s; using template", body.get("done_reason"))
        return None

    text = ((body.get("message") or {}).get("content") or "").strip()
    return text or None
