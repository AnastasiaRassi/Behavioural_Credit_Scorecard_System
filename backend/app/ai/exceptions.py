"""Errors from the AI layer: provider failures and policy blocks.

Nothing here carries a prompt, a reply or an account's figures: `context` is
logged, so it holds only the model name, a latency and a failure code.
`llm_client.chat()` still never raises: it returns an `LlmCall` with `failure`
set, and `error_for` turns that code into one of these for logging.
"""
from app.core.exceptions import AppError, InfraError


class LlmProviderError(InfraError):
    code = "LLM_PROVIDER_ERROR"
    user_message = "The assistant is unavailable right now."


class LlmTimeout(LlmProviderError):
    code = "LLM_TIMEOUT"


class LlmUnreachable(LlmProviderError):
    code = "LLM_UNREACHABLE"


class LlmAuthRejected(LlmProviderError):
    code = "LLM_AUTH_REJECTED"


class LlmRateLimited(LlmProviderError):
    code = "LLM_RATE_LIMITED"


class LlmBadResponse(LlmProviderError):
    """The provider answered, but empty, cut off, or not in the expected shape."""

    code = "LLM_BAD_RESPONSE"


class GuardrailBlocked(AppError):
    """A request or reply was stopped by policy. `context['reason']` says which rule."""

    code = "GUARDRAIL_BLOCKED"
    status = 422
    user_message = "I can't help with that request."


_BY_FAILURE: dict[str, type[LlmProviderError]] = {
    "llm_timeout": LlmTimeout,
    "llm_unreachable": LlmUnreachable,
    "llm_auth": LlmAuthRejected,
    "llm_rate_limited": LlmRateLimited,
    "llm_empty": LlmBadResponse,
    "llm_truncated": LlmBadResponse,
    "llm_bad_response": LlmBadResponse,
}


def error_for(failure: str, model: str | None = None, latency_ms: int | None = None) -> LlmProviderError:
    """The typed error for an `LlmCall.failure` code."""
    kind = _BY_FAILURE.get(failure, LlmProviderError)
    return kind(failure, context={"failure": failure, "model": model, "latency_ms": latency_ms})
