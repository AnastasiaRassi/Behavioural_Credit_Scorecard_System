"""Domain errors: one base, a stable code, and a message that is safe to show.

Each layer (`ml/`, `db/`, `ai/`) defines its own subclasses in its own
`exceptions.py` and raises them where an invariant breaks. The API turns any
`AppError` into `{code, message, request_id}`; the technical text and `context`
only ever go to the log.
"""
from typing import Any


class AppError(Exception):
    """Base for every error this service raises on purpose.

    `message` is for the log: precise, may name files and ids. `user_message` is
    what a screen may show. `context` holds ids and counts, never prompts, raw
    inputs or personal data, because it is logged.
    """

    code = "APP_ERROR"
    status = 500
    user_message = "Something went wrong on our side."

    def __init__(
        self,
        message: str | None = None,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message or self.user_message)
        if user_message is not None:
            self.user_message = user_message
        self.context = context or {}

    def log_line(self) -> str:
        context = " ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.code}: {self} {context}".rstrip()


class ConfigError(AppError):
    code = "CONFIG_ERROR"
    status = 500
    user_message = "The service is not configured correctly."


class AuthError(AppError):
    code = "AUTH_ERROR"
    status = 401
    user_message = "You need to sign in again."


class InfraError(AppError):
    """A dependency the service needs (database, model files, a provider) is not usable."""

    code = "INFRA_ERROR"
    status = 503
    user_message = "A service we depend on is unavailable. Try again in a moment."


class NotFoundError(AppError):
    code = "NOT_FOUND"
    status = 404
    user_message = "We could not find that."


class ConflictError(AppError):
    code = "CONFLICT"
    status = 409
    user_message = "That conflicts with what is already stored."
