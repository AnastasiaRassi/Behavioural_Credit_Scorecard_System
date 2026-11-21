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
    # Set on errors that are the caller's to fix: their message is safe and useful to show.
    expose_message = False
    headers: dict[str, str] = {}

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
        elif self.expose_message and message:
            self.user_message = message
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


class InvalidRequest(AppError):
    """The request is well-formed HTTP but wrong in substance. The message says how."""

    code = "INVALID_REQUEST"
    status = 422
    expose_message = True


class Unauthenticated(AuthError):
    code = "AUTH_REQUIRED"
    status = 401
    expose_message = True
    headers = {"WWW-Authenticate": "Bearer"}


class Forbidden(AuthError):
    code = "FORBIDDEN"
    status = 403
    expose_message = True


class ResourceNotFound(NotFoundError):
    """A path id or asserted identity names nothing. The message names the id."""

    code = "RESOURCE_NOT_FOUND"
    expose_message = True


class RequestTooLarge(InvalidRequest):
    code = "REQUEST_TOO_LARGE"
    status = 413


class EmptyPassword(InvalidRequest, ValueError):
    code = "EMPTY_PASSWORD"


def failure_line(exception: Exception, code: str) -> str:
    """A log line for any exception: its own code if it is ours, else `code` and its type."""
    if isinstance(exception, AppError):
        return exception.log_line()
    return f"{code}: {type(exception).__name__}: {exception}"
