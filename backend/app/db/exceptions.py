"""Errors from the database layer: connection, lookups, integrity, ingestion.

The built-in bases (`LookupError`, `FileNotFoundError`, `RuntimeError`) are kept so
existing handlers still catch these.
"""
from app.core.exceptions import AppError, ConfigError, ConflictError, InfraError, NotFoundError


class DatabaseUnavailable(InfraError):
    """No connection, or the connection dropped mid-request."""

    code = "DATABASE_UNAVAILABLE"
    user_message = "We cannot reach our records right now."


class SchemaOutOfDate(ConfigError, RuntimeError):
    """The database is on an earlier schema than the code."""

    code = "SCHEMA_OUT_OF_DATE"
    user_message = "The service is being upgraded. Try again later."


class PersistenceFailed(AppError):
    """The database rejected a write. `context` names the table; the cause is chained."""

    code = "PERSISTENCE_FAILED"
    user_message = "We could not save that."


class RecordConflict(ConflictError):
    """A write collides with an existing row (a unique key, a second live version)."""

    code = "RECORD_CONFLICT"


class CustomerNotFound(NotFoundError, LookupError):
    code = "CUSTOMER_NOT_FOUND"
    user_message = "We could not find that account."


class CustomerNotIngested(NotFoundError, RuntimeError):
    """A queue row names a customer the database never ingested."""

    code = "CUSTOMER_NOT_INGESTED"
    status = 500


class QueueIdMismatch(AppError, RuntimeError):
    """The worklist artifact and the ingested data describe the same account differently."""

    code = "QUEUE_ID_MISMATCH"


class IncompleteHistory(AppError, RuntimeError):
    """A customer is missing statement months the model needs."""

    code = "INCOMPLETE_HISTORY"
    user_message = "This account does not have enough history to score."


class NoLiveModel(InfraError):
    """No model version is registered as live."""

    code = "NO_LIVE_MODEL"
    user_message = "The scoring model is not available right now."


class SourceUnavailable(InfraError, FileNotFoundError):
    """The raw dataset file is missing or malformed, so ingestion cannot run."""

    code = "SOURCE_UNAVAILABLE"
