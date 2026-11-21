"""Application logging: console, a rotating app log, and a longer-kept audit log.

Every line carries the id of the request that produced it (`request_id`), set by
the middleware in `main.py`, so one request can be followed across modules.
"""
import logging
import os
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_ROOT = Path(__file__).resolve().parents[2] / "logs"


def log_dir() -> Path:
    """`logs/test/` under pytest (conftest sets APP_ENV=test), else `logs/app/`."""
    return LOGS_ROOT / ("test" if os.environ.get("APP_ENV") == "test" else "app")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s"

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def _rotating(name: str, max_bytes: int, backups: int) -> RotatingFileHandler:
    return RotatingFileHandler(
        log_dir() / name, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    # Idempotent: --reload and repeated imports must not stack handlers.
    if any(getattr(h, "_app_handler", False) for h in root.handlers):
        return

    log_dir().mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    request_filter = _RequestIdFilter()

    for handler in (_rotating("app.log", 5_000_000, 10), logging.StreamHandler()):
        handler.setFormatter(formatter)
        handler.addFilter(request_filter)
        handler._app_handler = True
        root.addHandler(handler)
    root.setLevel(level)

    # Audit lines also reach app.log through propagation; this copy is kept longer
    # and holds nothing else, so "who did what" can be read on its own.
    audit_handler = _rotating("audit.log", 10_000_000, 20)
    audit_handler.setFormatter(formatter)
    audit_handler.addFilter(request_filter)
    logging.getLogger("audit").addHandler(audit_handler)
