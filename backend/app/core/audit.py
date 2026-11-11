"""Audit events: who did what to whom, as one greppable line each.

Written to the `audit` logger, which `logging_config` also sends to `logs/audit.log`
(kept longer than the application log). Fields are `key=value`; the request id
the middleware stamps on every line ties an event to its HTTP request.

Log identifiers, never content: no names, passwords, justification or message text.
"""
import logging

audit_logger = logging.getLogger("audit")


def _render(value: object) -> str:
    text = "-" if value is None else str(value)
    return repr(text) if " " in text or not text else text


def event(action: str, actor: str | None = None, level: int = logging.INFO, **fields) -> None:
    """Record one auditable action. `actor` is the email or role the request names."""
    pairs = " ".join(f"{key}={_render(value)}" for key, value in fields.items())
    audit_logger.log(level, "action=%s actor=%s %s", action, _render(actor), pairs)
