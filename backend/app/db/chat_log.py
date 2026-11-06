"""Writing and summarising the chat log (`chat_turn`)."""
import logging
from typing import Mapping

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import ChatTurn

logger = logging.getLogger(__name__)


def record(session: Session, turn: Mapping) -> None:
    """Store one turn. Never raises: a failed log write must not cost the cardholder their reply."""
    try:
        # A savepoint, so a failed insert (such as the table not existing yet) rolls
        # back alone instead of poisoning the request's own transaction.
        with session.begin_nested():
            session.add(ChatTurn(**turn))
    except SQLAlchemyError:
        logger.exception("chat log: could not store the turn; is the chat_turn table created?")


def summary(session: Session) -> dict:
    """Volume, fallback rate and cost of the assistant, from the stored turns.

    Latency and tokens describe turns the model answered; a timeout's wait would
    otherwise be averaged in with replies that were actually produced.
    """
    turns = session.query(func.count(ChatTurn.id)).scalar()
    if not turns:
        return {"turns": 0, "reason": "No chat turns have been recorded yet."}

    model_turns = session.query(func.count(ChatTurn.id)).filter(ChatTurn.source == "model").scalar()
    reasons = dict(
        session.query(ChatTurn.fallback_reason, func.count(ChatTurn.id))
        .filter(ChatTurn.fallback_reason.is_not(None))
        .group_by(ChatTurn.fallback_reason)
        .all()
    )

    result = {
        "turns": turns,
        "model_turns": model_turns,
        "template_turns": turns - model_turns,
        "template_rate": round((turns - model_turns) / turns, 4),
        "fallback_reasons": reasons,
    }
    if model_turns:
        median, slowest, prompt_tokens, completion_tokens = (
            session.query(
                func.percentile_cont(0.5).within_group(ChatTurn.latency_ms),
                func.percentile_cont(0.95).within_group(ChatTurn.latency_ms),
                func.avg(ChatTurn.prompt_tokens),
                func.avg(ChatTurn.completion_tokens),
            )
            .filter(ChatTurn.source == "model")
            .one()
        )
        result["model_latency_ms_p50"] = round(median)
        result["model_latency_ms_p95"] = round(slowest)
        result["avg_prompt_tokens"] = None if prompt_tokens is None else round(prompt_tokens)
        result["avg_completion_tokens"] = None if completion_tokens is None else round(completion_tokens)
    return result
