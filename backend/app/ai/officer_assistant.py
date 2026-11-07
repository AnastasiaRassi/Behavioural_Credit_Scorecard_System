"""Loana AI for the loan officer: questions about one customer's account.

Same model client, fact sheet and guardrails as the cardholder chat
(`assistant.py`); only the audience differs. The sheet is built from the
cardholder's reduced view, so the probability never reaches OpenAI either. The
officer reads it in the prediction panel; the assistant explains the band and the
reasons behind it.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Mapping, Sequence

from app.ai import llm_client
from app.ai.assistant import (
    MAX_HISTORY,
    TEMPLATE_NOTE,
    ChatReply,
    TurnRecord,
    check_reply,
    fact_sheet,
)
from app.ai.coaching import DISCLAIMER, build_payload
from app.ml import artifacts, worklist

logger = logging.getLogger(__name__)

# Recorded with every reply; bump whenever SYSTEM_PROMPT changes.
PROMPT_VERSION = "v1-officer"

SYSTEM_PROMPT = """You are Loana AI, an assistant in CrediTrait, a credit card prototype. You help a loan officer understand one existing customer's account.
Everything you know is in FACTS below. It comes from a risk model that estimates whether the customer's next payment will be missed, and from a search for the smallest change this month that would improve the rating.

Rules:
1. Tone: calm, objective and concise. Refer to "the customer", never "applicant".
2. Use only the FACTS. Quote amounts exactly as written there. Never invent or estimate a number, percentage, amount, date or scenario.
3. Changes: only the steps listed under one_month_counterfactual.steps. If the steps list is empty, say plainly that no single-month action clears this risk band.
4. The rating is a band. FACTS do not include a probability; if asked for one, say the officer can read it in the prediction panel.
5. Do not approve, deny or recommend a decision, promise an outcome, or say what any lender will decide. The officer decides.
6. At most five short sentences.

FACTS:
{facts}"""


def answer(
    row: dict,
    messages: Sequence[Mapping[str, str]],
    stored_prediction=None,
    record: Mapping[str, float] | None = None,
    on_turn: Callable[[TurnRecord], None] | None = None,
) -> ChatReply:
    """The assistant's next turn for one customer. Always returns a reply.

    `messages` is the conversation so far and must end with the officer's question.
    """
    view = worklist.cardholder_view(row, stored_prediction=stored_prediction, record=record)
    payload = build_payload(view["risk_band"], view["reasons"], record)
    sheet = fact_sheet(view, row.get("history") or [], payload)

    conversation = [{"role": m["role"], "content": m["content"]} for m in messages]
    question = conversation[-1]["content"]
    call = llm_client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT.format(facts=json.dumps(sheet, indent=1))}]
        + conversation[-MAX_HISTORY:]
    )

    reply, reason, rejected, guard = None, call.failure, None, None
    if call.text is not None:
        guard = check_reply(call.text, sheet, payload)
        reason = guard.fallback_reason
        if reason is None:
            reply = ChatReply(call.text, "model", "", DISCLAIMER)
        else:
            rejected = call.text
            logger.warning("officer assistant: reply failed %s; using template", reason)
    if reply is None:
        reply = ChatReply(_template(view, payload), "template", TEMPLATE_NOTE, DISCLAIMER)

    if on_turn is not None:
        _log(on_turn, row, conversation, question, reply, call, reason, rejected, guard)
    return reply


def _template(view: dict, payload) -> str:
    """A fixed summary of the same facts, for when the model cannot answer."""
    raising = [r["description"] for r in view["reasons"] if r["raises_risk"]]
    lowering = [r["description"] for r in view["reasons"] if not r["raises_risk"]]
    sentences = [f"The customer is rated {view['risk_band']} risk on this model."]
    if raising:
        sentences.append("Raising it: " + "; ".join(raising) + ".")
    if lowering:
        sentences.append("Lowering it: " + "; ".join(lowering) + ".")
    if payload.steps:
        sentences.append("A one-month change that would help: " + "; ".join(payload.steps) + ".")
    return " ".join(sentences)


def _log(on_turn, row, conversation, question, reply, call, reason, rejected, guard) -> None:
    """Hand the turn to the caller's log. A failing log never costs the reply."""
    try:
        on_turn(
            TurnRecord(
                customer_id=row["customer_id"],
                turn_number=sum(1 for m in conversation if m["role"] == "user"),
                user_message=question,
                reply=reply.reply,
                source=reply.source,
                fallback_reason=reason if reply.source == "template" else None,
                rejected_reply=rejected,
                llm_model=call.model,
                prompt_version=PROMPT_VERSION,
                scoring_model_version=artifacts.scoring_config()["model_version"],
                latency_ms=call.latency_ms,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                grounding_passed=guard.grounding_passed if guard else None,
                claim_passed=guard.claim_passed if guard else None,
            )
        )
    except Exception:
        logger.exception("officer assistant: could not record the chat turn")
