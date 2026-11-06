"""Chat with a cardholder about their own account, on a free local LLM (see local_llm.py).

The model re-words facts, it never produces them. Each turn it is given a fact sheet
built here from what the system already computed for this account: the risk band, the
reasons with their direction, the one-month counterfactual, and the account's own
figures and six months of history. A reply that brings in a number found neither in
the sheet nor in the cardholder's own messages is discarded, and so is any model
failure; a template answer from the same sheet takes its place, so the chat always
answers.

The sheet is built from `worklist.cardholder_view`, the reduced view, so the
probability cannot reach it: the cardholder is told a band, never a number.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from app.ai import local_llm
from app.ai.coaching import (
    DISCLAIMER,
    LOWEST_BAND,
    REACHABLE,
    UNCHECKED,
    UNREACHABLE,
    CoachingPayload,
    action_sentence,
    build_payload,
    numbers_in,
    render,
)
from app.ml import artifacts, worklist

logger = logging.getLogger(__name__)

# Recorded with every reply. Bump it whenever SYSTEM_PROMPT changes, so logged
# answers can be compared before and after.
PROMPT_VERSION = "v1"

# Turns sent to the model. Older ones add prompt time on CPU and rarely matter.
MAX_HISTORY = 10

TEMPLATE_NOTE = (
    "Standard answer: the assistant model is not running, or its reply could not be "
    "checked against your account."
)

SYSTEM_PROMPT = """You are the account assistant in a credit card prototype. You talk to one cardholder about their own account.
Everything you know is in FACTS below. It comes from a risk model that estimates whether next month's payment will be missed, and from a search for the smallest change this month that would improve the rating.

Rules:
- Answer in plain, calm English, in at most four sentences.
- Use only the FACTS. Quote amounts exactly as they are written there. Never invent a number, amount, date or scenario.
- If asked about something the FACTS do not cover, such as the effect of an amount that is not listed or a numeric probability, say you can only speak to what the model found, and point to the factors or the one-month steps.
- Do not give financial advice, promise an outcome, or say what any lender will decide.
- The rating is a model's estimate on a prototype, not a decision.

FACTS:
{facts}"""

OUTCOME_WORDS = {
    LOWEST_BAND: "The account is already in the lowest risk band; there is nothing to change.",
    REACHABLE: "Doing every step below this month would move the account to {target} risk on this model.",
    UNREACHABLE: "No single-month action reaches {target} risk: nothing this month's bill, payment or repayment status can change is enough on this model.",
    UNCHECKED: "The one-month changes could not be checked for this account.",
}

PROBABILITY_WORDS = ("probab", "percent", "%", "chance", "likel", "score", "number")
WHY_WORDS = ("why", "reason", "cause", "factor", "because", "driv", "raising", "affect")
ACCOUNT_WORDS = ("balance", "limit", "owe", "bill", "last payment", "arrears", "history")
ACTION_WORDS = ("what can", "how can", "how do", "improve", "help", "change", "lower", "better", "fix", "reduce", "pay", "should")


@dataclass(frozen=True)
class ChatReply:
    reply: str
    source: str  # "model" or "template"
    note: str
    disclaimer: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TurnRecord:
    """One answered turn, as the chat log stores it. Field names match `ChatTurn`."""

    customer_id: str
    turn_number: int
    user_message: str
    reply: str
    source: str
    fallback_reason: str | None
    rejected_reply: str | None
    llm_model: str
    prompt_version: str
    scoring_model_version: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


def answer(
    row: dict,
    messages: Sequence[Mapping[str, str]],
    stored_prediction=None,
    record: Mapping[str, float] | None = None,
    on_turn: Callable[[TurnRecord], None] | None = None,
) -> ChatReply:
    """The assistant's next turn for one account. Always returns a reply.

    `on_turn` receives each turn that reached the model; the greeting does not.
    """
    view = worklist.cardholder_view(row, stored_prediction=stored_prediction, record=record)
    if not messages:
        return ChatReply(greeting(view["risk_band"]), "template", "", DISCLAIMER)

    payload = build_payload(view["risk_band"], view["reasons"], record)
    sheet = fact_sheet(view, row.get("history") or [], payload)
    conversation = [{"role": message["role"], "content": message["content"]} for message in messages]

    call = local_llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT.format(facts=json.dumps(sheet, indent=1))}]
        + conversation[-MAX_HISTORY:]
    )

    fallback_reason = call.failure
    rejected = None
    if call.text is not None:
        invented = _invented_numbers(call.text, sheet, conversation)
        if not invented:
            result = ChatReply(call.text, "model", "", DISCLAIMER)
            _report(on_turn, row, conversation, result, call, None, None)
            return result
        logger.warning("assistant: reply used numbers %s not in the facts; using template", sorted(invented))
        fallback_reason, rejected = "invented_numbers", call.text

    question = conversation[-1]["content"]
    result = ChatReply(template_reply(question, view, payload), "template", TEMPLATE_NOTE, DISCLAIMER)
    _report(on_turn, row, conversation, result, call, fallback_reason, rejected)
    return result


def _report(
    on_turn: Callable[[TurnRecord], None] | None,
    row: dict,
    conversation: Sequence[Mapping[str, str]],
    result: ChatReply,
    call: local_llm.LlmCall,
    fallback_reason: str | None,
    rejected_reply: str | None,
) -> None:
    """Hand the finished turn to the caller's log. A failing log never costs the reply."""
    if on_turn is None:
        return
    try:
        on_turn(
            TurnRecord(
                customer_id=row["customer_id"],
                turn_number=sum(1 for message in conversation if message["role"] == "user"),
                user_message=conversation[-1]["content"],
                reply=result.reply,
                source=result.source,
                fallback_reason=fallback_reason,
                rejected_reply=rejected_reply,
                llm_model=call.model,
                prompt_version=PROMPT_VERSION,
                scoring_model_version=artifacts.scoring_config()["model_version"],
                latency_ms=call.latency_ms,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
            )
        )
    except Exception:
        logger.exception("assistant: could not record the chat turn")


def greeting(band: str) -> str:
    return (
        f"Hi. I can answer questions about your account using what the risk model found: "
        f"why it is rated {band} risk, what is working in your favour, and what one month "
        "could change. What would you like to know?"
    )


def fact_sheet(view: dict, history: Sequence[Mapping], payload: CoachingPayload) -> dict:
    """Everything the model may say, with amounts pre-formatted so it can quote them exactly."""
    return {
        "risk_band_for_next_month": view["risk_band"],
        "flagged_for_attention": view["needs_attention"],
        "account": {
            "credit_limit": f"{view['credit_limit']:,.0f}",
            "current_balance": f"{view['current_balance']:,.0f}",
            "last_payment": f"{view['last_payment']:,.0f}",
            "months_in_arrears_out_of_last_six": view["months_in_arrears"],
        },
        "factors_raising_risk": [
            reason["description"] for reason in view["reasons"] if reason["raises_risk"]
        ],
        "factors_lowering_risk": [
            reason["description"] for reason in view["reasons"] if not reason["raises_risk"]
        ],
        "one_month_counterfactual": {
            "target_band": payload.target_band,
            "result": OUTCOME_WORDS[payload.outcome].format(target=payload.target_band),
            "steps": list(payload.steps),
        },
        "recent_months": [_month(entry) for entry in history],
    }


def _month(entry: Mapping) -> dict:
    month = {
        "month": entry["label"],
        "repayment": entry["payment_status_reading"],
        "bill": f"{entry['bill_amount']:,.0f}",
        "paid": f"{entry['amount_paid']:,.0f}",
    }
    if entry.get("utilisation") is not None:
        month["share_of_limit_used"] = f"{entry['utilisation']:.0%}"
    return month


def _invented_numbers(reply: str, sheet: dict, conversation: Sequence[Mapping[str, str]]) -> set[str]:
    """Numbers in the reply found neither in the sheet nor in what the cardholder typed."""
    allowed = numbers_in(json.dumps(sheet))
    for message in conversation:
        if message["role"] == "user":
            allowed |= numbers_in(message["content"])
    return numbers_in(reply) - allowed


def template_reply(question: str, view: dict, payload: CoachingPayload) -> str:
    """A fixed answer from the same facts, picked by what the question is about."""
    asked = question.lower()
    band = view["risk_band"]

    if any(word in asked for word in PROBABILITY_WORDS):
        return (
            f"This service describes your risk as a band rather than a number: {band} risk "
            "for next month, on this model. The factors behind it and the one-month steps "
            "are what that band is based on."
        )

    if any(word in asked for word in WHY_WORDS):
        raising = [r["description"] for r in view["reasons"] if r["raises_risk"]]
        lowering = [r["description"] for r in view["reasons"] if not r["raises_risk"]]
        sentences = [f"Your account is rated {band} risk on this model."]
        if raising:
            sentences.append("Raising it: " + "; ".join(raising) + ".")
        if lowering:
            sentences.append("Working in your favour: " + "; ".join(lowering) + ".")
        return " ".join(sentences)

    if any(word in asked for word in ACCOUNT_WORDS):
        return (
            f"Your credit limit is {view['credit_limit']:,.0f} and your current balance is "
            f"{view['current_balance']:,.0f}. Your last payment was {view['last_payment']:,.0f}, "
            f"and the account was in arrears in {view['months_in_arrears']} of the last six months."
        )

    if any(word in asked for word in ACTION_WORDS):
        return action_sentence(payload)

    return (
        render(payload)
        + " You can ask why your account is rated this way, or what would help."
    )
