"""Loana AI: chat with a cardholder about their own account (LLM in llm_client.py).

The model re-words facts, it never produces them. Each turn it is given a fact sheet
built here from what the system already computed for this account: the risk band, the
reasons with their direction, the one-month counterfactual, and the account's own
figures and six months of history. Data leaves the machine when OPENAI_API_KEY is set:
the sheet carries the band, reasons, account figures and history, never the
probability or any protected attribute.

Every model reply passes two checks, in order: number grounding (a number that is not
in the sheet is invented) and claim-level safety (no promises, and no advice the
counterfactual search did not produce). A failed check, or any model failure, is
replaced by a template answer from the same sheet, so the chat always answers.

The sheet is built from `worklist.cardholder_view`, the reduced view, so the
probability cannot reach it: the cardholder is told a band, never a number.
"""
from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from app.ai import llm_client
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
from app.ml.explainability import MONTHS_OBSERVED

logger = logging.getLogger(__name__)

# Recorded with every reply. Bump it whenever SYSTEM_PROMPT changes, so logged
# answers can be compared before and after.
PROMPT_VERSION = "v2.1-loana"

# Turns sent to the model. Older ones add prompt time and rarely matter.
MAX_HISTORY = 10

TEMPLATE_NOTE = (
    "Standard answer: the assistant model is not running, or its reply could not be "
    "checked against your account."
)

SYSTEM_PROMPT = """You are Loana AI, the financial health copilot in CrediTrait, a credit card prototype. You talk to one cardholder about their own account.
Everything you know is in FACTS below. It comes from a risk model that estimates whether next month's payment will be missed, and from a search for the smallest change this month that would improve the rating.

Rules:
1. Tone: calm, objective, supportive and constructive. Never punitive, judgmental or alarming.
2. Use only the FACTS. Quote amounts exactly as written there. Never invent or estimate a number, percentage, amount, date or scenario.
3. Actions: only the steps listed under one_month_counterfactual.steps. Never suggest any other payment strategy. If the steps list is empty, say plainly and gently that no single-month action clears this risk band, and do not suggest anything to do this month.
4. Never state, estimate or hint at a probability, percentage chance or score. The rating is a band. If asked for a number, say the service describes risk as a band and point to the factors.
5. Do not give financial advice, promise an outcome, or say what any lender will decide. The rating is a model's estimate on a prototype, not a decision.
6. At most four short sentences.

FACTS:
{facts}"""

# Sent as the user turn when the page opens, so the model has something to answer.
# It is never logged as the cardholder's message.
OPENING_REQUEST = (
    "Give me a plain-language summary of my account in three or four sentences: my "
    "risk band, my repayment summary, and what one month can or cannot change."
)

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

# --- claim-level safety ----------------------------------------------------
# Kept as constants so they can be reviewed and tested. A false rejection only
# costs a template answer, so the lists lean towards rejecting.

# "this is not a guarantee" and "I cannot promise" are the right things to say.
NEGATED_PROMISE = re.compile(
    r"\b(?:not|no|never|cannot|can't|can not|isn't|doesn't|does not|don't|do not|won't|without)\b"
    r"(?:\s+\w+){0,3}?\s+(?:guarantee|guarantees|guaranteed|promise|promises|promised)\b",
    re.IGNORECASE,
)

# Rejected whatever the outcome: promises about the result, and lender predictions.
OUTCOME_PROMISE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bguarantee[sd]?\b",
        r"\bpromis(?:e|es|ed|ing)\b",
        r"\b(?:will|would|'ll) (?:definitely|certainly|surely|absolutely|undoubtedly)\b",
        r"\bdefinitely (?:will|would)\b",
        r"\bwill be (?:approved|declined|denied|accepted|rejected|refused)\b",
        r"\b(?:bank|lender|issuer|creditor|provider)\s+(?:will|would|won't|is going to|is likely to|is unlikely to)\b",
        r"\byou(?: will|'ll|'re going to) (?:avoid|prevent|escape|not default|not miss|never miss|never default|be safe|be fine)\b",
        r"\b(?:will|would) (?:avoid|prevent|eliminate|remove) (?:the )?(?:risk|default|missed payments?)\b",
        r"\byou(?: will|'ll) (?:be|become|move to|reach|get to|qualify for|end up in)\b[^.!?]*\b(?:low|medium)\b",
        r"\b(?:is|are) (?:certain|sure|bound|guaranteed) to\b",
        r"\bno (?:risk|chance) of (?:default|missing|a missed)\b",
    )
)

# Rejected whatever the outcome: payment strategies the search did not produce.
OTHER_STRATEGY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bminimum payments?\b",
        r"\bpayment plans?\b",
        r"\bbalance transfers?\b",
        r"\b(?:take out|taking out|apply for|get|getting) (?:a |an )?(?:loan|credit|overdraft)\b",
        r"\bborrow",
        r"\bconsolidat",
        r"\brefinanc",
        r"\binstal?lments?\b",
        r"\b(?:direct debit|auto-?pay|standing order)\b",
    )
)

# Rejected unless the search found a feasible action: when there is nothing to do this
# month (unreachable, unchecked, already lowest) the reply must not invent something.
PRESCRIPTIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpay(?:ing)? (?:something|anything|more|extra|down|off|any|at least|your|the|towards?|in full|early|promptly|immediately|on time|before|what you can|a little)\b",
        # Only as an instruction: "you've been making payments" describes the past.
        r"(?:^|[.!?;] |\b(?:should|could|can|must|need to|try to|please|to|consider|let's|by|might want to|may want to) )(?:also |first |then )?mak(?:e|ing) (?:an? |any |some |the )?(?:extra|additional |larger |bigger |small |partial |full |minimum |early |on-time )?payments?\b",
        r"\bput(?:ting)?\b[^.!?]{0,40}\binto (?:the|your) (?:card|account)\b",
        r"\bbring(?:ing)? (?:the |your )?(?:card |account )?(?:balance|account|payments?)\b",
        r"\b(?:clear|settle|catch up|reduce|lower|cut|decrease|pay down) (?:on )?(?:the |your |some |any |all )?(?:overdue |outstanding |card )?(?:arrears|balance|bill|debt|payments?|spending)\b",
        r"\b(?:increas(?:e|ing)|rais(?:e|ing)) (?:the |your )?(?:payments?|repayments?|amount)\b",
        # Obligation words take any verb ("you need to improve ..."): the model must not
        # invent a to-do when the search found none.
        r"\byou (?:will )?(?:need to|ought to|must|have to|should|might want to|may want to)\b",
        r"\byou (?:could|can) (?:also |first |try to |consider )?(?:pay|make|put|bring|clear|settle|catch|reduce|lower|increase|contact|call|speak|talk|deposit|transfer)\b",
        r"\b(?:recommend|suggest|advise)\b",
        r"\btry to (?:pay|make|put|bring|clear|reduce|lower|settle)\b",
        r"\b(?:consider|think about|focus(?:ing)? on|work(?:ing)? on|aim(?:ing)? to|start(?:ing)? by) (?:paying|making|putting|bringing|clearing|reducing|lowering|increasing|raising|using)\b",
    )
)

# A recommended action with an amount, for the reachable case: the amount must be
# one a listed step carries. "paid" is left out, since it describes the past.
ACTION_WITH_AMOUNT = re.compile(
    r"\b(?:pay(?:ing)?|put(?:ting)?|bring(?:ing)?|rais(?:e|ing)|increas(?:e|ing)|reduc(?:e|ing)|"
    r"lower(?:ing)?|clear(?:ing)?|deposit(?:ing)?|transfer(?:ring)?|send(?:ing)?|settl(?:e|ing)|"
    r"add(?:ing)?|contribut(?:e|ing))\b[^.!?;:\d]{0,30}?(\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatReply:
    reply: str
    source: str  # "model" or "template"
    note: str
    disclaimer: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GuardResult:
    """How a model reply fared against each check. None on both: no text to check."""

    grounding_passed: bool | None
    claim_passed: bool | None

    @property
    def fallback_reason(self) -> str | None:
        """The first failing check, in the order they run."""
        if self.grounding_passed is False:
            return "invented_numbers"
        if self.claim_passed is False:
            return "unsafe_claim"
        return None


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
    grounding_passed: bool | None = None
    claim_passed: bool | None = None


@dataclass(frozen=True)
class _Outcome:
    """What one pass through the model and the guards produced."""

    reply: ChatReply
    call: llm_client.LlmCall
    fallback_reason: str | None
    rejected_reply: str | None
    guard: GuardResult


# Streamlit reruns the page on every click and the opening costs an LLM call, so a
# generated opening is kept. Bounded, like the payload memo in coaching.py.
OPENING_MEMO_SIZE = 256
_opening_memo: OrderedDict[tuple, ChatReply] = OrderedDict()


def answer(
    row: dict,
    messages: Sequence[Mapping[str, str]],
    stored_prediction=None,
    record: Mapping[str, float] | None = None,
    on_turn: Callable[[TurnRecord], None] | None = None,
) -> ChatReply:
    """The assistant's next turn for one account. Always returns a reply.

    An empty conversation asks for Loana's opening interpretation. `on_turn` receives
    each turn that reached the model, the opening (turn 0) included.
    """
    view = worklist.cardholder_view(row, stored_prediction=stored_prediction, record=record)
    payload = build_payload(view["risk_band"], view["reasons"], record)
    sheet = fact_sheet(view, row.get("history") or [], payload)

    if not messages:
        return _opening(row, payload, sheet, on_turn)

    conversation = [{"role": message["role"], "content": message["content"]} for message in messages]
    question = conversation[-1]["content"]
    outcome = _generate(
        sheet, payload, conversation[-MAX_HISTORY:],
        fallback=lambda: template_reply(question, view, payload),
    )
    turn_number = sum(1 for message in conversation if message["role"] == "user")
    _report(on_turn, row, turn_number, question, outcome)
    return outcome.reply


def _opening(
    row: dict,
    payload: CoachingPayload,
    sheet: dict,
    on_turn: Callable[[TurnRecord], None] | None,
) -> ChatReply:
    """Loana's first message: the account interpreted in plain language.

    Only a model-written opening is kept. A template opening means the LLM was down,
    and keeping it would hide a recovery until the entry was evicted.
    """
    key = (
        row["customer_id"],
        artifacts.scoring_config()["model_version"],
        PROMPT_VERSION,
        payload,
        # The payload does not carry balances or history, which the opening may quote.
        json.dumps(sheet, sort_keys=True),
    )
    if key in _opening_memo:
        _opening_memo.move_to_end(key)
        return _opening_memo[key]

    outcome = _generate(
        sheet, payload, [{"role": "user", "content": OPENING_REQUEST}],
        fallback=lambda: render(payload),
    )
    _report(on_turn, row, 0, "", outcome)

    if outcome.reply.source == "model":
        _opening_memo[key] = outcome.reply
        if len(_opening_memo) > OPENING_MEMO_SIZE:
            _opening_memo.popitem(last=False)
    return outcome.reply


def _generate(
    sheet: dict,
    payload: CoachingPayload,
    conversation: Sequence[Mapping[str, str]],
    fallback: Callable[[], str],
) -> _Outcome:
    """Ask the model, check the reply, and fall back to `fallback()` on any failure."""
    call = llm_client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT.format(facts=json.dumps(sheet, indent=1))}]
        + list(conversation)
    )

    guard = GuardResult(None, None)
    if call.text is not None:
        guard = check_reply(call.text, sheet, payload)
        if guard.fallback_reason is None:
            return _Outcome(ChatReply(call.text, "model", "", DISCLAIMER), call, None, None, guard)
        logger.warning("assistant: reply failed %s; using template", guard.fallback_reason)
        return _Outcome(
            ChatReply(fallback(), "template", TEMPLATE_NOTE, DISCLAIMER),
            call, guard.fallback_reason, call.text, guard,
        )

    return _Outcome(
        ChatReply(fallback(), "template", TEMPLATE_NOTE, DISCLAIMER), call, call.failure, None, guard
    )


def _report(
    on_turn: Callable[[TurnRecord], None] | None,
    row: dict,
    turn_number: int,
    user_message: str,
    outcome: _Outcome,
) -> None:
    """Hand the finished turn to the caller's log. A failing log never costs the reply."""
    if on_turn is None:
        return
    try:
        on_turn(
            TurnRecord(
                customer_id=row["customer_id"],
                turn_number=turn_number,
                user_message=user_message,
                reply=outcome.reply.reply,
                source=outcome.reply.source,
                fallback_reason=outcome.fallback_reason,
                rejected_reply=outcome.rejected_reply,
                llm_model=outcome.call.model,
                prompt_version=PROMPT_VERSION,
                scoring_model_version=artifacts.scoring_config()["model_version"],
                latency_ms=outcome.call.latency_ms,
                prompt_tokens=outcome.call.prompt_tokens,
                completion_tokens=outcome.call.completion_tokens,
                grounding_passed=outcome.guard.grounding_passed,
                claim_passed=outcome.guard.claim_passed,
            )
        )
    except Exception:
        logger.exception("assistant: could not record the chat turn")


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
        # The model may say "6 months" because the sheet does; no digit is allowed
        # globally, so the grounding check stays strict.
        "billing_window_months": MONTHS_OBSERVED,
        "repayment_summary": repayment_summary(history),
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


def repayment_summary(history: Sequence[Mapping]) -> str:
    """One sentence on how often the account was behind, counted in code, not by the model.

    `history` is most recent first; a status above zero is months in arrears.
    """
    if not history:
        return "No repayment history is available for this account."

    cycles = len(history)
    behind = sum(1 for entry in history if entry["payment_status"] > 0)
    if behind == 0:
        return f"Not behind on payments in any of the last {cycles} billing cycles."

    latest = history[0]["payment_status"]
    if latest > 0:
        recent = f"the most recent month was {latest} month{'s' if latest != 1 else ''} overdue"
    else:
        recent = "the most recent month was not overdue"
    return f"Behind on payments in {behind} of the last {cycles} billing cycles; {recent}."


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


# --- guardrails ------------------------------------------------------------


def check_reply(reply: str, sheet: dict, payload: CoachingPayload) -> GuardResult:
    """Run both checks. Both always run, so the log records each outcome separately."""
    return GuardResult(
        grounding_passed=validate_number_grounding(reply, sheet),
        claim_passed=validate_claim_level_safety(reply, payload),
    )


def _invented_numbers(reply: str, sheet: dict) -> set[str]:
    """Numbers in the reply that are not in the sheet.

    What the cardholder typed does not count: otherwise "can I pay 5,000?" would let
    the model answer "paying 5,000 moves you to Medium" and still pass.
    """
    return numbers_in(reply) - numbers_in(json.dumps(sheet))


def validate_number_grounding(reply: str, sheet: dict) -> bool:
    """True when every number in the reply is one the sheet states."""
    invented = _invented_numbers(reply, sheet)
    if invented:
        logger.warning("assistant: reply used numbers %s not in the facts", sorted(invented))
    return not invented


def validate_claim_level_safety(reply: str, payload: CoachingPayload) -> bool:
    """True when the reply makes no promise and recommends nothing the search did not find."""
    text = _normalise(reply)
    text = NEGATED_PROMISE.sub(" ", text)

    if any(pattern.search(text) for pattern in OUTCOME_PROMISE_PATTERNS):
        return False
    if any(pattern.search(text) for pattern in OTHER_STRATEGY_PATTERNS):
        return False

    if payload.outcome != REACHABLE:
        # The correct "nothing to do" message itself mentions this month's bill,
        # payment and repayment status, so it is taken out before scanning.
        text = _without_own_wording(text, payload)
        return not any(pattern.search(text) for pattern in PRESCRIPTIVE_PATTERNS)

    allowed = numbers_in(" ".join(payload.steps))
    return all(
        match.group(1).replace(",", "") in allowed for match in ACTION_WITH_AMOUNT.finditer(text)
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip()


def _without_own_wording(text: str, payload: CoachingPayload) -> str:
    """`text` with the payload's own outcome sentences removed."""
    own = (
        action_sentence(payload),
        OUTCOME_WORDS[payload.outcome].format(target=payload.target_band),
    )
    for sentence in own:
        text = re.sub(re.escape(_normalise(sentence).rstrip(".")), " ", text, flags=re.IGNORECASE)
    return text


# --- templates -------------------------------------------------------------


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
