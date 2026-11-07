"""Loana AI triage: turns an officer's request into a cohort filter.

The LLM's only job is to fill a `TriageFilter`. Its prose is never shown: the
reply is written from the validated filter (`worklist.describe_cohort`), so it
cannot claim a count, a customer or a risk the data does not back. It never
scores anyone and never sees a probability; it sees no customer data at all.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from pydantic import ValidationError

from app.ai import llm_client
from app.ai.assistant import MAX_HISTORY
from app.ml import worklist
from app.schemas.worklist import TriageFilter

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1.1-triage"

SYSTEM_PROMPT = """You are Loana AI in CrediTrait, a credit card prototype. You help a loan officer choose which existing customers to look at.
You do not see any customer, score or risk. You only translate the officer's request into a filter.
You are a filter, not an adviser: anything else (explaining an account, a decision, an approval, a review request, policy) gets {"filter": null}.

Answer with one JSON object and nothing else, in this shape:
{"filter": {...}}
or, when the request is not about choosing customers:
{"filter": null}

Filter fields. Leave out any field the officer did not ask for:
- min_months_in_arrears: integer 0-6. "behind", "late", "in arrears" mean this.
- trajectory: "improving", "steady" or "deteriorating". "getting worse" is deteriorating.
- min_utilisation: number, 0.9 means 90% of the limit used.
- min_months_with_no_payment: integer 0-6, months where money was owed and nothing was paid.
- min_credit_limit, max_credit_limit: numbers.
- not_scored: true when the officer wants customers nobody has predicted yet.
- sort_by: one of "months_in_arrears", "utilisation_now", "current_balance", "months_with_no_payment".
- limit: integer 1-25, how many customers to show. Default 10.

Example: "10 customers at least 3 months behind and getting worse, not scored yet"
{"filter": {"min_months_in_arrears": 3, "trajectory": "deteriorating", "not_scored": true, "limit": 10}}"""

TEMPLATE_NOTE = (
    "Standard answer: the assistant model is not running, or it did not return a "
    "usable filter. The quick cohorts still work."
)

OFF_TOPIC = (
    "I can pull groups of customers by months in arrears, whether arrears are "
    "getting worse, utilisation, months without a payment, credit limit, or "
    "whether anyone has scored them yet."
)


@dataclass(frozen=True)
class TriageTurn:
    reply: str
    source: str
    note: str
    filter: TriageFilter | None
    failure: str | None


def interpret(messages: Sequence[Mapping[str, str]]) -> TriageTurn:
    """A filter for the officer's latest request. Always returns a turn.

    `messages` is the conversation so far and must end with the officer's request.
    """
    conversation = [{"role": m["role"], "content": m["content"]} for m in messages]
    call = llm_client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-MAX_HISTORY:]
    )

    failure = call.failure
    if call.text is not None:
        parsed, failure = parse(call.text)
        if failure is None:
            if parsed is None:
                return TriageTurn(OFF_TOPIC, "model", "", None, None)
            criteria = parsed.model_dump()
            return TriageTurn(worklist.describe_cohort(criteria), "model", "", parsed, None)

    logger.warning("triage assistant: no usable filter (%s); using template", failure)
    return TriageTurn(OFF_TOPIC, "template", TEMPLATE_NOTE, None, failure)


def parse(text: str) -> tuple[TriageFilter | None, str | None]:
    """(filter, None) on success, (None, None) for a deliberate null, (None, reason) on failure."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "triage_not_json"
    try:
        body = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, "triage_not_json"
    if not isinstance(body, dict) or "filter" not in body:
        return None, "triage_wrong_shape"
    if body["filter"] is None:
        return None, None
    try:
        return TriageFilter.model_validate(body["filter"]), None
    except ValidationError:
        return None, "triage_invalid_filter"
