"""Smoke checks for the cardholder chat.

The contract: the assistant always answers, the model only restates the fact sheet,
and a model failure, an invented number or an unsafe claim becomes a template
answer. No LLM or network is needed; conftest stubs the model and each test replaces it.

Run from the backend/ directory:

    python -m pytest tests/test_assistant.py
"""
import json

import uuid

import pytest
import requests

from app.ai import assistant, coaching, llm_client, local_llm
from app.ml import worklist
from tests.test_coaching import STRESSED


@pytest.fixture
def row():
    rows = worklist.all_rows()
    if not rows:
        pytest.skip("no queue artifact; run: python -m app.ml.train")
    # The API adds the customer's UUID to a queue row; there is no database here.
    return {**next(r for r in rows if r["risk_band"] == "High"), "customer_id": str(uuid.uuid4())}


def _ask(question):
    return [{"role": "user", "content": question}]


def _llm(text=None, failure=None):
    return llm_client.LlmCall(
        text=text, model="test-model", latency_ms=12, failure=failure,
        prompt_tokens=100 if text else None, completion_tokens=20 if text else None,
    )


def _sheet(row):
    view = worklist.cardholder_view(row, record=STRESSED)
    payload = coaching.build_payload(view["risk_band"], view["reasons"], STRESSED)
    return assistant.fact_sheet(view, row["history"], payload)


def test_empty_conversation_without_the_model_opens_with_the_coaching_template(row):
    reply = assistant.answer(row, [], record=STRESSED)

    view = worklist.cardholder_view(row, record=STRESSED)
    payload = coaching.build_payload(view["risk_band"], view["reasons"], STRESSED)
    assert (reply.source, reply.reply) == ("template", coaching.render(payload))
    assert reply.note


def test_answers_without_the_model_and_every_number_is_grounded(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(failure="llm_unreachable"))

    for question in ("Why am I high risk?", "What can I do?", "What is my balance?",
                     "What is my probability?", "Hello"):
        reply = assistant.answer(row, _ask(question), record=STRESSED)
        assert reply.source == "template"
        assert reply.reply.strip()
        assert reply.note
        assert coaching.numbers_in(reply.reply) <= coaching.numbers_in(json.dumps(_sheet(row)))


def test_fact_sheet_carries_counterfactual_reasons_and_history_but_no_probability(row):
    sheet = _sheet(row)

    assert sheet["one_month_counterfactual"]["target_band"] == "Medium"
    assert sheet["factors_raising_risk"]
    assert len(sheet["recent_months"]) == len(row["history"])
    assert str(row["probability"]) not in json.dumps(sheet)
    assert "probability" not in json.dumps(sheet)


def test_model_reply_with_an_invented_number_is_replaced(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm("Pay 987,654 and you will be Low risk."))

    reply = assistant.answer(row, _ask("What can I do?"), record=STRESSED)

    assert reply.source == "template"
    assert "987" not in reply.reply


def test_grounded_model_reply_is_used(row, monkeypatch):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return _llm(f"Your account is rated {row['risk_band']} risk on this model.")

    monkeypatch.setattr(llm_client, "chat", fake_chat)

    reply = assistant.answer(row, _ask("Why?"), record=STRESSED)

    assert reply.source == "model"
    assert reply.note == ""
    assert seen["messages"][0]["role"] == "system"
    assert "one_month_counterfactual" in seen["messages"][0]["content"]


def test_a_number_only_the_cardholder_typed_is_rejected(row, monkeypatch):
    # Echoing a typed amount used to pass. It must not: "can I pay 4,321?" would let the
    # model answer "paying 4,321 moves you to Medium" and still be called grounded.
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm("I cannot estimate what paying 4,321 would do."))
    turns = []

    reply = assistant.answer(row, _ask("What if I pay 4321?"), record=STRESSED, on_turn=turns.append)

    assert reply.source == "template"
    assert (turns[0].fallback_reason, turns[0].grounding_passed) == ("invented_numbers", False)
    assert turns[0].rejected_reply == "I cannot estimate what paying 4,321 would do."


def test_local_llm_returns_none_when_ollama_is_down(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", refuse)
    assert local_llm.chat([{"role": "user", "content": "hi"}]).text is None


def test_local_llm_returns_none_on_timeout(monkeypatch):
    def slow(*args, **kwargs):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(requests, "post", slow)
    assert local_llm.chat([{"role": "user", "content": "hi"}]).text is None


def test_local_llm_records_failure_reason_and_latency(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", refuse)
    call = local_llm.chat([{"role": "user", "content": "hi"}])

    assert call.failure == "llm_unreachable"
    assert call.latency_ms >= 0
    assert call.prompt_tokens is None


def test_local_llm_reads_token_counts_from_ollama(monkeypatch):
    class Reply:
        status_code = 200
        ok = True

        def json(self):
            return {"message": {"content": " Hello. "}, "done_reason": "stop",
                    "prompt_eval_count": 321, "eval_count": 45}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Reply())
    call = local_llm.chat([{"role": "user", "content": "hi"}])

    assert (call.text, call.failure) == ("Hello.", None)
    assert (call.prompt_tokens, call.completion_tokens) == (321, 45)


def test_on_turn_receives_a_model_turn(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(f"Rated {row['risk_band']} risk."))
    turns = []

    assistant.answer(row, _ask("Why?"), record=STRESSED, on_turn=turns.append)

    (turn,) = turns
    assert (turn.source, turn.fallback_reason, turn.rejected_reply) == ("model", None, None)
    assert (turn.customer_id, turn.turn_number, turn.user_message) == (row["customer_id"], 1, "Why?")
    assert (turn.llm_model, turn.prompt_version) == ("test-model", assistant.PROMPT_VERSION)
    assert (turn.prompt_tokens, turn.completion_tokens) == (100, 20)
    assert (turn.grounding_passed, turn.claim_passed) == (True, True)


def test_on_turn_records_why_the_template_was_used(row, monkeypatch):
    turns = []

    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(failure="llm_timeout"))
    assistant.answer(row, _ask("Why?"), record=STRESSED, on_turn=turns.append)

    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm("Pay 987,654 now."))
    assistant.answer(row, _ask("What can I do?"), record=STRESSED, on_turn=turns.append)

    assert [(t.source, t.fallback_reason) for t in turns] == [
        ("template", "llm_timeout"), ("template", "invented_numbers"),
    ]
    assert turns[0].rejected_reply is None
    assert turns[1].rejected_reply == "Pay 987,654 now."


def test_a_failing_log_does_not_cost_the_reply(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(f"Rated {row['risk_band']} risk."))

    def broken(turn):
        raise RuntimeError("database down")

    reply = assistant.answer(row, _ask("Why?"), record=STRESSED, on_turn=broken)

    assert reply.source == "model"


# --- the opening interpretation ---------------------------------------------


def test_opening_is_asked_of_the_model_with_the_sheet_and_not_logged_as_a_question(row, monkeypatch):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return _llm(f"Your account is rated {row['risk_band']} risk on this model.")

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    turns = []

    reply = assistant.answer(row, [], record=STRESSED, on_turn=turns.append)

    assert reply.source == "model"
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert "repayment_summary" in seen["messages"][0]["content"]
    (turn,) = turns
    assert (turn.turn_number, turn.user_message) == (0, "")


def test_opening_is_cached_so_a_second_call_does_not_reach_the_llm(row, monkeypatch):
    calls = []

    def fake_chat(messages):
        calls.append(messages)
        return _llm(f"Your account is rated {row['risk_band']} risk on this model.")

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    turns = []

    first = assistant.answer(row, [], record=STRESSED, on_turn=turns.append)
    second = assistant.answer(row, [], record=STRESSED, on_turn=turns.append)

    assert second == first
    assert len(calls) == 1
    # One generation, one row: a cache hit is not a new turn.
    assert [t.turn_number for t in turns] == [0]


def test_a_template_opening_is_not_cached_so_a_recovered_llm_is_used(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(failure="llm_timeout"))
    assert assistant.answer(row, [], record=STRESSED).source == "template"

    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(f"Rated {row['risk_band']} risk."))
    assert assistant.answer(row, [], record=STRESSED).source == "model"


def test_opening_goes_through_the_guards(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm("This will definitely fix everything."))
    turns = []

    reply = assistant.answer(row, [], record=STRESSED, on_turn=turns.append)

    assert reply.source == "template"
    assert turns[0].fallback_reason == "unsafe_claim"


# --- model failures become the right fallback_reason ------------------------


@pytest.mark.parametrize("failure", ["llm_timeout", "llm_truncated", "llm_unreachable", "llm_auth"])
def test_each_llm_failure_is_a_template_reply_with_its_reason(row, monkeypatch, failure):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm(failure=failure))
    turns = []

    reply = assistant.answer(row, _ask("Why?"), record=STRESSED, on_turn=turns.append)

    assert reply.source == "template"
    assert (turns[0].fallback_reason, turns[0].grounding_passed, turns[0].claim_passed) == (
        failure, None, None,
    )


# --- number grounding --------------------------------------------------------


def test_grounding_accepts_numbers_in_the_sheet_and_rejects_others(row):
    sheet = _sheet(row)
    stated = f"{row['credit_limit']:,.0f}"

    assert assistant.validate_number_grounding(f"Your limit is {stated}.", sheet)
    assert not assistant.validate_number_grounding("Your limit is 987,654.", sheet)


def test_fact_sheet_has_a_repayment_summary_and_the_window_but_no_global_digits(row):
    sheet = _sheet(row)

    assert sheet["billing_window_months"] == 6
    assert "billing cycles" in sheet["repayment_summary"]
    # "6 months" is allowed because the sheet says it; "7" is not.
    assert assistant.validate_number_grounding("Over the last 6 months.", sheet)
    assert not assistant.validate_number_grounding("Over the last 7777 months.", sheet)


def test_repayment_summary_counts_behind_cycles_and_names_the_latest_month():
    def month(position, status):
        return {"month": position, "payment_status": status}

    history = [month(1, 2), month(2, 2), month(3, 1), month(4, 0), month(5, -1), month(6, -2)]

    assert assistant.repayment_summary(history) == (
        "Behind on payments in 3 of the last 6 billing cycles; "
        "the most recent month was 2 months overdue."
    )
    assert "most recent month was 1 month overdue" in assistant.repayment_summary([month(1, 1)])
    assert "not overdue" in assistant.repayment_summary([month(1, 0), month(2, 3)])
    assert "Not behind" in assistant.repayment_summary([month(1, 0), month(2, -1)])
    assert "No repayment history" in assistant.repayment_summary([])


# --- claim-level safety ------------------------------------------------------

UNREACHABLE_PAYLOAD = coaching.CoachingPayload(
    band="High", reasons=("Last month: 2 months in arrears",), target_band="Medium",
    outcome=coaching.UNREACHABLE, steps=(),
)
REACHABLE_PAYLOAD = coaching.CoachingPayload(
    band="High", reasons=("Last month: 2 months in arrears",), target_band="Medium",
    outcome=coaching.REACHABLE,
    steps=("Put 5,000 into the card this month, instead of nothing",),
)


@pytest.mark.parametrize("reply", [
    "You should pay something this month.",
    "Try to make a payment before the due date.",
    "Putting some money into the card would help.",
    "Bring the balance down as far as you can.",
    "I suggest you pay more.",
])
def test_prescriptive_advice_is_rejected_when_nothing_is_feasible(reply):
    assert not assistant.validate_claim_level_safety(reply, UNREACHABLE_PAYLOAD)


def test_the_payloads_own_unreachable_sentence_is_accepted_verbatim():
    # It mentions this month's bill, payment and repayment status, which is correct.
    for sentence in (
        coaching.action_sentence(UNREACHABLE_PAYLOAD),
        assistant.OUTCOME_WORDS[coaching.UNREACHABLE].format(target="Medium"),
    ):
        assert assistant.validate_claim_level_safety(sentence, UNREACHABLE_PAYLOAD)

    framed = "Your account is rated High risk. " + coaching.action_sentence(UNREACHABLE_PAYLOAD)
    assert assistant.validate_claim_level_safety(framed, UNREACHABLE_PAYLOAD)


def test_advice_is_rejected_for_every_outcome_that_is_not_reachable():
    for outcome in (coaching.LOWEST_BAND, coaching.UNCHECKED):
        payload = coaching.CoachingPayload("Low", (), None, outcome, ())
        assert not assistant.validate_claim_level_safety("You could pay more this month.", payload)


@pytest.mark.parametrize("payload", [UNREACHABLE_PAYLOAD, REACHABLE_PAYLOAD])
@pytest.mark.parametrize("reply", [
    "This will definitely lower your risk.",
    "I guarantee a better rating.",
    "The bank will approve your next request.",
    "You will avoid default.",
    "You will be Low risk next month.",
])
def test_outcome_promises_are_rejected_whatever_the_outcome(payload, reply):
    assert not assistant.validate_claim_level_safety(reply, payload)


def test_saying_there_is_no_guarantee_is_fine():
    reply = "This is a model's estimate and not a guarantee; I cannot promise any outcome."

    assert assistant.validate_claim_level_safety(reply, REACHABLE_PAYLOAD)


def test_a_reachable_reply_may_restate_a_listed_step():
    reply = "Putting 5,000 into the card this month would move the account to Medium risk on this model."

    assert assistant.validate_claim_level_safety(reply, REACHABLE_PAYLOAD)


def test_a_reachable_reply_may_not_attach_an_action_to_an_amount_outside_the_steps():
    assert not assistant.validate_claim_level_safety("You could pay 7,777 this month.", REACHABLE_PAYLOAD)
    # An amount that is in the sheet (the limit, say) is grounded but still not a step.
    assert not assistant.validate_claim_level_safety("Pay 200,000 to the card.", REACHABLE_PAYLOAD)


def test_other_payment_strategies_are_rejected():
    for reply in ("A payment plan could help.", "Just cover the minimum payment.", "Consider a balance transfer."):
        assert not assistant.validate_claim_level_safety(reply, REACHABLE_PAYLOAD)


def test_an_unsafe_claim_becomes_a_template_and_both_checks_are_logged(row, monkeypatch):
    monkeypatch.setattr(assistant, "build_payload", lambda *args, **kwargs: UNREACHABLE_PAYLOAD)
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm("Please pay something this month."))
    turns = []

    reply = assistant.answer(row, _ask("What can I do?"), record=STRESSED, on_turn=turns.append)

    assert reply.source == "template"
    (turn,) = turns
    assert (turn.fallback_reason, turn.grounding_passed, turn.claim_passed) == (
        "unsafe_claim", True, False,
    )
    assert turn.rejected_reply == "Please pay something this month."


def test_grounding_failure_wins_the_reason_but_the_claim_check_still_runs(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda messages: _llm("I guarantee you should pay 987,654."))
    turns = []

    assistant.answer(row, _ask("What can I do?"), record=STRESSED, on_turn=turns.append)

    assert (turns[0].fallback_reason, turns[0].grounding_passed, turns[0].claim_passed) == (
        "invented_numbers", False, False,
    )


def test_the_system_prompt_is_loana_v2_1():
    assert assistant.PROMPT_VERSION == "v2.1-loana"
    assert assistant.SYSTEM_PROMPT.startswith("You are Loana AI")
    assert "{facts}" in assistant.SYSTEM_PROMPT


def test_describing_past_payments_is_not_advice_but_telling_them_to_increase_them_is():
    # Seen in a real qwen opening: the first is a fact, the second is advice.
    described = "You've been making payments that cover only a small portion of your bills."
    advised = "Focusing on increasing your payments could help."

    assert assistant.validate_claim_level_safety(described, UNREACHABLE_PAYLOAD)
    assert not assistant.validate_claim_level_safety(advised, UNREACHABLE_PAYLOAD)
    assert not assistant.validate_claim_level_safety("Make a payment before the due date.", UNREACHABLE_PAYLOAD)


def test_an_obligation_with_any_verb_is_rejected_when_nothing_is_feasible():
    # Seen in a real qwen opening for an unreachable account.
    reply = "You need to improve your repayment status consistently over time."

    assert not assistant.validate_claim_level_safety(reply, UNREACHABLE_PAYLOAD)
    assert assistant.validate_claim_level_safety("You can ask me why the account is rated this way.", UNREACHABLE_PAYLOAD)
