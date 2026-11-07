"""Checks for the chat log: a turn is stored, a failed write is contained, the summary adds up.

Each test runs in a transaction that is rolled back. Skipped when no database is reachable.
"""
import pytest
from sqlalchemy import text

from app.ai import llm_client, local_llm
from app.db import chat_log
from app.db.models import ChatTurn
from tests.conftest import bearer
from tests.test_api import (  # noqa: F401  (fixture imports)
    api_session,
    client,
    quietest_customer_id,
    riskiest_customer_id,
)
from tests.test_api_roles import any_customer_id, cardholder_identity  # noqa: F401


def _turn(customer_id, **overrides) -> dict:
    turn = dict(
        customer_id=customer_id, turn_number=1, user_message="Why?", reply="Because.",
        source="model", fallback_reason=None, rejected_reply=None,
        llm_model="test-model", prompt_version="v1", scoring_model_version="m1",
        latency_ms=1000, prompt_tokens=100, completion_tokens=20,
    )
    return {**turn, **overrides}


@pytest.fixture
def session(api_session):  # noqa: F811
    # Created inside the test's transaction, so it is rolled back with everything else.
    ChatTurn.__table__.create(api_session.connection(), checkfirst=True)
    return api_session


@pytest.fixture
def turn(session, riskiest_customer_id):  # noqa: F811
    """Builds a turn about a real customer: the log's customer_id is a foreign key."""
    return lambda **overrides: _turn(riskiest_customer_id, **overrides)


def test_record_stores_the_turn(session, turn):
    chat_log.record(session, turn())

    stored = session.query(ChatTurn).one()
    assert (stored.source, stored.latency_ms) == ("model", 1000)


def test_a_bad_row_is_contained_and_the_session_stays_usable(session, turn):
    # A template turn with no reason breaks the table's own rule.
    chat_log.record(session, turn(source="template", fallback_reason=None))

    assert session.query(ChatTurn).count() == 0
    assert session.execute(text("select 1")).scalar() == 1


def test_summary_is_empty_without_turns(session):
    assert chat_log.summary(session)["turns"] == 0


def test_summary_counts_fallbacks_and_times_only_model_turns(session, turn):
    chat_log.record(session, turn(latency_ms=1000))
    chat_log.record(session, turn(latency_ms=3000, prompt_tokens=200, completion_tokens=40))
    chat_log.record(session, turn(source="template", fallback_reason="llm_timeout", latency_ms=60000))
    chat_log.record(
        session, turn(source="template", fallback_reason="invented_numbers", rejected_reply="x")
    )

    result = chat_log.summary(session)

    assert (result["turns"], result["model_turns"], result["template_rate"]) == (4, 2, 0.5)
    assert result["fallback_reasons"] == {"llm_timeout": 1, "invented_numbers": 1}
    assert result["model_latency_ms_p50"] == 2000
    assert result["avg_prompt_tokens"] == 150


def test_a_chat_request_is_logged_and_shows_in_the_summary(
    session, client, cardholder_identity, any_customer_id, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(
        llm_client, "chat",
        lambda messages: local_llm.LlmCall(None, "test-model", 5, failure="llm_unreachable"),
    )

    response = client.post(
        f"/cardholders/{any_customer_id}/chat",
        json={"messages": [{"role": "user", "content": "Why am I at risk?"}]},
        headers=bearer(cardholder_identity),
    )

    assert response.status_code == 200
    stored = session.query(ChatTurn).one()
    assert (str(stored.customer_id), stored.source, stored.fallback_reason) == (
        any_customer_id, "template", "llm_unreachable",
    )
    assert client.get("/monitoring/assistant").json()["fallback_reasons"] == {"llm_unreachable": 1}


def test_both_guard_verdicts_are_stored(session, turn):
    chat_log.record(session, turn(grounding_passed=True, claim_passed=False))
    chat_log.record(session, turn(source="template", fallback_reason="llm_timeout"))

    passed, unchecked = session.query(ChatTurn).order_by(ChatTurn.source).all()
    assert (passed.grounding_passed, passed.claim_passed) == (True, False)
    # No LLM text, so nothing was checked: null, not "passed".
    assert (unchecked.grounding_passed, unchecked.claim_passed) == (None, None)


def test_the_opening_is_stored_as_turn_zero_with_an_empty_message(session, turn):
    chat_log.record(session, turn(turn_number=0, user_message=""))

    stored = session.query(ChatTurn).one()
    assert (stored.turn_number, stored.user_message) == (0, "")


def test_summary_reports_guard_rejection_rates_and_a_breakdown_per_provider(session, turn):
    chat_log.record(session, turn(llm_model="gpt-4o-mini", grounding_passed=True, claim_passed=True))
    chat_log.record(
        session,
        turn(llm_model="gpt-4o-mini", source="template", fallback_reason="invented_numbers",
              rejected_reply="x", grounding_passed=False, claim_passed=True),
    )
    chat_log.record(
        session,
        turn(llm_model="gpt-4o-mini", source="template", fallback_reason="unsafe_claim",
              rejected_reply="y", grounding_passed=True, claim_passed=False),
    )
    # No LLM text: not part of the denominator.
    chat_log.record(
        session, turn(llm_model="qwen2.5:3b", source="template", fallback_reason="llm_unreachable"),
    )

    result = chat_log.summary(session)

    assert result["checked_turns"] == 3
    assert result["grounding_rejection_rate"] == round(1 / 3, 4)
    assert result["claim_rejection_rate"] == round(1 / 3, 4)
    assert result["providers"]["gpt-4o-mini"]["turns"] == 3
    assert result["providers"]["gpt-4o-mini"]["template_rate"] == round(2 / 3, 4)
    assert result["providers"]["qwen2.5:3b"] == {
        "turns": 1, "template_rate": 1.0, "model_latency_ms_p50": None,
    }


def test_summary_rates_are_none_when_nothing_was_checked(session, turn):
    chat_log.record(session, turn(source="template", fallback_reason="llm_timeout"))

    result = chat_log.summary(session)

    assert (result["grounding_rejection_rate"], result["claim_rejection_rate"]) == (None, None)


def test_a_logged_chat_request_stores_the_guard_verdicts_and_the_opening_as_turn_zero(
    session, client, cardholder_identity, any_customer_id, monkeypatch  # noqa: F811
):
    reply = lambda messages: local_llm.LlmCall("Rated Medium risk.", "gpt-4o-mini", 5)  # noqa: E731
    monkeypatch.setattr(llm_client, "chat", reply)
    headers = bearer(cardholder_identity)

    client.post(f"/cardholders/{any_customer_id}/chat", json={"messages": []}, headers=headers)
    client.post(
        f"/cardholders/{any_customer_id}/chat",
        json={"messages": [{"role": "user", "content": "Why?"}]},
        headers=headers,
    )

    opening, question = session.query(ChatTurn).order_by(ChatTurn.turn_number).all()
    assert (opening.turn_number, opening.user_message) == (0, "")
    assert question.turn_number == 1
    assert (question.grounding_passed, question.claim_passed, question.llm_model) == (
        True, True, "gpt-4o-mini",
    )
