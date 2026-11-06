"""Checks for the chat log: a turn is stored, a failed write is contained, the summary adds up.

Each test runs in a transaction that is rolled back. Skipped when no database is reachable.
"""
import pytest
from sqlalchemy import text

from app.ai import local_llm
from app.db import chat_log
from app.db.models import ChatTurn
from tests.conftest import bearer
from tests.test_api import api_session, client  # noqa: F401  (fixture import)
from tests.test_api_roles import any_customer_id, cardholder_identity  # noqa: F401


def _turn(**overrides) -> dict:
    turn = dict(
        customer_id="C1", turn_number=1, user_message="Why?", reply="Because.",
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


def test_record_stores_the_turn(session):
    chat_log.record(session, _turn())

    stored = session.query(ChatTurn).one()
    assert (stored.customer_id, stored.source, stored.latency_ms) == ("C1", "model", 1000)


def test_a_bad_row_is_contained_and_the_session_stays_usable(session):
    # A template turn with no reason breaks the table's own rule.
    chat_log.record(session, _turn(source="template", fallback_reason=None))

    assert session.query(ChatTurn).count() == 0
    assert session.execute(text("select 1")).scalar() == 1


def test_summary_is_empty_without_turns(session):
    assert chat_log.summary(session)["turns"] == 0


def test_summary_counts_fallbacks_and_times_only_model_turns(session):
    chat_log.record(session, _turn(latency_ms=1000))
    chat_log.record(session, _turn(latency_ms=3000, prompt_tokens=200, completion_tokens=40))
    chat_log.record(session, _turn(source="template", fallback_reason="llm_timeout", latency_ms=60000))
    chat_log.record(
        session, _turn(source="template", fallback_reason="invented_numbers", rejected_reply="x")
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
        local_llm, "chat",
        lambda messages: local_llm.LlmCall(None, "test-model", 5, failure="llm_unreachable"),
    )

    response = client.post(
        f"/cardholders/{any_customer_id}/chat",
        json={"messages": [{"role": "user", "content": "Why am I at risk?"}]},
        headers=bearer(cardholder_identity),
    )

    assert response.status_code == 200
    stored = session.query(ChatTurn).one()
    assert (stored.customer_id, stored.source, stored.fallback_reason) == (
        any_customer_id, "template", "llm_unreachable",
    )
    assert client.get("/monitoring/assistant").json()["fallback_reasons"] == {"llm_unreachable": 1}
