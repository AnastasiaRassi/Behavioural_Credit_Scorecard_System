"""Loana's triage: request -> validated filter -> cohort, and "predict selected".

The filter and parsing checks need only the queue artifact. The API checks need an
ingested database, since a cohort can name any held-out customer.
"""
import json

import pytest

from app.ai import llm_client, triage_assistant
from app.ml import worklist
from tests.test_api import (  # noqa: F401  (fixture imports)
    OFFICER_EMAIL,
    api_session,
    client,
    riskiest_customer_id,
)


@pytest.fixture
def rows():
    rows = worklist.all_rows()
    if not rows:
        pytest.skip("no queue artifact; run: python -m app.ml.train")
    return rows


def _llm(text=None, failure=None):
    return llm_client.LlmCall(
        text=text, model="test-model", latency_ms=5, failure=failure,
        prompt_tokens=None, completion_tokens=None,
    )


def _ask(text="customers 3 months behind and getting worse"):
    return [{"role": "user", "content": text}]


# --- the filter ---------------------------------------------------------------


def test_a_cohort_row_carries_no_score(rows):
    row = worklist.cohort_row(rows[0])
    assert not {"probability", "risk_band", "flagged", "reasons"} & row.keys()


def test_the_filter_selects_only_matching_customers(rows):
    chosen = worklist.select_cohort(
        rows, {"min_months_in_arrears": 2, "trajectory": "deteriorating"}, set()
    )
    assert chosen
    assert all(r["months_in_arrears"] >= 2 and r["trajectory"] == "deteriorating" for r in chosen)


def test_not_scored_leaves_out_scored_accounts(rows):
    everyone = worklist.select_cohort(rows, {}, set())
    first = everyone[0]["account_number"]
    unscored = worklist.select_cohort(rows, {"not_scored": True}, {first})
    assert first not in {r["account_number"] for r in unscored}
    assert len(unscored) == len(everyone) - 1


def test_the_same_filter_lists_the_same_people_in_the_same_order(rows):
    criteria = {"min_months_with_no_payment": 3, "sort_by": "current_balance"}
    assert worklist.select_cohort(rows, criteria, set()) == worklist.select_cohort(
        rows, criteria, set()
    )


def test_the_description_is_written_from_the_filter():
    text = worklist.describe_cohort(
        {"min_months_in_arrears": 3, "trajectory": "deteriorating", "not_scored": True}
    )
    assert text == (
        "Customers with 3+ months in arrears, arrears deteriorating, not scored yet, "
        "most months in arrears first."
    )


# --- Loana's output -----------------------------------------------------------


def test_a_valid_filter_becomes_the_reply(monkeypatch):
    sent = {"filter": {"min_months_in_arrears": 3, "trajectory": "deteriorating"}}
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm(json.dumps(sent)))

    turn = triage_assistant.interpret(_ask())

    assert turn.source == "model"
    assert turn.filter.min_months_in_arrears == 3
    # The LLM's own words never reach the officer.
    assert turn.reply == worklist.describe_cohort(turn.filter.model_dump())


def test_prose_around_the_json_is_ignored(monkeypatch):
    monkeypatch.setattr(
        llm_client, "chat",
        lambda m: _llm('Sure! 42 customers match: {"filter": {"min_months_in_arrears": 2}}'),
    )
    turn = triage_assistant.interpret(_ask())
    assert turn.filter.min_months_in_arrears == 2 and "42" not in turn.reply


@pytest.mark.parametrize(
    "text, reason",
    [
        ("I think you want the risky ones.", "triage_not_json"),
        ('{"customers": [1, 2]}', "triage_wrong_shape"),
        ('{"filter": {"probability": 0.9}}', "triage_invalid_filter"),
        ('{"filter": {"min_months_in_arrears": 12}}', "triage_invalid_filter"),
    ],
)
def test_anything_but_a_valid_filter_falls_back(monkeypatch, text, reason):
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm(text))
    turn = triage_assistant.interpret(_ask())
    assert (turn.source, turn.filter, turn.failure) == ("template", None, reason)


def test_without_a_model_the_template_answers(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm(failure="llm_unreachable"))
    turn = triage_assistant.interpret(_ask())
    assert turn.source == "template" and turn.note and turn.filter is None


def test_an_off_topic_request_gets_no_cohort(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm('{"filter": null}'))
    turn = triage_assistant.interpret(_ask("approve this loan"))
    assert (turn.source, turn.filter) == ("model", None)


def test_scored_band_keeps_only_that_latest_band(rows):
    first, second = rows[0]["account_number"], rows[1]["account_number"]
    bands = {first: "Low", second: "High"}
    chosen = worklist.select_cohort(rows, {"scored_band": "Low"}, set(bands), bands)
    assert [r["account_number"] for r in chosen] == [first]
    assert "a Low risk score from an officer" in worklist.describe_cohort({"scored_band": "Low"})


def test_low_risk_becomes_a_band_filter(monkeypatch):
    sent = {"filter": {"scored_band": "Low"}}
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm(json.dumps(sent)))
    turn = triage_assistant.interpret(_ask("show me low risk users"))
    assert turn.filter.scored_band == "Low"
    assert turn.llm_model == "test-model" and turn.latency_ms == 5


# --- the briefing -------------------------------------------------------------


def test_the_briefing_counts_each_quick_cohort(rows):
    cohorts, lead = worklist.briefing(rows, set())
    for cohort, (label, criteria) in zip(cohorts, worklist.QUICK_COHORTS):
        assert cohort["label"] == label
        assert cohort["total"] == len(worklist.select_cohort(rows, criteria, set()))
    if lead:
        worse = worklist.select_cohort(rows, worklist.QUICK_COHORTS[0][1], set())
        assert lead["current_balance"] == max(r["current_balance"] for r in worse)


def test_the_briefing_text_is_written_from_the_counts():
    cohorts = [{"total": 14}, {"total": 0}, {"total": 40}]
    lead = {"full_name": "Hiro Williams", "account_number": "20164",
            "months_in_arrears": 4, "current_balance": 20235.0}
    text = worklist.describe_briefing(cohorts, lead)
    assert "**14** getting worse" in text and "**40** using 90%+" in text
    assert "no payment" not in text
    assert "**Hiro Williams** (account 20164)" in text and "20,235" in text


def test_an_empty_briefing_says_nothing_is_waiting():
    text = worklist.describe_briefing([{"total": 0}] * 3, None)
    assert text.startswith("Nothing is waiting")


# --- the API ------------------------------------------------------------------


def _cohort_or_skip(response):
    if response.status_code == 503:
        pytest.skip("database not ingested; run: python -m app.db.ingest")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_cohort_endpoint_respects_the_limit(client):
    body = _cohort_or_skip(
        client.post("/officer/cohort", json={"min_months_in_arrears": 2, "limit": 5})
    )
    assert len(body["rows"]) <= 5 <= body["total"]
    assert all(r["months_in_arrears"] >= 2 for r in body["rows"])
    assert all("probability" not in r for r in body["rows"])


def test_the_briefing_endpoint_names_a_lead_with_a_customer_id(client):
    body = _cohort_or_skip(client.get("/officer/briefing"))
    assert len(body["cohorts"]) == len(worklist.QUICK_COHORTS)
    if body["lead"]:
        assert body["lead"]["customer_id"] and body["lead"]["full_name"] in body["text"]
    assert "probability" not in str(body)


def test_triage_turns_are_logged_with_the_officer(client, api_session, monkeypatch):
    from app.db.models import TriageTurn, User

    sent = {"filter": {"min_months_in_arrears": 2, "limit": 3}}
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm(json.dumps(sent)))
    officer = api_session.query(User).filter_by(email=OFFICER_EMAIL).one_or_none()
    if officer is None:
        pytest.skip("officer not seeded; run: python -m app.db.seed")

    body = _cohort_or_skip(client.post(
        "/officer/triage", json={"messages": _ask("2+ months behind"), "asked_by": OFFICER_EMAIL}
    ))

    logged = api_session.query(TriageTurn).order_by(TriageTurn.created_at.desc()).first()
    assert logged.user_message == "2+ months behind"
    assert logged.officer_id == officer.id
    assert logged.filter["min_months_in_arrears"] == 2
    assert logged.matched == body["cohort"]["total"]
    assert (logged.source, logged.llm_model) == ("model", "test-model")


def test_a_failed_triage_turn_is_logged_with_the_discarded_text(client, api_session, monkeypatch):
    from app.db.models import TriageTurn

    monkeypatch.setattr(llm_client, "chat", lambda m: _llm("not json at all"))
    response = client.post("/officer/triage", json={"messages": _ask("anything")})
    if response.status_code == 503:
        pytest.skip("database not available")

    logged = api_session.query(TriageTurn).order_by(TriageTurn.created_at.desc()).first()
    assert logged.source == "template" and logged.fallback_reason == "triage_not_json"
    assert logged.rejected_reply == "not json at all" and logged.officer_id is None


def test_the_briefing_is_built_once_a_day_and_rebuilt_when_stale(client, api_session):
    from datetime import date, timedelta

    from app.db.models import DailyBriefing

    first = _cohort_or_skip(client.get("/officer/briefing"))
    stored = api_session.get(DailyBriefing, date.today())
    assert stored is not None and stored.payload["text"] == first["text"]

    # Today's copy is served as stored, even if the counts have since moved.
    stored.payload = {**stored.payload, "text": "stored copy"}
    api_session.flush()
    assert client.get("/officer/briefing").json()["text"] == "stored copy"

    # A copy from yesterday is stale: today's request builds a new one.
    stored.day = date.today() - timedelta(days=1)
    api_session.flush()
    assert client.get("/officer/briefing").json()["text"] != "stored copy"


def test_the_cohort_endpoint_rejects_unknown_fields(client):
    assert client.post("/officer/cohort", json={"probability": 0.5}).status_code == 422


def test_triage_returns_a_cohort(client, monkeypatch):
    sent = {"filter": {"min_months_in_arrears": 2, "limit": 3}}
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm(json.dumps(sent)))

    body = _cohort_or_skip(client.post("/officer/triage", json={"messages": _ask()}))

    assert body["source"] == "model"
    assert len(body["cohort"]["rows"]) <= 3


def test_predict_selected_stores_one_score_each(client, riskiest_customer_id):
    response = client.post(
        "/officer/predictions",
        json={"customer_ids": [riskiest_customer_id], "scored_by": OFFICER_EMAIL},
    )
    assert response.status_code == 201
    [score] = response.json()
    assert score["risk_band"] in worklist.BANDS and score["prediction_id"]


def test_predict_selected_is_capped_at_ten(client, riskiest_customer_id):
    response = client.post(
        "/officer/predictions",
        json={"customer_ids": [riskiest_customer_id] * 11, "scored_by": OFFICER_EMAIL},
    )
    assert response.status_code == 422


def test_predict_selected_needs_a_known_officer(client, riskiest_customer_id):
    response = client.post(
        "/officer/predictions",
        json={"customer_ids": [riskiest_customer_id], "scored_by": "nobody@example.com"},
    )
    assert response.status_code in (403, 404)
