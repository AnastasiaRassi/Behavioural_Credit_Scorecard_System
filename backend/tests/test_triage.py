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
