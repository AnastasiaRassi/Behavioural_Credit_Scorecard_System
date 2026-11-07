"""Checks for the customer detail page, segmentation, and the queue-to-trail link.

Covers the backend changes the new frontend needs: the six-month history, the
month vocabulary, derived highlights, the behavioural segment, recording a
decision against a queued customer, and malformed ids answering 4xx not 500.
"""
import uuid

import pytest

from app.ml import worklist
from app.ml.explainability import MONTH_LABELS, MONTHS_OBSERVED, month_label
from tests.test_api import (  # noqa: F401  (fixture imports)
    OFFICER_EMAIL,
    api_session,
    client,
    quietest_customer_id,
    riskiest_customer_id,
)


# --- B. the month vocabulary --------------------------------------------------

def test_month_labels_cover_the_observed_window():
    assert MONTHS_OBSERVED == 6
    assert len(MONTH_LABELS) == MONTHS_OBSERVED
    assert month_label(1) == "Last month"
    assert month_label(3) == "3 months ago"


def test_an_unknown_month_still_reads_as_something():
    assert month_label(9) == "month 9"


# --- the customer's identity --------------------------------------------------

def test_the_detail_names_the_customer_by_uuid_and_account_number(
    client, riskiest_customer_id
):
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()
    riskiest = worklist.all_rows()[0]

    assert body["customer_id"] == riskiest_customer_id
    assert body["account_number"] == riskiest["account_number"]
    assert body["full_name"], "every customer is a user with a name"


def test_an_account_number_is_not_a_customer_id(client, riskiest_customer_id):
    number = worklist.all_rows()[0]["account_number"]
    assert client.get(f"/officer/customers/{number}").status_code == 404


def test_search_finds_any_queued_customer_by_name(client, riskiest_customer_id):
    detail = client.get(f"/officer/customers/{riskiest_customer_id}").json()

    found = client.get(
        "/officer/customers",
        params={"q": detail["full_name"], "limit": 50},
    ).json()

    assert riskiest_customer_id in {match["customer_id"] for match in found}
    assert all(match["account_number"] for match in found)


def test_search_finds_a_customer_by_account_number_and_lists_it_first(
    client, riskiest_customer_id
):
    number = worklist.all_rows()[0]["account_number"]

    found = client.get("/officer/customers", params={"q": number}).json()

    assert found[0]["customer_id"] == riskiest_customer_id
    assert found[0]["account_number"] == number


def test_search_lists_customers_who_can_sign_in_before_those_who_cannot(
    client, api_session
):
    from app.core.passwords import hash_password
    from app.db import customer_registry
    from tests.conftest import ensure_queued_customer

    # The last queue row gets a login; its name is then searched for.
    last = worklist.all_rows()[-1]
    ensure_queued_customer(api_session, last)
    customer = customer_registry.find_by_account_number(api_session, last["account_number"])
    customer_registry.give_login(
        api_session, customer, "searchable@example.test", hash_password("not-a-real-password")
    )

    found = client.get(
        "/officer/customers", params={"q": customer.user.display_name, "limit": 50}
    ).json()

    flags = [match["known"] for match in found]
    assert str(customer.id) in {m["customer_id"] for m in found if m["known"]}
    assert flags == sorted(flags, reverse=True), "signed-up customers come first"


def test_search_only_returns_customers_who_can_be_opened(client, api_session):
    # A walk-in is a customer and a user, but is not in the worklist.
    from app.db import customer_registry
    from app.ml.scorer import example_customer

    customer_registry.register_from_raw(
        api_session, example_customer(), "W-searchme"
    )

    assert client.get("/officer/customers", params={"q": "W-searchme"}).json() == []


# --- A. the six-month history ------------------------------------------------

def test_the_history_is_six_months_most_recent_first(client, riskiest_customer_id):
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()
    history = body["history"]

    assert [entry["month"] for entry in history] == [1, 2, 3, 4, 5, 6]
    assert history[0]["label"] == "Last month"


def test_each_month_carries_the_raw_status_and_its_reading(client, riskiest_customer_id):
    history = client.get(f"/officer/customers/{riskiest_customer_id}").json()["history"]

    for entry in history:
        assert isinstance(entry["payment_status"], int)
        assert entry["payment_status_reading"]
        assert "bill_amount" in entry
        assert "amount_paid" in entry


def test_utilisation_is_the_bill_over_the_limit(client, riskiest_customer_id):
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()
    limit = body["credit_limit"]

    for entry in body["history"]:
        if entry["utilisation"] is None:
            continue
        assert entry["utilisation"] == pytest.approx(
            entry["bill_amount"] / limit, abs=1e-4
        )


def test_the_most_recent_month_agrees_with_the_scalars(client, riskiest_customer_id):
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()
    recent = body["history"][0]

    assert recent["bill_amount"] == body["current_balance"]
    assert recent["amount_paid"] == body["last_payment"]


# --- C. derived highlights ----------------------------------------------------

def test_highlights_are_derived_not_invented(client, riskiest_customer_id):
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()
    highlights, history = body["highlights"], body["history"]

    assert highlights["utilisation_now"] == history[0]["utilisation"]
    assert highlights["worst_payment_status"] == max(
        entry["payment_status"] for entry in history
    )
    assert highlights["months_with_no_payment"] == sum(
        1 for entry in history if entry["amount_paid"] <= 0
    )
    assert highlights["trajectory"] in {"improving", "steady", "deteriorating"}


def test_trajectory_reads_the_direction_of_arrears():
    assert worklist._trajectory([4, 3, 2, 1, 0, 0]) == "deteriorating"
    assert worklist._trajectory([0, 1, 2, 3, 4, 5]) == "improving"
    assert worklist._trajectory([2, 2, 2, 2, 2, 2]) == "steady"
    assert worklist._trajectory([]) is None


def test_months_since_full_payment_counts_back_from_now():
    paid_now = [
        {"month": 1, "bill_amount": 500.0, "amount_paid": 500.0},
        {"month": 2, "bill_amount": 400.0, "amount_paid": 10.0},
    ]
    assert worklist._months_since_full_payment(paid_now) == 0

    never = [
        {"month": month, "bill_amount": 500.0, "amount_paid": 1.0}
        for month in range(1, 7)
    ]
    assert worklist._months_since_full_payment(never) is None


# --- D. segmentation ----------------------------------------------------------

def test_the_detail_carries_the_behavioural_segment(client, riskiest_customer_id):
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()

    assert body["segment"] is not None, "run python -m app.ml.segmentation"
    assert body["segment"]["name"]
    assert isinstance(body["segment"]["label"], int)


def test_segments_are_exposed_with_how_k_was_chosen(client):
    body = client.get("/segments").json()

    assert body["k"] == len(body["segments"])
    assert body["chosen_by"]
    assert str(body["k"]) in body["silhouette_scores"]
    assert all(segment["name"] for segment in body["segments"])
    assert all("default_rate" in segment for segment in body["segments"])


def test_a_queue_segment_is_one_of_the_published_segments(client, riskiest_customer_id):
    published = {
        segment["name"] for segment in client.get("/segments").json()["segments"]
    }
    body = client.get(f"/officer/customers/{riskiest_customer_id}").json()
    assert body["segment"]["name"] in published


# --- F. the queue reaches the decision trail ----------------------------------

def test_a_queued_customer_can_be_decided(client, riskiest_customer_id):
    response = client.post(
        f"/officer/customers/{riskiest_customer_id}/decisions",
        json={
            "action": "contact_customer",
            "justification": "Six months in arrears; calling today.",
            "decided_by": OFFICER_EMAIL,
        },
    )
    assert response.status_code == 201
    assert response.json()["action"] == "contact_customer"


def test_deciding_twice_keeps_one_score_with_two_decisions(client, riskiest_customer_id):
    for action in ("monitor", "reduce_limit"):
        client.post(
            f"/officer/customers/{riskiest_customer_id}/decisions",
            json={
                "action": action,
                "justification": f"escalating to {action}",
                "decided_by": OFFICER_EMAIL,
            },
        )

    trail = client.get(f"/officer/customers/{riskiest_customer_id}/decisions").json()
    assert [row["action"] for row in trail] == ["monitor", "reduce_limit"]
    assert len({row["prediction_id"] for row in trail}) == 1


def test_the_stored_decision_withholds_protected_attributes(client, riskiest_customer_id):
    created = client.post(
        f"/officer/customers/{riskiest_customer_id}/decisions",
        json={
            "action": "monitor",
            "justification": "watching next cycle",
            "decided_by": OFFICER_EMAIL,
        },
    ).json()

    stored = client.get(f"/predictions/{created['prediction_id']}").json()
    assert stored["probability"] == pytest.approx(
        client.get(f"/officer/customers/{riskiest_customer_id}").json()["probability"]
    )

    raw = client.get(f"/predictions/{created['prediction_id']}").json()
    assert raw["risk_band"] in {"Low", "Medium", "High"}


def test_an_untouched_queued_customer_has_an_empty_trail(client, quietest_customer_id):
    assert client.get(f"/officer/customers/{quietest_customer_id}/decisions").json() == []


def test_deciding_on_an_unknown_customer_is_a_404(client):
    response = client.post(
        f"/officer/customers/{uuid.uuid4()}/decisions",
        json={
            "action": "monitor",
            "justification": "x",
            "decided_by": OFFICER_EMAIL,
        },
    )
    assert response.status_code == 404


# --- G. malformed ids ---------------------------------------------------------

def test_a_malformed_prediction_id_is_not_a_500(client):
    assert client.get("/predictions/not-a-uuid").status_code == 422


def test_deciding_against_a_malformed_prediction_id_is_not_a_500(client):
    response = client.post(
        "/predictions/not-a-uuid/decisions",
        json={
            "action": "monitor",
            "justification": "x",
            "decided_by": OFFICER_EMAIL,
        },
    )
    assert response.status_code == 422


def test_listing_decisions_for_a_malformed_id_is_not_a_500(client):
    assert client.get("/predictions/not-a-uuid/decisions").status_code == 422


def test_a_well_formed_but_absent_id_is_still_a_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/predictions/{missing}").status_code == 404
