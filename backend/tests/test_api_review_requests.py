"""A cardholder asks for a human review; the officer sees it."""
import uuid

from app.core.passwords import hash_password
from app.db import customer_registry
from app.db.models import Customer
from tests.conftest import bearer
from tests.test_api import (  # noqa: F401  (fixture imports)
    OFFICER_EMAIL,
    api_session,
    client,
    quietest_customer_id,
    riskiest_customer_id,
)

CARDHOLDER_EMAIL = "asks-for-review@example.test"
OTHER_EMAIL = "someone-else@example.test"


def _cardholder(api_session, customer_id: str, email: str) -> dict[str, str]:
    customer_registry.give_login(
        api_session,
        api_session.get(Customer, uuid.UUID(customer_id)),
        email,
        hash_password("not-a-real-password"),
    )
    return bearer(email)


def _url(customer_id: str) -> str:
    return f"/cardholders/{customer_id}/review-request"


def test_a_cardholder_can_ask_and_the_officer_sees_the_comment(
    client, api_session, quietest_customer_id
):
    headers = _cardholder(api_session, quietest_customer_id, CARDHOLDER_EMAIL)

    assert client.get(_url(quietest_customer_id), headers=headers).json() is None

    made = client.post(
        _url(quietest_customer_id), json={"reason": "Please check my account."},
        headers=headers,
    )
    assert made.status_code == 201
    assert made.json()["status"] == "pending"

    mine = client.get(_url(quietest_customer_id), headers=headers).json()
    assert mine["reason"] == "Please check my account."

    listed = client.get("/officer/review-requests").json()
    ours = [item for item in listed if item["customer_id"] == quietest_customer_id]
    assert [item["reason"] for item in ours] == ["Please check my account."]
    assert ours[0]["full_name"]


def test_a_second_message_joins_the_thread(client, api_session, quietest_customer_id):
    headers = _cardholder(api_session, quietest_customer_id, CARDHOLDER_EMAIL)

    client.post(_url(quietest_customer_id), json={"reason": "First."}, headers=headers)
    client.post(_url(quietest_customer_id), json={"reason": "Second."}, headers=headers)

    # The oldest unanswered message is still the one an officer reads first.
    assert client.get(_url(quietest_customer_id), headers=headers).json()["reason"] == "First."
    thread = client.get(f"/cardholders/{quietest_customer_id}/thread", headers=headers).json()
    assert [entry["body"] for entry in thread] == ["First.", "Second."]
    assert {entry["kind"] for entry in thread} == {"message"}


def test_a_decision_closes_the_conversation_and_both_sides_see_it(
    client, api_session, quietest_customer_id
):
    headers = _cardholder(api_session, quietest_customer_id, CARDHOLDER_EMAIL)
    client.post(_url(quietest_customer_id), json={"reason": "Can you look?"}, headers=headers)

    decided = client.post(
        f"/officer/customers/{quietest_customer_id}/decisions",
        json={"decided_by": OFFICER_EMAIL, "action": "monitor", "justification": "Looked; monitoring."},
    )
    assert decided.status_code == 201

    assert client.get(_url(quietest_customer_id), headers=headers).json() is None
    listed = client.get("/officer/review-requests").json()
    assert not [item for item in listed if item["customer_id"] == quietest_customer_id]

    mine = client.get(f"/cardholders/{quietest_customer_id}/thread", headers=headers).json()
    theirs = client.get(f"/officer/customers/{quietest_customer_id}/thread").json()
    assert mine == theirs
    assert [entry["kind"] for entry in mine] == ["message", "decision"]
    assert mine[0]["status"] == "resolved"
    assert mine[1]["action"] == "monitor"
    assert mine[1]["body"] == "Looked; monitoring."

    # Writing again reopens it.
    client.post(_url(quietest_customer_id), json={"reason": "One more thing."}, headers=headers)
    assert client.get(_url(quietest_customer_id), headers=headers).json()["reason"] == "One more thing."


def test_a_blank_comment_is_refused(client, api_session, quietest_customer_id):
    headers = _cardholder(api_session, quietest_customer_id, CARDHOLDER_EMAIL)

    response = client.post(_url(quietest_customer_id), json={"reason": "   "}, headers=headers)

    assert response.status_code == 422


def test_a_cardholder_cannot_ask_for_someone_elses_account(
    client, api_session, quietest_customer_id, riskiest_customer_id
):
    headers = _cardholder(api_session, quietest_customer_id, OTHER_EMAIL)

    response = client.post(
        _url(riskiest_customer_id), json={"reason": "Not mine."}, headers=headers
    )

    assert response.status_code == 403


def test_asking_needs_a_signed_in_caller(client, quietest_customer_id):
    response = client.post(_url(quietest_customer_id), json={"reason": "Anonymous."})

    assert response.status_code == 401
