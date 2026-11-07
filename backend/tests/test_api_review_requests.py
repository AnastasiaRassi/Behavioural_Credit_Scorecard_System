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


def test_asking_twice_keeps_the_first_request(client, api_session, quietest_customer_id):
    headers = _cardholder(api_session, quietest_customer_id, CARDHOLDER_EMAIL)

    first = client.post(_url(quietest_customer_id), json={"reason": "First."}, headers=headers)
    second = client.post(_url(quietest_customer_id), json={"reason": "Second."}, headers=headers)

    assert second.json()["request_id"] == first.json()["request_id"]
    assert second.json()["reason"] == "First."


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
