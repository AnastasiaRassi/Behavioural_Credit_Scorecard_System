"""Recourse over HTTP: what a customer is told to do, and what they are not told.

The maths has its own tests in test_counterfactuals.py. These are about the
endpoint: that it answers for a real account, that the answer is in plain
language, that it stays inside the bands, and that it carries no probability.
"""
import pytest

from app.ml import worklist
from tests.conftest import bearer, ensure_queued_customer
from tests.test_api import (  # noqa: F401  (fixture imports)
    OFFICER_EMAIL,
    api_session,
    client,
)


@pytest.fixture
def scored_customer(api_session):
    """A customer the queue knows about, present in the database."""
    rows = worklist.all_rows()
    if not rows:
        pytest.skip("no queue artifact; run: python -m app.ml.train")
    customer_id = ensure_queued_customer(api_session, rows[0])
    api_session.flush()
    return customer_id


def _recourse(client, customer_id: str):
    return client.get(
        f"/cardholders/{customer_id}/recourse", headers=bearer(OFFICER_EMAIL)
    )


def test_recourse_answers_for_a_scored_account(client, scored_customer):
    response = _recourse(client, scored_customer)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["customer_id"] == scored_customer
    assert body["current_band"] in {"Low", "Medium", "High"}
    assert body["target_band"] in {"Low", "Medium", "High"}
    assert body["note"]


def test_the_target_is_one_band_better(client, scored_customer):
    body = _recourse(client, scored_customer).json()
    better = {"High": "Medium", "Medium": "Low", "Low": "Low"}
    assert body["target_band"] == better[body["current_band"]]


def test_steps_are_written_for_a_person(client, scored_customer):
    body = _recourse(client, scored_customer).json()
    if not body["steps"]:
        pytest.skip("this account needs no change to reach the next band")

    for step in body["steps"]:
        # The whole point of describe_change: never "PAY_AMT1 = 0".
        assert step["feature"] not in step["description"]
        assert step["description"][0].isupper()
        assert step["before"] != step["after"]


def test_the_payload_carries_no_probability(client, scored_customer):
    """Same rule as the cardholder view: what a role cannot see is not sent."""
    body = _recourse(client, scored_customer).json()
    leaked = {"probability", "threshold", "score", "original_score", "achieved_score"}
    assert leaked.isdisjoint(body)


def test_an_unknown_account_is_not_found(client):
    assert _recourse(client, "no-such-customer").status_code == 404


def test_a_cardholder_cannot_read_another_account(client, scored_customer, api_session):
    """The guard is account_reader's, and it applies here too."""
    from app.db.models import UserAccount
    from app.core.passwords import hash_password

    other = UserAccount(
        email="someone-else@example.test",
        display_name="Someone Else",
        password_hash=hash_password("not-a-real-password"),
        role="cardholder",
    )
    api_session.add(other)
    api_session.flush()

    response = client.get(
        f"/cardholders/{scored_customer}/recourse",
        headers=bearer("someone-else@example.test"),
    )
    assert response.status_code == 403
