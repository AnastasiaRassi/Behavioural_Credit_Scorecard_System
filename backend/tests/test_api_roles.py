"""Smoke checks for the three role surfaces the frontend drives.

The important assertions here are about what each role is *not* sent. A payload
that carries a number the screen declines to draw has still disclosed it.
"""
import pytest

from app.core.passwords import hash_password
from app.db import customer_registry
from app.db.models import Customer, UserAccount
from app.ml import worklist
from app.ml.scorer import example_customer
from tests.conftest import bearer, ensure_queued_customer
from tests.test_api import (  # noqa: F401  (fixture imports)
    OFFICER_EMAIL,
    api_session,
    client,
)

CARDHOLDER_EMAIL = "test-cardholder@example.test"


@pytest.fixture(scope="module")
def any_customer_id() -> str:
    return worklist.all_rows()[0]["customer_id"]


@pytest.fixture(scope="module")
def other_customer_id() -> str:
    return worklist.all_rows()[-1]["customer_id"]


@pytest.fixture
def cardholder_identity(api_session, any_customer_id) -> str:
    """A cardholder registered as `any_customer_id`, the way the seed links one."""
    account = UserAccount(
        email=CARDHOLDER_EMAIL,
        display_name="Test Cardholder",
        password_hash=hash_password("not-a-real-password"),
        role="cardholder",
    )
    api_session.add(account)
    api_session.flush()

    ensure_queued_customer(api_session, worklist.find(any_customer_id))
    customer_registry.link_account(
        api_session, api_session.get(Customer, any_customer_id), account
    )
    return account.email


def _read_account(client, customer_id: str, email: str):
    return client.get(f"/cardholders/{customer_id}", headers=bearer(email))


# --- landing ------------------------------------------------------------------

def test_roles_lists_the_three_roles_with_a_starting_point(client):
    body = client.get("/roles").json()
    assert {row["role"] for row in body} == {"officer", "engineer", "cardholder"}
    assert all(row["landing_endpoint"] for row in body)
    assert all(row["description"] for row in body)


def test_users_omits_cardholders_by_default(client):
    body = client.get("/users").json()
    assert body, "the seeded staff should be listed"
    assert all(user["role"] != "cardholder" for user in body)


def test_users_can_be_filtered_to_officers(client):
    body = client.get("/users", params={"role": "officer"}).json()
    assert all(user["role"] == "officer" for user in body)


# --- loan officer -------------------------------------------------------------

def test_the_queue_is_paged_and_riskiest_first(client):
    body = client.get("/officer/queue", params={"limit": 10}).json()

    assert body["returned"] == 10
    assert body["total"] > 10, "the queue should be larger than one page"
    probabilities = [row["probability"] for row in body["rows"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_the_queue_reports_filtered_and_unfiltered_totals(client):
    body = client.get("/officer/queue", params={"band": "High", "limit": 5}).json()

    assert all(row["risk_band"] == "High" for row in body["rows"])
    assert body["summary"]["in_queue"] <= body["summary_all"]["in_queue"]
    assert body["summary_all"]["bands"]["High"] == body["summary"]["in_queue"]


def test_the_queue_rejects_an_unknown_band(client):
    assert client.get("/officer/queue", params={"band": "Severe"}).status_code == 422


def test_queue_paging_does_not_repeat_rows(client):
    first = client.get("/officer/queue", params={"limit": 5, "offset": 0}).json()
    second = client.get("/officer/queue", params={"limit": 5, "offset": 5}).json()

    ids = {row["customer_id"] for row in first["rows"]}
    assert ids.isdisjoint({row["customer_id"] for row in second["rows"]})


def test_an_officer_sees_the_effect_size(client, any_customer_id):
    body = client.get(f"/officer/queue/{any_customer_id}").json()
    assert "probability" in body
    assert all("effect" in reason for reason in body["reasons"])


def test_an_unknown_queued_customer_is_a_404(client):
    assert client.get("/officer/queue/C-nobody").status_code == 404


# --- cardholder ---------------------------------------------------------------

# These read as the officer, who is entitled to any account: they are about the
# shape of the payload, not about who may ask for it.

def test_a_cardholder_is_not_sent_the_number(client, any_customer_id):
    body = _read_account(client, any_customer_id, OFFICER_EMAIL).json()

    assert "probability" not in body
    assert "threshold" not in body
    assert all("effect" not in reason for reason in body["reasons"])


def test_a_cardholder_gets_a_band_and_plain_reasons(client, any_customer_id):
    body = _read_account(client, any_customer_id, OFFICER_EMAIL).json()

    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert isinstance(body["needs_attention"], bool)
    assert body["reasons"]
    assert all(reason["description"] for reason in body["reasons"])
    assert all(isinstance(reason["raises_risk"], bool) for reason in body["reasons"])


def test_the_two_views_describe_the_same_account(client, any_customer_id):
    officer_view = client.get(f"/officer/queue/{any_customer_id}").json()
    cardholder = _read_account(client, any_customer_id, OFFICER_EMAIL).json()

    assert cardholder["risk_band"] == officer_view["risk_band"]
    assert cardholder["needs_attention"] == officer_view["flagged"]
    assert cardholder["months_in_arrears"] == officer_view["months_in_arrears"]


def test_an_unknown_account_is_a_404(client):
    assert _read_account(client, "C-nobody", OFFICER_EMAIL).status_code == 404


# --- who may read an account --------------------------------------------------

def test_a_cardholder_cannot_read_another_account(
    client, cardholder_identity, other_customer_id
):
    """The defect this guard exists for: any id was readable by any caller."""
    response = _read_account(client, other_customer_id, cardholder_identity)
    assert response.status_code == 403


def test_a_cardholder_can_read_their_own_account(
    client, cardholder_identity, any_customer_id
):
    response = _read_account(client, any_customer_id, cardholder_identity)

    assert response.status_code == 200
    assert response.json()["risk_band"] in {"Low", "Medium", "High"}


def test_an_officer_can_read_any_account(client, any_customer_id, other_customer_id):
    for customer_id in (any_customer_id, other_customer_id):
        assert _read_account(client, customer_id, OFFICER_EMAIL).status_code == 200


def test_an_account_cannot_be_read_without_signing_in(client, any_customer_id):
    """The boundary is required, not advisory: no token is not an anonymous read."""
    response = client.get(f"/cardholders/{any_customer_id}")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


# --- ML engineer --------------------------------------------------------------

def test_the_report_carries_the_curves_the_dashboard_draws(client):
    body = client.get("/report").json()

    assert len(body["deciles"]) == 10
    assert body["calibration"], "the reliability curve must be available"
    assert set(body["split_sizes"]) == {"train", "validation", "test"}


def test_explain_scores_without_storing(client):
    before = client.get("/decisions").json()
    body = client.post("/explain", json=example_customer()).json()

    assert body["stored"] is False
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["reasons"]) > 6, "the engineer sees more than a screen shows"
    assert client.get("/decisions").json() == before


def test_explain_validates_the_customer(client):
    customer = example_customer()
    customer["LIMIT_BAL"] = 0  # a limit of zero is not a cardholder
    assert client.post("/explain", json=customer).status_code == 422
