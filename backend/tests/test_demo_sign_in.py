"""Every demo account signs in, and lands on the screen its role is for.

The bug this exists to catch: nothing in the suite ever signed in and then
opened a role's own landing endpoint, so a frontend calling a route the running
backend did not serve looked fine until someone typed a password. Each test
here goes through POST /session for real and uses the token it returns.
"""
import pytest

from app.db import seed
from app.db.models import UserAccount
from tests.test_api import (  # noqa: F401  (fixture imports)
    api_session,
    client,
)

# Where each role starts, as roles.py publishes it. A cardholder's landing
# endpoint is keyed on the customer the account is linked to, so it is built
# from the sign-in response rather than listed here.
LANDING = {"officer": "/officer/queue", "engineer": "/report"}


@pytest.fixture
def demo_accounts(api_session, monkeypatch):
    """The demo accounts, inside the fixture's transaction so they roll back."""
    monkeypatch.setenv("DEMO_MODE", "1")
    seed.seed_demo_accounts(api_session)
    api_session.flush()
    return seed.DEMO_ACCOUNTS


def _sign_in(client, email: str, password: str = seed.DEMO_PASSWORD):
    return client.post("/session", json={"email": email, "password": password})


def test_every_demo_account_signs_in(client, demo_accounts):
    for email, display_name, role in demo_accounts:
        response = _sign_in(client, email)
        assert response.status_code == 200, f"{email}: {response.text}"

        body = response.json()
        assert body["email"] == email
        assert body["display_name"] == display_name
        assert body["role"] == role
        assert body["token"]


def test_each_demo_account_lands_on_its_own_screen(client, demo_accounts):
    for email, _, role in demo_accounts:
        body = _sign_in(client, email).json()
        headers = {"Authorization": f"Bearer {body['token']}"}

        if role == "cardholder":
            if body["customer_id"] is None:
                pytest.skip("no unlinked customer for the demo cardholder")
            landing = f"/cardholders/{body['customer_id']}"
        else:
            landing = LANDING[role]

        response = client.get(landing, headers=headers)
        assert response.status_code == 200, f"{role} -> {landing}: {response.text}"


def test_the_role_comes_from_the_account(client, demo_accounts):
    """A caller cannot ask to be something else; the account decides."""
    for email, _, role in demo_accounts:
        body = _sign_in(client, email).json()
        headers = {"Authorization": f"Bearer {body['token']}"}
        assert client.get("/session", headers=headers).json()["role"] == role


@pytest.mark.xfail(
    reason="officer/queue takes no auth dependency, so any token reaches it",
    strict=False,
)
def test_a_demo_account_cannot_open_another_roles_screen(client, demo_accounts):
    """What the sign-in change does not yet buy.

    The role is decided by the account, but the officer router never asks who
    is calling, so a cardholder's token reads the worklist. Guarding that is a
    change to the officer endpoints, not to signing in; this records the gap
    until it is closed.
    """
    body = _sign_in(client, "cardholder@demo.test").json()
    headers = {"Authorization": f"Bearer {body['token']}"}

    assert client.get("/officer/queue", headers=headers).status_code == 403


def test_a_wrong_demo_password_is_rejected(client, demo_accounts):
    response = _sign_in(client, "officer@demo.test", password="not-the-password")
    assert response.status_code == 401


def test_reseeding_does_not_duplicate_or_crash(api_session, monkeypatch):
    """Safe to rerun. Says nothing about whether the accounts existed before."""
    monkeypatch.setenv("DEMO_MODE", "1")
    emails = [email for email, _, _ in seed.DEMO_ACCOUNTS]

    first_added, first_updated = seed.seed_demo_accounts(api_session)
    second_added, second_updated = seed.seed_demo_accounts(api_session)
    api_session.flush()

    assert sorted(first_added + first_updated) == sorted(emails)
    # The second pass can only update: the first one left them all there.
    assert second_added == []
    assert sorted(second_updated) == sorted(emails)

    for email in emails:
        count = api_session.query(UserAccount).filter_by(email=email).count()
        assert count == 1, f"{email} was seeded {count} times"


def test_nothing_is_seeded_outside_demo_mode(api_session, monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    demo = UserAccount.email.like("%@demo.test")
    before = api_session.query(UserAccount).filter(demo).count()

    assert seed.seed_demo_accounts(api_session) == ([], [])
    api_session.flush()

    assert api_session.query(UserAccount).filter(demo).count() == before
