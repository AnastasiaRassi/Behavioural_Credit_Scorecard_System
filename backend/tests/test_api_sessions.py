"""Signing in: what a credential buys, and what a token cannot be made to say.

Runs against the account the `client` fixture creates, so nothing here depends
on a seeded password or on SEED_PASSWORD being configured.
"""
import pytest

from app.core import sessions
from tests.test_api import (  # noqa: F401  (fixture imports)
    OFFICER_EMAIL,
    OFFICER_PASSWORD,
    api_session,
    client,
)


def _sign_in(client, email: str = OFFICER_EMAIL, password: str = OFFICER_PASSWORD):
    return client.post("/session", json={"email": email, "password": password})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- the credential -----------------------------------------------------------

def test_signing_in_returns_the_account_and_a_token(client):
    response = _sign_in(client)
    assert response.status_code == 200

    body = response.json()
    assert body["email"] == OFFICER_EMAIL
    assert body["display_name"] == "Test Officer"
    assert body["role"] == "officer"
    assert body["token"]
    assert body["expires_at"] > 0


def test_the_role_comes_from_the_account_not_the_request(client):
    """The point of replacing the role picker: a caller cannot choose their role."""
    response = client.post(
        "/session",
        json={
            "email": OFFICER_EMAIL,
            "password": OFFICER_PASSWORD,
            "role": "engineer",
        },
    )
    assert response.json()["role"] == "officer"


def test_a_wrong_password_is_refused(client):
    assert _sign_in(client, password="not it").status_code == 401


def test_an_unknown_email_is_refused_identically(client):
    """Neither answer may reveal which accounts exist."""
    unknown = _sign_in(client, email="nobody@example.test")
    wrong = _sign_in(client, password="not it")

    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"]
    assert "nobody@example.test" not in unknown.json()["detail"]


def test_a_capitalised_email_is_the_same_account(client):
    assert _sign_in(client, email=OFFICER_EMAIL.upper()).status_code == 200


def test_the_password_is_never_echoed(client):
    assert OFFICER_PASSWORD not in _sign_in(client).text


# --- the token ----------------------------------------------------------------

def test_a_token_identifies_the_account_that_signed_in(client):
    token = _sign_in(client).json()["token"]
    body = client.get("/session", headers=_bearer(token)).json()

    assert body["email"] == OFFICER_EMAIL
    assert body["role"] == "officer"
    assert "token" not in body, "reading a session must not re-issue one"


def test_a_tampered_token_is_refused(client):
    scheme, payload, signature = _sign_in(client).json()["token"].split(".")
    forged = f"{scheme}.{payload[:-2]}AA.{signature}"

    assert client.get("/session", headers=_bearer(forged)).status_code == 401


def test_an_expired_token_is_refused(client):
    stale, _ = sessions.issue(OFFICER_EMAIL, lifetime_seconds=-1)
    response = client.get("/session", headers=_bearer(stale))

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_a_token_for_an_unknown_account_is_refused(client):
    """A signature alone is not enough: the account still has to exist."""
    orphan, _ = sessions.issue("nobody@example.test")
    assert client.get("/session", headers=_bearer(orphan)).status_code == 401


def test_something_that_is_not_a_bearer_token_is_refused(client):
    for header in ({"Authorization": "Basic abc"}, {"Authorization": "Bearer"}):
        assert client.get("/session", headers=header).status_code == 401


# --- the signing itself -------------------------------------------------------

def test_the_signature_covers_the_expiry():
    """Re-dating a token must not survive, or expiry would be caller-controlled."""
    token, _ = sessions.issue("someone@example.test", lifetime_seconds=1)
    scheme, payload, signature = token.split(".")
    extended = sessions._encode(
        sessions._decode(payload).decode().rpartition(":")[0].encode()
        + b":99999999999"
    )

    with pytest.raises(sessions.InvalidSession):
        sessions.identity_from(f"{scheme}.{extended}.{signature}")


def test_a_valid_token_round_trips():
    token, expires_at = sessions.issue("someone@example.test")

    assert sessions.identity_from(token) == "someone@example.test"
    assert expires_at > 0


def test_a_malformed_token_is_not_an_unhandled_error():
    for rubbish in ("", "nonsense", "a.b", "v1.!!.!!", "v9.aa.bb"):
        with pytest.raises(sessions.InvalidSession):
            sessions.identity_from(rubbish)
