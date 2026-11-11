"""Signing in: verify a credential, then say who the caller is.

The role is read off the account and returned, never accepted from the request.
A caller therefore cannot choose which screens open for them, which is the whole
point of moving role selection out of the landing page.
"""
import logging
from functools import lru_cache
from secrets import token_urlsafe

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, signed_in_user
from app.core import audit, sessions
from app.core.passwords import hash_password, verify_password
from app.db.models import User
from app.schemas.sessions import Account, Credentials, SignedIn

router = APIRouter(tags=["session"])


@lru_cache(maxsize=1)
def _decoy_hash() -> str:
    """A hash nothing matches, so an unknown email costs what a wrong password does."""
    return hash_password(token_urlsafe(32))


def _client(http: Request) -> str:
    return http.client.host if http.client else "-"


def _rejected() -> HTTPException:
    # One message for both an unknown email and a wrong password: a response that
    # distinguished them would turn this endpoint into a way to discover which
    # accounts exist.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Email or password is incorrect",
    )


@router.post("/session", response_model=SignedIn)
def sign_in(
    request: Credentials, http: Request, session: Session = Depends(get_session)
):
    """Exchange an email and password for a signed session token."""
    # Folded to lower case because that is how the seed and registration store
    # an email, so a capitalised one is the same account rather than no account.
    email = request.email.strip().lower()
    account = session.query(User).filter_by(email=email).one_or_none()

    # A customer who never registered has a user but no password, so there is
    # nothing to verify: treated exactly like an unknown email.
    if account is None or account.password_hash is None:
        # Hashed anyway, so a missing account does not answer faster than a wrong
        # password and reveal itself by timing.
        verify_password(request.password, _decoy_hash())
        # The reason stays in the log; the response gives nothing away.
        audit.event(
            "sign_in_failed",
            actor=email,
            level=logging.WARNING,
            reason="unknown_or_unregistered",
            client=_client(http),
        )
        raise _rejected()

    if not verify_password(request.password, account.password_hash):
        audit.event(
            "sign_in_failed",
            actor=email,
            level=logging.WARNING,
            reason="wrong_password",
            role=account.role,
            client=_client(http),
        )
        raise _rejected()

    token, expires_at = sessions.issue(account.email)
    audit.event("sign_in", actor=email, role=account.role, client=_client(http))
    return SignedIn(
        **_account(account).model_dump(), token=token, expires_at=expires_at
    )


@router.get("/session", response_model=Account)
def read_signed_in_account(account: User = Depends(signed_in_user)):
    """Who the presented token belongs to, for a client checking it is still valid."""
    return _account(account)


def _account(account: User) -> Account:
    # A user id is not a customer id, and the cardholder screen needs the latter.
    registered_as = account.customer
    details = account.details
    return Account(
        email=account.email,
        display_name=account.display_name,
        role=account.role,
        customer_id=str(registered_as.id) if registered_as else None,
        account_number=registered_as.account_number if registered_as else None,
        age_years=details.age_years if details else None,
        sex=details.sex if details else None,
        education=details.education if details else None,
        marriage=details.marriage if details else None,
    )
