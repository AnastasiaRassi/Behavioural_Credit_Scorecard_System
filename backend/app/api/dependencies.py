"""Shared request dependencies: the scorer, the explainer, a session, the caller.

The scorer and explainer are process-wide and cached, since loading the ensemble
costs far more than a request should. The session is per request.

Authorization also lives here. The caller's identity arrives as an asserted
claim on a header rather than from a local login, because credential
verification belongs at an identity provider in front of this service (§1). What
that identity is *allowed to read* is still this layer's decision, and is made in
one place so a real OIDC validator can replace the claim without the rules
moving.
"""
import uuid
from functools import lru_cache
from typing import Iterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import _session_factory
from app.db.models import ModelVersion, Prediction, UserAccount
from app.ml.explainability import Explainer
from app.ml.scorer import ArtifactsMissing, CreditScorer

# Where the gateway puts the upstream claim. Named for what it is so a reader
# does not mistake it for a credential.
IDENTITY_HEADER = "X-User-Identity"

# Staff entitled to read any cardholder's account. A cardholder reads only their
# own, and no other role reaches the account at all.
ACCOUNT_READERS: tuple[str, ...] = ("officer", "engineer")


@lru_cache(maxsize=1)
def get_scorer() -> CreditScorer:
    try:
        return CreditScorer()
    except ArtifactsMissing as exception:
        # 503 rather than 500: the service is correctly built but not provisioned,
        # and the message says which command fixes it.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exception)
        ) from exception


@lru_cache(maxsize=1)
def get_explainer() -> Explainer:
    return Explainer(get_scorer())


def get_session() -> Iterator[Session]:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def live_model_version(session: Session) -> ModelVersion:
    """The registered live model, or a 503 telling the caller to seed it."""
    version = session.query(ModelVersion).filter_by(is_live=True).one_or_none()
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No live model version registered. Run: python -m app.db.seed",
        )
    return version


def prediction_or_404(session: Session, prediction_id: str) -> Prediction:
    """Look up a prediction, distinguishing a malformed id from a missing one.

    The id is parsed before it reaches the driver. Passing a non-UUID string
    straight to a uuid column raises inside psycopg, which surfaces as a 500 for
    what is really a bad request (§9).
    """
    try:
        parsed = uuid.UUID(prediction_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{prediction_id!r} is not a valid prediction id",
        ) from None

    prediction = session.get(Prediction, parsed)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction {prediction_id}",
        )
    return prediction


def user_by_email(
    session: Session, email: str, expected_role: str | tuple[str, ...]
) -> UserAccount:
    """The account behind an asserted email: 404 if unknown, 403 if the wrong role."""
    allowed = (expected_role,) if isinstance(expected_role, str) else expected_role

    user = session.query(UserAccount).filter_by(email=email).one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown user {email}"
        )
    if user.role not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{email} is a {user.role}, not a {' or a '.join(allowed)}",
        )
    return user


def authorise_account_read(
    session: Session, identity: str, customer_id: str
) -> UserAccount:
    """Who may read one cardholder's account: that cardholder, or any staff reader.

    The only place this rule is written, so a route never decides it for itself.
    """
    user = user_by_email(session, identity, ACCOUNT_READERS + ("cardholder",))
    if user.role in ACCOUNT_READERS:
        return user

    # A cardholder is identified by email, never by customer id, so the claim is
    # matched against the customer this account actually registered as rather
    # than compared to the path.
    registered_as = user.customer
    if registered_as is None or registered_as.customer_id != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{identity} may only read their own account",
        )
    return user


def account_reader(
    customer_id: str,
    identity: str = Header(alias=IDENTITY_HEADER),
    session: Session = Depends(get_session),
) -> UserAccount:
    """Route guard: the asserted caller, authorised to read this one account."""
    # The header is an unverified claim injected by the API gateway, not a local
    # login: nothing here checks a signature, a token or a password. Verifying
    # who the caller is belongs upstream at the identity provider; this layer
    # only decides what the asserted identity may read.
    return authorise_account_read(session, identity, customer_id)
