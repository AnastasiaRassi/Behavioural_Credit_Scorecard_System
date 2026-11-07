"""Shared request dependencies: the scorer, the explainer, a session, the caller.

The scorer and explainer are process-wide and cached, since loading the ensemble
costs far more than a request should. The session is per request.

Authentication and authorization both live here, and are kept apart:
`signed_in_user` establishes *who* the caller is from the bearer token
`POST /session` minted, and `authorise_account_read` decides what that caller
may read. Each rule is written once, so a route never decides it for itself.
"""
import uuid
from functools import lru_cache
from typing import Iterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core import sessions
from app.db.database import _session_factory
from app.db.models import Customer, ModelVersion, Prediction, User
from app.ml.counterfactuals import CounterfactualExplainer
from app.ml import worklist
from app.ml.explainability import Explainer
from app.ml.scorer import ArtifactsMissing, CreditScorer

BEARER_PREFIX = "Bearer "

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


@lru_cache(maxsize=1)
def get_recourse_explainer() -> CounterfactualExplainer:
    """Shares the scorer: a counterfactual search runs the model many times."""
    return CounterfactualExplainer(get_scorer())


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
) -> User:
    """The account behind an asserted email: 404 if unknown, 403 if the wrong role."""
    allowed = (expected_role,) if isinstance(expected_role, str) else expected_role

    user = session.query(User).filter_by(email=email).one_or_none()
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


def _unauthenticated(detail: str) -> HTTPException:
    # Declared optional above and refused here, rather than required by FastAPI:
    # a missing credential is a 401 with a challenge, not a 422 about a header.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def signed_in_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    session: Session = Depends(get_session),
) -> User:
    """The account a presented bearer token belongs to, or 401.

    Authentication only: what the caller may then reach is decided per resource.
    """
    if authorization is None or not authorization.startswith(BEARER_PREFIX):
        raise _unauthenticated('Authorization header must read "Bearer <token>"')

    try:
        identity = sessions.identity_from(authorization[len(BEARER_PREFIX) :])
    except sessions.InvalidSession as exception:
        raise _unauthenticated(str(exception)) from exception

    # The signature proves this service minted the token, so the email in it can
    # be trusted. The account is still re-read rather than carried in the token,
    # so a changed role or a deleted account takes effect on the next request
    # instead of at the next expiry.
    account = session.query(User).filter_by(email=identity).one_or_none()
    if account is None:
        raise _unauthenticated("This session belongs to an account that no longer exists")
    return account


def customer_or_404(session: Session, customer_id: str) -> Customer:
    """The customer behind a path id. A malformed id names nobody, so it is a 404 too."""
    try:
        parsed = uuid.UUID(customer_id)
    except (ValueError, AttributeError, TypeError):
        parsed = None

    customer = None if parsed is None else session.get(Customer, parsed)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No customer {customer_id}",
        )
    return customer


def queued_account(session: Session, customer_id: str) -> tuple[Customer, dict]:
    """A customer and their queue row, the row carrying the customer's UUID.

    The queue artifact is keyed by account number and knows nothing of UUIDs, so
    this is the one place the two are joined.
    """
    customer = customer_or_404(session, customer_id)
    row = worklist.find(customer.account_number)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No queued customer {customer_id}",
        )
    return customer, {**row, "customer_id": str(customer.id)}


def authorise_account_read(user: User, customer_id: str) -> User:
    """Who may read one cardholder's account: that cardholder, or any staff reader.

    The only place this rule is written. Takes the resolved account, so changing
    how callers are authenticated does not touch the rule.
    """
    if user.role in ACCOUNT_READERS:
        return user

    # A cardholder is identified by email, never by customer id, so the caller is
    # matched against the customer this account actually registered as rather
    # than compared to the path.
    registered_as = user.customer
    if registered_as is None or str(registered_as.id) != customer_id.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{user.email} may only read their own account",
        )
    return user


def account_reader(
    customer_id: str, user: User = Depends(signed_in_user)
) -> User:
    """Route guard: the signed-in caller, authorised to read this one account."""
    return authorise_account_read(user, customer_id)
