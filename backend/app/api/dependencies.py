"""Shared request dependencies: the scorer, the explainer and a session.

The scorer and explainer are process-wide and cached, since loading the ensemble
costs far more than a request should. The session is per request.
"""
from functools import lru_cache
from typing import Iterator

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import _session_factory
from app.db.models import ModelVersion, User
from app.ml.explainability import Explainer
from app.ml.scorer import ArtifactsMissing, CreditScorer


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


def user_by_email(session: Session, email: str, expected_role: str) -> User:
    user = session.query(User).filter_by(email=email).one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown user {email}"
        )
    if user.role != expected_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{email} is a {user.role}, not a {expected_role}",
        )
    return user
