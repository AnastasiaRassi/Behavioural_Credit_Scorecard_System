"""Engine and session handling.

The engine is created once per process and cached, like the model artifacts.
Sessions are per request: `session_scope()` commits on success and rolls back on
any exception, so a half-written decision never reaches the audit trail.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import database_url
from app.db.models import Base


@lru_cache(maxsize=1)
def engine() -> Engine:
    # pool_pre_ping because a local Postgres is routinely stopped and started
    # between sittings, which otherwise surfaces as a stale-connection error on
    # the first query rather than a reconnect.
    return create_engine(database_url(), pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """Create any missing tables.

    Enough for a prototype with no schema history. A migration tool would be the
    answer if this schema had to evolve against data worth keeping.
    """
    Base.metadata.create_all(engine())
