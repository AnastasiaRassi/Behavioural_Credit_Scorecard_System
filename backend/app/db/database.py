"""Engine and session handling.

The engine is created once per process and cached, like the model artifacts.
Sessions are per request: `session_scope()` commits on success and rolls back on
any exception, so a half-written decision never reaches the audit trail.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, text
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


class SchemaOutOfDate(RuntimeError):
    pass


# Columns that only exist in the pre-separation schema, where one user_account
# table was both the login and the scoring subject. `create_all` never alters an
# existing table, so a database carrying these would keep the old shape and fail
# on first query rather than at startup.
LEGACY_COLUMNS = (
    ("user_account", "full_name"),
    ("prediction", "subject_user_id"),
)


def _legacy_columns() -> list[str]:
    found = []
    with engine().connect() as connection:
        for table, column in LEGACY_COLUMNS:
            # Qualified by current_schema(): an identically named table in
            # another schema is not this application's and must not trip the
            # guard.
            present = connection.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = :table AND column_name = :column"
                ),
                {"table": table, "column": column},
            ).first()
            if present:
                found.append(f"{table}.{column}")
    return found


def create_all() -> None:
    """Create any missing tables, refusing to run against the pre-separation schema.

    Enough for a prototype with no schema history. A migration tool would be the
    answer if this schema had to evolve against data worth keeping.
    """
    stale = _legacy_columns()
    if stale:
        # Not migrated automatically: the subject of a prediction moved from a
        # user account to a customer, and no rewrite of the old rows can invent
        # the customer ids that were never stored.
        raise SchemaOutOfDate(
            f"Database still has the pre-separation schema ({', '.join(stale)}). "
            "It holds demo output only, so rebuild it: "
            "python -c \"from app.db.database import drop_all; drop_all()\" "
            "&& python -m app.db.ingest && python -m app.db.seed"
        )

    Base.metadata.create_all(engine())


def drop_all() -> None:
    """Drop every table this schema defines. Destructive; used to rebuild a prototype."""
    Base.metadata.drop_all(engine())
