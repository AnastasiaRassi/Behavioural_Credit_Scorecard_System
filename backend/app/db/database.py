"""Engine, session handling, and creating or dropping the schema.

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


# Columns that only exist in an earlier schema: one where a user_account was both
# the login and the scoring subject, and the one where customer was keyed by the
# dataset's text id with a mostly-null user_account_id. `create_all` never alters an
# existing table, so a database carrying these would keep the old shape and fail
# on first query rather than at startup.
LEGACY_COLUMNS = (
    ("user_account", "full_name"),
    ("prediction", "subject_user_id"),
    ("customer", "customer_id"),
    ("customer", "user_account_id"),
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


# Columns added to a table after it first shipped. `create_all` never alters an existing
# table, so a database created earlier would lack them and every insert would fail.
# IF NOT EXISTS makes the step safe to repeat; Postgres only, like the rest of the schema.
ADDED_COLUMNS = (
    ("chat_turn", "grounding_passed", "boolean"),
    ("chat_turn", "claim_passed", "boolean"),
    ("prediction", "scored_by_user_id", "uuid REFERENCES users(id)"),
)


def _add_missing_columns() -> None:
    with engine().begin() as connection:
        for table, column, sql_type in ADDED_COLUMNS:
            connection.execute(
                text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {sql_type}")
            )


def create_all() -> None:
    """Create missing tables and columns, refusing to run against an earlier schema.

    Columns added since a table first shipped are listed in `ADDED_COLUMNS`. Anything
    bigger needs a rebuild; a migration tool would replace this if the schema had to
    evolve against data worth keeping.
    """
    stale = _legacy_columns()
    if stale:
        # Not migrated automatically: keys changed type and every customer gained a
        # user, and no rewrite of the old rows can invent the ids that were never
        # stored.
        raise SchemaOutOfDate(
            f"Database still has an earlier schema ({', '.join(stale)}). "
            "It holds demo output only, so rebuild it: "
            "python -c \"from app.db.database import drop_all; drop_all()\" "
            "&& python -m app.db.ingest && python -m app.db.seed"
        )

    Base.metadata.create_all(engine())
    _add_missing_columns()


# Tables an earlier schema had and this one does not, dropped too so a rebuild leaves
# nothing behind. CASCADE because the old customer and decision tables point at it.
LEGACY_TABLES = ("user_account",)


def drop_all() -> None:
    """Drop every table this schema defines, and any an earlier one left.

    Destructive; used to rebuild a prototype.
    """
    with engine().begin() as connection:
        for table in LEGACY_TABLES:
            connection.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
    Base.metadata.drop_all(engine())
