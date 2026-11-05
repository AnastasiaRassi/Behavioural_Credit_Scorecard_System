"""Shared database guards.

The suite runs against a local Postgres when one is there and skips when it is
not, so a machine without a database still runs everything else. A database on
the pre-separation schema is skipped the same way, with the command that fixes it.
"""
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db.database import engine
from app.db.models import Customer, MonthlyStatement
from app.db.queue_ids import source_id_for

REBUILD = (
    'python -c "from app.db.database import drop_all; drop_all()" '
    "&& python -m app.db.ingest && python -m app.db.seed"
)


def skip_unless_current(connection: Connection) -> None:
    """Skip the caller unless the connected schema is the current one."""
    present = connection.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        )
    ).scalars()
    missing = {"customer", "monthly_statement", "credit_outcome"} - set(present)
    if missing:
        pytest.skip(f"database is missing {sorted(missing)}; rebuild it: {REBUILD}")


def connect_or_skip() -> Connection:
    """A live connection on the current schema, or a skipped test."""
    try:
        connection = engine().connect()
        connection.execute(text("select 1"))
    except OperationalError as exception:
        pytest.skip(f"no database reachable: {exception}")

    skip_unless_current(connection)

    # The probes autobegan a transaction. Returned open, it would collide with a
    # caller that opens its own, which is how every API fixture here works.
    connection.rollback()
    return connection


def ensure_queued_customer(session: Session, row: dict) -> str:
    """Give a worklist row the customer row ingestion would have created.

    Lets the API tests run against a database nobody has ingested, and exercises
    the queue-id mapping on the way: the figures are taken from the queue row, so
    the bridge's own verification has to agree with them.
    """
    customer_id = source_id_for(row["customer_id"])
    if session.get(Customer, customer_id) is not None:
        return customer_id

    session.add(
        Customer(
            customer_id=customer_id,
            credit_limit=row["credit_limit"],
            # The queue is built blind, so it carries no protected attributes.
            # Valid placeholders: these tests are about the officer flow, and a
            # fabricated demographic must not read as the dataset's own.
            sex=1,
            age_years=30,
            education=4,
            marriage=3,
        )
    )
    for entry in row["history"]:
        session.add(
            MonthlyStatement(
                customer_id=customer_id,
                month_offset=entry["month"],
                repayment_status=entry["payment_status"],
                bill_amount=entry["bill_amount"],
                payment_amount=entry["amount_paid"],
            )
        )
    session.flush()
    return customer_id
