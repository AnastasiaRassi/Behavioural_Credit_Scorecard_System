"""Rebuild a cardholder's wide scoring frame from their stored statements.

The pipeline wants one row per customer with six months side by side; the
database stores one row per month. This module is the only place that converts
between the two, and it takes the target column names from
`ml/feature_engineering` rather than spelling them out, so a stored history
cannot drift from what `engineer()` expects.

Reads only. Nothing here scores, and nothing here writes.
"""
import uuid

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MONTHS_OBSERVED, Customer, MonthlyStatement
from app.ml.feature_engineering import (
    AMT_COLS,
    BILL_COLS,
    PAY_COLS,
    RAW_COLUMNS,
)

# Statement column -> the pipeline's column prefix for that reading.
PIVOTED_COLUMNS = {
    "repayment_status": "PAY_",
    "bill_amount": "BILL_AMT",
    "payment_amount": "PAY_AMT",
}

# The customer's attributes the pipeline reads, under the dataset's own names, with
# the type each one keeps. The limit is on `customer`, the four codes on
# `user_details`; they stay integral because the audit groups and labels by them.
CUSTOMER_COLUMNS = {"credit_limit": ("LIMIT_BAL", float)}
DETAIL_COLUMNS = {
    "sex": ("SEX", int),
    "age_years": ("AGE", int),
    "education": ("EDUCATION", int),
    "marriage": ("MARRIAGE", int),
}


class CustomerNotFound(LookupError):
    pass


class IncompleteHistory(RuntimeError):
    pass


def monthly_statements(session: Session, customer_id: uuid.UUID) -> pd.DataFrame:
    """A customer's statement rows in long form, most recent month first."""
    rows = session.scalars(
        select(MonthlyStatement)
        .where(MonthlyStatement.customer_id == customer_id)
        .order_by(MonthlyStatement.month_offset)
    ).all()

    if not rows:
        raise CustomerNotFound(
            f"No statements stored for customer {customer_id!r}. "
            "Run: python -m app.db.ingest"
        )

    return pd.DataFrame(
        [
            {
                "customer_id": row.customer_id,
                "month_offset": row.month_offset,
                "repayment_status": row.repayment_status,
                "bill_amount": row.bill_amount,
                "payment_amount": row.payment_amount,
            }
            for row in rows
        ]
    )


def wide_statements(session: Session, customer_id: uuid.UUID) -> pd.DataFrame:
    """The eighteen history columns for one customer, as a single-row frame."""
    long = monthly_statements(session, customer_id)
    _require_a_full_window(customer_id, long["month_offset"])

    wide = long.pivot(
        index="customer_id",
        columns="month_offset",
        values=list(PIVOTED_COLUMNS),
    )
    wide.columns = [
        f"{PIVOTED_COLUMNS[reading]}{month}" for reading, month in wide.columns
    ]
    return _typed(wide[[*PAY_COLS, *BILL_COLS, *AMT_COLS]])


def scoring_frame(
    session: Session, customer_id: uuid.UUID, include_protected: bool = False
) -> pd.DataFrame:
    """One customer as the pipeline's input row: the limit plus the pivoted history.

    `include_protected` adds the four audited attributes; the default leaves them
    out, which is the blind frame the officer screens are built from.
    """
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise CustomerNotFound(
            f"No customer {customer_id!r}. Run: python -m app.db.ingest"
        )

    frame = wide_statements(session, customer_id)
    for attribute, (column, kind) in CUSTOMER_COLUMNS.items():
        frame[column] = kind(getattr(customer, attribute))
    if include_protected:
        for attribute, (column, kind) in DETAIL_COLUMNS.items():
            frame[column] = kind(getattr(customer.details, attribute))

    # Column order follows RAW_COLUMNS so the frame lines up with the order
    # training used, whichever subset of columns was asked for.
    return frame[[column for column in RAW_COLUMNS if column in frame.columns]]


def raw_input_for(
    session: Session, customer_id: uuid.UUID, include_protected: bool = False
) -> dict:
    """The same row as a plain dict, for storing on a prediction."""
    frame = scoring_frame(session, customer_id, include_protected)
    # Column by column, not `.iloc[0].to_dict()`: that collapses the row to one
    # dtype and would store every repayment status as a float.
    return {column: frame[column].iloc[0].item() for column in frame.columns}


def _require_a_full_window(customer_id: uuid.UUID, months: pd.Series) -> None:
    expected = set(range(1, MONTHS_OBSERVED + 1))
    missing = sorted(expected - set(months.astype(int)))
    if missing:
        # Without this the pivot silently returns fewer columns and the failure
        # surfaces as a KeyError inside engineer(), far from its cause.
        raise IncompleteHistory(
            f"Customer {customer_id!r} has {len(months)} of {MONTHS_OBSERVED} "
            f"months; missing month_offset {missing}"
        )


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    # NUMERIC arrives as Decimal, which raises on division against the float
    # limit inside engineer(). Statuses stay integral because the features count
    # and difference them.
    out = frame.copy()
    out[PAY_COLS] = out[PAY_COLS].astype(int)
    out[[*BILL_COLS, *AMT_COLS]] = out[[*BILL_COLS, *AMT_COLS]].astype(float)
    return out
