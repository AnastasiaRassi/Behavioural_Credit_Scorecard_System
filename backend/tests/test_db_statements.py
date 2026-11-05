"""Checks for ingestion and the pivot back out to the pipeline's wide shape.

Two halves. The row builders are pure functions and are checked against the
source file with no database. The pivot needs rows, so it writes a customer into
a savepoint it rolls back, rather than depending on ingestion having been run.
"""
import uuid

import pytest

from app.db.database import _session_factory
from app.db.ingest import (
    SourceUnavailable,
    _customer_rows,
    _outcome_rows,
    _statement_rows,
    read_source,
    source_path,
)
from app.db.models import MONTHS_OBSERVED, Customer, MonthlyStatement
from app.db.models.base import now
from app.db.statements import (
    CustomerNotFound,
    IncompleteHistory,
    monthly_statements,
    raw_input_for,
    scoring_frame,
    wide_statements,
)
from app.ml.feature_engineering import AMT_COLS, BILL_COLS, PAY_COLS, RAW_COLUMNS
from tests.conftest import connect_or_skip

# The first row of the source file, which the published codebook documents.
FIRST_CUSTOMER = {
    "customer_id": "1",
    "LIMIT_BAL": 20000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_1": 2,
    "PAY_5": -2,
    "BILL_AMT1": 3913.0,
    "PAY_AMT2": 689.0,
}


@pytest.fixture(scope="module")
def source():
    if not source_path().exists():
        pytest.skip(f"source .xls not present at {source_path()}")
    return read_source()


@pytest.fixture(scope="module")
def session():
    connect_or_skip().close()

    active = _session_factory()()
    try:
        yield active
    finally:
        active.rollback()
        active.close()


@pytest.fixture
def stored(session):
    """One customer with a complete six-month history, rolled back afterwards."""
    savepoint = session.begin_nested()

    customer_id = f"TEST-{uuid.uuid4().hex[:8]}"
    session.add(
        Customer(
            customer_id=customer_id,
            credit_limit=FIRST_CUSTOMER["LIMIT_BAL"],
            sex=FIRST_CUSTOMER["SEX"],
            age_years=FIRST_CUSTOMER["AGE"],
            education=FIRST_CUSTOMER["EDUCATION"],
            marriage=FIRST_CUSTOMER["MARRIAGE"],
        )
    )
    # Distinct values per month, so a pivot that transposed or reversed the
    # window would not still pass.
    for month in range(1, MONTHS_OBSERVED + 1):
        session.add(
            MonthlyStatement(
                customer_id=customer_id,
                month_offset=month,
                repayment_status=month - 2,
                bill_amount=1000 * month,
                payment_amount=10 * month,
            )
        )
    session.flush()

    yield customer_id

    savepoint.rollback()


# ---- reading the source ------------------------------------------------------


def test_the_source_id_is_kept_as_the_customer_id(source):
    assert source["customer_id"].iloc[0] == "1"
    assert source["customer_id"].iloc[-1] == "30000"
    # Natural ids, not positions: there is no '0' and the count matches.
    assert source["customer_id"].nunique() == len(source) == 30_000


def test_undocumented_codes_are_folded_at_read(source):
    assert set(source["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(source["MARRIAGE"].unique()) <= {1, 2, 3}


def test_a_missing_source_says_what_to_do():
    with pytest.raises(SourceUnavailable) as raised:
        read_source("does-not-exist.xls")
    assert "archive.ics.uci.edu" in str(raised.value)


def test_customer_rows_carry_the_source_values(source):
    rows = {row["customer_id"]: row for row in _customer_rows(source, now())}
    first = rows["1"]
    assert first["credit_limit"] == FIRST_CUSTOMER["LIMIT_BAL"]
    assert first["sex"] == FIRST_CUSTOMER["SEX"]
    assert first["age_years"] == FIRST_CUSTOMER["AGE"]
    # Nobody is registered by ingestion.
    assert first["user_account_id"] is None


def test_statement_rows_are_six_per_customer_with_month_one_most_recent(source):
    rows = _statement_rows(source.head(3))
    assert len(rows) == 3 * MONTHS_OBSERVED

    first = {row["month_offset"]: row for row in rows if row["customer_id"] == "1"}
    assert sorted(first) == [1, 2, 3, 4, 5, 6]
    assert first[1]["repayment_status"] == FIRST_CUSTOMER["PAY_1"]
    assert first[5]["repayment_status"] == FIRST_CUSTOMER["PAY_5"]
    assert first[1]["bill_amount"] == FIRST_CUSTOMER["BILL_AMT1"]
    assert first[2]["payment_amount"] == FIRST_CUSTOMER["PAY_AMT2"]


def test_row_values_are_python_scalars_not_numpy(source):
    # psycopg does not adapt numpy scalars, so a leaked one fails at insert.
    row = _statement_rows(source.head(1))[0]
    assert type(row["repayment_status"]) is int
    assert type(row["bill_amount"]) is float


def test_outcome_rows_label_one_period(source):
    rows = _outcome_rows(source.head(5))
    assert {row["observed_period"] for row in rows} == {"2005-10"}
    assert all(isinstance(row["defaulted"], bool) for row in rows)


# ---- pivoting back out -------------------------------------------------------


def test_statements_come_back_in_month_order(session, stored):
    long = monthly_statements(session, stored)
    assert list(long["month_offset"]) == [1, 2, 3, 4, 5, 6]


def test_the_pivot_produces_the_pipeline_column_names(session, stored):
    wide = wide_statements(session, stored)
    assert list(wide.columns) == [*PAY_COLS, *BILL_COLS, *AMT_COLS]
    assert len(wide) == 1


def test_the_pivot_puts_each_month_in_its_own_column(session, stored):
    row = wide_statements(session, stored).iloc[0]
    for month in range(1, MONTHS_OBSERVED + 1):
        assert row[f"PAY_{month}"] == month - 2
        assert row[f"BILL_AMT{month}"] == 1000 * month
        assert row[f"PAY_AMT{month}"] == 10 * month


def test_money_arrives_as_float_not_decimal(session, stored):
    wide = wide_statements(session, stored)
    assert wide["BILL_AMT1"].dtype.kind == "f"
    assert wide["PAY_AMT1"].dtype.kind == "f"
    assert wide["PAY_1"].dtype.kind == "i"


def test_the_scoring_frame_is_blind_by_default(session, stored):
    frame = scoring_frame(session, stored)
    assert "LIMIT_BAL" in frame.columns
    assert not {"SEX", "AGE", "EDUCATION", "MARRIAGE"} & set(frame.columns)


def test_the_scoring_frame_can_include_the_audited_attributes(session, stored):
    frame = scoring_frame(session, stored, include_protected=True)
    assert list(frame.columns) == RAW_COLUMNS
    assert frame.iloc[0]["AGE"] == FIRST_CUSTOMER["AGE"]


def test_the_frame_goes_straight_through_the_pipeline(session, stored):
    from app.ml.feature_engineering import engineer

    features = engineer(scoring_frame(session, stored))
    assert len(features) == 1
    assert "pay_max" in features.columns
    assert features["pay_max"].iloc[0] == 4  # month 6 is 6 - 2


def test_raw_input_is_a_plain_dict_that_keeps_its_types(session, stored):
    raw = raw_input_for(session, stored)
    assert set(raw) == set(scoring_frame(session, stored).columns)
    assert isinstance(raw["LIMIT_BAL"], float)
    # A status stored as 2.0 would read as a float all the way into the JSONB.
    assert isinstance(raw["PAY_1"], int)


def test_the_audited_attributes_stay_integral(session, stored):
    frame = scoring_frame(session, stored, include_protected=True)
    assert frame["AGE"].dtype.kind == "i"
    assert frame["EDUCATION"].dtype.kind == "i"


def test_an_incomplete_window_fails_loudly(session, stored):
    savepoint = session.begin_nested()
    session.execute(
        MonthlyStatement.__table__.delete().where(
            MonthlyStatement.customer_id == stored,
            MonthlyStatement.month_offset == 4,
        )
    )
    session.flush()

    # Silently pivoting to 15 columns would surface as a KeyError inside
    # engineer(), far from the cause.
    with pytest.raises(IncompleteHistory) as raised:
        scoring_frame(session, stored)
    assert "[4]" in str(raised.value)

    savepoint.rollback()


def test_an_unknown_customer_says_what_to_run(session):
    with pytest.raises(CustomerNotFound) as raised:
        scoring_frame(session, "TEST-not-ingested")
    assert "app.db.ingest" in str(raised.value)
