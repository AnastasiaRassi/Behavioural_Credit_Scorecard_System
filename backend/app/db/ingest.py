"""Load the UCI Taiwan dataset into customer, monthly_statement and credit_outcome.

Run from the backend/ directory:

    python -m app.db.ingest

Reads the original .xls, not the cached CSV. The CSV that
`ml/feature_engineering.load_taiwan` maintains drops the source `ID` column, and
without it a customer could only be identified by row position, which stops being
a stable identity the moment the file is re-sorted or filtered.

Category codes are folded through `feature_engineering.clean()` — the same call
training makes — so the codes stored here and the codes the model was fitted on
cannot diverge. The CHECK constraints on `customer` are the second line of
defence: if that folding ever stopped, ingestion would fail loudly instead of
writing codes the pipeline has no level for.

Idempotent. Customers already present are left untouched, so rerunning adds only
what is missing and never rewrites a row a prediction points at.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from app.db.models import (
    MONTHS_OBSERVED,
    SOURCE_OBSERVED_PERIOD,
    CreditOutcome,
    Customer,
    MonthlyStatement,
)
from app.db.models.base import now
from app.ml.feature_engineering import TARGET, clean, project_root

# Rows per executemany. The full load is 30,000 customers and 180,000 statements;
# sending it as one statement works but holds the whole parameter list in memory
# twice, once in pandas and once in the driver.
BATCH_ROWS = 10_000

SOURCE_ID_COLUMN = "ID"


class SourceUnavailable(FileNotFoundError):
    pass


@dataclass
class IngestionReport:
    customers: int
    statements: int
    outcomes: int
    already_present: int

    def to_dict(self) -> dict:
        return asdict(self)


def source_path() -> Path:
    return project_root() / "data" / "raw" / "default of credit card clients.xls"


def read_source(path: Path | str | None = None) -> pd.DataFrame:
    """The dataset with its source IDs intact and its category codes folded.

    Separate from `feature_engineering.load_taiwan`, which returns the modelling
    frame and has no use for an identifier.
    """
    xls_path = Path(path) if path else source_path()
    if not xls_path.exists():
        raise SourceUnavailable(
            f"Dataset not found at {xls_path}. Download the .xls from "
            "https://archive.ics.uci.edu/dataset/350/ — ingestion needs the "
            "original file because the cached CSV has no ID column."
        )

    df = pd.read_excel(xls_path, header=1)
    df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})

    if SOURCE_ID_COLUMN not in df.columns:
        raise SourceUnavailable(
            f"{xls_path} has no {SOURCE_ID_COLUMN} column, so customers could "
            "only be keyed by row position. Re-download the original file."
        )

    # The identity stored in the database: the source's own integer id as text.
    df["customer_id"] = df[SOURCE_ID_COLUMN].astype(int).astype(str)
    return clean(df.drop(columns=[SOURCE_ID_COLUMN]))


def load_taiwan(session: Session, path: Path | str | None = None) -> IngestionReport:
    """Ingest the dataset into customer, monthly_statement and credit_outcome.

    Takes a session rather than opening one, so a caller can ingest inside a
    transaction it also rolls back.
    """
    frame = read_source(path)
    known = _existing_customer_ids(session)
    fresh = frame[~frame["customer_id"].isin(known)]

    if fresh.empty:
        return IngestionReport(0, 0, 0, already_present=len(frame))

    ingested_at = now()
    customers = _customer_rows(fresh, ingested_at)
    statements = _statement_rows(fresh)
    outcomes = _outcome_rows(fresh)

    # Customers first: the other two tables are foreign-keyed to them.
    _insert_batched(session, Customer, customers)
    _insert_batched(session, MonthlyStatement, statements)
    _insert_batched(session, CreditOutcome, outcomes)

    return IngestionReport(
        customers=len(customers),
        statements=len(statements),
        outcomes=len(outcomes),
        already_present=len(frame) - len(fresh),
    )


def _existing_customer_ids(session: Session) -> set[str]:
    return set(session.scalars(select(Customer.customer_id)).all())


def _customer_rows(frame: pd.DataFrame, ingested_at: datetime) -> list[dict]:
    return [
        {
            "customer_id": row.customer_id,
            "credit_limit": float(row.LIMIT_BAL),
            "sex": int(row.SEX),
            "age_years": int(row.AGE),
            "education": int(row.EDUCATION),
            "marriage": int(row.MARRIAGE),
            # Left null: a dataset entity has no login until someone registers.
            "user_account_id": None,
            "ingested_at": ingested_at,
        }
        for row in frame.itertuples(index=False)
    ]


def _statement_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.itertuples(index=False):
        for month in range(1, MONTHS_OBSERVED + 1):
            rows.append(
                {
                    "customer_id": row.customer_id,
                    "month_offset": month,
                    "repayment_status": int(getattr(row, f"PAY_{month}")),
                    "bill_amount": float(getattr(row, f"BILL_AMT{month}")),
                    "payment_amount": float(getattr(row, f"PAY_AMT{month}")),
                }
            )
    return rows


def _outcome_rows(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "customer_id": row.customer_id,
            "defaulted": bool(getattr(row, TARGET)),
            "observed_period": SOURCE_OBSERVED_PERIOD,
        }
        for row in frame.itertuples(index=False)
    ]


def _insert_batched(session: Session, model: type, rows: Sequence[dict]) -> None:
    for batch in _batches(rows, BATCH_ROWS):
        session.execute(insert(model), batch)


def _batches(rows: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> IngestionReport:
    from app.db.database import create_all, session_scope

    create_all()
    with session_scope() as session:
        report = load_taiwan(session)

    if report.customers:
        print(
            f"ingested       {report.customers} customers, "
            f"{report.statements} statements, {report.outcomes} outcomes"
        )
    else:
        print(f"nothing to add; {report.already_present} customers already present")
    return report


if __name__ == "__main__":
    main()
