"""Create a customer from columns submitted to the API, and link cardholder logins.

Ingestion covers the dataset; this covers the other way a customer appears — an
officer keying in someone who was never in the file. Both end at the same three
tables, so a walk-in is queryable by exactly the same helpers.

Writes only what was submitted. A customer that already exists is returned
unchanged: their stored statements came from the source file, and a re-score is
not grounds for rewriting the history the label was observed against. What was
actually submitted is kept on `prediction.raw_input` either way.
"""
import uuid
from typing import Sequence

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import MONTHS_OBSERVED, Customer, MonthlyStatement, UserAccount
from app.ml.feature_engineering import clean

# Walk-ins are prefixed so they can never collide with a source id, which is
# always bare digits.
WALK_IN_PREFIX = "W-"


def register_from_raw(
    session: Session, raw: dict, customer_id: str | None = None
) -> Customer:
    """Find or create the customer a submitted score is about."""
    if customer_id:
        existing = session.get(Customer, customer_id)
        if existing is not None:
            return existing
    else:
        customer_id = f"{WALK_IN_PREFIX}{uuid.uuid4().hex[:8]}"

    # Folded through the same call training uses, so the codes stored match the
    # levels the model was fitted on and the CHECK constraints accept them.
    folded = clean(pd.DataFrame([raw])).iloc[0]

    customer = Customer(
        customer_id=customer_id,
        credit_limit=float(folded["LIMIT_BAL"]),
        sex=int(folded["SEX"]),
        age_years=int(folded["AGE"]),
        education=int(folded["EDUCATION"]),
        marriage=int(folded["MARRIAGE"]),
    )
    session.add(customer)

    for month in range(1, MONTHS_OBSERVED + 1):
        session.add(
            MonthlyStatement(
                customer_id=customer_id,
                month_offset=month,
                repayment_status=int(folded[f"PAY_{month}"]),
                bill_amount=float(folded[f"BILL_AMT{month}"]),
                payment_amount=float(folded[f"PAY_AMT{month}"]),
            )
        )

    session.flush()
    return customer


def link_account(session: Session, customer: Customer, account: UserAccount) -> Customer:
    """Attach a registered cardholder's login to the customer they are."""
    if account.role != "cardholder":
        raise ValueError(
            f"{account.email} is a {account.role}; only a cardholder can be "
            "linked to a customer"
        )
    customer.user_account_id = account.id
    session.flush()
    return customer


def display_name_for(session: Session, customer_id: str) -> str | None:
    """The registered cardholder's name for a customer, or None if unregistered."""
    customer = session.get(Customer, customer_id)
    if customer is None or customer.account is None:
        return None
    return customer.account.display_name


def display_names_for(session: Session, customer_ids: Sequence[str]) -> dict[str, str]:
    """Names for several customers at once, omitting the unregistered ones."""
    if not customer_ids:
        return {}

    rows = (
        session.query(Customer.customer_id, UserAccount.display_name)
        .join(UserAccount, Customer.user_account_id == UserAccount.id)
        .filter(Customer.customer_id.in_(list(customer_ids)))
        .all()
    )
    return {customer_id: name for customer_id, name in rows}
