"""The scoring subject: who is scored, their six months of history, and the label.

Normalised into three tables rather than the source file's one wide row. The
repayment history repeats six times per customer, so it is a child table keyed by
month; storing `PAY_1`..`PAY_6` as eighteen columns would make "the most recent
month" a column name rather than a value, and no query could ask for it.

A customer is a person (`users`, role 'cardholder', with a `user_details` row) and
`customer` is what is scored about them: the limit, the history and the label.
`user_id` is therefore never null. Most customers have no login; staff have a user
and no customer row.

The wide shape the model wants is rebuilt on read by db/statements.py.
"""
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Numeric,
    SmallInteger,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, new_id, now

# The observation window the scorecard was built on. Matches the six PAY_n /
# BILL_AMTn / PAY_AMTn groups in ml/feature_engineering.py; changing one without
# the other makes the stored history unreadable by the pipeline.
MONTHS_OBSERVED = 6

# The source data runs April to September 2005 and its target is a default in the
# month after, so every ingested label describes the same single period.
SOURCE_OBSERVED_PERIOD = "2005-10"


class Customer(Base):
    """A cardholder being scored.

    Keyed by a UUID like every other table. The dataset's own ID survives as
    `account_number`, so a row still traces back to the source file.
    """

    __tablename__ = "customer"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )

    # The source dataset's ID ('1', '42') or a walk-in's 'W-...', never a row
    # index: positions shift when the file is re-sorted and would silently
    # repoint every statement. What an officer types, and what the ML queue
    # artifact is keyed on.
    account_number: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    # The person. Unique, so one user can never be two customers.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"), unique=True, nullable=False
    )
    credit_limit: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    user: Mapped["User"] = relationship(back_populates="customer")  # noqa: F821
    statements: Mapped[list["MonthlyStatement"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        order_by="MonthlyStatement.month_offset",
    )
    outcome: Mapped["CreditOutcome | None"] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(  # noqa: F821
        back_populates="subject"
    )

    __table_args__ = (
        CheckConstraint("credit_limit > 0", name="credit_limit_is_positive"),
        CheckConstraint(
            "length(trim(account_number)) > 0", name="account_number_is_present"
        ),
    )

    @property
    def details(self) -> "UserDetails":  # noqa: F821
        return self.user.details

    @property
    def display_name(self) -> str:
        return self.user.display_name

    @property
    def is_registered(self) -> bool:
        """Whether this customer has a login."""
        return self.user.can_sign_in


class MonthlyStatement(Base):
    """One month of a customer's billing history: status, bill and payment."""

    __tablename__ = "monthly_statement"

    customer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("customer.id", ondelete="CASCADE"), primary_key=True
    )

    # 1 is the most recent month, matching the source's PAY_1/BILL_AMT1 suffix, so
    # the pivot in db/statements.py is a rename and not a reordering.
    month_offset: Mapped[int] = mapped_column(SmallInteger, primary_key=True)

    # -2 no balance, -1 paid in full, 0 revolving, 1..9 months in arrears.
    repayment_status: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # No lower bound: a bill is negative when the account is in credit.
    bill_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    payment_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="statements")

    __table_args__ = (
        CheckConstraint(
            f"month_offset BETWEEN 1 AND {MONTHS_OBSERVED}",
            name="month_offset_is_within_the_window",
        ),
        CheckConstraint(
            "repayment_status BETWEEN -2 AND 9",
            name="repayment_status_is_a_codebook_value",
        ),
        CheckConstraint(
            "payment_amount >= 0", name="payment_amount_is_not_negative"
        ),
    )


class CreditOutcome(Base):
    """The dataset's ground-truth label: did this customer default in the period.

    Training input, not system output. Distinct from `observed_outcome`, which
    records what happened after a live prediction; joining this into a frame
    alongside a stored prediction would be target leakage.
    """

    __tablename__ = "credit_outcome"

    customer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("customer.id", ondelete="CASCADE"), primary_key=True
    )
    defaulted: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Which period the label describes, so a second dataset cannot be ingested
    # on top of the first and have its labels read as the same thing.
    observed_period: Mapped[str] = mapped_column(Text, nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="outcome")

    __table_args__ = (
        CheckConstraint(
            "length(trim(observed_period)) > 0", name="observed_period_is_present"
        ),
    )
