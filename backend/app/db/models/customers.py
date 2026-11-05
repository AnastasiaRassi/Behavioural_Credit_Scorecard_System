"""The scoring subject: who is scored, their six months of history, and the label.

Normalised into three tables rather than the source file's one wide row. The
repayment history repeats six times per customer, so it is a child table keyed by
month; storing `PAY_1`..`PAY_6` as eighteen columns would make "the most recent
month" a column name rather than a value, and no query could ask for it.

`customer` is not `user_account`. A customer is an entity being scored and needs
no credentials; `user_account_id` is filled only if that cardholder registers an
account, which most never do. Staff have an account and no customer row.

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

from app.db.models.base import Base, now

# The observation window the scorecard was built on. Matches the six PAY_n /
# BILL_AMTn / PAY_AMTn groups in ml/feature_engineering.py; changing one without
# the other makes the stored history unreadable by the pipeline.
MONTHS_OBSERVED = 6

# The source data runs April to September 2005 and its target is a default in the
# month after, so every ingested label describes the same single period.
SOURCE_OBSERVED_PERIOD = "2005-10"


class Customer(Base):
    """A cardholder being scored, with the attributes the fairness audit needs.

    Keyed by the dataset's own ID so a row traces back to the source file.
    """

    __tablename__ = "customer"

    # The source dataset's ID ('1', '42'), never a row index: positions shift
    # when the file is re-sorted and would silently repoint every statement.
    customer_id: Mapped[str] = mapped_column(Text, primary_key=True)

    credit_limit: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)

    # Protected attributes. Stored because the fairness audit needs them, and
    # dropped inside CreditScorer.prepare() before anything reaches the model.
    # Codes are the published codebook's, folded to documented levels at ingest.
    sex: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    age_years: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    education: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    marriage: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # Null for everyone the system has not met, which is most of them. Unique, so
    # one account can never claim two customers.
    user_account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user_account.id"), unique=True, nullable=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    account: Mapped["UserAccount | None"] = relationship(  # noqa: F821
        back_populates="customer"
    )
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
        CheckConstraint("sex IN (1, 2)", name="sex_is_a_codebook_value"),
        CheckConstraint(
            "age_years BETWEEN 18 AND 120", name="age_years_is_plausible"
        ),
        CheckConstraint(
            "education IN (1, 2, 3, 4)", name="education_is_a_codebook_value"
        ),
        CheckConstraint("marriage IN (1, 2, 3)", name="marriage_is_a_codebook_value"),
    )

    @property
    def is_registered(self) -> bool:
        return self.user_account_id is not None


class MonthlyStatement(Base):
    """One month of a customer's billing history: status, bill and payment."""

    __tablename__ = "monthly_statement"

    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customer.customer_id", ondelete="CASCADE"), primary_key=True
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

    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customer.customer_id", ondelete="CASCADE"), primary_key=True
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
