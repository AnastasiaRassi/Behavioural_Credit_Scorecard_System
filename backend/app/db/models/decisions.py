"""The audit trail: what an officer did about a score, and what then happened.

Both tables hang off a prediction rather than off a customer, because a decision
is only defensible against the number that was on screen when it was taken.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, new_id, now, sql_tuple

# A behavioural scorecard scores customers who already hold the card, so there is
# nothing to approve or decline. The officer chooses an intervention on an
# existing account.
DECISION_ACTIONS = ("no_action", "monitor", "contact_customer", "reduce_limit")


class Decision(Base):
    """An officer's intervention on a scored account, with the justification kept.

    Separate from Prediction because one score can be acted on more than once
    (monitored, then escalated to contact), and a re-score may carry no decision
    at all. Whether the officer overrode the model is derived by comparing the
    action against the prediction's flagged value, never stored, so it cannot
    contradict the row it describes.
    """

    __tablename__ = "decision"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction.id"), nullable=False, index=True
    )
    decided_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user_account.id"), nullable=False, index=True
    )
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    action: Mapped[str] = mapped_column(Text, nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)

    prediction: Mapped["Prediction"] = relationship(  # noqa: F821
        back_populates="decisions"
    )
    decided_by: Mapped["UserAccount"] = relationship(  # noqa: F821
        back_populates="decisions"
    )

    __table_args__ = (
        CheckConstraint(
            f"action IN {sql_tuple(DECISION_ACTIONS)}", name="action_is_known"
        ),
        # Required on every decision, not only overrides: an audit trail with holes
        # is harder to defend than one that is always complete.
        CheckConstraint(
            "length(trim(justification)) > 0", name="justification_is_present"
        ),
    )

    @property
    def overrode_model(self) -> bool:
        """Whether the officer acted against what the score suggested.

        Derived rather than stored, so it cannot drift from the rows it
        describes. A flagged account left alone is an override; so is an
        intervention on an account the model did not flag. Monitoring sits in
        between and counts as agreement either way.
        """
        if self.prediction.flagged:
            return self.action == "no_action"
        return self.action in ("contact_customer", "reduce_limit")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "prediction_id": str(self.prediction_id),
            "decided_at": self.decided_at.isoformat(),
            "action": self.action,
            "justification": self.justification,
            "overrode_model": self.overrode_model,
        }


class ObservedOutcome(Base):
    """Whether a customer defaulted after the system scored them.

    The only route to measuring production performance or recomputing fairness,
    both of which need labels. Keyed per prediction, not per customer, which is
    what makes it distinct from `credit_outcome`: that holds the dataset's
    training label, this holds what happened after a live score.

    Drop this table and the monitoring tab can still show input drift (PSI over
    features) but never performance drift.
    """

    __tablename__ = "observed_outcome"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    # Unique: one truth per prediction.
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction.id"), unique=True, nullable=False
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    defaulted: Mapped[bool] = mapped_column(Boolean, nullable=False)

    prediction: Mapped["Prediction"] = relationship(  # noqa: F821
        back_populates="observed_outcome"
    )
