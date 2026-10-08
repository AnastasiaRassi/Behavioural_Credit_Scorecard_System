"""What the system produces: scores, the reasons shown, and the decisions taken.

The dataset, the splits and the model artifacts are files and stay files. This
schema holds outputs only, so losing the database loses no training input and
nothing here needs regenerating to reproduce a published number.

Rows here must never be fed back into training. A scored row carries the model's
own output, and its outcome is observed later than the training window, so
joining this into a training frame is both target leakage and a time violation.

Metrics are not stored. Production performance is a join over prediction and
observed_outcome, computed on demand; training metrics live in
artifacts/scoring_config.json, which stays the single source of truth for them.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

ROLES = ("officer", "engineer", "cardholder")

# A behavioural scorecard scores customers who already hold the card, so there is
# nothing to approve or decline. The officer chooses an intervention on an
# existing account.
DECISION_ACTIONS = ("no_action", "monitor", "contact_customer", "reduce_limit")


def _new_id() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sql_tuple(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


class Base(DeclarativeBase):
    pass


class User(Base):
    """Everyone the system knows: officers, engineers and cardholders.

    One table rather than separate user and customer tables. Only cardholders
    actually scored through the application appear here, not the 30,000 dataset
    rows, so the table stays small. Email is nullable because most cardholders
    never sign in; such a row gains credentials only if the person does.

    UUID keys rather than serial integers: these identifiers reach
    cardholder-facing URLs, where a sequence would leak volume and allow
    enumeration.
    """

    __tablename__ = "user_account"  # "user" is reserved in Postgres.

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    display_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="subject")

    __table_args__ = (
        CheckConstraint(f"role IN {_sql_tuple(ROLES)}", name="role_is_known"),
    )


class ModelVersion(Base):
    """Identity of a trained model, so every prediction says what produced it.

    Identity only. Metrics, thresholds and fairness results stay in artifacts/,
    which train.py writes and everything else reads. Duplicating them here would
    create a second source of truth able to disagree with the served model.
    """

    __tablename__ = "model_version"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    version: Mapped[str] = mapped_column(String(64), unique=True)
    trained_on: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model_version"
    )

    __table_args__ = (
        # At most one live model, enforced by the database rather than by convention.
        Index(
            "index_one_live_model_version",
            "is_live",
            unique=True,
            postgresql_where="is_live",
        ),
    )


class Prediction(Base):
    """One scoring event, with the explanation shown alongside it."""

    __tablename__ = "prediction"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    subject_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user_account.id"), index=True
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_version.id"), index=True
    )
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
    flagged: Mapped[bool] = mapped_column(Boolean)

    # The operating point in force when this score was produced. Denormalised
    # because the threshold is tunable, so a past decision must not be re-read
    # against a later value.
    threshold: Mapped[float] = mapped_column(Float)

    # The raw columns as submitted. engineer() reproduces the 46 features from
    # these deterministically, so storing the engineered frame would duplicate
    # derivable data and diverge the day engineer() changes.
    raw_input: Mapped[dict] = mapped_column(JSONB)

    # The contributions shown to the user, stored rather than recomputed: train.py
    # prunes artifacts/, so an earlier model's binary does not survive a retrain
    # and a past explanation cannot be reproduced from raw_input alone.
    explanation: Mapped[dict] = mapped_column(JSONB)

    subject: Mapped["User"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="prediction")
    observed_outcome: Mapped["ObservedOutcome | None"] = relationship(
        back_populates="prediction"
    )

    __table_args__ = (
        CheckConstraint(
            "probability >= 0.0 AND probability <= 1.0", name="probability_is_a_rate"
        ),
        Index("index_prediction_scored_at", "scored_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "scored_at": self.scored_at.isoformat(),
            "probability": self.probability,
            "risk_band": self.risk_band,
            "threshold": self.threshold,
            "flagged": self.flagged,
            "explanation": self.explanation,
        }


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
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction.id"), index=True
    )
    decided_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user_account.id"), index=True
    )
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    action: Mapped[str] = mapped_column(String(32))
    justification: Mapped[str] = mapped_column(Text)

    prediction: Mapped["Prediction"] = relationship(back_populates="decisions")

    __table_args__ = (
        CheckConstraint(
            f"action IN {_sql_tuple(DECISION_ACTIONS)}", name="action_is_known"
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
    """Whether the customer actually defaulted the following month.

    The target of a behavioural scorecard, and the only route to measuring
    production performance or recomputing fairness, both of which need labels.
    Named observed_outcome to stay distinct from Decision.action, which is what a
    human chose to do rather than what actually happened.

    Drop this table and the monitoring tab can still show input drift (PSI over
    features) but never performance drift.
    """

    __tablename__ = "observed_outcome"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    # Unique: one truth per prediction.
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction.id"), unique=True
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    defaulted: Mapped[bool] = mapped_column(Boolean)

    prediction: Mapped["Prediction"] = relationship(back_populates="observed_outcome")
