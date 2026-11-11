"""What the system produced: which model was live, and what it said about whom.

A prediction points at a `customer`, not at a `users` row. The subject of a score
is the entity being scored, which usually has no login at all, so keying this on
the login identity would tie every score to credentials that mostly do not exist.

Metrics are not stored. Production performance is a join over prediction and
observed_outcome, computed on demand; training metrics live in
artifacts/scoring_config.json, which stays the single source of truth for them.
"""
import uuid
from datetime import datetime

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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, new_id, now


class ModelVersion(Base):
    """Identity of a trained model, so every prediction says what produced it.

    Identity only. Metrics, thresholds and fairness results stay in artifacts/,
    which train.py writes and everything else reads. Duplicating them here would
    create a second source of truth able to disagree with the served model.
    """

    __tablename__ = "model_version"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    trained_on: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_live: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
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
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    subject_customer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("customer.id"), nullable=False, index=True
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_version.id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    # The officer who ran this score. Null for scores the system produced itself
    # (the overnight queue, and rows from before this was recorded).
    scored_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )

    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str] = mapped_column(Text, nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # The operating point in force when this score was produced. Denormalised
    # because the threshold is tunable, so a past decision must not be re-read
    # against a later value.
    threshold: Mapped[float] = mapped_column(Float, nullable=False)

    # The raw columns as submitted. engineer() reproduces the 46 features from
    # these deterministically, so storing the engineered frame would duplicate
    # derivable data and diverge the day engineer() changes.
    raw_input: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # The contributions shown to the user, stored rather than recomputed: train.py
    # prunes artifacts/, so an earlier model's binary does not survive a retrain
    # and a past explanation cannot be reproduced from raw_input alone.
    explanation: Mapped[dict] = mapped_column(JSONB, nullable=False)

    subject: Mapped["Customer"] = relationship(  # noqa: F821
        back_populates="predictions"
    )
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    decisions: Mapped[list["Decision"]] = relationship(  # noqa: F821
        back_populates="prediction"
    )
    observed_outcome: Mapped["ObservedOutcome | None"] = relationship(  # noqa: F821
        back_populates="prediction"
    )

    __table_args__ = (
        CheckConstraint(
            "probability >= 0.0 AND probability <= 1.0", name="probability_is_a_rate"
        ),
        Index("index_prediction_created_at", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "customer_id": str(self.subject_customer_id),
            # Key stays "scored_at": it is the published API field name, and the
            # screens read it.
            "scored_at": self.created_at.isoformat(),
            "probability": self.probability,
            "risk_band": self.risk_band,
            "threshold": self.threshold,
            "flagged": self.flagged,
            "explanation": self.explanation,
        }
