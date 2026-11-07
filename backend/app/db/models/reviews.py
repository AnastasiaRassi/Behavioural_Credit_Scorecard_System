"""A cardholder's request for a person to look again at the model's rating."""
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base, new_id, now, sql_tuple

REVIEW_STATUSES = ("pending", "in_review", "resolved")
OPEN_REVIEW_STATUSES = ("pending", "in_review")


class ReviewRequest(Base):
    __tablename__ = "review_request"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("customer.id"), nullable=False
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )
    # The customer's own words, kept as typed.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")

    __table_args__ = (
        CheckConstraint(
            f"status IN {sql_tuple(REVIEW_STATUSES)}", name="status_is_known"
        ),
        CheckConstraint("length(trim(reason)) > 0", name="reason_is_present"),
        Index("index_review_request_customer_id", "customer_id"),
    )
