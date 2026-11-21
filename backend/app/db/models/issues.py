"""Problems and feedback sent from the app's Report an Issue page.

One row per submission, from any signed-in role. `details` is the person's own
words, kept as typed; nothing reads it back yet, so it is only a place to triage from.
"""
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base, new_id, now, sql_tuple

ISSUE_KINDS = ("Something is broken", "Something looks wrong", "Feedback or idea")


class IssueReport(Base):
    __tablename__ = "issue_report"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(f"kind IN {sql_tuple(ISSUE_KINDS)}", name="kind_is_known"),
        CheckConstraint("length(trim(details)) > 0", name="details_are_present"),
        Index("index_issue_report_created_at", "created_at"),
    )
