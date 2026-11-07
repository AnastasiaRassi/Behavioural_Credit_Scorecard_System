"""What the cardholder assistant said, and how: one row per answered chat turn.

The LLM equivalent of `prediction`. Every reply is stored with the model, the prompt
version and the scoring-model version that produced it, so a bad answer can be traced
and a change to any of them can be compared against the rows before it.

`user_message` is personal data and is kept as typed. No retention limit is enforced.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base, new_id, now


class ChatTurn(Base):
    __tablename__ = "chat_turn"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    # A real foreign key: the chat resolves its customer through the database, so a
    # turn always has one. A failed log write is still swallowed (db/chat_log.py).
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("customer.id"), nullable=False
    )
    # 0 is the assistant's opening message, which answers no question.
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Empty for the opening (turn 0).
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    reply: Mapped[str] = mapped_column(Text, nullable=False)

    # "model" when the LLM's reply was shown, "template" when the fallback was.
    source: Mapped[str] = mapped_column(Text, nullable=False)
    # Why the template was used; null exactly when the model answered.
    fallback_reason: Mapped[str | None] = mapped_column(Text)
    # The LLM output that was discarded, kept because it is the most useful row to read.
    rejected_reply: Mapped[str | None] = mapped_column(Text)

    # Each guard's verdict on the LLM's text. Null when the LLM produced none (it
    # failed), so a null means "not checked", never "passed".
    grounding_passed: Mapped[bool | None] = mapped_column(Boolean)
    claim_passed: Mapped[bool | None] = mapped_column(Boolean)

    llm_model: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    scoring_model_version: Mapped[str] = mapped_column(Text, nullable=False)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        CheckConstraint("source IN ('model', 'template')", name="source_is_known"),
        CheckConstraint(
            "(source = 'model') = (fallback_reason IS NULL)",
            name="fallback_reason_iff_template",
        ),
        Index("index_chat_turn_created_at", "created_at"),
        Index("index_chat_turn_customer_id", "customer_id"),
    )
