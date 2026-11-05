"""People who can sign in: credentials, role and the name a human goes by.

Deliberately not the scoring subject. A row here is someone the system has met
and holds credentials for; the 30,000 dataset entities are not people with
logins and live in `customer`. Staff never appear in `customer` at all, and a
cardholder only reaches an account by registering, so the two tables cannot be
confused for one another.

UUID keys rather than serial integers: these identifiers reach cardholder-facing
URLs, where a sequence would leak volume and allow enumeration.
"""
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, new_id, now, sql_tuple

# The system's own role vocabulary. 'officer' is the loan officer; 'engineer'
# opens the model report, which no cardholder or officer screen exposes.
ROLES = ("officer", "engineer", "cardholder")


class UserAccount(Base):
    """An authenticable account: credentials, role, and who the person is."""

    __tablename__ = "user_account"  # "user" is reserved in Postgres.

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    # The real human identity: a staff member's name, or the name a cardholder
    # registered under. Never a customer id — that lives in `customer`, which is
    # what this account links to.
    display_name: Mapped[str] = mapped_column(Text, nullable=False)

    # Only ever a hash. core/passwords.py produces it and is the only module
    # that handles the plaintext.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    # One-to-one and optional: set only for a cardholder who registered.
    customer: Mapped["Customer | None"] = relationship(  # noqa: F821
        back_populates="account"
    )
    decisions: Mapped[list["Decision"]] = relationship(  # noqa: F821
        back_populates="decided_by"
    )

    __table_args__ = (
        CheckConstraint(f"role IN {sql_tuple(ROLES)}", name="role_is_known"),
        # An empty hash would make an empty password verify against anything a
        # future login path compared loosely.
        CheckConstraint(
            "length(password_hash) > 0", name="password_hash_is_present"
        ),
        CheckConstraint("length(trim(display_name)) > 0", name="display_name_is_present"),
    )

    def to_dict(self) -> dict:
        # password_hash is omitted on purpose: this feeds API responses.
        return {
            "id": str(self.id),
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
        }
