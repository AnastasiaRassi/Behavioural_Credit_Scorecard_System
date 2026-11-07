"""Everyone the system knows as a person: staff, and every customer.

`users` is the identity table. A customer is a user with role 'cardholder', so
`customer.user_id` is never null. Whether that person can sign in is a property of
the row, not of the table: staff always can, a cardholder only once they register
an email and password, and until then `email` and `password_hash` are both null.

`user_details` holds what is known about a customer as a person (the audited
demographics) and exists only for cardholders. Staff have no row there. Names are
not duplicated into it: `users.display_name` is the one name, for everybody.

UUID keys rather than serial integers: these identifiers reach cardholder-facing
URLs, where a sequence would leak volume and allow enumeration. The table is
`users`, not `user`, because `user` is reserved in Postgres and a bare
`SELECT * FROM user` returns the database role instead of rows.
"""
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, SmallInteger, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, new_id, now, sql_tuple

# The system's own role vocabulary. 'officer' is the loan officer; 'engineer'
# opens the model report, which no cardholder or officer screen exposes.
ROLES = ("officer", "engineer", "cardholder")


class User(Base):
    """A person: credentials if they can sign in, a role, and the name they go by."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_id
    )

    # Null for a customer who has not registered. Unique, and Postgres allows many
    # nulls, so thousands of unregistered customers do not collide.
    email: Mapped[str | None] = mapped_column(Text, unique=True)

    # The one name: a staff member's, or the customer's. Never an account number.
    display_name: Mapped[str] = mapped_column(Text, nullable=False)

    # Only ever a hash. core/passwords.py produces it and is the only module
    # that handles the plaintext.
    password_hash: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now
    )

    # Cardholders only: one customer and one details row each.
    customer: Mapped["Customer | None"] = relationship(  # noqa: F821
        back_populates="user"
    )
    details: Mapped["UserDetails | None"] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    decisions: Mapped[list["Decision"]] = relationship(  # noqa: F821
        back_populates="decided_by"
    )

    __table_args__ = (
        CheckConstraint(f"role IN {sql_tuple(ROLES)}", name="role_is_known"),
        # A login is an email and a password together. One without the other could
        # never sign in, and would read as a half-registered account.
        CheckConstraint(
            "(email IS NULL) = (password_hash IS NULL)", name="login_is_complete"
        ),
        # Only a cardholder may exist without a login.
        CheckConstraint(
            "role = 'cardholder' OR email IS NOT NULL", name="staff_can_sign_in"
        ),
        # An empty hash would make an empty password verify against anything a
        # future login path compared loosely.
        CheckConstraint(
            "password_hash IS NULL OR length(password_hash) > 0",
            name="password_hash_is_not_empty",
        ),
        CheckConstraint("length(trim(display_name)) > 0", name="display_name_is_present"),
    )

    @property
    def can_sign_in(self) -> bool:
        return self.password_hash is not None

    def to_dict(self) -> dict:
        # password_hash is omitted on purpose: this feeds API responses.
        return {
            "id": str(self.id),
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
        }


class UserDetails(Base):
    """What is known about a customer as a person. Cardholders only.

    The protected attributes are stored because the fairness audit needs them, and
    dropped inside CreditScorer.prepare() before anything reaches the model. Codes
    are the published codebook's, folded to documented levels at ingest.
    """

    __tablename__ = "user_details"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    sex: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    age_years: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    education: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    marriage: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    user: Mapped["User"] = relationship(back_populates="details")

    __table_args__ = (
        CheckConstraint("sex IN (1, 2)", name="sex_is_a_codebook_value"),
        CheckConstraint(
            "age_years BETWEEN 18 AND 120", name="age_years_is_plausible"
        ),
        CheckConstraint(
            "education IN (1, 2, 3, 4)", name="education_is_a_codebook_value"
        ),
        CheckConstraint("marriage IN (1, 2, 3)", name="marriage_is_a_codebook_value"),
    )
