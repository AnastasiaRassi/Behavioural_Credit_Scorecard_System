"""The schema, in six modules by what each table is for.

    accounts   every person: users, and the details of those who are customers
    customers  what is scored about them, their history, and the dataset's label
    scoring    which model was live, and what it predicted
    decisions  what an officer did about it, and what then happened
    chat       what the cardholder assistant answered, and how
    reviews    cardholders asking a person to look again at their rating

Import from this package rather than from a module: SQLAlchemy resolves
relationships by class name at first use, so every mapper has to be registered
before any of them is queried, and importing the package is what guarantees that.
"""
from app.db.models.accounts import ROLES, User, UserDetails
from app.db.models.base import Base, new_id, now
from app.db.models.chat import ChatTurn
from app.db.models.customers import (
    MONTHS_OBSERVED,
    SOURCE_OBSERVED_PERIOD,
    CreditOutcome,
    Customer,
    MonthlyStatement,
)
from app.db.models.decisions import DECISION_ACTIONS, Decision, ObservedOutcome
from app.db.models.reviews import OPEN_REVIEW_STATUSES, REVIEW_STATUSES, ReviewRequest
from app.db.models.scoring import ModelVersion, Prediction

__all__ = [
    "OPEN_REVIEW_STATUSES",
    "REVIEW_STATUSES",
    "ReviewRequest",
    "Base",
    "ChatTurn",
    "CreditOutcome",
    "Customer",
    "DECISION_ACTIONS",
    "Decision",
    "MONTHS_OBSERVED",
    "ModelVersion",
    "MonthlyStatement",
    "ObservedOutcome",
    "Prediction",
    "ROLES",
    "SOURCE_OBSERVED_PERIOD",
    "User",
    "UserDetails",
    "new_id",
    "now",
]
