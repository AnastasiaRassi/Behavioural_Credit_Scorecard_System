"""Declarative base and the column defaults every table shares.

Imported by every model module. Holds no tables of its own, so importing it
cannot pull an unrelated table into the metadata.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def new_id() -> uuid.UUID:
    return uuid.uuid4()


def now() -> datetime:
    return datetime.now(timezone.utc)


def sql_tuple(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"
