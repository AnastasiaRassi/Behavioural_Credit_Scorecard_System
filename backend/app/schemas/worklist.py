"""Shapes for the officer worklist and the cardholder's own view."""
from pydantic import BaseModel, Field


class QueueReason(BaseModel):
    """A driver of a score, as an officer sees it: with the effect size."""

    feature: str
    effect: float = Field(description="Log-odds; positive raises risk")
    description: str


class QueueRow(BaseModel):
    customer_id: str
    probability: float
    risk_band: str
    flagged: bool
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    reasons: list[QueueReason]


class MonthEntry(BaseModel):
    """One observed month. Positional, not dated: month 1 is the most recent."""

    month: int
    label: str
    payment_status: int
    payment_status_reading: str
    bill_amount: float
    amount_paid: float
    utilisation: float | None = Field(
        default=None, description="None when the credit limit is zero"
    )


class Segment(BaseModel):
    label: int
    name: str


class Highlights(BaseModel):
    """Derived from the history, never stored alongside it."""

    utilisation_now: float | None = None
    utilisation_mean: float | None = None
    trajectory: str | None = Field(
        default=None, description="improving, steady or deteriorating"
    )
    months_since_full_payment: int | None = Field(
        default=None, description="None when the bill was not cleared in six months"
    )
    months_with_no_payment: int
    worst_payment_status: int | None = None


class CustomerDetail(BaseModel):
    """The dedicated customer page: a queue row plus history, segment, highlights."""

    customer_id: str
    full_name: str | None = Field(
        default=None,
        description="Known only for cardholders the system has met; the id otherwise",
    )
    probability: float
    risk_band: str
    flagged: bool
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    segment: Segment | None = Field(
        default=None, description="None when segmentation has not been run"
    )
    history: list[MonthEntry]
    highlights: Highlights
    reasons: list["QueueReason"]
    latest_prediction: dict | None = Field(
        default=None,
        description="Most recent score an officer ran, or None if never run",
    )


class QueueSummary(BaseModel):
    in_queue: int
    flagged: int
    not_flagged: int
    bands: dict[str, int]


class QueuePage(BaseModel):
    """One page of the worklist, with the totals the summary tiles need.

    `summary` describes the filtered set, `summary_all` the whole queue, so a
    screen can show "41 of 4,500" without a second request.
    """

    total: int
    offset: int
    returned: int
    threshold: float
    summary: QueueSummary
    summary_all: QueueSummary
    rows: list[QueueRow]


class CardholderReason(BaseModel):
    """A driver of a score, as the cardholder sees it: no effect size."""

    feature: str
    description: str
    raises_risk: bool


class CardholderView(BaseModel):
    """Deliberately narrower than QueueRow.

    No probability and no threshold: the number is not sent, not merely hidden.
    """

    customer_id: str
    risk_band: str
    needs_attention: bool
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    reasons: list[CardholderReason]


class RoleOut(BaseModel):
    role: str
    label: str
    description: str
    landing_endpoint: str
    users: int
