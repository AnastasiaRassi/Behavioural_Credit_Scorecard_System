"""What a customer is told they could do, and what is deliberately left out.

No probability and no threshold, for the same reason `CardholderView` carries
none: a payload a screen merely declines to draw is still one devtools panel
away. Recourse is answerable in bands and actions, so it is.
"""
from pydantic import BaseModel, Field


class RecourseStep(BaseModel):
    """One change, in plain language, with the numbers behind it."""

    description: str
    feature: str
    # The raw pair is kept so a screen can render its own wording or a bar; it
    # is the customer's own account data, which they may already see.
    before: float
    after: float


class RecourseView(BaseModel):
    """The smallest set of changes that would move this account to a better band."""

    customer_id: str
    current_band: str
    target_band: str
    # False when the search could not reach the target within the policy's
    # bounds. The steps then show the best it managed, and `note` says so.
    reachable: bool
    achieved_band: str
    steps: list[RecourseStep] = Field(default_factory=list)
    note: str = ""
