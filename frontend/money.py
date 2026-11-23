"""Money as shown on screen. The source data is in New Taiwan Dollars (TWD), shown as USD."""

CURRENCY = "US$"

# Same rate as the backend's core/money.py: Fed G.5 average, Apr-Sep 2005.
TWD_PER_USD = 31.83


def to_usd(value: float) -> float:
    return value / TWD_PER_USD


def money(value: float) -> str:
    """`US$387` from a TWD value, matching the backend's formatting."""
    dollars = to_usd(value)
    sign = "-" if dollars < 0 else ""
    return f"{sign}{CURRENCY}{abs(dollars):,.0f}"


def money_or_none(value: float | None) -> str:
    """A missing amount (no payment made) reads as a word, never US$0."""
    return "None in six months" if value is None else money(value)


def markdown_safe(text: str) -> str:
    """Escape `$` so two amounts on one line are not read as LaTeX by st.markdown."""
    return text.replace("$", r"\$")


def html_safe(text: str) -> str:
    """For already HTML-escaped text: an entity, since a backslash would show inside raw HTML."""
    return text.replace("$", "&#36;")
