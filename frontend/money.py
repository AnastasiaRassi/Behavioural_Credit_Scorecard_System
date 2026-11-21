"""Money as shown on screen. The source data is in New Taiwan Dollars."""

CURRENCY = "NT$"


def money(value: float) -> str:
    """`NT$12,345`, matching the backend's formatting."""
    sign = "-" if value < 0 else ""
    return f"{sign}{CURRENCY}{abs(value):,.0f}"


def markdown_safe(text: str) -> str:
    """Escape `$` so two amounts on one line are not read as LaTeX by st.markdown."""
    return text.replace("$", r"\$")


def html_safe(text: str) -> str:
    """For already HTML-escaped text: an entity, since a backslash would show inside raw HTML."""
    return text.replace("$", "&#36;")
