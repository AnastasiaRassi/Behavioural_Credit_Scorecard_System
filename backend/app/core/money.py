"""Money as shown to people. The source data is in New Taiwan Dollars."""

CURRENCY = "NT$"


def money(value: float, signed: bool = False) -> str:
    """`NT$12,345`; with `signed`, `+NT$12,345` or `-NT$12,345`."""
    sign = "-" if value < 0 else ("+" if signed else "")
    return f"{sign}{CURRENCY}{abs(value):,.0f}"
