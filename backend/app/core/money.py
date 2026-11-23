"""Money as shown to people. The source data is in New Taiwan Dollars (TWD).

Stored values and the model stay in TWD; only display converts to US dollars.
"""

CURRENCY = "US$"

# Mean of the Fed G.5 monthly averages, Apr-Sep 2005 (the dataset's window):
# 31.48, 31.27, 31.35, 31.89, 32.08, 32.92 TWD per USD.
TWD_PER_USD = 31.83


def to_usd(value: float) -> float:
    return value / TWD_PER_USD


def money(value: float, signed: bool = False) -> str:
    """`US$387`; with `signed`, `+US$387` or `-US$387`. `value` is in TWD."""
    dollars = to_usd(value)
    sign = "-" if dollars < 0 else ("+" if signed else "")
    return f"{sign}{CURRENCY}{abs(dollars):,.0f}"
