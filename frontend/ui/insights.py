"""Account health: the overview panels, built from what the risk model weighs most.

Permutation importance ranks last month's repayment status and the worst status in
six months first, then credit left, then how much of each bill is repaid. Those are
the panels, in that order. Billed alone is not a model feature (a big bill on a big
limit is not risky), so it is shown only as plain amounts beside what was paid.
Shared by the cardholder and officer overviews; styles are in insights.css.
"""
import plotly.graph_objects as go
import streamlit as st

import theme
from money import CURRENCY, to_usd
from ui import loana
from ui.components import flatten, show, stylesheet

TICK = "#475569"
STATUS_LEVELS = {-2: "No balance", -1: "Paid in full", 0: "Minimum paid", 1: "1 month late",
                 2: "2 months late", 3: "3+ months late"}
QUESTIONS = {
    "status": "Explain my Repayment status chart. What does it show and how does it affect my risk?",
    "use": "Explain my Credit used chart. What does it show and how does it affect my risk?",
    "repaid": "Explain my Share of each bill repaid chart. What does it show and how does it affect my risk?",
    "amounts": "Explain my Billed and paid chart. What does it show and does a large bill affect my risk?",
}
CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def render(history: list[dict], credit_limit: float, title: str = "Account health") -> None:
    if not history:
        return
    stylesheet("insights.css")
    # The API sends the most recent month first; charts read oldest to newest.
    months = sorted(history, key=lambda month: month["month"], reverse=True)
    labels = [month["label"] for month in months]
    status = [month["payment_status"] for month in months]
    bill = [max(to_usd(month["bill_amount"]), 0.0) for month in months]
    paid = [to_usd(month["amount_paid"]) for month in months]
    limit = to_usd(credit_limit)
    used = [value / limit * 100 if limit > 0 else None for value in bill]
    repaid = [p / b * 100 if b > 0 else None for p, b in zip(paid, bill)]

    show(flatten(f"""
        <div class="insights-head"><div class="insights-title">{title}</div>
        <div class="insights-hint">The signals the risk model weighs most: repayment status,
        credit use and how much of each bill is paid.</div></div>"""))
    _tiles(months, status, used, repaid)

    first, second = st.columns(2)
    with first, st.container(border=True, key="insight_status"):
        _heading("Repayment status", "Most important signal. Paying late raises risk; paying the minimum or more does not.")
        _plot(_status_figure(labels, status), "insight_status_chart")
        _ask("status")
    with second, st.container(border=True, key="insight_use"):
        _heading("Credit used", "Statement balance as a share of your limit.")
        _plot(_use_figure(labels, used), "insight_use_chart")
        _ask("use")
    third, fourth = st.columns(2)
    with third, st.container(border=True, key="insight_repaid"):
        _heading("Share of each bill repaid", "Paying a bigger share each month lowers risk.")
        _plot(_repaid_figure(labels, repaid), "insight_repaid_chart")
        _ask("repaid")
    with fourth, st.container(border=True, key="insight_amounts"):
        _heading("Billed and paid", "Plain amounts. A large bill is not a risk signal by itself.")
        _plot(_amounts_figure(labels, bill, paid), "insight_amounts_chart")
        _ask("amounts")


def _heading(title: str, hint: str) -> None:
    show(flatten(f'<div class="insight-title">{title}</div><div class="insight-hint">{hint}</div>'))


def _ask(chart: str) -> None:
    """Sends a question about this chart to the assistant and opens it."""
    st.button("Ask AI to explain this chart", icon=":material/auto_awesome:", type="tertiary",
              key=f"insight_ask_{chart}", on_click=loana.ask, args=(QUESTIONS[chart],))


def _plot(figure: go.Figure, key: str) -> None:
    figure.update_layout(
        height=250, margin=dict(l=8, r=16, t=8, b=8), showlegend=False, dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=theme.FONT, color=theme.TEXT, size=13),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=theme.GRID, font_size=13),
    )
    figure.update_xaxes(automargin=True, showgrid=False, fixedrange=True, tickfont=dict(color=TICK, size=12),
                        range=[-0.5, len(figure.data[0].x) - 0.5])
    figure.update_yaxes(automargin=True, fixedrange=True, gridcolor=theme.GRID, zeroline=False,
                        tickfont=dict(color=TICK, size=12), ticklabelstandoff=8)
    st.plotly_chart(figure, width="stretch", theme=None, config=CONFIG, key=key)


def _tone(status: int) -> str:
    return theme.LOW if status <= 0 else theme.MEDIUM if status == 1 else theme.HIGH


def _tiles(months: list[dict], status: list[int], used: list, repaid: list) -> None:
    late = sum(1 for value in status if value > 0)
    valid = [value for value in repaid if value is not None]
    tiles = [
        ("Last month", months[-1]["payment_status_reading"].capitalize(), _tone(status[-1])),
        ("Months late", f"{late} of {len(status)}", theme.LOW if late == 0 else theme.HIGH),
        ("Credit used now", f"{used[-1]:.0f}%" if used[-1] is not None else "n/a", theme.ACCENT),
        ("Avg share repaid", f"{sum(valid) / len(valid):.0f}%" if valid else "n/a", theme.ACCENT),
    ]
    for column, (label, value, tone) in zip(st.columns(4), tiles):
        column.markdown(
            flatten(f'<div class="insight-tile" style="border-top-color:{tone}">'
                    f'<div class="insight-tile-label">{label}</div>'
                    f'<div class="insight-tile-value">{value}</div></div>'),
            unsafe_allow_html=True,
        )


def _status_figure(labels: list[str], status: list[int]) -> go.Figure:
    levels = [min(max(value, -2), 3) for value in status]
    figure = go.Figure(go.Scatter(
        x=labels, y=levels, mode="lines+markers",
        line=dict(color="#94A3B8", width=2, shape="hv"),
        marker=dict(size=14, color=[_tone(v) for v in status], line=dict(color="#FFFFFF", width=2)),
        customdata=status, hovertemplate="%{x}: status %{customdata}<extra></extra>",
    ))
    figure.add_hrect(y0=0.5, y1=3.5, fillcolor=theme.HIGH, opacity=0.06, line_width=0)
    figure.update_yaxes(tickvals=list(STATUS_LEVELS), ticktext=list(STATUS_LEVELS.values()), range=[-2.5, 3.5])
    return figure


def _use_figure(labels: list[str], used: list) -> go.Figure:
    top = max([value for value in used if value is not None] + [100])
    figure = go.Figure(go.Scatter(
        x=labels, y=used, mode="lines+markers", fill="tozeroy",
        line=dict(color=theme.ACCENT, width=3),
        marker=dict(size=9, color="#FFFFFF", line=dict(color=theme.ACCENT, width=2.5)),
        fillcolor="rgba(76,141,246,0.12)", hovertemplate="%{x}: %{y:.0f}% of limit<extra></extra>",
    ))
    figure.add_hline(y=100, line=dict(color=theme.HIGH, width=1, dash="dot"),
                     annotation_text="Limit", annotation_position="top left",
                     annotation_font=dict(color=theme.HIGH, size=11))
    figure.update_yaxes(range=[0, top * 1.15], ticksuffix="%", nticks=5)
    return figure


def _repaid_figure(labels: list[str], repaid: list) -> go.Figure:
    figure = go.Figure(go.Bar(
        x=labels, y=repaid, marker_color=theme.LOANA, width=0.5,
        text=[f"{value:.0f}%" if value is not None else "" for value in repaid],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:.0f}% of the bill<extra></extra>",
    ))
    top = max([value for value in repaid if value is not None] + [10])
    figure.update_yaxes(range=[0, top * 1.3], ticksuffix="%", nticks=4)
    return figure


def _amounts_figure(labels: list[str], bill: list[float], paid: list[float]) -> go.Figure:
    figure = go.Figure([
        go.Bar(x=labels, y=bill, name="Billed", marker_color="#94A3B8",
               hovertemplate=f"Billed: {CURRENCY}%{{y:,.0f}}<extra></extra>"),
        go.Bar(x=labels, y=paid, name="Paid", marker_color=theme.LOANA,
               hovertemplate=f"Paid: {CURRENCY}%{{y:,.0f}}<extra></extra>"),
    ])
    figure.update_layout(barmode="group", hovermode="x unified")
    figure.update_yaxes(tickprefix=CURRENCY, tickformat="~s", nticks=4)
    return figure
