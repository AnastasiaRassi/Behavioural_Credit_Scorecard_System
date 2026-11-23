"""Risk prediction: the model's call on the next payment, and why.

One layout for both roles, drawn from the same pieces: the cardholder's Risk
Prediction page, and the Risk assessment mode of an officer's customer page
(ui/customer_mode.py). An officer sees the probability; a cardholder is never sent
it (`worklist.cardholder_view`), so their score card shows the band alone. The six-month history chart stays on the
overview rather than repeating here. Styles are in ui/prediction.css.
"""
from collections.abc import Callable
from html import escape

import plotly.graph_objects as go
import streamlit as st

import api
import theme
from theme import BAND_ICONS
from ui import customer_mode, loana, nav
from ui.components import flatten, reason_line, show, stylesheet

# No model is certain, so the extremes are shown as bounds.
CERTAINTY_CAP = 0.99

PROBLEM_KEY = "prediction_problem"

CARDHOLDER_QUESTION = "How can I improve my risk of missing the next payment?"
OFFICER_QUESTION = "What would most lower this customer's risk of missing the next payment?"

# Why the score matters, worded to match what officers can actually do (review.ACTIONS).
CARDHOLDER_STAKES = (
    "This prediction helps the bank decide whether to keep your credit limit as it is, "
    "check in with you early, or watch the account more closely. A lower risk keeps "
    "your options open. Nothing changes automatically: a member of staff reviews it first."
)
OFFICER_STAKES = (
    "This score feeds the review queue and backs the decision you record: no action, "
    "monitor, contact the customer, or reduce the limit. Use it as evidence alongside "
    "the history, not as a verdict."
)


def as_risk(probability: float) -> str:
    if probability >= CERTAINTY_CAP:
        return f"over {CERTAINTY_CAP:.0%}"
    if probability <= 1 - CERTAINTY_CAP:
        return f"under {1 - CERTAINTY_CAP:.0%}"
    return f"{probability:.0%}"


def _raises(reason: dict) -> bool:
    return reason.get("raises_risk", reason.get("effect", 0) > 0)


# --- the page ---------------------------------------------------------------


def officer_result(detail: dict) -> tuple[dict | None, bool]:
    """This session's fresh score, else the last one stored, and whether it is fresh."""
    fresh = st.session_state.get(f"prediction_{detail['customer_id']}")
    if fresh is not None:
        return fresh, True
    stored = detail.get("latest_prediction")
    if stored is None:
        return None, False
    return {
        "probability": stored["probability"],
        "risk_band": stored["risk_band"],
        "flagged": stored["flagged"],
        "threshold": stored["threshold"],
        "reasons": stored["explanation"]["reasons"],
    }, False


def officer_page(detail: dict, officer_email: str) -> None:
    result, is_fresh = officer_result(detail)
    band = detail["risk_band"]
    _layout(
        result,
        on_predict=_officer_predict, args=(detail["customer_id"], officer_email),
        show_probability=True, question=OFFICER_QUESTION, stakes=OFFICER_STAKES,
        follow_up=("See the account overview", customer_mode.go, (customer_mode.OVERVIEW,)),
        scored_note=(
            "Scored just now and recorded." if is_fresh
            else "The last score recorded for this customer."
        ),
        not_run=(
            f"This account sits in the {BAND_ICONS[band]} {band} band in this month's "
            "batch. Run the prediction to see the risk and the reasons behind it."
        ),
        problem=st.session_state.pop(PROBLEM_KEY, None),
    )


def _officer_predict(customer_id: str, officer_email: str) -> None:
    """A callback, so the page already shows the stored score when it redraws."""
    try:
        st.session_state[f"prediction_{customer_id}"] = api.predict(
            customer_id, scored_by=officer_email
        )
    except api.ServiceUnavailable as exception:
        st.session_state[PROBLEM_KEY] = str(exception)
        return
    api.customer_detail.clear()


def cardholder_page(
    account: dict, on_predict: Callable[[], None], problem: str | None = None
) -> None:
    result = None
    if account.get("predicted", True):
        result = {
            "risk_band": account["risk_band"],
            "flagged": account["needs_attention"],
            "reasons": account["reasons"],
        }
    _layout(
        result, on_predict=on_predict, args=(), show_probability=False, scored_note="",
        question=CARDHOLDER_QUESTION, stakes=CARDHOLDER_STAKES,
        follow_up=("Request an officer review", nav.go, (nav.CHAT,)),
        not_run=(
            "Run the model on your last six months to see how your next payment "
            "looks and what is driving it."
        ),
        problem=problem,
    )


def _layout(
    result: dict | None, on_predict: Callable[..., None], args: tuple,
    show_probability: bool, scored_note: str, not_run: str, problem: str | None,
    question: str, stakes: str, follow_up: tuple[str, Callable[..., None], tuple],
) -> None:
    """`follow_up` is the label, callback and its args for the step after reading the score."""
    stylesheet("prediction.css")
    show(flatten(f"""
<div class="prediction-stakes">
<div class="prediction-stakes-title">Why this prediction matters</div>
<p>{escape(stakes)}</p>
</div>
"""))
    score, why = st.columns([2, 3], gap="large")
    with score, st.container(border=True, key="prediction_score"):
        if result is None:
            show(f'<div class="prediction-eyebrow">Next payment</div><p class="prediction-empty">{escape(not_run)}</p>')
        else:
            _score(result, show_probability, scored_note)
        st.button(
            "Re-run prediction" if result else "Predict next payment",
            type="primary", key="predict_next_payment",
            on_click=on_predict, args=args, width="stretch",
        )
        _next_steps(question, follow_up)
    with why, st.container(border=True, key="prediction_why"):
        _why(result["reasons"] if result else None)
    if problem:
        st.error(problem)
    _charts(result)


def _next_steps(question: str, follow_up: tuple[str, Callable[..., None], tuple]) -> None:
    label, on_click, args = follow_up
    with st.container(key="prediction_next"):
        st.button(
            ":material/auto_awesome: Ask Loana AI how to improve this",
            key="prediction_ask_loana", type="tertiary",
            on_click=loana.ask, args=(question,),
        )
        st.button(
            f"{label} →", key="prediction_follow_up", type="tertiary",
            on_click=on_click, args=args,
        )


def _score(result: dict, show_probability: bool, scored_note: str) -> None:
    band = result["risk_band"]
    flag = "Flagged for attention" if result["flagged"] else "Not flagged for attention"
    # The calibrated probability; a cardholder's payload never carries it.
    probability = (
        f'<div class="prediction-sub"><b>{as_risk(result["probability"])}</b> risk of '
        "missing the next payment</div>"
        if show_probability else ""
    )
    show(flatten(f"""
<div class="prediction-eyebrow">Next payment</div>
<div class="prediction-figure prediction-figure--{escape(band)}">{BAND_ICONS[band]} {escape(band)} risk</div>
{probability}
<div class="prediction-flag">{flag}.</div>
"""))
    if scored_note:
        st.caption(scored_note)


def _why(reasons: list[dict] | None) -> None:
    show('<div class="prediction-title">Why this score</div>')
    if reasons is None:
        show('<p class="prediction-empty">The reasons appear here once the model has run.</p>')
        return
    raising = sum(_raises(reason) for reason in reasons)
    show(
        f'<p class="prediction-summary">{raising} of {len(reasons)} factors raise the risk, '
        f"{len(reasons) - raising} lower it.</p>"
    )
    rows = "".join(
        '<li class="prediction-reason">'
        f'<span class="prediction-rank">{rank}</span>'
        f'<span class="prediction-reason-text">{reason_line(reason["description"], _raises(reason))}</span></li>'
        for rank, reason in enumerate(reasons, start=1)
    )
    show(f'<ol class="prediction-reasons">{rows}</ol>')
    st.caption("▲ raises the risk, ▼ lowers it, strongest first.")


# --- chart --------------------------------------------------------------------


def _charts(result: dict | None) -> None:
    """Officers only: the factor sizes. The history chart lives on the overview."""
    reasons = (result or {}).get("reasons") or []
    sized = [reason for reason in reasons if "effect" in reason]
    if sized:
        _factor_chart(sized)


def _factor_chart(reasons: list[dict]) -> None:
    """How far each factor pushes the score, on one shared axis around zero."""
    # Plotly draws the first bar at the bottom; reverse so the strongest is on top.
    ordered = list(reversed(reasons))
    effects = [reason["effect"] for reason in ordered]
    with st.container(border=True, key="prediction_factor_card"):
        show(
            '<div class="prediction-chart-title">What moves the score</div>'
            '<div class="prediction-chart-hint">Bars to the right raise the risk, '
            "to the left lower it. Hover a bar for the full reason.</div>"
        )
        figure = go.Figure(go.Bar(
            x=effects, y=[_short(reason["description"]) for reason in ordered],
            orientation="h",
            marker=dict(
                color=[theme.HIGH if effect > 0 else theme.LOW for effect in effects],
                cornerradius=4,
            ),
            customdata=[reason["description"] for reason in ordered],
            hovertemplate="%{customdata}<extra></extra>",
        ))
        figure.update_layout(
            height=290, bargap=0.35,
            margin=dict(l=8, r=16, t=8, b=24),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=theme.FONT, color=theme.TEXT, size=13),
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=theme.GRID, font_size=13),
            dragmode=False, showlegend=False,
            xaxis=dict(
                fixedrange=True, zeroline=True, zerolinecolor=theme.MUTED,
                gridcolor=theme.GRID, showticklabels=False,
            ),
            yaxis=dict(fixedrange=True, automargin=True),
        )
        st.plotly_chart(
            figure, width="stretch", theme=None, key="prediction_factor_chart",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )


def _short(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
