"""The Risk Prediction page: the model's call on the next payment, and why.

One layout for both roles, drawn from the same pieces. An officer sees the
probability; a cardholder is never sent it (`worklist.cardholder_view`), so their
score card shows the band alone. Styles are in ui/prediction.css.
"""
from collections.abc import Callable
from html import escape

import streamlit as st

import api
from theme import BAND_ICONS
from ui import nav
from ui.components import band_badge, flatten, reason_line, show, stylesheet

# No model is certain, so the extremes are shown as bounds.
CERTAINTY_CAP = 0.99

PROBLEM_KEY = "prediction_problem"
NOT_RUN_SUMMARY = "Not run yet. Run the model to see the risk and the reasons."


def as_risk(probability: float) -> str:
    if probability >= CERTAINTY_CAP:
        return f"over {CERTAINTY_CAP:.0%}"
    if probability <= 1 - CERTAINTY_CAP:
        return f"under {1 - CERTAINTY_CAP:.0%}"
    return f"{probability:.0%}"


def _raises(reason: dict) -> bool:
    return reason.get("raises_risk", reason.get("effect", 0) > 0)


# --- the entry card on the overview -----------------------------------------


def entry_card(summary: str) -> None:
    """A compact link from the overview to the full analysis."""
    stylesheet("prediction.css")
    with st.container(key="prediction_entry"):
        text, action = st.columns([4, 1], vertical_alignment="center")
        text.markdown(
            flatten(f"""
<div class="prediction-entry-text"><b>Risk prediction</b>
<span>{escape(summary)}</span></div>
"""),
            unsafe_allow_html=True,
        )
        action.button(
            "View analysis :material/arrow_forward:", key="open_prediction",
            on_click=nav.go, args=(nav.PREDICTION,), width="stretch",
        )


def cardholder_summary(account: dict) -> str:
    if not account.get("predicted", True):
        return NOT_RUN_SUMMARY
    return f"{account['risk_band']} risk. See what is driving it."


def officer_summary(detail: dict) -> str:
    result, _ = officer_result(detail)
    if result is None:
        return NOT_RUN_SUMMARY
    return (
        f"{result['risk_band']} risk · {as_risk(result['probability'])} chance "
        "of missing the next payment"
    )


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
        show_probability=True,
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
        not_run=(
            "Run the model on your last six months to see how your next payment "
            "looks and what is driving it."
        ),
        problem=problem,
    )


def _layout(
    result: dict | None, on_predict: Callable[..., None], args: tuple,
    show_probability: bool, scored_note: str, not_run: str, problem: str | None,
) -> None:
    stylesheet("prediction.css")
    score, why = st.columns([2, 3], gap="large")
    with score, st.container(border=True, key="prediction_score"):
        if result is None:
            show(f'<div class="prediction-eyebrow">Next payment</div><p class="prediction-empty">{escape(not_run)}</p>')
        else:
            _score(result, show_probability, scored_note)
        st.button(
            "Predict next payment", type="primary", key="predict_next_payment",
            on_click=on_predict, args=args, width="stretch",
        )
    with why, st.container(border=True, key="prediction_why"):
        _why(result["reasons"] if result else None)
    if problem:
        st.error(problem)


def _score(result: dict, show_probability: bool, scored_note: str) -> None:
    band = result["risk_band"]
    flag = "Flagged for attention" if result["flagged"] else "Not flagged for attention"
    if show_probability:
        figure = (
            f'<div class="prediction-figure">{as_risk(result["probability"])}</div>'
            '<div class="prediction-sub">chance of missing the next payment</div>'
            f'<div class="prediction-badge">{band_badge(band)}</div>'
        )
    else:
        figure = (
            f'<div class="prediction-figure prediction-figure--{escape(band)}">'
            f"{BAND_ICONS[band]} {escape(band)} risk</div>"
        )
    show(flatten(f"""
<div class="prediction-eyebrow">Next payment</div>
{figure}
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
        f'<li class="prediction-reason prediction-reason--{"up" if _raises(reason) else "down"}">'
        f'{reason_line(reason["description"], _raises(reason))}</li>'
        for reason in reasons
    )
    show(f'<ol class="prediction-reasons">{rows}</ol>')
    st.caption("▲ raises the risk, ▼ lowers it, strongest first.")
