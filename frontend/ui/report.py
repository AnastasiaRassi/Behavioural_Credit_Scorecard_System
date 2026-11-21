"""The Report an Issue page: a short form, then a thank-you once it is stored."""
import streamlit as st

import api

# Keep equal to ISSUE_KINDS in backend/app/db/models/issues.py, which validates it.
ISSUE_KINDS = ("Something is broken", "Something looks wrong", "Feedback or idea")

SENT_KEY = "issue_report_sent"
KIND_KEY = "issue_report_kind"
DETAILS_KEY = "issue_report_details"


def _send(token: str) -> None:
    try:
        api.report_issue(token, st.session_state[KIND_KEY], st.session_state[DETAILS_KEY])
    except api.ServiceUnavailable as exception:
        api.guard(exception)
    st.session_state[SENT_KEY] = True
    st.session_state[DETAILS_KEY] = ""


def _another() -> None:
    st.session_state[SENT_KEY] = False


def render_page(session: dict) -> None:
    st.markdown('<div class="page-title">Report an issue</div>', unsafe_allow_html=True)
    if st.session_state.get(SENT_KEY):
        st.success("Thank you, your feedback is with us. We read every report and will look into it.")
        st.button("Send another", key="issue_report_another", on_click=_another)
        return

    st.caption("Tell us what went wrong or what could be better. A sentence or two is enough.")
    st.selectbox("What kind of issue?", ISSUE_KINDS, key=KIND_KEY)
    details = st.text_area(
        "What happened?", key=DETAILS_KEY, max_chars=2000,
        placeholder="Describe it in a sentence or two.",
    )
    st.button(
        "Send report", type="primary", key="issue_report_send",
        disabled=not details.strip(), on_click=_send, args=(session["token"],),
    )
