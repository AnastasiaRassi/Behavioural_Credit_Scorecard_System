"""The profile page, opened from the nav drawer."""
import html

import streamlit as st

import api

PROFILE_OPEN_KEY = "profile_open"

# Hardcoded for the demo UI: one bank and one officer for every cardholder.
BANK_NAME = "Cedar Crest Bank"
LOAN_OFFICER = "Rana Haddad"

# The dataset codebook's codes (education and marriage are folded to these at ingest).
SEX = {1: "Male", 2: "Female"}
EDUCATION = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE = {1: "Married", 2: "Single", 3: "Other"}

_CSS = f"""
<style>
.ct-profile-badge {{
    flex: none; width: 40px; height: 40px; border-radius: 50%; background: #059669;
    color: #FFFFFF; display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 14px; letter-spacing: .5px;
}}
.ct-profile-name {{ font-weight: 600; font-size: 14.5px; color: #111827; line-height: 1.3; }}
.ct-profile-role {{ font-size: 12.5px; color: #6B7280; }}
/* Profile page: slate for text and borders, emerald for accents. */
.ct-page-title {{ font-size: 1.6rem; font-weight: 700; color: #0F172A; margin: .25rem 0 1rem; }}
.ct-card {{
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 8px 24px -12px rgba(15,23,42,.12);
}}
.ct-card-head {{
    display: flex; align-items: center; gap: 1rem; padding-bottom: 1.25rem;
    margin-bottom: 1.25rem; border-bottom: 1px solid #E2E8F0;
}}
.ct-badge-lg {{ width: 56px; height: 56px; font-size: 18px; }}
.ct-card-name {{ font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: .25rem; }}
.ct-pill {{
    display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
    font-weight: 600; background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0;
}}
.ct-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .9rem; }}
.ct-stat {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: .85rem 1rem; }}
.ct-stat-label {{
    font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: #64748B;
}}
.ct-stat-value {{ font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-top: .2rem; overflow-wrap: anywhere; }}
.st-key-profile_edit button {{
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    box-shadow: 0 1px 3px rgba(15,23,42,.06); padding: .45rem 1rem;
}}
.st-key-profile_edit button p {{ font-size: 13.5px; font-weight: 600; color: #0F172A; }}
.st-key-profile_edit button:hover, .st-key-profile_edit button:focus-visible {{
    border-color: #A7F3D0; background: #ECFDF5;
}}
.st-key-profile_edit button:hover p {{ color: #047857; }}
</style>
"""


def initials(display_name: str) -> str:
    """First letters of the first and last name: "Omar Fares" -> "OF"."""
    words = display_name.split()
    if not words:
        return "?"
    return (words[0][0] + (words[-1][0] if len(words) > 1 else "")).upper()


def is_open() -> bool:
    return bool(st.session_state.get(PROFILE_OPEN_KEY))


def set_open(value: bool) -> None:
    st.session_state[PROFILE_OPEN_KEY] = value


def render_page(session: dict, role_label: str) -> None:
    """Who is signed in, and a place for preferences."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="page-title">Your profile</div>', unsafe_allow_html=True)

    fields = [("Email", session["email"])]
    if session.get("account_number"):
        fields = [
            ("Customer ID", session["account_number"]),
            *fields,
            ("Age", session.get("age_years")),
            ("Gender", SEX.get(session.get("sex"))),
            ("Education", EDUCATION.get(session.get("education"))),
            ("Marital status", MARRIAGE.get(session.get("marriage"))),
            ("Bank", BANK_NAME),
            ("Loan officer", LOAN_OFFICER),
        ]
    tiles = "".join(
        f'<div class="ct-stat"><div class="ct-stat-label">{html.escape(label)}</div>'
        f'<div class="ct-stat-value">'
        f'{html.escape("-" if value is None else str(value))}</div></div>'
        for label, value in fields
    )
    st.markdown(
        '<div class="ct-card"><div class="ct-card-head">'
        f'<div class="ct-profile-badge ct-badge-lg">{html.escape(initials(session["display_name"]))}</div>'
        f'<div><div class="ct-card-name">{html.escape(session["display_name"])}</div>'
        f'<span class="ct-pill">{html.escape(role_label)}</span></div></div>'
        f'<div class="ct-grid">{tiles}</div></div>',
        unsafe_allow_html=True,
    )
    if session.get("account_number"):
        if st.button("Edit my information", icon=":material/edit:", key="profile_edit"):
            _edit_dialog(session)


def _choice(options: dict, current) -> int:
    codes = list(options)
    return codes.index(current) if current in codes else 0


@st.dialog("Edit my information")
def _edit_dialog(session: dict) -> None:
    name = st.text_input("Name", value=session["display_name"])
    age = st.number_input("Age", 18, 120, int(session.get("age_years") or 18))
    sex = st.selectbox("Gender", list(SEX), _choice(SEX, session.get("sex")), format_func=SEX.get)
    education = st.selectbox(
        "Education", list(EDUCATION), _choice(EDUCATION, session.get("education")),
        format_func=EDUCATION.get,
    )
    marriage = st.selectbox(
        "Marital status", list(MARRIAGE), _choice(MARRIAGE, session.get("marriage")),
        format_func=MARRIAGE.get,
    )
    if st.button("Save", type="primary", key="profile_save"):
        if not name.strip():
            st.error("Name cannot be blank.")
            return
        try:
            account = api.edit_profile(session["token"], {
                "display_name": name, "age_years": int(age), "sex": sex,
                "education": education, "marriage": marriage,
            })
        except api.ServiceUnavailable as exception:
            st.error(str(exception))
            return
        session.update(account)
        st.rerun()
