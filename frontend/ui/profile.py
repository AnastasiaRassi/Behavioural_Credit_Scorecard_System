"""Top-right "More Actions..." menu and the profile page it opens.

The menu is drawn inside the fixed `top_actions` container in app.py.
"""
import html
import logging

import streamlit as st

LOGGER = logging.getLogger(__name__)

PROFILE_OPEN_KEY = "profile_open"
MENU_KEY = "profile_menu"
VISIT_KEY = "profile_visit"
REPORT_KEY = "profile_report"
SIGN_OUT_KEY = "sign_out"

# Hardcoded for the demo UI: one bank and one officer for every cardholder.
BANK_NAME = "Cedar Crest Bank"
LOAN_OFFICER = "Rana Haddad"

# The dataset codebook's codes (education and marriage are folded to these at ingest).
SEX = {1: "Male", 2: "Female"}
EDUCATION = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE = {1: "Married", 2: "Single", 3: "Other"}

ISSUE_KINDS = ("Something is broken", "Something looks wrong", "Feedback or idea")

_CSS = f"""
<style>
.st-key-{MENU_KEY} button {{
    background: #FFFFFF; border: 1px solid #E6E2DE; border-radius: 8px;
    padding: 6px 12px; min-height: 0; transition: background .15s, border-color .15s;
}}
.st-key-{MENU_KEY} button p {{ font-size: 13px; font-weight: 500; color: #1C1917; }}
.st-key-{MENU_KEY} button:hover, .st-key-{MENU_KEY} button:focus-visible {{
    background: rgba(28,25,23,.05); border-color: rgba(28,25,23,.35);
}}
[data-testid="stPopoverBody"] {{
    min-width: 270px; padding: .6rem; background: #FFFFFF;
    border: 1px solid #E5E7EB; border-radius: 12px;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,.1), 0 4px 6px -4px rgba(0,0,0,.1);
}}
[data-testid="stPopoverBody"] hr {{ margin: .5rem 0; border-color: #E5E7EB; }}
.ct-profile-card {{
    display: flex; align-items: center; gap: .75rem; padding: .7rem .75rem;
    background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px;
}}
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
.ct-section-title {{ font-weight: 700; color: #0F172A; margin-bottom: .25rem; }}
.ct-muted {{ color: #64748B; font-size: 14px; }}
.st-key-{VISIT_KEY} button, .st-key-{REPORT_KEY} button, .st-key-{SIGN_OUT_KEY} button {{
    justify-content: flex-start; gap: .6rem; border-radius: 8px; box-shadow: none;
    transition: background .15s, border-color .15s;
}}
.st-key-{VISIT_KEY} button, .st-key-{REPORT_KEY} button {{
    border: none; background: transparent;
}}
.st-key-{VISIT_KEY} button p, .st-key-{REPORT_KEY} button p {{
    color: #374151; font-weight: 500;
}}
.st-key-{VISIT_KEY} button:hover, .st-key-{REPORT_KEY} button:hover {{ background: #F3F4F6; }}
/* Icons sit in small tinted tiles. */
.st-key-{VISIT_KEY} [data-testid="stIconMaterial"],
.st-key-{REPORT_KEY} [data-testid="stIconMaterial"],
.st-key-{SIGN_OUT_KEY} [data-testid="stIconMaterial"] {{
    width: 30px; height: 30px; border-radius: 8px; font-size: 18px;
    display: inline-flex; align-items: center; justify-content: center;
}}
.st-key-{VISIT_KEY} [data-testid="stIconMaterial"] {{ background: #ECFDF5; color: #059669; }}
.st-key-{REPORT_KEY} [data-testid="stIconMaterial"] {{ background: #FFF7ED; color: #D97706; }}
/* Muted red, so signing out reads as the way out and not as a risk verdict. */
.st-key-{SIGN_OUT_KEY} button {{ background: #FEF2F2; border: 1px solid #FECACA; }}
.st-key-{SIGN_OUT_KEY} button p {{ color: #B91C1C; font-weight: 600; }}
.st-key-{SIGN_OUT_KEY} [data-testid="stIconMaterial"] {{ background: #FEE2E2; color: #B91C1C; }}
.st-key-{SIGN_OUT_KEY} button:hover {{ background: #FEE2E2; border-color: #FCA5A5; }}
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


def _set_open(value: bool) -> None:
    st.session_state[PROFILE_OPEN_KEY] = value


def render(display_name: str, role_label: str, on_sign_out) -> None:
    """The trigger and its popover; `on_sign_out` runs when Sign out is clicked."""
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.popover("More Actions", key=MENU_KEY):
        st.markdown(
            f'<div class="ct-profile-card"><div class="ct-profile-badge">'
            f"{html.escape(initials(display_name))}</div><div>"
            f'<div class="ct-profile-name">{html.escape(display_name)}</div>'
            f'<div class="ct-profile-role">{html.escape(role_label)}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.button(
            ":material/person: Visit Profile", key=VISIT_KEY,
            on_click=_set_open, args=(True,), width="stretch",
        )
        if st.button(":material/flag: Report an Issue", key=REPORT_KEY, width="stretch"):
            _report_dialog(display_name)
        st.divider()
        st.button(":material/logout: Sign Out", key=SIGN_OUT_KEY, on_click=on_sign_out, width="stretch")


@st.dialog("Report an issue")
def _report_dialog(display_name: str) -> None:
    kind = st.selectbox("What kind of issue?", ISSUE_KINDS)
    details = st.text_area("What happened?", placeholder="Describe it in a sentence or two.")
    if st.button("Send report", type="primary", disabled=not details.strip()):
        # No tracker is wired up yet; the log is where reports land.
        LOGGER.info("Issue report from %s [%s]: %s", display_name, kind, details.strip())
        st.toast("Thanks, your report was sent.")
        st.rerun()


def render_page(session: dict, role_label: str) -> None:
    """Who is signed in, and a place for preferences."""
    st.button(
        ":material/arrow_back: Back", key="profile_back",
        on_click=_set_open, args=(False,),
    )
    st.markdown('<div class="ct-page-title">Your profile</div>', unsafe_allow_html=True)

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
    st.markdown(
        '<div class="ct-card"><div class="ct-section-title">Settings</div>'
        '<div class="ct-muted">Preferences will appear here.</div></div>',
        unsafe_allow_html=True,
    )
