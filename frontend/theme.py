"""One palette, one chart theme, applied once at startup.

Colour carries meaning on this dashboard. Green, amber and red mean risk band and
audit outcome; blue is the neutral series colour and the interface accent. Keeping
the accent out of the traffic-light range stops a button or an active tab from
reading as a verdict.

Import colours from here rather than repeating literals, so a chart cannot drift
away from the band colour it is meant to match.
"""
import altair as alt
import streamlit as st

# Risk bands and audit outcomes. The only three colours allowed to mean something.
LOW = "#35B36A"
MEDIUM = "#D97706"
HIGH = "#DC2626"

# Neutral series and interface accent. Mirrors primaryColor in .streamlit/config.toml.
ACCENT = "#4C8DF6"

# Loana AI's card. Not a risk colour: it only ever fills the assistant's column.
LOANA = "#00C588"
LOANA_INK = "#073B2D"

# Cardholder timeline: neon mint for what was paid, red for what was billed.
PAID = "#00E699"
BILLED = "#EF4444"

BAND_COLOURS ={"Low": LOW, "Medium": MEDIUM, "High": HIGH}
BAND_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# Chart furniture: espresso text and warm stone on white cards.
TEXT = "#1C1917"
MUTED = "#78716C"
GRID = "#E6E2DE"
SURFACE = "#FFFFFF"
# Captions sit straight on the grey backdrop, where MUTED is too faint to read.
CAPTION = "#0F172A"

FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Helvetica Neue", '
    "Arial, sans-serif"
)

_CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
/* Inter everywhere: banner, Loana and the rest of the app. Icon fonts are kept. */
html, body, [data-testid="stApp"], .ct-banner,
[data-testid="stApp"] :is(p, span, div, label, button, input, textarea, h1, h2, h3, h4, li, td, th):not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
    font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
}}
/* Warm ecru page; the header is transparent so it shows through. */
[data-testid="stApp"] {{ background: #FBF9F6; color: {TEXT}; }}
[data-testid="stApp"] :is(h1, h2, h3, h4, p, li, label) {{ color: {TEXT}; }}
[data-testid="stCaptionContainer"] p {{ color: {CAPTION}; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* Loana AI (ui/loana.py): a fixed bubble, or a fixed 380x550 card, bottom-right. */
.st-key-loana_launch {{
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    z-index: 1000002;
    width: auto;
    opacity: 1 !important;
}}
/* A pill: the invitation on the left, the avatar (a CSS background, so the button
   stays a real button) on the right. */
.st-key-loana_launch button {{
    width: auto;
    height: 112px;
    min-height: 0;
    padding: 0 132px 0 1.6rem;
    border: 3px solid #00A669;
    border-radius: 999px;
    background-color: #FFFFFF !important;
    opacity: 1 !important;
    background-size: auto 100%;
    background-repeat: no-repeat;
    background-position: right 8px center;
    box-shadow: 0 8px 22px rgba(7, 59, 45, 0.3);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    animation: loana-nudge 2.6s ease-in-out 1.5s 3;
}}
.st-key-loana_launch button p {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F172A;
    white-space: nowrap;
}}
.st-key-loana_launch button:hover {{
    transform: scale(1.05);
    border-color: #34D399;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.3), 0 0 16px rgba(0, 166, 105, 0.35);
}}
@keyframes loana-nudge {{
    0%, 100% {{ transform: none; }}
    50% {{ transform: translateY(-6px); }}
}}
@keyframes loana-pop {{
    from {{ opacity: 0; transform: translateY(16px) scale(0.96); }}
    to {{ opacity: 1; transform: none; }}
}}
.st-key-loana_panel {{
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    z-index: 1000;
    width: min(460px, calc(100vw - 2rem));
    height: min(760px, calc(100vh - 3rem));
    gap: 0.6rem;
    padding: 1.1rem;
    overflow-y: auto;
    background: {LOANA};
    border-radius: 22px;
    box-shadow: 0 16px 44px rgba(7, 59, 45, 0.35);
    animation: loana-pop 0.22s ease-out;
}}
.st-key-loana_close {{
    position: absolute;
    top: 0.7rem;
    right: 0.7rem;
    z-index: 2;
    width: auto;
}}
.st-key-loana_panel .st-key-loana_close button {{
    width: 32px;
    height: 32px;
    min-height: 0;
    padding: 0;
    background: rgba(255, 255, 255, 0.85);
}}
.st-key-loana_panel > * {{ flex-shrink: 0; }}
/* The messages scroll inside the card; everything under them (presets, input)
   keeps its place at the bottom. */
[data-testid="stLayoutWrapper"]:has(> .st-key-loana_chat) {{
    flex: 1 1 0;
    min-height: 200px;
}}
.st-key-loana_chat {{ height: 100% !important; }}
/* The messages fade out at the bottom edge of their fixed-height box. The padding
   is the fade's own height, so the last message can still scroll clear of it. */
.st-key-loana_chat {{
    padding-bottom: 32px;
    -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - 32px), transparent);
    mask-image: linear-gradient(to bottom, #000 calc(100% - 32px), transparent);
}}

.loana-head {{ display: flex; align-items: center; gap: 0.85rem; }}
.loana-head img {{
    width: 88px;
    height: 88px;
    flex: none;
    padding: 4px;
    background: #FFFFFF;
    border-radius: 50%;
    box-shadow: 0 4px 14px rgba(7, 59, 45, 0.25);
}}
/* Padding, not margin: Streamlit trims a markdown block's last 1rem. */
.loana-head {{ min-width: 0; padding-bottom: 1rem; }}
.loana-head .loana-name {{
    font-size: clamp(1.1rem, 1.9vw, 1.4rem);
    font-weight: 600;
    letter-spacing: 0.5px;
    line-height: 1.1;
    white-space: nowrap;
    color: #0F172A;
}}
.loana-head .loana-sub {{
    margin-top: 0.15rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: rgba(15, 23, 42, 0.78);
}}

.st-key-loana_panel p, .st-key-officer_briefing p {{ color: {LOANA_INK}; }}
.st-key-loana_panel [data-testid="stCaptionContainer"] p,
.st-key-officer_briefing [data-testid="stCaptionContainer"] p {{
    color: rgba(7, 59, 45, 0.78);
}}
.st-key-loana_panel [data-testid="stChatMessage"],
.st-key-officer_briefing [data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.92);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    align-items: flex-start;
}}
.st-key-loana_panel [data-testid="stChatMessage"] p,
.st-key-officer_briefing [data-testid="stChatMessage"] p {{
    line-height: 1.55;
    text-align: left;
}}
.st-key-loana_panel [data-testid="stChatMessage"] p,
.st-key-officer_briefing [data-testid="stChatMessage"] p {{ color: #1D2B27; }}
.st-key-loana_panel [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p,
.st-key-officer_briefing [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {{
    color: {MUTED};
    text-align: left;
}}
.st-key-loana_panel button,
.st-key-officer_briefing button {{
    background: #FFFFFF;
    border: none;
    color: {LOANA_INK};
    border-radius: 999px;
}}
.st-key-loana_panel button:hover,
.st-key-officer_briefing button:hover {{ background: #E9FFF7; color: {LOANA_INK}; }}

/* Cardholder status banner (views/cardholder.py): the tone follows the risk. */
div.account-alert {{
    padding: 0.9rem 1.1rem;
    border-radius: 12px;
    border: 1px solid;
    border-left: 4px solid;
    line-height: 1.5;
    margin-bottom: 1.25rem;
}}
.account-alert-title {{ font-weight: 700; }}
div.account-alert p {{ margin: 0.3rem 0 0; color: inherit; }}
div.account-alert--high {{ background: #FEF2F2; color: #991B1B; border-color: #FECACA; border-left-color: #DC2626; }}
div.account-alert--medium {{ background: #FFFBEB; color: #92400E; border-color: #FDE68A; border-left-color: #D97706; }}
div.account-alert--good {{ background: #ECFDF5; color: #065F46; border-color: #A7F3D0; border-left-color: #059669; }}

/* Officer inbox alert (views/officer.py): customers waiting for a reply. */
div.inbox-alert {{
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.75rem 1rem; margin-bottom: 0.6rem;
    border-radius: 12px; border: 1px solid #FECACA; border-left: 4px solid {HIGH};
    background: #FEF2F2; color: #991B1B; line-height: 1.45;
}}
div.inbox-alert--clear {{ background: #F5F5F4; color: {MUTED}; border-color: {GRID}; border-left-color: {GRID}; }}
.inbox-badge {{
    flex: none; min-width: 1.9rem; height: 1.9rem; padding: 0 0.45rem;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 999px; background: {HIGH}; color: #FFFFFF;
    font-weight: 800; font-size: 0.95rem;
    box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.15);
}}

/* Review record (ui/thread.py): full-width entries, a status stepper and a decision block. */
.thread {{ display: flex; flex-direction: column; gap: 0.75rem; margin: 0.4rem 0 1rem; }}
.case-steps {{
    display: flex; list-style: none; margin: 0 0 0.25rem; padding: 0; gap: 0.5rem;
}}
.case-step {{
    flex: 1; display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.75rem;
    border-radius: 10px; font-size: 0.85rem; font-weight: 600;
    border: 1px solid {GRID}; background: {SURFACE}; color: {MUTED};
}}
.case-step span {{
    flex: none; width: 22px; height: 22px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-size: 0.75rem;
    background: #E2E8F0; color: #475569;
}}
.case-step--done {{ color: #065F46; }}
.case-step--done span {{ background: #A7F3D0; color: #065F46; }}
.case-step--current {{ background: #ECFDF5; border-color: #6EE7B7; color: #065F46; }}
.case-step--current span {{ background: #008F63; color: #FFFFFF; }}
.case-entry {{
    padding: 0.85rem 1.1rem; border-radius: 10px; border: 1px solid {GRID};
    background: {SURFACE}; color: {TEXT};
}}
.case-entry--open {{ background: #FFFBEB; border-color: #FDE68A; }}
.case-entry--decision {{ background: #F0FDF4; border-color: #A7F3D0; border-left: 5px solid #008F63; }}
.case-meta, .case-decision-head {{
    display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;
    margin-bottom: 0.4rem;
}}
.case-label {{
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    color: {MUTED};
}}
.case-outcome {{ font-size: 1.15rem; font-weight: 700; color: #065F46; }}
.case-tag {{
    padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
}}
.case-tag--open {{ background: #FDE68A; color: #92400E; }}
.case-tag--done {{ background: #F3F4F6; color: #4B5563; }}
.case-body {{ white-space: pre-wrap; line-height: 1.5; }}
.case-sign {{ margin-top: 0.5rem; font-size: 0.8rem; color: {MUTED}; }}
.thread-empty {{ color: {CAPTION}; font-style: italic; margin: 0.4rem 0 0.8rem; }}

/* Review requests (views/cardholder.py): header, scrolling record and request form
   in one rounded card. */
.st-key-chat_card {{
    width: 100%; max-width: 920px; gap: 0; overflow: hidden;
    background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 16px;
    box-shadow: 0 10px 30px -12px rgba(15, 23, 42, 0.35);
}}
.chat-head {{
    display: flex; align-items: center; gap: 0.6rem; padding: 0.85rem 1.1rem;
    background: #F8FAFC; border-bottom: 1px solid #E2E8F0; color: {TEXT};
}}
.chat-head b {{ font-size: 0.95rem; }}
.chat-status {{
    margin-left: auto; padding: 0.15rem 0.65rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600;
}}
.chat-status--active {{ background: #D1FAE5; color: #065F46; }}
.chat-status--closed {{ background: #F3F4F6; color: #4B5563; }}
.st-key-chat_messages {{ min-height: 360px; padding: 0.5rem 1.1rem; background: #FFFFFF; }}
.st-key-chat_card .thread {{ margin: 0; padding: 0.5rem 0; }}
.chat-empty {{
    display: flex; flex-direction: column; align-items: center; gap: 0.35rem;
    padding: 1.5rem 1rem 0.75rem; text-align: center; color: #334155;
}}
.chat-empty-icon {{
    width: 56px; height: 56px; border-radius: 50%; background: #D1FAE5; color: #047857;
    display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
}}
.chat-empty b {{ font-size: 1.05rem; color: {TEXT}; }}
.chat-empty span {{ max-width: 460px; font-size: 0.9rem; color: {MUTED}; }}
.st-key-chat_chips {{ align-items: center; gap: 0.5rem; }}
.st-key-chat_chips button {{
    border-radius: 999px; padding: 0.3rem 1rem; min-height: 0;
    background: #F0FDF4; border: 1px solid #A7F3D0;
}}
.st-key-chat_chips button p {{ font-size: 0.85rem; font-weight: 600; color: #065F46; }}
.st-key-chat_chips button:hover {{ background: #D1FAE5; border-color: #059669; }}
.chat-closed {{
    display: flex; flex-direction: column; align-items: center; gap: 0.1rem;
    margin: 0.5rem 0; padding: 0.6rem 1rem; text-align: center;
    background: #F3F4F6; border: 1px solid #E5E7EB; border-radius: 10px; color: #374151;
}}
.chat-closed span {{ font-size: 0.8rem; color: {MUTED}; }}
.st-key-chat_card [data-testid="stCaptionContainer"] {{ padding: 0.2rem 0 0.4rem; }}
.st-key-chat_composer {{
    padding: 0.8rem 1.1rem; background: #F8FAFC; border-top: 1px solid #E2E8F0;
}}
.st-key-chat_composer textarea {{ resize: none !important; padding: 0.6rem 0.85rem; }}
.st-key-chat_composer [data-baseweb="textarea"],
.st-key-chat_composer [data-testid="stTextAreaRootElement"] {{ border-radius: 12px; }}
[class*="st-key-review_send_"] button {{
    background: #059669; border: 1px solid #059669; color: #FFFFFF; border-radius: 10px;
    min-height: 44px; transition: background 0.15s;
}}
[class*="st-key-review_send_"] button p {{ color: #FFFFFF; font-weight: 600; }}
[class*="st-key-review_send_"] button:hover {{ background: #047857; border-color: #047857; }}
.st-key-chat_card [data-testid="stAlert"] {{ margin: 0 1.1rem 1rem; }}

/* Cardholder timeline card (views/cardholder.py). */
.st-key-history_card {{
    background: #FFFFFF;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(28, 25, 23, 0.06);
    padding: 0.6rem 0.9rem;
}}
.history-head {{ display: flex; align-items: center; gap: 0.9rem; }}
.history-icon {{
    flex: none;
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font: italic 700 1.35rem Georgia, serif;
    color: #00754F;
    background: #E3FFF4;
    border: 2px solid {PAID};
    box-shadow: 0 0 12px rgba(0, 230, 153, 0.45);
}}
.history-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }}
.history-hint {{ font-size: 0.95rem; color: {MUTED}; }}
.loana-hint {{
    margin: 0.4rem 0 1.2rem;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: #F0FFF9;
    border-left: 4px solid {LOANA};
    color: {LOANA_INK};
}}

/* Shorter run-in at the top. The page uses the full width: the cardholder screen is
   two columns, and a 1180px cap left the copilot column too narrow to read. The
   sign-in page sets its own width (ui/auth.css). */
[data-testid="stMainBlockContainer"] {{
    padding-top: 3rem;
    padding-bottom: 2rem;
    padding-left: clamp(1.5rem, 4vw, 4rem);
    padding-right: clamp(1.5rem, 4vw, 4rem);
    max-width: none;
}}

/* The page title competes with the metrics underneath it at Streamlit's default
   size, and this screen is about the numbers. */
h1 {{ font-size: 2.1rem !important; }}
h2 {{ font-size: 1.5rem !important; }}
h3 {{ font-size: 1.25rem !important; }}
/* One type scale for every screen: bold tight headings, medium muted subtext,
   semibold labels. Streamlit's own heading and caption styles are weaker. */
[data-testid="stApp"] {{
    line-height: 1.55;
    letter-spacing: -0.003em;
    -webkit-font-smoothing: antialiased;
}}
[data-testid="stApp"] :is(h1, h2, h3, h4) {{
    font-weight: 700;
    line-height: 1.25;
    letter-spacing: -0.02em;
    color: {TEXT};
}}
[data-testid="stHeading"] {{ margin-bottom: 0.15rem; }}
[data-testid="stCaptionContainer"] {{ opacity: 1; }}
[data-testid="stCaptionContainer"] p {{
    font-size: 0.9rem;
    font-weight: 500;
    line-height: 1.5;
}}
[data-testid="stWidgetLabel"] p {{
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0;
    color: {TEXT};
}}
[data-testid="stMarkdownContainer"] strong {{ font-weight: 600; }}

/* Metric tiles: the number is the content, the label is the furniture. */
[data-testid="stMetricValue"] {{
    font-size: 2.3rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.78rem !important;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
}}

/* The 💭 triggers are an aid, not an action. As default secondary buttons there
   was a row of grey pills under every metric, louder than the numbers. */
[data-testid="stPopoverButton"] {{
    background: transparent !important;
    border: 1px solid {GRID} !important;
    color: {MUTED} !important;
    padding: 0.1rem 0.5rem !important;
    min-height: 0 !important;
    border-radius: 999px !important;
}}
[data-testid="stPopoverButton"]:hover {{
    border-color: {ACCENT} !important;
    color: {TEXT} !important;
}}

/* Tabs read as navigation rather than as links in a paragraph. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
[data-testid="stTab"] {{
    font-weight: 600;
    letter-spacing: 0.01em;
    padding-left: 0.85rem;
    padding-right: 0.85rem;
}}

/* Bordered containers become cards rather than boxes drawn on the background. */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background: {SURFACE};
    border: 1px solid {GRID};
    border-radius: 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.04);
}}
[data-testid="stMetric"] {{ padding: 0.9rem 1.1rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0.35rem 0.5rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
}}
/* Only the review-request cards lift: they are the ones with a Review button. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.officer-card-head--lift):hover {{
    transform: translateY(-2px);
    border-color: #D6D0CA;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04), 0 10px 24px rgba(0, 0, 0, 0.08);
}}
/* The Loana card and its chat are not white cards. */
.st-key-loana_panel [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent; border: none; box-shadow: none;
}}

/* Officer cards (ui/components.py). Red, amber and green appear only on the band
   badge and the reason arrows, where they mean risk. */
.officer-card-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
}}
.officer-card-title {{
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    line-height: 1.3;
}}
.officer-card-meta {{ font-size: 0.85rem; font-weight: 500; color: {MUTED}; margin-top: 0.15rem; }}
.officer-badge {{
    flex: none;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    border: 1px solid {MUTED};
    white-space: nowrap;
}}
.officer-badge--Low {{ border-color: {LOW}; }}
.officer-badge--Medium {{ border-color: {MEDIUM}; }}
.officer-badge--High {{ border-color: {HIGH}; }}
.officer-quote {{
    margin: 0.25rem 0 0;
    padding-left: 0.8rem;
    border-left: 3px solid {GRID};
    line-height: 1.5;
}}
.officer-arrow--up {{ color: {HIGH}; }}
.officer-arrow--down {{ color: {LOW}; }}

[data-testid="stCaptionContainer"] p {{ color: {CAPTION}; line-height: 1.5; }}
hr {{ border-color: {GRID} !important; opacity: 1; }}
[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

/* Landing hero: room between the sections, and the illustration as a soft card. */
[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) {{ gap: 3.5rem; }}
[data-testid="stImage"] img {{
    border-radius: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.04);
}}
[data-testid="stMainBlockContainer"] hr {{ margin: 1.75rem 0; }}

/* Officer landing: Loana's briefing leads the page as her emerald card. */
.st-key-officer_briefing {{
    background: linear-gradient(135deg, {LOANA} 0%, #19D3A2 55%, #4FD8C9 100%);
    border-radius: 22px;
    padding: 1.4rem 1.6rem 1.2rem;
    box-shadow: 0 16px 44px rgba(7, 59, 45, 0.28);
}}
.st-key-officer_briefing .loana-head img {{ width: 64px; height: 64px; }}
.st-key-officer_briefing .loana-head {{ padding-bottom: 0.4rem; }}
.st-key-officer_briefing h3 {{ color: #0F172A; padding: 0.2rem 0 0.1rem; }}
.st-key-officer_briefing p {{ font-size: 1.05rem; line-height: 1.6; max-width: 62rem; }}
.st-key-officer_briefing strong {{ color: #04261C; }}
.st-key-officer_suggestions {{ gap: 0.5rem; padding-top: 0.4rem; }}
.st-key-officer_briefing button {{
    box-shadow: 0 2px 8px rgba(7, 59, 45, 0.15);
    padding: 0.35rem 1rem;
}}
.st-key-officer_briefing button[kind="primary"] {{
    background: #0F172A;
    color: #FFFFFF;
}}
.st-key-officer_briefing button[kind="primary"] p {{ color: #FFFFFF; }}
.st-key-officer_briefing button[kind="primary"]:hover {{ background: #1E293B; color: #FFFFFF; }}

/* Customer page: the emerald action. */
.st-key-predict_next_payment button {{
    background: #00A669; border-color: #00A669; color: #FFFFFF;
}}
.st-key-predict_next_payment button:hover {{
    background: #047857; border-color: #047857; color: #FFFFFF;
}}
.st-key-predict_next_payment button p {{ color: #FFFFFF; }}

/* One command bar for search and Loana: the biggest input on the page. */
.st-key-officer_command_bar {{ margin-top: 0.6rem; }}
.st-key-officer_command_bar [data-testid="stChatInput"] {{
    background: #FFFFFF;
    border: 1px solid {GRID};
    border-radius: 16px;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.08);
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}}
.st-key-officer_command_bar [data-testid="stChatInput"]:focus-within {{
    border-color: {LOANA};
    box-shadow: 0 0 0 4px rgba(0, 197, 136, 0.18), 0 6px 24px rgba(15, 23, 42, 0.08);
}}
.st-key-officer_command_bar textarea {{ font-size: 1.05rem; }}
.st-key-officer_reply [data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.85);
    border-radius: 14px;
    padding: 0.7rem 1rem;
}}
.st-key-officer_matches {{ gap: 0.5rem; }}

/* The worklist as a card of rows, each with its own way into the dashboard. */
.st-key-officer_worklist {{
    background: #FFFFFF;
    border: 1px solid {GRID};
    border-radius: 14px;
    padding: 0.4rem 1rem;
    gap: 0;
}}
.st-key-officer_worklist [data-testid="stHorizontalBlock"] {{ gap: 0.75rem; }}
.st-key-officer_worklist p {{
    font-size: 0.9rem;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.worklist-heading {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: {MUTED};
    white-space: nowrap;
}}
[class*="st-key-worklist_row_"] {{
    border-top: 1px solid {GRID};
    padding: 0.15rem 0;
    transition: background 0.12s ease;
}}
[class*="st-key-worklist_row_"]:hover {{ background: #F3FBF8; }}
[class*="st-key-worklist_row_"] [data-testid="stColumn"]:last-child {{
    display: flex;
    justify-content: flex-end;
}}
[class*="st-key-officer_open_"] button p {{ color: #008F63; font-weight: 600; white-space: nowrap; }}
[class*="st-key-officer_open_"] button:hover p {{ color: {LOANA_INK}; }}
</style>
"""


@alt.theme.register("credit_risk", enable=True)
def _chart_theme() -> alt.theme.ThemeConfig:
    """Shared chart styling, so no tab has to restate fonts, grid or axis colour."""
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": "transparent",
                "font": FONT,
                "view": {"stroke": "transparent", "continuousHeight": 300},
                "axis": {
                    "labelColor": MUTED,
                    "titleColor": MUTED,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "titleFontWeight": 600,
                    "titlePadding": 10,
                    "gridColor": GRID,
                    "domainColor": GRID,
                    "tickColor": GRID,
                },
                "legend": {
                    "labelColor": TEXT,
                    "titleColor": MUTED,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "symbolType": "square",
                },
                "title": {"color": TEXT, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
                "range": {"category": [ACCENT, LOW, MEDIUM, HIGH, MUTED]},
                "bar": {"color": ACCENT, "cornerRadiusEnd": 3},
                "line": {"color": ACCENT, "strokeWidth": 2},
                "point": {"color": ACCENT, "filled": True},
                "rule": {"color": MUTED},
            }
        }
    )


def apply() -> None:
    """Inject the stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
