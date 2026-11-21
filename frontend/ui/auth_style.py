"""Style helper for the sign-in screen: injects ui/auth.css with its images inlined.

Kept apart from `components.stylesheet` because the logo is inlined too. Both
images are declared once (`__BACKDROP__` in the CSS, `--brand-logo` on :root)
rather than per use, so the markup carries one copy of each.
"""
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

UI = Path(__file__).resolve().parent
ASSETS = UI.parent / "assets"
CSS_PATH = UI / "auth.css"
BACKDROP_PATH = ASSETS / "gradient-bg.jpg"
LOGO_PATH = ASSETS / "creditrait-logo.png"
PHOTO_PATH = ASSETS / "login-bg.jpg"


def inject() -> None:
    st.markdown(_stylesheet(), unsafe_allow_html=True)


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    for path in (CSS_PATH, BACKDROP_PATH, LOGO_PATH, PHOTO_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. The sign-in screen ships with it; restore it "
                "from version control."
            )
    backdrop = base64.b64encode(BACKDROP_PATH.read_bytes()).decode("ascii")
    logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    css = CSS_PATH.read_text(encoding="utf-8").replace(
        "__BACKDROP__", f"data:image/jpeg;base64,{backdrop}"
    )
    photo = base64.b64encode(PHOTO_PATH.read_bytes()).decode("ascii")
    root = (
        ":root { "
        f'--brand-logo: url("data:image/png;base64,{logo}"); '
        f'--login-photo: url("data:image/jpeg;base64,{photo}"); }}'
    )
    return f"<style>{root}\n{css}</style>"
