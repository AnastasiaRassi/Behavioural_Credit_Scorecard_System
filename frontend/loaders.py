"""Artifact readers. The dashboard never recomputes or retrains.

Everything on screen is loaded from `backend/artifacts`, written by `app.ml.train`.
That keeps the UI fast and guarantees the screen agrees with what was actually
evaluated. Regenerate with `python -m app.ml.train` from `backend/`.
"""
import json
import sys
from pathlib import Path

import streamlit as st

REGENERATE = "cd backend && python -m app.ml.train"

# Recorded baseline. The status light goes amber if test AUC drifts from this.
BASELINE_AUC = 0.781
AUC_TOLERANCE = 0.02
FOUR_FIFTHS = 0.80


def project_root() -> Path:
    """Repo root, resolved from this file rather than the working directory."""
    return Path(__file__).resolve().parents[1]


def artifacts_dir() -> Path:
    return project_root() / "backend" / "artifacts"


# CreditScorer lives under backend/, which is not a package from the repo root.
if str(project_root() / "backend") not in sys.path:
    sys.path.insert(0, str(project_root() / "backend"))


class ArtifactMissing(RuntimeError):
    pass


def _read(name: str) -> dict | list:
    path = artifacts_dir() / name
    if not path.exists():
        st.error(
            f"Missing artifact `{name}`.\n\nRegenerate it with:\n\n`{REGENERATE}`"
        )
        st.stop()
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def config() -> dict:
    """Threshold, band edges, metric suite, confusion matrix, deciles, calibration."""
    return _read("scoring_config.json")


@st.cache_data(show_spinner=False)
def sweep() -> list[dict]:
    return _read("threshold_sweep.json")


@st.cache_data(show_spinner=False)
def cost_curves() -> dict:
    return _read("cost_curve.json")


@st.cache_data(show_spinner=False)
def fairness() -> dict:
    return _read("fairness_detail.json")


@st.cache_data(show_spinner=False)
def importance() -> list[dict]:
    return _read("feature_importance.json")


@st.cache_data(show_spinner=False)
def registry() -> list[dict]:
    return _read("registry.json")


@st.cache_resource(show_spinner="Loading model…")
def explainer():
    """Live model, for the single-prediction explainer only."""
    from app.ml.explainability import Explainer
    from app.ml.scorer import CreditScorer

    scorer = CreditScorer()
    return scorer, Explainer(scorer)


def health() -> tuple[str, str, str]:
    """Status light: (colour, label, reason).

    Green when every fairness check passes and test AUC is within tolerance of the
    recorded baseline. Red on a fairness breach, amber on AUC drift.
    """
    audit = fairness()
    breaches = [name for name, v in audit.items() if not v.get("passes_four_fifths")]
    if breaches:
        return "🔴", "FAIRNESS BREACH", f"{', '.join(breaches)} below the four-fifths rule"

    auc = config()["test_metrics"]["roc_auc"]
    if abs(auc - BASELINE_AUC) > AUC_TOLERANCE:
        return "🟠", "DRIFT WARNING", f"test AUC {auc:.3f} against baseline {BASELINE_AUC:.3f}"

    return "🟢", "HEALTHY", "all fairness checks pass, AUC within tolerance"


def nearest_sweep_row(threshold: float) -> dict:
    """Closest swept row to a threshold, since the slider moves continuously."""
    return min(sweep(), key=lambda row: abs(row["threshold"] - threshold))
