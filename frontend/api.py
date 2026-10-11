"""HTTP client for the backend. The frontend reads nothing from disk.

Replaces the earlier artifact readers. Reading `artifacts/` directly would let a
screen hold data its role is not entitled to and merely decline to draw it: the
cardholder's probability would still be one devtools panel away. The API decides
what each role is sent, so the frontend asks for it.

Shapes are adapted here rather than in the tabs, so the screens keep one vocabulary
whichever endpoint feeds them.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

import streamlit as st

BASE_URL = "http://127.0.0.1:8012"
TIMEOUT_SECONDS = 30

START_COMMAND = "cd backend && uvicorn app.main:app --port 8012"

# Recorded baseline. The status light goes amber if test AUC drifts from this.
BASELINE_AUC = 0.781
AUC_TOLERANCE = 0.02
FOUR_FIFTHS = 0.80


class ServiceUnavailable(RuntimeError):
    pass


def _request(path: str, payload: dict | None = None, method: str | None = None):
    url = BASE_URL + path
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exception:
        detail = exception.read().decode()[:400]
        raise ServiceUnavailable(f"{exception.code} from {path}: {detail}") from exception
    except urllib.error.URLError as exception:
        raise ServiceUnavailable(
            f"Cannot reach the API at {BASE_URL}. Start it with:\n\n`{START_COMMAND}`"
        ) from exception


def _get(path: str, **params):
    supplied = {k: v for k, v in params.items() if v is not None}
    query = f"?{urllib.parse.urlencode(supplied)}" if supplied else ""
    return _request(path + query)


def _post(path: str, payload: dict | None = None, **params):
    supplied = {k: v for k, v in params.items() if v is not None}
    query = f"?{urllib.parse.urlencode(supplied)}" if supplied else ""
    return _request(path + query, payload=payload or {})


def guard(exception: ServiceUnavailable) -> None:
    """Show a failure the user can act on, then stop the script."""
    st.error(str(exception))
    st.stop()


# --- roles ----------------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def roles() -> list[dict]:
    return _get("/roles")


@st.cache_data(ttl=60, show_spinner=False)
def users(role: str | None = None) -> list[dict]:
    return _get("/users", role=role)


# --- ML engineer ----------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def config() -> dict:
    """The model report, keyed as the engineer tabs already expect."""
    report = dict(_get("/report"))
    # /report publishes the operating point as `threshold`; the tabs grew up on
    # the artifact's name for it.
    report["threshold_f1_optimal"] = report["threshold"]
    return report


@st.cache_data(ttl=60, show_spinner=False)
def sweep() -> list[dict]:
    return _get("/report/threshold-sweep")


@st.cache_data(ttl=60, show_spinner=False)
def cost_curves() -> dict:
    return _get("/report/cost-curve")


@st.cache_data(ttl=60, show_spinner=False)
def importance() -> list[dict]:
    return _get("/report/feature-importance")


@st.cache_data(ttl=60, show_spinner=False)
def fairness() -> dict:
    """Just the per-attribute detail, which is what the fairness tab renders."""
    return _get("/fairness")["detail"]


@st.cache_data(ttl=60, show_spinner=False)
def fairness_verdict() -> dict:
    return _get("/fairness")


@st.cache_data(ttl=60, show_spinner=False)
def registry() -> list[dict]:
    """Registered versions, flattened into the row shape the registry tab draws."""
    rows = []
    for version in _get("/model-versions"):
        metrics = version.get("metrics") or {}
        rows.append(
            {
                "version": version["version"],
                "trained": version["trained_on"][:10],
                "roc_auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "fairness_passed": metrics.get("fairness_passed"),
                "fairness_total": metrics.get("fairness_total"),
                "status": "live" if version["is_live"] else "candidate",
            }
        )
    return rows


@st.cache_data(ttl=30, show_spinner=False)
def drift() -> dict:
    return _get("/monitoring/drift")


@st.cache_data(ttl=30, show_spinner=False)
def performance() -> dict:
    return _get("/monitoring/performance")


def explain(customer: dict) -> dict:
    """Score a hypothetical customer without recording it. Never cached."""
    return _post("/explain", customer)


# --- officer --------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner="Loading the queue…")
def queue(
    band: str | None = None,
    flagged: bool | None = None,
    search: str | None = None,
    minimum_months_in_arrears: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    return _get(
        "/officer/queue",
        band=band,
        flagged=flagged,
        search=search,
        minimum_months_in_arrears=minimum_months_in_arrears,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=30, show_spinner=False)
def search_customers(q: str, limit: int = 10) -> list[dict]:
    """Find a customer by name, email or account number."""
    return _get("/officer/customers", q=q, limit=limit)


@st.cache_data(ttl=10, show_spinner=False)
def customer_detail(customer_id: str) -> dict:
    return _get(f"/officer/customers/{customer_id}")


def predict(customer_id: str, scored_by: str) -> dict:
    """The officer's predict action. Stores a prediction, so never cached."""
    return _post(f"/officer/customers/{customer_id}/predict", scored_by=scored_by)


def decide(customer_id: str, decided_by: str, action: str, justification: str) -> dict:
    return _post(
        f"/officer/customers/{customer_id}/decisions",
        {"decided_by": decided_by, "action": action, "justification": justification},
    )


@st.cache_data(ttl=5, show_spinner=False)
def decisions(customer_id: str) -> list[dict]:
    return _get(f"/officer/customers/{customer_id}/decisions")


# --- cardholder -----------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def cardholder(customer_id: str) -> dict:
    """Deliberately narrower than the officer's view: no probability is sent."""
    return _get(f"/cardholders/{customer_id}")


# --- derived --------------------------------------------------------------


def health() -> tuple[str, str, str]:
    """Status light: (colour, label, reason)."""
    verdict = fairness_verdict()
    if not verdict["passes"]:
        return (
            "🔴", "FAIRNESS BREACH",
            f"{', '.join(verdict['breaches'])} below the four-fifths rule",
        )

    auc = config()["test_metrics"]["roc_auc"]
    if abs(auc - BASELINE_AUC) > AUC_TOLERANCE:
        return "🟠", "DRIFT WARNING", f"test AUC {auc:.3f} against baseline {BASELINE_AUC:.3f}"

    return "🟢", "HEALTHY", "all fairness checks pass, AUC within tolerance"


def nearest_sweep_row(threshold: float) -> dict:
    """Closest swept row to a threshold, since the slider moves continuously."""
    return min(sweep(), key=lambda row: abs(row["threshold"] - threshold))
