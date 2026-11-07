"""HTTP client for the backend. The frontend reads nothing from disk.

Replaces the earlier artifact readers. Reading `artifacts/` directly would let a
screen hold data its role is not entitled to and merely decline to draw it: the
cardholder's probability would still be one devtools panel away. The API decides
what each role is sent, so the frontend asks for it.

Shapes are adapted here rather than in the tabs, so the screens keep one vocabulary
whichever endpoint feeds them.
"""
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import review
import streamlit as st

logger = logging.getLogger(__name__)

# The only thing a screen says when the service fails, whatever the cause.
UNREACHABLE = "CrediTrait can't connect right now. Try again in a moment."

# Overridable so the dashboard can be pointed at an API on another port without
# editing it, which is also what lets two instances run side by side.
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8014")
TIMEOUT_SECONDS = 30

START_COMMAND = "docker compose up --build"

# Recorded baseline. The status light goes amber if test AUC drifts from this.
BASELINE_AUC = 0.781
AUC_TOLERANCE = 0.02
FOUR_FIFTHS = 0.80


class ServiceUnavailable(RuntimeError):
    """A failure of the service, not of the person using it.

    `str(...)` is the one sentence a screen may show. The status code, the URL
    and the body stay in `technical`, which goes to the log: a route that moved
    or a process that died is an operator's problem, and putting FastAPI's JSON
    in front of a loan officer tells them nothing they can act on.
    """

    def __init__(self, technical: str, status: int | None = None):
        super().__init__(UNREACHABLE)
        # Carried so a caller can tell a rejected credential from a dead service
        # without parsing the message.
        self.status = status
        self.technical = technical


class SignInRejected(RuntimeError):
    pass


def _request(
    path: str,
    payload: dict | None = None,
    method: str | None = None,
    token: str | None = None,
    timeout: float = TIMEOUT_SECONDS,
):
    url = BASE_URL + path
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exception:
        detail = exception.read().decode()[:400]
        technical = f"{exception.code} from {method or 'GET'} {url}: {detail}"
        logger.warning("%s", technical)
        raise ServiceUnavailable(technical, status=exception.code) from exception
    except TimeoutError as exception:
        # A read that outlives the timeout raises this bare, not wrapped in URLError.
        # The chat turn is the call most likely to, so it must not take the page down.
        technical = f"no reply from {method or 'GET'} {url} within {timeout}s"
        logger.warning("%s", technical)
        raise ServiceUnavailable(technical) from exception
    except urllib.error.URLError as exception:
        technical = (
            f"cannot reach the API at {BASE_URL} ({exception}). "
            f"Start it with: {START_COMMAND}"
        )
        logger.warning("%s", technical)
        raise ServiceUnavailable(technical) from exception


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


# --- signing in -----------------------------------------------------------


def sign_in(email: str, password: str) -> dict:
    """Exchange credentials for a session: the token, and who the caller is.

    Never cached. A sign-in is not a read, and caching one would hold a
    credential answer longer than the attempt it belongs to.
    """
    try:
        return _request("/session", payload={"email": email, "password": password})
    except ServiceUnavailable as exception:
        if exception.status == 401:
            raise SignInRejected("Email or password is incorrect") from exception
        raise


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


DEMO_REQUESTS = 4


def review_requests() -> list[dict]:
    """Customers asking for a human review: request_id, customer_id, full_name,
    requested_at, reason, status (pending, in_review or resolved).

    DEMO DATA until the backend has the endpoint. Replacing this body with a `_get`
    is the whole change; the officer screen reads only this shape.
    """
    # Registered cardholders first, so the demo has names; then the flagged queue.
    named = {
        user["customer_id"]: user["display_name"]
        for user in users(role="cardholder")
        if user["customer_id"]
    }
    flagged = [row["customer_id"] for row in queue(flagged=True, limit=DEMO_REQUESTS)["rows"]]
    customer_ids = list(dict.fromkeys([*named, *flagged]))[:DEMO_REQUESTS]
    return review.demo_requests(customer_ids, named, datetime.now(timezone.utc))


def chat_before_request(customer_id: str, requested_at: str) -> list[dict]:
    """The customer's assistant turns before they asked for a review: role, content,
    at (ISO). Empty until the backend keeps a readable log for staff."""
    return []


def officer_chat(customer_id: str, messages: list[dict]) -> dict:
    """The officer assistant's next turn. Never cached.

    Its own endpoint, not the cardholder's: the two are scoped to different readers.
    """
    return _request(
        f"/officer/customers/{customer_id}/chat",
        payload={"messages": messages},
        timeout=CHAT_TIMEOUT_SECONDS,
    )


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
def cardholder(customer_id: str, token: str) -> dict:
    """Deliberately narrower than the officer's view: no probability is sent."""
    # The token is an argument rather than read from session state here, so it is
    # part of the cache key: two cardholders must never share a response.
    return _request(f"/cardholders/{customer_id}", token=token)


@st.cache_data(ttl=60, show_spinner=False)
def recourse(customer_id: str, token: str) -> dict:
    """The smallest change that would move this account to a better band.

    Cached longer than the account itself: the search runs the model dozens of
    times, and the answer only moves when the account's own figures do.
    """
    return _request(f"/cardholders/{customer_id}/recourse", token=token)


# OpenAI is cut off at 4 s by the API and the one-month search behind a turn takes
# about a second, so 20 s leaves room for both. A local Ollama model on CPU can
# take longer; the call then fails here with the usual message while the API goes
# on to log its own template fallback.
CHAT_TIMEOUT_SECONDS = 20


def chat(customer_id: str, token: str, messages: list[dict]) -> dict:
    """The assistant's next turn. Never cached: every turn is a new question."""
    return _request(
        f"/cardholders/{customer_id}/chat",
        payload={"messages": messages},
        token=token,
        timeout=CHAT_TIMEOUT_SECONDS,
    )


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
