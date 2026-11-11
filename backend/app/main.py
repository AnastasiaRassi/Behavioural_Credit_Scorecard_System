"""FastAPI application.

Run from the backend/ directory:

    uvicorn app.main:app --reload

Routes translate HTTP and nothing else: scoring lives in `ml/scorer.py`,
explanations in `ml/explainability.py`, drift in `ml/drift.py`, and every
reported metric is read from artifacts/ rather than recomputed.

`POST /session` verifies a password and returns a signed, expiring token; the
caller presents it as `Authorization: Bearer <token>` and the role is read from
the account, never from the request. Reading a cardholder's account is guarded
by it (`dependencies.account_reader`). The officer's own write paths still name
the acting officer by email in the request body, which records who acted without
proving it — the next thing to close.
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from app.api import (
    admin,
    cardholders,
    decisions,
    fairness,
    monitoring,
    officer,
    predictions,
    recourse,
    roles,
    sessions,
)
from app.core.logging_config import configure_logging, request_id
from app.db.database import create_all
from app.ml import artifacts

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Creating missing tables on startup keeps a prototype runnable from a clean
    # database. A migration tool would replace this if the schema had to evolve
    # against data worth keeping.
    create_all()
    try:
        config = artifacts.scoring_config()
        logger.info(
            "startup: model=%s trained=%s threshold=%s",
            config["model_version"],
            config["trained"],
            config.get("threshold_f1_optimal"),
        )
    except artifacts.ArtifactsMissing as exception:
        logger.error("startup: artifacts missing, scoring will return 503: %s", exception)
    yield
    logger.info("shutdown")


app = FastAPI(
    lifespan=lifespan,
    title="Credit Risk · Behavioural Scorecard",
    description=(
        "Scores existing cardholders for the risk of missing their next payment, "
        "explains each score, and audits itself for fairness."
    ),
    version="1.0.0",
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request: method, path, status, time. The query string is left out
    because it carries emails and search terms."""
    token = request_id.set(uuid.uuid4().hex[:8])
    started = time.perf_counter()
    client = request.client.host if request.client else "-"
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("%s %s -> unhandled error client=%s", request.method, request.url.path, client)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
    # Health checks poll constantly; keep them out of the INFO stream.
    level = logging.DEBUG if request.url.path == "/health" else (
        logging.WARNING if response.status_code >= 400 else logging.INFO
    )
    logger.log(
        level,
        "%s %s -> %s %dms client=%s",
        request.method, request.url.path, response.status_code, elapsed_ms, client,
    )
    response.headers["X-Request-ID"] = request_id.get()
    request_id.reset(token)
    return response


app.include_router(sessions.router)
app.include_router(roles.router)
app.include_router(officer.router)
app.include_router(cardholders.router)
app.include_router(recourse.router)
app.include_router(predictions.router)
app.include_router(decisions.router)
app.include_router(fairness.router)
app.include_router(admin.router)
app.include_router(monitoring.router)


@app.get("/health", tags=["service"])
def health() -> dict:
    """Whether the service can actually serve: artifacts present, model live."""
    try:
        config = artifacts.scoring_config()
    except artifacts.ArtifactsMissing as exception:
        return {"status": "unprovisioned", "detail": str(exception)}

    return {
        "status": "ok",
        "model_version": config["model_version"],
        "trained": config["trained"],
        "n_features": config["n_features"],
    }
