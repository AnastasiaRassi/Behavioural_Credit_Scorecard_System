"""FastAPI application.

Run from the backend/ directory:

    uvicorn app.main:app --reload

Routes translate HTTP and nothing else: scoring lives in `ml/scorer.py`,
explanations in `ml/explainability.py`, drift in `ml/drift.py`, and every
reported metric is read from artifacts/ rather than recomputed.

No authentication. The role selector is a demo affordance, and the officer email
on a request identifies who acted rather than proving it. A real deployment would
put an identity provider in front of this; saying so is more honest than a login
form that checks nothing.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api import admin, decisions, fairness, monitoring, predictions
from app.db.database import create_all
from app.ml import artifacts


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Creating missing tables on startup keeps a prototype runnable from a clean
    # database. A migration tool would replace this if the schema had to evolve
    # against data worth keeping.
    create_all()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Credit Risk · Behavioural Scorecard",
    description=(
        "Scores existing cardholders for the risk of missing their next payment, "
        "explains each score, and audits itself for fairness."
    ),
    version="1.0.0",
)

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
