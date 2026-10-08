"""Seed the demo accounts and register the model that artifacts/ currently holds.

Run from the backend/ directory:

    python -m app.db.seed

Idempotent: rerunning changes nothing, so it is safe after every retrain. The
model version is read from scoring_config.json rather than typed here, so the
registered version can never claim to be a model the artifacts do not contain.

Seeds people and the model only. Predictions are written by the API when a
customer is actually scored.
"""
from datetime import datetime, time, timezone

from app.db.database import create_all, session_scope
from app.db.models import ModelVersion, User
from app.ml.train import artifacts_dir

import json

# Demo staff. Cardholders are created by the API as they are scored, so none are
# seeded here; a behavioural scorecard has no queue of applicants waiting.
STAFF = [
    ("officer@example.test", "Rana Haddad", "officer"),
    ("officer2@example.test", "Samir Khoury", "officer"),
    ("engineer@example.test", "Anastasia Al Rassi", "engineer"),
]


def live_model() -> dict:
    """Version identity from the saved config, which train.py wrote."""
    with open(artifacts_dir() / "scoring_config.json", encoding="utf-8") as handle:
        config = json.load(handle)

    # scoring_config stores the training date as a plain day.
    trained = datetime.combine(
        datetime.strptime(config["trained"], "%Y-%m-%d").date(),
        time.min,
        tzinfo=timezone.utc,
    )
    return {"version": config["model_version"], "trained_on": trained}


def seed() -> dict:
    create_all()
    added = {"users": [], "model_versions": []}

    with session_scope() as session:
        for email, display_name, role in STAFF:
            if session.query(User).filter_by(email=email).one_or_none():
                continue
            session.add(User(email=email, display_name=display_name, role=role))
            added["users"].append(email)

        identity = live_model()
        existing = (
            session.query(ModelVersion)
            .filter_by(version=identity["version"])
            .one_or_none()
        )
        if existing is None:
            # The partial unique index allows only one live row, so demote any
            # earlier model before registering this one as live.
            session.query(ModelVersion).filter_by(is_live=True).update(
                {"is_live": False}
            )
            session.flush()
            session.add(ModelVersion(**identity, is_live=True))
            added["model_versions"].append(identity["version"])

    return added


def main() -> dict:
    added = seed()
    if not added["users"] and not added["model_versions"]:
        print("Nothing to add; the database is already seeded.")
    else:
        for email in added["users"]:
            print(f"user           {email}")
        for version in added["model_versions"]:
            print(f"model version  {version} (live)")
    return added


if __name__ == "__main__":
    main()
