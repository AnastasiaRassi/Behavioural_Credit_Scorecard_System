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

# Demo staff.
STAFF = [
    ("officer@example.test", "Rana Haddad", "officer"),
    ("officer2@example.test", "Samir Khoury", "officer"),
    ("engineer@example.test", "Anastasia Al Rassi", "engineer"),
]

# A handful of cardholders so the cardholder role has accounts to open without
# an officer having scored anyone first. The rest of the queue still becomes a
# cardholder on first contact, as before.
#
# display_name stays the queue's customer id, because the bridge in
# db/queue_predictions.py finds a cardholder by it. The person's name lives in
# full_name, so an officer can search for someone the way they would be asked
# for them on the phone.
DEMO_CARDHOLDERS = 5

# Invented people. Drawn from a fixed list rather than a generator library so
# the names are reproducible, reviewable, and add no dependency. Obviously
# fictional, because the dataset's real customers are anonymous and inventing
# plausible identities for them would misrepresent what the data contains.
FIRST_NAMES = (
    "Nour", "Karim", "Layla", "Fadi", "Maya", "Tarek", "Rima", "Jad",
    "Hiba", "Omar", "Dana", "Ziad",
)
LAST_NAMES = (
    "Haddad", "Khoury", "Nassar", "Saab", "Gerges", "Mansour", "Aoun",
    "Chalhoub", "Daher", "Fares", "Rizk", "Sleiman",
)


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


def invent_name(customer_id: str) -> str:
    """A stable fictional name for a customer id.

    Derived from the id itself, so the same account is always the same person
    and no state has to be kept to make that true.
    """
    digits = int("".join(filter(str.isdigit, customer_id)) or 0)
    first = FIRST_NAMES[digits % len(FIRST_NAMES)]
    last = LAST_NAMES[(digits // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def demo_cardholders(count: int = DEMO_CARDHOLDERS) -> list[tuple[str, str, str]]:
    """A spread of queued customers: the riskiest, the safest, and the middle.

    Picked by position in the risk ordering rather than at random, so the same
    accounts are seeded on every run and a demo shows both a warning and an
    all-clear.
    """
    from app.ml import worklist

    rows = worklist.all_rows()
    if not rows:
        return []

    last = len(rows) - 1
    positions = {
        round(last * fraction) for fraction in (0.0, 0.08, 0.5, 0.9, 1.0)
    }
    chosen = [rows[position] for position in sorted(positions)][:count]

    people = []
    for row in chosen:
        customer_id = row["customer_id"]
        name = invent_name(customer_id)
        handle = name.lower().replace(" ", ".")
        people.append((customer_id, name, f"{handle}@example.test"))
    return people


def seed() -> dict:
    create_all()
    added = {"users": [], "model_versions": [], "cardholders": []}

    with session_scope() as session:
        for email, display_name, role in STAFF:
            if session.query(User).filter_by(email=email).one_or_none():
                continue
            session.add(
                User(
                    email=email,
                    display_name=display_name,
                    full_name=display_name,
                    role=role,
                )
            )
            added["users"].append(email)

        for customer_id, name, email in demo_cardholders():
            existing = session.query(User).filter_by(
                display_name=customer_id, role="cardholder"
            ).one_or_none()
            if existing is not None:
                # Backfill a row seeded before full_name existed.
                if existing.full_name is None:
                    existing.full_name = name
                    added["cardholders"].append(f"{customer_id} ({name}, named)")
                continue
            session.add(
                User(
                    email=email,
                    display_name=customer_id,
                    full_name=name,
                    role="cardholder",
                )
            )
            added["cardholders"].append(f"{customer_id} ({name})")

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
    if not any(added.values()):
        print("Nothing to add; the database is already seeded.")
    else:
        for email in added["users"]:
            print(f"user           {email}")
        for customer_id in added["cardholders"]:
            print(f"cardholder     {customer_id}")
        for version in added["model_versions"]:
            print(f"model version  {version} (live)")
    return added


if __name__ == "__main__":
    main()
