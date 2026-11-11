"""Seed the demo accounts and register the model that artifacts/ currently holds.

Run from the backend/ directory, after ingestion:

    python -m app.db.ingest
    python -m app.db.seed

Idempotent: rerunning adds nothing twice, so it is safe after every retrain. The
one thing it does rewrite is a seeded account's password, and only when
SEED_PASSWORD is set, so accounts created before it was configured can be signed
into without deleting them. The model version is read from scoring_config.json
rather than typed here, so the registered version can never claim to be a model
the artifacts do not contain.

With DEMO_MODE=1 it also creates one account per role on the password `demo`,
so the app can be shown without anyone typing a generated credential.

Seeds staff, the model, the demo cardholders' logins and coaching cache, and one
sample message to the officer. Customers
(each already a user) come from the dataset via db/ingest.py, so "registering" a
cardholder here means giving an ingested customer's user an email and password.
Predictions are written by the API when someone is scored.
"""
import json
import os
from datetime import datetime, time, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.passwords import hash_password
from app.db import reviews
from app.db.customer_registry import (
    find_by_account_number,
    give_login,
    invent_name,
)
from app.db.database import create_all, session_scope
from app.db.models import Customer, ModelVersion, User
from app.ml.train import artifacts_dir

# Demo staff: email, name, role.
STAFF = [
    ("officer@example.test", "Rana Haddad", "officer"),
    ("officer2@example.test", "Samir Khoury", "officer"),
    ("engineer@example.test", "Anastasia Al Rassi", "engineer"),
]

# A handful of cardholders so the cardholder role has accounts to open without
# an officer having scored anyone first. Each one is a customer whose user is
# given a login, which is how a login reaches a scoring subject at all.
DEMO_CARDHOLDERS = 5

SEED_REVIEW_NOTE = (
    "I have fallen behind because my income dropped this year. "
    "Could someone talk me through my options?"
)

# One account per role on a shared, memorable password, for showing the app.
# DEMO_MODE=1 only: these are the credentials printed on a slide, so they must
# never exist in an environment that is not a demo.
DEMO_PASSWORD = "demo"
DEMO_ACCOUNTS = [
    ("officer@demo.test", "Demo Officer", "officer"),
    ("engineer@demo.test", "Demo Engineer", "engineer"),
    ("cardholder@demo.test", "Demo Cardholder", "cardholder"),
]


def email_for(name: str, account_number: str, taken: set[str]) -> str:
    """`first.last@example.test`; the account number breaks a tie between namesakes."""
    handle = name.lower().replace(" ", ".")
    email = f"{handle}@example.test"
    if email in taken:
        email = f"{handle}.{account_number}@example.test"
    taken.add(email)
    return email


def demo_mode() -> bool:
    return os.environ.get("DEMO_MODE") == "1"


def demo_password() -> str:
    """A password for a seeded account: SEED_PASSWORD, else the shared demo password."""
    return os.environ.get("SEED_PASSWORD") or DEMO_PASSWORD


def _set_credential(account: User) -> bool:
    """Bring an existing account to the seed password on every run."""
    account.password_hash = hash_password(demo_password())
    return True


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


def demo_cardholders(count: int = DEMO_CARDHOLDERS) -> list[tuple[str, str, str]]:
    """(account number, name, email) for a spread of queued customers.

    The riskiest, the safest, and the middle. Picked by position in the risk ordering rather than at random, so the same
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
    taken: set[str] = set()
    for row in chosen:
        account_number = row["account_number"]
        name = invent_name(account_number)
        people.append((account_number, name, email_for(name, account_number, taken)))
    return people


def _seed_staff(session: Session) -> tuple[list[str], list[str]]:
    added, recredentialed = [], []
    for email, display_name, role in STAFF:
        existing = session.query(User).filter_by(email=email).one_or_none()
        if existing:
            if _set_credential(existing):
                recredentialed.append(email)
            continue
        session.add(
            User(
                email=email,
                display_name=display_name,
                password_hash=hash_password(demo_password()),
                role=role,
            )
        )
        added.append(email)
    return added, recredentialed


def _give_every_cardholder_a_login(session: Session) -> int:
    """Give each cardholder a name-based email and the seed password; returns rows changed.

    Emails are rewritten when the name changes. One hash is shared across the rows:
    scrypt per row would take many minutes for the whole dataset.
    """
    rows = session.execute(
        select(User.id, User.display_name, User.email, User.password_hash, Customer.account_number)
        .join(Customer, Customer.user_id == User.id)
        .where(User.role == "cardholder")
        .where(User.email.is_(None) | User.email.not_like("%@demo.test"))
        .order_by(User.email.is_(None), Customer.account_number)  # keep existing holders
    ).all()
    shared_hash = hash_password(demo_password())
    taken = {
        email
        for (email,) in session.execute(
            select(User.email).where(User.email.isnot(None), User.role != "cardholder")
        )
    }
    updates = []
    for user_id, name, email, password_hash, account_number in rows:
        wanted = email_for(name, account_number, taken)
        if email != wanted or password_hash is None:
            updates.append(
                {"id": user_id, "email": wanted, "password_hash": password_hash or shared_hash}
            )
    if updates:
        session.execute(update(User), updates)
    return len(updates)


def _seed_cardholders(session: Session) -> tuple[list[str], list[str], list[str]]:
    added, not_ingested, recredentialed = [], [], []
    for account_number, name, email in demo_cardholders():
        customer = find_by_account_number(session, account_number)
        if customer is None:
            not_ingested.append(account_number)
            continue

        existing = session.query(User).filter_by(email=email).one_or_none()
        if existing is not None:
            if _set_credential(existing):
                recredentialed.append(f"{account_number} ({name})")
            continue
        if customer.user.can_sign_in:
            continue

        give_login(session, customer, email, hash_password(demo_password()))
        added.append(f"{account_number} ({name})")
    return added, not_ingested, recredentialed


def seed_demo_accounts(session: Session) -> tuple[list[str], list[str]]:
    """The one-click demo accounts. Nothing at all outside demo mode.

    `demo` is far too weak for a real credential, which is the point: it is
    typed in front of an audience. It is hashed with scrypt like every other
    account, so the stored form is no weaker than the rest — and there is no
    strength rule anywhere in the system to relax, only this choice of value.

    Rerunnable: an existing demo account is brought back to the known password
    and role rather than duplicated.
    """
    if not demo_mode():
        return [], []

    added, updated = [], []
    for email, display_name, role in DEMO_ACCOUNTS:
        account = session.query(User).filter_by(email=email).one_or_none()
        if account is None:
            if role == "cardholder":
                account = _spare_customer_user(session)
                if account is None:
                    continue
                account.email = email
            else:
                account = User(email=email, display_name=display_name, role=role)
                session.add(account)
            added.append(email)
        else:
            updated.append(email)
        account.display_name = display_name
        account.role = role
        account.password_hash = hash_password(DEMO_PASSWORD)
        session.flush()

    return added, updated


def _spare_customer_user(session: Session) -> User | None:
    """A customer nobody can sign in as yet, to be the demo cardholder.

    Any queued customer without a login will do; without one the cardholder
    screen has no scoring subject to show.
    """
    from app.ml import worklist

    for row in worklist.all_rows()[:200]:
        customer = find_by_account_number(session, row["account_number"])
        if customer is not None and not customer.user.can_sign_in:
            return customer.user
    return None


def _seed_review_request(session: Session) -> list[str]:
    """One request from the riskiest demo cardholder, so the officer's list is never empty.

    A flagged customer asking for help is the case the feature exists for. Anyone else
    makes theirs from the cardholder screen.
    """
    people = demo_cardholders()
    if not people:
        return []

    account_number, name, _ = people[0]
    customer = find_by_account_number(session, account_number)
    if customer is None or reviews.open_for(session, customer.id) is not None:
        return []

    reviews.open_request(session, customer.id, SEED_REVIEW_NOTE)
    return [f"{account_number} ({name})"]


def _pregenerate_coaching(session: Session) -> list[str]:
    """Cache the demo cardholders' coaching, so the demo renders it without running the search."""
    from app.ai import coaching
    from app.db import statements
    from app.ml import worklist

    people = demo_cardholders()
    # Keyed by account number, the dataset's id, so the cache survives a database
    # rebuild that mints new customer UUIDs.
    account_numbers = [account_number for account_number, _, _ in people]
    if coaching.cache_covers(account_numbers):
        return []

    entries = {}
    for account_number in account_numbers:
        row = worklist.find(account_number)
        if row is None:
            continue
        customer = find_by_account_number(session, account_number)
        try:
            record = (
                statements.raw_input_for(session, customer.id)
                if customer is not None
                else None
            )
        except statements.CustomerNotFound:
            record = None
        entries[account_number] = coaching.cache_entry(
            account_number, row["risk_band"], row["reasons"], record
        )

    coaching.write_cache(entries)
    return sorted(entries)


def _seed_model_version(session: Session) -> list[str]:
    identity = live_model()
    existing = (
        session.query(ModelVersion)
        .filter_by(version=identity["version"])
        .one_or_none()
    )
    if existing is not None:
        return []

    # The partial unique index allows only one live row, so demote any earlier
    # model before registering this one as live.
    session.query(ModelVersion).filter_by(is_live=True).update({"is_live": False})
    session.flush()
    session.add(ModelVersion(**identity, is_live=True))
    return [identity["version"]]


def seed() -> dict:
    create_all()

    with session_scope() as session:
        users, staff_recredentialed = _seed_staff(session)
        _give_every_cardholder_a_login(session)
        cardholders, not_ingested, held_recredentialed = _seed_cardholders(session)
        demo_added, demo_updated = seed_demo_accounts(session)
        versions = _seed_model_version(session)
        reviews_added = _seed_review_request(session)
        coaching_cached = _pregenerate_coaching(session)

    return {
        "review_requests": reviews_added,
        "users": users,
        "cardholders": cardholders,
        "demo": demo_added,
        "demo_updated": demo_updated,
        "model_versions": versions,
        "coaching": coaching_cached,
        "not_ingested": not_ingested,
        "recredentialed": staff_recredentialed + held_recredentialed,
    }


def main() -> dict:
    added = seed()
    if not any(added.values()):
        print("Nothing to add; the database is already seeded.")
        return added

    for email in added["users"]:
        print(f"user           {email}")
    for account_number in added["cardholders"]:
        print(f"cardholder     {account_number}")
    for email in added["demo"]:
        print(f"demo account   {email} / {DEMO_PASSWORD}")
    for email in added["demo_updated"]:
        print(f"demo reset     {email} / {DEMO_PASSWORD}")
    for account_number in added["review_requests"]:
        print(f"review request {account_number}")
    for version in added["model_versions"]:
        print(f"model version  {version} (live)")
    if added["coaching"]:
        print(f"coaching       {len(added['coaching'])} demo cardholders cached to coaching.json")
    for email in added["recredentialed"]:
        print(f"password set   {email}")
    if added["not_ingested"]:
        missing = ", ".join(added["not_ingested"])
        # Plain hyphen: a Windows console on cp1252 renders an em dash as mojibake.
        print(f"skipped        {missing} - not ingested. Run: python -m app.db.ingest")
    return added


if __name__ == "__main__":
    main()
