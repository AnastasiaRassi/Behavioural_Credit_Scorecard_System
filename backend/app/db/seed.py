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

Seeds accounts, the model and the demo cardholders' coaching cache. Customers come from the dataset via
db/ingest.py, and predictions are written by the API when someone is scored.
"""
import json
import os
from datetime import datetime, time, timezone
from secrets import token_urlsafe

from sqlalchemy.orm import Session

from app.core.passwords import hash_password
from app.db.customer_registry import link_account
from app.db.database import create_all, session_scope
from app.db.models import Customer, ModelVersion, UserAccount
from app.ml.train import artifacts_dir

# Demo staff: email, name, role.
STAFF = [
    ("officer@example.test", "Rana Haddad", "officer"),
    ("officer2@example.test", "Samir Khoury", "officer"),
    ("engineer@example.test", "Anastasia Al Rassi", "engineer"),
]

# A handful of cardholders so the cardholder role has accounts to open without
# an officer having scored anyone first. Each one is linked to the customer it
# represents, which is how a login reaches a scoring subject at all.
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


# One account per role on a shared, memorable password, for showing the app.
# DEMO_MODE=1 only: these are the credentials printed on a slide, so they must
# never exist in an environment that is not a demo.
DEMO_PASSWORD = "demo"
DEMO_ACCOUNTS = [
    ("officer@demo.test", "Demo Officer", "officer"),
    ("engineer@demo.test", "Demo Engineer", "engineer"),
    ("cardholder@demo.test", "Demo Cardholder", "cardholder"),
]


def demo_mode() -> bool:
    return os.environ.get("DEMO_MODE") == "1"


def demo_password() -> str:
    """A password for a seeded account: from SEED_PASSWORD, else unguessable.

    A random fallback keeps `password_hash` honest without putting a shared
    credential in version control, but nothing can then sign in to these
    accounts. Set SEED_PASSWORD in .env to be able to.
    """
    return os.environ.get("SEED_PASSWORD") or token_urlsafe(32)


def _set_credential(account: UserAccount) -> bool:
    """Bring an existing account to the configured password. True if it changed.

    Rerunning the seed is otherwise a no-op per account, which would leave the
    accounts created before SEED_PASSWORD was set holding a hash of a random
    string nobody recorded — unsignable, with no way to repair them short of
    deleting the rows and the cardholder links hanging off them.
    """
    configured = os.environ.get("SEED_PASSWORD")
    if not configured:
        return False
    account.password_hash = hash_password(configured)
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


def _seed_staff(session: Session) -> tuple[list[str], list[str]]:
    added, recredentialed = [], []
    for email, display_name, role in STAFF:
        existing = session.query(UserAccount).filter_by(email=email).one_or_none()
        if existing:
            if _set_credential(existing):
                recredentialed.append(email)
            continue
        session.add(
            UserAccount(
                email=email,
                display_name=display_name,
                password_hash=hash_password(demo_password()),
                role=role,
            )
        )
        added.append(email)
    return added, recredentialed


def _seed_cardholders(session: Session) -> tuple[list[str], list[str], list[str]]:
    added, not_ingested, recredentialed = [], [], []
    for customer_id, name, email in demo_cardholders():
        customer = session.get(Customer, customer_id)
        if customer is None:
            not_ingested.append(customer_id)
            continue

        existing = session.query(UserAccount).filter_by(email=email).one_or_none()
        if existing is not None:
            if _set_credential(existing):
                recredentialed.append(f"{customer_id} ({name})")
            continue
        if customer.user_account_id is not None:
            continue

        account = UserAccount(
            email=email,
            display_name=name,
            password_hash=hash_password(demo_password()),
            role="cardholder",
        )
        session.add(account)
        session.flush()
        link_account(session, customer, account)
        added.append(f"{customer_id} ({name})")
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
        account = session.query(UserAccount).filter_by(email=email).one_or_none()
        if account is None:
            account = UserAccount(email=email, display_name=display_name, role=role)
            session.add(account)
            added.append(email)
        else:
            account.display_name = display_name
            account.role = role
            updated.append(email)
        account.password_hash = hash_password(DEMO_PASSWORD)
        session.flush()

        if role == "cardholder":
            _link_spare_customer(session, account)

    return added, updated


def _link_spare_customer(session: Session, account: UserAccount) -> None:
    """Give the demo cardholder a scoring subject, so its screen has content.

    Any customer nobody else is linked to will do; without one the cardholder
    screen can only say the account is not registered yet.
    """
    if account.customer is not None:
        return

    from app.ml import worklist

    for row in worklist.all_rows()[:200]:
        customer = session.get(Customer, row["customer_id"])
        if customer is not None and customer.user_account_id is None:
            link_account(session, customer, account)
            return


def _pregenerate_coaching(session: Session) -> list[str]:
    """Cache the demo cardholders' coaching, so the demo renders it without running the search."""
    from app.ai import coaching
    from app.db import statements
    from app.ml import worklist

    people = demo_cardholders()
    customer_ids = [customer_id for customer_id, _, _ in people]
    if coaching.cache_covers(customer_ids):
        return []

    entries = {}
    for customer_id in customer_ids:
        row = worklist.find(customer_id)
        if row is None:
            continue
        try:
            record = statements.raw_input_for(session, customer_id)
        except statements.CustomerNotFound:
            record = None
        entries[customer_id] = coaching.cache_entry(
            customer_id, row["risk_band"], row["reasons"], record
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
        cardholders, not_ingested, held_recredentialed = _seed_cardholders(session)
        demo_added, demo_updated = seed_demo_accounts(session)
        versions = _seed_model_version(session)
        coaching_cached = _pregenerate_coaching(session)

    return {
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
    for customer_id in added["cardholders"]:
        print(f"cardholder     {customer_id}")
    for email in added["demo"]:
        print(f"demo account   {email} / {DEMO_PASSWORD}")
    for email in added["demo_updated"]:
        print(f"demo reset     {email} / {DEMO_PASSWORD}")
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
