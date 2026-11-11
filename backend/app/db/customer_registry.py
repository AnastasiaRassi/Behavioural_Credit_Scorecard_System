"""Create a customer, find one by account number, and give one a login.

Ingestion covers the dataset; this covers the other way a customer appears — an
officer keying in someone who was never in the file. Both end at the same tables
(`users`, `user_details`, `customer`, `monthly_statement`), so a walk-in is
queryable by exactly the same helpers.

Every customer is a user, with or without a login. Writes only what was
submitted. A customer that already exists is returned unchanged: their stored
statements came from the source file, and a re-score is not grounds for rewriting
the history the label was observed against. What was actually submitted is kept
on `prediction.raw_input` either way.
"""
import uuid
from typing import Sequence

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    MONTHS_OBSERVED,
    Customer,
    MonthlyStatement,
    User,
    UserDetails,
)
from app.ml.feature_engineering import clean

# Walk-ins are prefixed so they can never collide with a source id, which is
# always bare digits.
WALK_IN_PREFIX = "W-"

# Invented people. Drawn from a fixed list rather than a generator library so
# the names are reproducible, reviewable, and add no dependency. Obviously
# fictional: the dataset's customers are anonymous, so these stand in for the
# name a real deployment would hold.
FIRST_NAMES = (
    "Nour", "Karim", "Layla", "Fadi", "Maya", "Tarek", "Rima", "Jad", "Hiba",
    "Omar", "Dana", "Ziad", "James", "Sofia", "Wei", "Priya", "Mateo", "Aisha",
    "Liam", "Yuki", "Elena", "Kwame", "Chloe", "Arjun", "Isabella", "Noah",
    "Fatima", "Lucas", "Mei", "Olivia", "Diego", "Amara", "Ethan", "Hana", "Pavel",
    "Zara", "Henrik", "Camila", "Ravi", "Grace", "Mohammed", "Ingrid", "Tomas",
    "Leila", "Daniel", "Anya", "Samuel", "Nia", "Rami", "Joelle", "Georges",
    "Carla", "Elie", "Joumana", "Bilal", "Yasmine", "Walid", "Lina", "Hadi",
    "Salma", "Marwan", "Reem", "Anwar", "Tala", "Samir", "Rana", "Kamal", "Nadine",
    "Ali", "Aya", "Hassan", "Mira", "Youssef", "Celine", "Ahmad", "Jana", "Rayan",
    "Lara", "Majd", "Noura", "Rabih", "Sara", "Tony", "Michelle", "Charbel",
    "Perla", "Mario", "Rita", "Antoine", "Maria", "Peter", "Emma", "Oliver",
    "Sophie", "Jack", "Hannah", "Henry", "Lea", "Felix", "Clara", "Hugo", "Alice",
    "Leo", "Nora", "Max", "Ida", "Oscar", "Ella", "Anton", "Mila", "Jonas", "Greta",
    "Marco", "Giulia", "Luca", "Chiara", "Andrei", "Irina", "Dmitri", "Katya",
    "Jin", "Sakura", "Hiro", "Aiko", "Min", "Soo", "Jun", "Hyun", "Anil", "Deepa",
    "Vikram", "Sunita", "Imran", "Nadia", "Tariq", "Samira", "Kofi", "Abena",
    "Chidi", "Ngozi", "Thabo", "Zola", "Juan", "Lucia", "Pablo", "Valeria",
)
LAST_NAMES = (
    "Haddad", "Khoury", "Nassar", "Saab", "Gerges", "Mansour", "Aoun", "Chalhoub",
    "Daher", "Fares", "Rizk", "Sleiman", "Smith", "Garcia", "Chen", "Patel",
    "Okafor", "Tanaka", "Muller", "Rossi", "Kowalski", "Nguyen", "Silva",
    "Johansson", "Ivanov", "Kim", "Hernandez", "Dubois", "Mensah", "Ahmed", "Lopez",
    "Brown", "Singh", "Andersen", "Costa", "Novak", "Park", "Williams", "Santos",
    "Petrov", "Yilmaz", "Reyes", "Fischer", "Abebe", "Larsen", "Moreau", "Sato",
    "Clarke", "Karam", "Abboud", "Hamade", "Azar", "Sfeir", "Boustany", "Frem",
    "Gemayel", "Kanaan", "Maalouf", "Tabet", "Zeidan", "Bitar", "Hajj", "Issa",
    "Jaber", "Kassab", "Lahoud", "Matta", "Nahas", "Obeid", "Qassem", "Raad",
    "Salameh", "Tawil", "Wehbe", "Yazbeck", "Zahar", "Atallah", "Baroudi", "Daou",
    "Eid", "Farhat", "Ghanem", "Hobeika", "Itani", "Jreij", "Khalil", "Mouawad",
    "Nakhoul", "Habib", "Najjar", "Sarkis", "Touma", "Younes", "Abi", "Rahme",
    "Murray", "Walker", "Young", "Hughes", "Foster", "Bennett", "Gray", "Reid",
    "Webb", "Stone", "Weber", "Schmidt", "Bauer", "Keller", "Hoffmann", "Wagner",
    "Bianchi", "Romano", "Conti", "Ricci", "Bruno", "Popov", "Sokolov", "Volkov",
    "Lee", "Wong", "Takahashi", "Watanabe", "Yamamoto", "Joshi", "Sharma", "Reddy",
    "Khan", "Hussain", "Mwangi", "Diallo", "Traore", "Osei", "Banda", "Mokoena",
    "Torres", "Ramirez", "Flores", "Vargas",
)

# Odd multiplier, coprime to the pair count, so neighbouring accounts look unrelated.
_SCRAMBLE = 7919


def invent_name(account_number: str) -> str:
    """A stable fictional name for an account number.

    Derived from the number itself, so the same account is always the same person
    and no state has to be kept to make that true. The number is scrambled across
    every first/last pairing, so consecutive accounts get unrelated names and no
    pairing is used more than twice per 2x the pair count of accounts.
    """
    digits = int("".join(filter(str.isdigit, account_number)) or 0)
    pairs = len(FIRST_NAMES) * len(LAST_NAMES)
    index = (digits * _SCRAMBLE) % pairs
    return f"{FIRST_NAMES[index % len(FIRST_NAMES)]} {LAST_NAMES[index // len(FIRST_NAMES)]}"


def refresh_names(session: Session) -> int:
    """Rewrite stored names that no longer match `invent_name`; returns the count.

    Names are stored at creation, so changing the name lists leaves old rows stale.
    Only customers are touched, so staff and demo logins keep their names.
    """
    rows = session.execute(
        select(User.id, User.display_name, Customer.account_number).join(
            Customer, Customer.user_id == User.id
        )
    ).all()
    stale = [
        {"id": user_id, "display_name": invent_name(account)}
        for user_id, current, account in rows
        if current != invent_name(account)
    ]
    if stale:
        session.execute(update(User), stale)
    return len(stale)


def find_by_account_number(session: Session, account_number: str) -> Customer | None:
    return session.scalars(
        select(Customer).where(Customer.account_number == account_number)
    ).first()


def register_from_raw(
    session: Session, raw: dict, account_number: str | None = None
) -> Customer:
    """Find or create the customer a submitted score is about."""
    if account_number:
        existing = find_by_account_number(session, account_number)
        if existing is not None:
            return existing
    else:
        account_number = f"{WALK_IN_PREFIX}{uuid.uuid4().hex[:8]}"

    # Folded through the same call training uses, so the codes stored match the
    # levels the model was fitted on and the CHECK constraints accept them.
    folded = clean(pd.DataFrame([raw])).iloc[0]

    user = User(
        display_name=invent_name(account_number),
        role="cardholder",
        details=UserDetails(
            sex=int(folded["SEX"]),
            age_years=int(folded["AGE"]),
            education=int(folded["EDUCATION"]),
            marriage=int(folded["MARRIAGE"]),
        ),
    )
    customer = Customer(
        account_number=account_number,
        user=user,
        credit_limit=float(folded["LIMIT_BAL"]),
    )
    session.add(customer)
    session.flush()

    for month in range(1, MONTHS_OBSERVED + 1):
        session.add(
            MonthlyStatement(
                customer_id=customer.id,
                month_offset=month,
                repayment_status=int(folded[f"PAY_{month}"]),
                bill_amount=float(folded[f"BILL_AMT{month}"]),
                payment_amount=float(folded[f"PAY_AMT{month}"]),
            )
        )

    session.flush()
    return customer


def give_login(
    session: Session, customer: Customer, email: str, password_hash: str
) -> User:
    """Let this customer sign in. Registration is attaching a login to their user."""
    user = customer.user
    if user.role != "cardholder":
        raise ValueError(
            f"{user.display_name} is a {user.role}; only a cardholder is a customer"
        )
    user.email = email
    user.password_hash = password_hash
    session.flush()
    return user


def display_name_for(session: Session, customer_id: uuid.UUID) -> str | None:
    """A customer's name, or None if there is no such customer."""
    customer = session.get(Customer, customer_id)
    return None if customer is None else customer.user.display_name


def customers_by_account_number(
    session: Session, account_numbers: Sequence[str]
) -> dict[str, Customer]:
    """Customers for several account numbers at once, omitting unknown ones."""
    if not account_numbers:
        return {}
    rows = session.scalars(
        select(Customer)
        .where(Customer.account_number.in_(list(account_numbers)))
        .options(selectinload(Customer.user))
    ).all()
    return {customer.account_number: customer for customer in rows}
