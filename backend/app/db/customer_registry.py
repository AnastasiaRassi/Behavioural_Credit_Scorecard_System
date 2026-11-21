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

from app.db.exceptions import NotACardholder
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
# name a real deployment would hold. Split by the dataset's SEX code so a name
# never contradicts the stored gender; unisex names are left out for that reason.
MALE_FIRST_NAMES = (
    "Karim", "Fadi", "Tarek", "Jad", "Omar", "Ziad", "James", "Wei", "Mateo",
    "Liam", "Kwame", "Arjun", "Noah", "Lucas", "Diego", "Ethan", "Pavel", "Henrik",
    "Ravi", "Mohammed", "Tomas", "Daniel", "Samuel", "Rami", "Georges", "Elie",
    "Bilal", "Walid", "Hadi", "Marwan", "Anwar", "Samir", "Kamal", "Ali", "Hassan",
    "Youssef", "Ahmad", "Rayan", "Majd", "Rabih", "Tony", "Charbel", "Mario",
    "Antoine", "Peter", "Oliver", "Jack", "Henry", "Felix", "Hugo", "Leo", "Max",
    "Oscar", "Anton", "Jonas", "Marco", "Luca", "Andrei", "Dmitri", "Hiro", "Kenji",
    "Anil", "Vikram", "Imran", "Tariq", "Kofi", "Chidi", "Thabo", "Juan", "Pablo",
)
FEMALE_FIRST_NAMES = (
    "Layla", "Maya", "Rima", "Hiba", "Dana", "Sofia", "Priya", "Aisha", "Elena",
    "Chloe", "Isabella", "Fatima", "Mei", "Olivia", "Amara", "Hana", "Zara",
    "Camila", "Grace", "Ingrid", "Leila", "Anya", "Nia", "Joelle", "Carla",
    "Joumana", "Yasmine", "Lina", "Salma", "Reem", "Tala", "Rana", "Nadine", "Aya",
    "Mira", "Celine", "Jana", "Lara", "Noura", "Sara", "Michelle", "Perla", "Rita",
    "Maria", "Emma", "Sophie", "Hannah", "Lea", "Clara", "Alice", "Nora", "Ida",
    "Ella", "Mila", "Greta", "Giulia", "Chiara", "Irina", "Katya", "Sakura", "Aiko",
    "Deepa", "Sunita", "Nadia", "Samira", "Abena", "Ngozi", "Lucia", "Valeria",
)
FIRST_NAMES_BY_SEX = {1: MALE_FIRST_NAMES, 2: FEMALE_FIRST_NAMES}
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


def invent_name(account_number: str, sex: int | None) -> str:
    """A stable fictional name for an account number, matching the dataset's SEX code.

    Derived from the number itself, so the same account is always the same person
    and no state has to be kept to make that true. The number is scrambled across
    every first/last pairing, so consecutive accounts get unrelated names.
    """
    first_names = FIRST_NAMES_BY_SEX.get(sex, MALE_FIRST_NAMES + FEMALE_FIRST_NAMES)
    digits = int("".join(filter(str.isdigit, account_number)) or 0)
    pairs = len(first_names) * len(LAST_NAMES)
    index = (digits * _SCRAMBLE) % pairs
    return f"{first_names[index % len(first_names)]} {LAST_NAMES[index // len(first_names)]}"


def refresh_names(session: Session) -> int:
    """Rewrite stored names that no longer match `invent_name`; returns the count.

    Names are stored at creation, so changing the name lists leaves old rows stale.
    Only customers are touched, so staff and demo logins keep their names.
    """
    rows = session.execute(
        select(User.id, User.display_name, Customer.account_number, UserDetails.sex)
        .join(Customer, Customer.user_id == User.id)
        .outerjoin(UserDetails, UserDetails.user_id == User.id)
    ).all()
    stale = [
        {"id": user_id, "display_name": invent_name(account, sex)}
        for user_id, current, account, sex in rows
        if current != invent_name(account, sex)
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
        display_name=invent_name(account_number, int(folded["SEX"])),
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
        raise NotACardholder(
            f"{user.display_name} is a {user.role}; only a cardholder is a customer",
            context={"role": user.role},
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
