"""Feature engineering for the Taiwan behavioural scorecard.

Single source of truth shared by the training notebook and the scoring API. If this
drifts from what the model was trained on, predictions become silently wrong, so the
notebook imports from here rather than keeping its own copy.

Target: will the customer miss their next monthly payment, given six months of history.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default_next_month"
PROTECTED = ["SEX", "AGE", "EDUCATION", "MARRIAGE"]

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = ["LIMIT_BAL", *PROTECTED, *PAY_COLS, *BILL_COLS, *AMT_COLS]

# Codebook labels. EDUCATION 0/5/6 and MARRIAGE 0 are undocumented in the source and
# are folded into the existing "other" level rather than dropped.
SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "other"}
AGE_BANDS = [20, 30, 40, 50, 60, 100]
AGE_LABELS = ["21-30", "31-40", "41-50", "51-60", "60+"]

# PAY_n encoding: -2 no balance, -1 paid in full, 0 revolving, 1..8 months in arrears.
PAY_STATUS_LABELS = {
    -2: "no balance",
    -1: "paid in full",
    0: "revolving credit",
}


def project_root() -> Path:
    """Repo root, resolved from this file's location rather than the working directory."""
    return Path(__file__).resolve().parents[3]


def load_taiwan(path: Path | str | None = None) -> pd.DataFrame:
    """Load the UCI dataset from CSV, falling back to the original .xls."""
    root = project_root()
    csv_path = Path(path) if path else root / "data" / "raw" / "taiwan_credit.csv"
    xls_path = root / "data" / "raw" / "default of credit card clients.xls"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xls_path.exists():
        df = pd.read_excel(xls_path, header=1)
        df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})
        df = df.drop(columns=["ID"])
        df.to_csv(csv_path, index=False)
    else:
        raise FileNotFoundError(
            f"Dataset not found at {csv_path} or {xls_path}. "
            "Download from https://archive.ics.uci.edu/dataset/350/"
        )

    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the raw columns into the model's feature set.

    Works with or without the target column so the same code serves training and
    live scoring. Every feature uses only the customer's own row, so there is no
    fitted population statistic and nothing can leak between splits.
    """
    out = df.drop(columns=[TARGET], errors="ignore").copy()
    limit = out["LIMIT_BAL"].replace(0, np.nan)

    # Delinquency profile across the six observed months.
    pay = out[PAY_COLS]
    out["pay_max"] = pay.max(axis=1)
    out["pay_mean"] = pay.mean(axis=1)
    out["pay_months_late"] = (pay > 0).sum(axis=1)
    out["pay_months_revolving"] = (pay == 0).sum(axis=1)
    out["pay_trend"] = out["PAY_1"] - out["PAY_6"]
    out["pay_worsening"] = (pay.diff(axis=1).iloc[:, 1:] > 0).sum(axis=1)

    # Credit utilisation: balance as a share of the limit.
    for i, column in enumerate(BILL_COLS, start=1):
        out[f"util_{i}"] = out[column] / limit
    util = out[[f"util_{i}" for i in range(1, 7)]]
    out["util_mean"] = util.mean(axis=1)
    out["util_max"] = util.max(axis=1)
    out["util_trend"] = out["util_1"] - out["util_6"]
    out["remaining_credit"] = out["LIMIT_BAL"] - out["BILL_AMT1"]

    # Repayment behaviour: share of the previous month's bill actually paid.
    for i in range(1, 6):
        prior_bill = out[f"BILL_AMT{i + 1}"].where(out[f"BILL_AMT{i + 1}"] > 0)
        out[f"repay_ratio_{i}"] = out[f"PAY_AMT{i}"] / prior_bill
    repay = out[[f"repay_ratio_{i}" for i in range(1, 6)]]
    out["repay_ratio_mean"] = repay.mean(axis=1)
    out["repay_ratio_min"] = repay.min(axis=1)
    out["months_zero_payment"] = (out[AMT_COLS] == 0).sum(axis=1)

    out["bill_mean"] = out[BILL_COLS].mean(axis=1)
    out["pay_amt_mean"] = out[AMT_COLS].mean(axis=1)
    out["pay_to_bill_gap"] = out["bill_mean"] - out["pay_amt_mean"]

    return out.replace([np.inf, -np.inf], np.nan)


def build_splits(seed: int = 42, blind: bool = True):
    """Stratified 70/15/15 splits.

    blind=True drops protected attributes from the model inputs and keeps them in a
    separate audit frame. That is the deployed configuration: it costs 0.0007 AUC and
    keeps demographics out of the decision.
    """
    from sklearn.model_selection import train_test_split

    df = load_taiwan()
    X = engineer(df)
    y = df[TARGET].astype("int8")
    audit = df[PROTECTED].copy()

    if blind:
        X = X.drop(columns=PROTECTED)

    idx_train, idx_rest = train_test_split(
        X.index, test_size=0.30, stratify=y, random_state=seed
    )
    idx_validation, idx_test = train_test_split(
        idx_rest, test_size=0.50, stratify=y.loc[idx_rest], random_state=seed
    )

    names = {"train": idx_train, "validation": idx_validation, "test": idx_test}
    frames = {name: X.loc[index] for name, index in names.items()}
    labels = {name: y.loc[index] for name, index in names.items()}
    audits = {name: audit.loc[index] for name, index in names.items()}
    return frames, labels, audits


def age_band(age: int | float) -> str:
    """Label an age into the bands used by the fairness audit."""
    return str(pd.cut([age], AGE_BANDS, labels=AGE_LABELS)[0])


def describe_pay_status(value: int) -> str:
    """Human-readable repayment status for explanations shown to users."""
    if value in PAY_STATUS_LABELS:
        return PAY_STATUS_LABELS[value]
    return f"{int(value)} month{'s' if value != 1 else ''} in arrears"
