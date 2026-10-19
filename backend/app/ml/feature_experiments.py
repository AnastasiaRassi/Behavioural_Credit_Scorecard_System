"""Feature-side comparison studies: which inputs are worth including.

`experiments.py` compares models on a fixed feature set. These compare feature sets on a
fixed model, plus the tuning that goes with them. None is a candidate for deployment;
`train.py` still builds the served model.

| Study | Question |
| --- | --- |
| `additions` | Do the extra features engineered in notebook 01 improve the final output? |
| `demographics` | What does excluding the protected attributes cost? |
| `retuning` | Does a configuration tuned on the larger feature set change the answer? |

**Everything is paired across five independent split seeds.** A single split cannot resolve
differences of 0.001 when its own noise is several times that, which is how two earlier
single-split figures came out misleading: 0.0007 for unawareness, and a +0.0008 blend gain
that reversed to -0.0014 once tested on data that had not chosen its weight. The paired
design measures each difference on the same splits, and the spread of that difference is
what decides whether an effect is real.

Each arm runs the deployed pipeline end to end: the three-seed ensemble on train, the
F1-optimal threshold from validation, a refit on train plus validation, then test.

Run `python -m app.ml.feature_experiments` to regenerate `feature_experiments.json`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.ml.feature_engineering import (
    AMT_COLS,
    BILL_COLS,
    PAY_COLS,
    PROTECTED,
    TARGET,
    engineer,
    load_taiwan,
    project_root,
)
from app.ml.train import HYPERPARAMETERS, SEEDS

SEED = 42
SPLIT_SEEDS = (42, 202, 777, 1337, 2024)
ALERT_RATE = 0.20
CONFIGURATIONS = 40

# Months oldest first, so a positive difference means rising towards the present.
RECENT, HISTORY = [1, 2], [3, 4, 5, 6]

# What the notebook calls ADDED: columns engineered beside the served set and tested
# rather than promoted. Grouped so a study can add one concept at a time.
ADDITION_GROUPS = {
    "repayment counts": ["months_cleared_bill", "months_token_payment"],
    "utilisation slope": ["util_slope_relative"],
    "consecutive delinquency": ["delinquency_run_current"],
    "volatility": ["payment_volatility", "util_volatility", "pay_status_volatility"],
    "recent vs historical": ["pay_status_shift", "payment_shift_ratio"],
    "inferred spending": ["spend_months", "spend_mean_to_limit", "spend_to_payment"],
    "roll-rate transitions": ["status_changes", "cure_events", "roll_events"],
    "minimum payment": ["months_at_minimum", "months_below_minimum"],
    "round-number payments": ["round_payment_share"],
    "limit pressure": ["months_at_limit", "months_over_limit"],
    "dormancy": ["dormant_months", "reactivated"],
    "predictability": ["status_entropy", "payment_responsiveness"],
}
ADDITIONS = [column for columns in ADDITION_GROUPS.values() for column in columns]

# Assumed minimum payment as a share of the balance. Unverified against the 2005
# Taiwanese regulation, so anything derived from it is reported as provisional.
MINIMUM_SHARE = 0.10


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def engineer_additions(df: pd.DataFrame) -> pd.DataFrame:
    """The extra columns defined in notebook 01, section 3.

    Kept here rather than in `feature_engineering.py` because they are not served. The
    notebook computes them inline and asserts this function agrees, so the two cannot
    drift apart silently.
    """
    out = pd.DataFrame(index=df.index)
    limit = df["LIMIT_BAL"].replace(0, np.nan)
    util = pd.DataFrame({i: df[f"BILL_AMT{i}"] / limit for i in range(1, 7)}, index=df.index)
    payments = df[AMT_COLS]

    # Repayment counts, over the same ratios feature_engineering already computes.
    repay = pd.DataFrame(
        {i: df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i + 1}"].where(df[f"BILL_AMT{i + 1}"] > 0)
         for i in range(1, 6)},
        index=df.index,
    )
    out["months_cleared_bill"] = (repay >= 0.99).sum(axis=1)
    out["months_token_payment"] = (repay < 0.05).sum(axis=1)

    # Utilisation slope across all six months, relative to the customer's own average.
    months_centred = np.arange(6, dtype=float) - 2.5
    oldest_first = util[[6, 5, 4, 3, 2, 1]].to_numpy()
    centred = oldest_first - np.nanmean(oldest_first, axis=1, keepdims=True)
    slope = np.nansum(centred * months_centred, axis=1) / np.sum(months_centred**2)
    out["util_slope_relative"] = slope / util.mean(axis=1).replace(0, np.nan)

    # Arrears still open at the most recent month.
    late = (df[PAY_COLS[::-1]] > 0).to_numpy()
    open_run = np.zeros(len(df), dtype="int8")
    still_open = np.ones(len(df), dtype=bool)
    for month in reversed(range(late.shape[1])):
        still_open &= late[:, month]
        open_run += still_open.astype("int8")
    out["delinquency_run_current"] = open_run

    # Volatility, scaled by the customer's own average where the scale is theirs.
    out["payment_volatility"] = (payments.std(axis=1)
                                 / payments.mean(axis=1).replace(0, np.nan))
    out["util_volatility"] = util.std(axis=1)
    out["pay_status_volatility"] = df[PAY_COLS].std(axis=1)

    # Recent versus historical.
    out["pay_status_shift"] = (df[[f"PAY_{i}" for i in RECENT]].mean(axis=1)
                               - df[[f"PAY_{i}" for i in HISTORY]].mean(axis=1))
    history_paid = df[[f"PAY_AMT{i}" for i in HISTORY]].mean(axis=1)
    out["payment_shift_ratio"] = (df[[f"PAY_AMT{i}" for i in RECENT]].mean(axis=1)
                                  / history_paid.where(history_paid > 0))

    # Inferred spending, from the balance identity: what was charged in a month is the
    # change in balance plus what was paid.
    spend = pd.DataFrame(
        {i: df[f"BILL_AMT{i}"] - df[f"BILL_AMT{i + 1}"] + df[f"PAY_AMT{i}"]
         for i in range(1, 6)},
        index=df.index,
    )
    out["spend_months"] = (spend > 0).sum(axis=1)
    out["spend_mean_to_limit"] = spend.mean(axis=1) / limit
    paid_total = df[[f"PAY_AMT{i}" for i in range(1, 6)]].sum(axis=1)
    out["spend_to_payment"] = spend.sum(axis=1) / paid_total.where(paid_total > 0)

    # Roll-rate transitions, as a collections team reads them.
    status = df[PAY_COLS[::-1]].to_numpy()
    step = np.diff(status, axis=1)
    was_late = status[:, :-1] > 0
    out["status_changes"] = (step != 0).sum(axis=1)
    out["cure_events"] = (was_late & (status[:, 1:] <= 0)).sum(axis=1)
    out["roll_events"] = (was_late & (step > 0)).sum(axis=1)

    # Minimum-payment behaviour.
    minimum_ratio = pd.DataFrame(
        {i: df[f"PAY_AMT{i}"]
            / (MINIMUM_SHARE * df[f"BILL_AMT{i + 1}"].where(df[f"BILL_AMT{i + 1}"] > 0))
         for i in range(1, 6)},
        index=df.index,
    )
    out["months_at_minimum"] = ((minimum_ratio >= 0.9) & (minimum_ratio <= 1.3)).sum(axis=1)
    out["months_below_minimum"] = (minimum_ratio < 0.9).sum(axis=1)

    # Round-number payments: budgeting rather than reading the statement.
    active = (payments > 0).sum(axis=1)
    out["round_payment_share"] = (((payments % 1000 == 0) & (payments > 0)).sum(axis=1)
                                  / active.replace(0, np.nan))

    # Limit pressure, which util_max averages away.
    out["months_at_limit"] = (util >= 0.95).sum(axis=1)
    out["months_over_limit"] = (util > 1).sum(axis=1)

    # Dormancy and reactivation.
    dormant = (df[BILL_COLS] <= 0).to_numpy() & (payments.to_numpy() == 0)
    out["dormant_months"] = dormant.sum(axis=1)
    out["reactivated"] = (dormant[:, 1:].any(axis=1)
                          & ((df["BILL_AMT1"] / limit) >= 0.5).to_numpy()).astype("int8")

    # Predictability of the status sequence. PAY_n runs -2 to 8.
    shares = np.stack([(status == value).sum(axis=1) for value in range(-2, 9)], axis=1) / 6
    # Unused states contribute nothing; substituting 1 inside the log keeps 0*log(0) = 0
    # without evaluating log(0) first.
    out["status_entropy"] = -(shares * np.log(np.where(shares > 0, shares, 1.0))).sum(axis=1)

    # Does the payment respond to the bill at all, month by month?
    paid = df[[f"PAY_AMT{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    billed = df[[f"BILL_AMT{i + 1}" for i in range(1, 6)]].to_numpy(dtype=float)
    paid_centred = paid - paid.mean(axis=1, keepdims=True)
    billed_centred = billed - billed.mean(axis=1, keepdims=True)
    denominator = np.sqrt((paid_centred**2).sum(axis=1) * (billed_centred**2).sum(axis=1))
    out["payment_responsiveness"] = np.where(
        denominator > 0, (paid_centred * billed_centred).sum(axis=1) / denominator, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)


def build_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Served features, protected attributes and additions in one frame, plus the target."""
    raw = load_taiwan()
    served = engineer(raw)
    additions = engineer_additions(raw)
    return pd.concat([served, additions], axis=1), raw[TARGET].astype("int8")


def split_indices(index: pd.Index, y: pd.Series, seed: int) -> dict:
    train, rest = train_test_split(index, test_size=0.30, stratify=y, random_state=seed)
    validation, test = train_test_split(
        rest, test_size=0.50, stratify=y.loc[rest], random_state=seed)
    return {"train": train, "validation": validation, "test": test}


def evaluate_arm(X: pd.DataFrame, y: pd.Series, index: dict, columns: list[str],
                 params: dict | None = None) -> dict:
    """The deployed pipeline on one split: threshold from validation, then one test read."""
    params = params or HYPERPARAMETERS

    def ensemble(fit_index, score_index):
        return np.mean([
            LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbose=-1)
            .fit(X.loc[fit_index, columns], y.loc[fit_index])
            .predict_proba(X.loc[score_index, columns])[:, 1] for seed in SEEDS], axis=0)

    validation_scores = ensemble(index["train"], index["validation"])
    grid = np.unique(np.quantile(validation_scores, np.linspace(0.01, 0.99, 197)))
    threshold = float(grid[int(np.argmax(
        [f1_score(y.loc[index["validation"]], validation_scores >= t) for t in grid]))])

    development = index["train"].union(index["validation"])
    test_scores = ensemble(development, index["test"])
    y_test = y.loc[index["test"]]
    flagged = test_scores >= np.quantile(test_scores, 1 - ALERT_RATE)
    return {
        "validation_auc": float(roc_auc_score(y.loc[index["validation"]], validation_scores)),
        "test_auc": float(roc_auc_score(y_test, test_scores)),
        "test_f1": float(f1_score(y_test, (test_scores >= threshold).astype(int))),
        "test_recall": float(recall_score(y_test, (test_scores >= threshold).astype(int))),
        "recall_at_alert_rate": float(recall_score(y_test, flagged)),
        "threshold": threshold,
        "features": len(columns),
    }


def paired_study(X: pd.DataFrame, y: pd.Series, arms: dict, reference: str) -> dict:
    """Every arm on every split, then the paired difference against the reference arm."""
    measured = {arm: [] for arm in arms}
    for split_seed in SPLIT_SEEDS:
        index = split_indices(X.index, y, split_seed)
        for arm, columns in arms.items():
            measured[arm].append(evaluate_arm(X, y, index, columns))
            print(f"    seed {split_seed} {arm:<34} "
                  f"auc {measured[arm][-1]['test_auc']:.4f}", flush=True)

    return _summarise(measured, reference)


def additions_study(X: pd.DataFrame, y: pd.Series) -> dict:
    """Do the extra features improve the final output? One arm per feature group."""
    served = [column for column in X.columns
              if column not in ADDITIONS and column not in PROTECTED]
    arms = {"served only": served}
    for group, columns in ADDITION_GROUPS.items():
        arms[f"+ {group}"] = served + columns
    arms["+ every addition"] = served + ADDITIONS
    print("  additions study", flush=True)
    return paired_study(X, y, arms, "served only")


def demographics_study(X: pd.DataFrame, y: pd.Series) -> dict:
    """What does fairness through unawareness cost? Protected attributes are inputs here
    and nowhere else in the project; this measures the price of leaving them out.

    A gain here is not a reason to include them: sex and marital status are prohibited
    inputs to a credit decision and age is restricted. The number tells the report what
    compliance costs, not what to deploy."""
    served = [column for column in X.columns
              if column not in ADDITIONS and column not in PROTECTED]
    arms = {
        "blind (deployed)": served,
        "+ AGE": served + ["AGE"],
        "+ EDUCATION": served + ["EDUCATION"],
        "+ SEX": served + ["SEX"],
        "+ all four": served + PROTECTED,
    }
    print("  demographics study", flush=True)
    return paired_study(X, y, arms, "blind (deployed)")


def retuning_study(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> dict:
    """Greedy group selection, then a fresh search on the selected feature set.

    The deployed hyperparameters were searched against the served features alone, so the
    additions have only ever been judged under a configuration chosen without them.
    """
    print("  retuning study", flush=True)
    index = split_indices(X.index, y, seed)
    served = [column for column in X.columns
              if column not in ADDITIONS and column not in PROTECTED]

    def validation_auc(columns, params=None):
        return float(np.mean([
            roc_auc_score(
                y.loc[index["validation"]],
                LGBMClassifier(**(params or HYPERPARAMETERS), n_jobs=-1,
                               random_state=model_seed, verbose=-1)
                .fit(X.loc[index["train"], columns], y.loc[index["train"]])
                .predict_proba(X.loc[index["validation"], columns])[:, 1])
            for model_seed in SEEDS]))

    baseline = validation_auc(served)
    chosen, current, remaining = [], baseline, dict(ADDITION_GROUPS)
    rounds = []
    while remaining:
        scored = sorted(
            ((validation_auc(served + [c for g in chosen for c in ADDITION_GROUPS[g]]
                             + columns), name)
             for name, columns in remaining.items()), reverse=True)
        best_auc, best_name = scored[0]
        rounds.append({"round": len(chosen) + 1, "best_group": best_name,
                       "validation_auc": best_auc, "delta": best_auc - current})
        print(f"    round {len(chosen) + 1}: {best_name} {best_auc:.4f} "
              f"({best_auc - current:+.4f})", flush=True)
        if best_auc <= current:
            break
        chosen.append(best_name)
        current = best_auc
        remaining.pop(best_name)

    selected = [column for group in chosen for column in ADDITION_GROUPS[group]]
    columns = served + selected

    grid = {
        "n_estimators": [300, 473, 700, 1000],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.4, 0.55, 0.7, 0.85],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [1.0, 10.0, 50.0, 100.0],
    }
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed)
                 .split(X.loc[index["train"], columns], y.loc[index["train"]]))

    def cross_validated(params):
        scores = []
        for fit_rows, score_rows in folds:
            fit_index = index["train"][fit_rows]
            score_index = index["train"][score_rows]
            model = LGBMClassifier(**params, n_jobs=-1, random_state=seed, verbose=-1)
            model.fit(X.loc[fit_index, columns], y.loc[fit_index])
            scores.append(roc_auc_score(
                y.loc[score_index], model.predict_proba(X.loc[score_index, columns])[:, 1]))
        return float(np.mean(scores))

    deployed_cv = cross_validated(HYPERPARAMETERS)
    print(f"    deployed params CV {deployed_cv:.4f}", flush=True)
    rng = np.random.default_rng(seed)
    trials = []
    for trial in range(CONFIGURATIONS):
        params = {key: rng.choice(values).item() for key, values in grid.items()}
        params["subsample_freq"] = 1
        started = time.time()
        trials.append({"params": params, "cv_auc": cross_validated(params),
                       "seconds": round(time.time() - started, 1)})
        print(f"    trial {trial + 1:>2}/{CONFIGURATIONS}: CV {trials[-1]['cv_auc']:.4f}",
              flush=True)

    best = max(trials, key=lambda row: row["cv_auc"])
    return {
        "split_seed": seed,
        "served_validation_auc": baseline,
        "greedy_rounds": rounds,
        "selected_groups": chosen,
        "selected_columns": selected,
        "configurations": CONFIGURATIONS,
        "deployed_cv_auc": deployed_cv,
        "best_search_cv_auc": best["cv_auc"],
        "best_search_params": best["params"],
        "retuning_beats_deployed": bool(best["cv_auc"] > deployed_cv),
        "validation_auc_with_best_params": validation_auc(columns, best["params"]),
    }


def segmentation_study(X: pd.DataFrame, y: pd.Series) -> dict:
    """Is the behavioural segmentation worth modelling on, or only describing with?

    Segmented scorecards are standard in credit: fit one model per behavioural archetype
    instead of one model for everyone. The expectation here is that it loses, because
    boosted trees already partition the space internally and splitting 21,000 training
    rows three ways leaves the smallest segment with a few thousand. Worth measuring
    rather than asserting, since the segments exist either way.

    The segmenter is refitted on each split's training rows, never on validation or test.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from app.ml.segmentation import prepare

    # k = 3, the silhouette winner recorded in segments.json. Fixed rather than re-chosen
    # per split, so every split's arms describe the same segmentation.
    segments = 3

    print("  segmentation study", flush=True)
    served = [column for column in X.columns
              if column not in ADDITIONS and column not in PROTECTED]
    rows = {"global": [], "global + segment": [], "per-segment models": []}

    for split_seed in SPLIT_SEEDS:
        index = split_indices(X.index, y, split_seed)
        prepared = prepare(X[served])
        imputer = SimpleImputer(strategy="median").fit(prepared.loc[index["train"]])
        scaler = StandardScaler().fit(imputer.transform(prepared.loc[index["train"]]))
        kmeans = KMeans(n_clusters=segments, n_init=10, random_state=SEED).fit(
            scaler.transform(imputer.transform(prepared.loc[index["train"]])))
        segment = pd.Series(
            kmeans.predict(scaler.transform(imputer.transform(prepared))),
            index=X.index, name="segment")

        rows["global"].append(evaluate_arm(X, y, index, served))

        with_segment = X.assign(segment=segment)
        rows["global + segment"].append(
            evaluate_arm(with_segment, y, index, served + ["segment"]))

        rows["per-segment models"].append(
            _per_segment_arm(X, y, index, served, segment))
        for arm in rows:
            print(f"    seed {split_seed} {arm:<22} auc {rows[arm][-1]['test_auc']:.4f}",
                  flush=True)

    return _summarise(rows, "global")


def _per_segment_arm(X, y, index, columns, segment) -> dict:
    """One model per segment, each scoring only the customers assigned to it."""

    def scores_for(fit_index, score_index):
        out = pd.Series(np.nan, index=score_index)
        for label in sorted(segment.unique()):
            fit_rows = fit_index[segment.loc[fit_index] == label]
            score_rows = score_index[segment.loc[score_index] == label]
            if len(score_rows) == 0:
                continue
            predictions = np.mean([
                LGBMClassifier(**HYPERPARAMETERS, n_jobs=-1, random_state=seed, verbose=-1)
                .fit(X.loc[fit_rows, columns], y.loc[fit_rows])
                .predict_proba(X.loc[score_rows, columns])[:, 1] for seed in SEEDS], axis=0)
            out.loc[score_rows] = predictions
        return out.to_numpy()

    validation_scores = scores_for(index["train"], index["validation"])
    grid = np.unique(np.quantile(validation_scores, np.linspace(0.01, 0.99, 197)))
    threshold = float(grid[int(np.argmax(
        [f1_score(y.loc[index["validation"]], validation_scores >= t) for t in grid]))])

    development = index["train"].union(index["validation"])
    test_scores = scores_for(development, index["test"])
    y_test = y.loc[index["test"]]
    flagged = test_scores >= np.quantile(test_scores, 1 - ALERT_RATE)
    return {
        "validation_auc": float(roc_auc_score(y.loc[index["validation"]], validation_scores)),
        "test_auc": float(roc_auc_score(y_test, test_scores)),
        "test_f1": float(f1_score(y_test, (test_scores >= threshold).astype(int))),
        "test_recall": float(recall_score(y_test, (test_scores >= threshold).astype(int))),
        "recall_at_alert_rate": float(recall_score(y_test, flagged)),
        "threshold": threshold,
        "features": len(columns),
    }


def selection_study(X: pd.DataFrame, y: pd.Series) -> dict:
    """Does an automatically selected subset beat using every column?

    Columns are ranked by permutation importance measured on validation, never on test,
    and the ranking is recomputed inside each split so no selection sees the data it is
    scored on. The arms walk down that ranking. This is the selection half of the claim
    that the feature space was searched rather than guessed; the extraction half, a
    feature bank such as tsfresh, is not done.
    """
    from sklearn.inspection import permutation_importance

    print("  selection study", flush=True)
    candidates = [column for column in X.columns if column not in PROTECTED]
    sizes = [10, 20, 30, 40, 50, 60]
    rows = {f"top {size}": [] for size in sizes}
    rows["all columns"] = []
    rows["positive importance only"] = []
    rankings = []

    for split_seed in SPLIT_SEEDS:
        index = split_indices(X.index, y, split_seed)
        ranker = LGBMClassifier(**HYPERPARAMETERS, n_jobs=-1, random_state=SEED, verbose=-1)
        ranker.fit(X.loc[index["train"], candidates], y.loc[index["train"]])
        importance = permutation_importance(
            ranker, X.loc[index["validation"], candidates], y.loc[index["validation"]],
            scoring="roc_auc", n_repeats=3, random_state=SEED)
        ranked = pd.Series(importance.importances_mean, index=candidates).sort_values(
            ascending=False)
        rankings.append({"split_seed": split_seed,
                         "order": ranked.index.tolist(),
                         "importance": [float(v) for v in ranked.to_numpy()]})

        arms = {f"top {size}": ranked.index[:size].tolist() for size in sizes}
        arms["all columns"] = candidates
        arms["positive importance only"] = ranked[ranked > 0].index.tolist() or candidates
        for arm, columns in arms.items():
            rows[arm].append(evaluate_arm(X, y, index, columns))
            print(f"    seed {split_seed} {arm:<26} {len(columns):>3} cols "
                  f"auc {rows[arm][-1]['test_auc']:.4f}", flush=True)

    summary = _summarise(rows, "all columns")
    summary["rankings"] = rankings
    return summary


def _summarise(rows: dict, reference: str) -> dict:
    """Paired differences against the reference arm, shared by the studies above."""
    metrics = ["test_auc", "test_f1", "recall_at_alert_rate"]
    summary = {}
    for arm, runs in rows.items():
        row = {metric: float(np.mean([run[metric] for run in runs])) for metric in metrics}
        row["features"] = float(np.mean([run["features"] for run in runs]))
        for metric in metrics:
            difference = np.array([run[metric] for run in runs]) - np.array(
                [run[metric] for run in rows[reference]])
            row[f"delta_{metric}"] = float(difference.mean())
            row[f"delta_{metric}_spread"] = float(difference.std())
            if metric == "test_auc":
                row["better_on_splits"] = int((difference > 0).sum())
        summary[arm] = row
    return {"splits": len(SPLIT_SEEDS), "reference": reference, "arms": summary}


def main() -> dict:
    X, y = build_frame()
    print(f"{X.shape[1]} columns: {X.shape[1] - len(ADDITIONS) - len(PROTECTED)} served, "
          f"{len(ADDITIONS)} additions, {len(PROTECTED)} protected", flush=True)

    results = {
        "split_seeds": list(SPLIT_SEEDS),
        "model_seeds": list(SEEDS),
        "hyperparameters": HYPERPARAMETERS,
        "alert_rate": ALERT_RATE,
        "minimum_share_assumed": MINIMUM_SHARE,
        "studies": {},
    }
    out = artifacts_dir() / "feature_experiments.json"

    # Written after each study rather than at the end: the whole run takes the better part
    # of an hour, and a partial artifact is more useful than none.
    for name, study in [("additions", additions_study), ("demographics", demographics_study),
                        ("retuning", retuning_study), ("segmentation", segmentation_study),
                        ("selection", selection_study)]:
        results["studies"][name] = study(X, y)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print(f"  wrote {name} to {out.name}", flush=True)

    print(f"\nwrote {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
