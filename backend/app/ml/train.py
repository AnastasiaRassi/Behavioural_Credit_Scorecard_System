"""Fit the deployable model and write every artifact the scoring API needs.

Run this to regenerate artifacts after changing features or hyperparameters:

    python -m app.ml.train        (from the backend/ directory)

Deliberately serves a single calibrated model rather than the notebook's rank blend.
The blend ranked marginally worse on test and its scores were not probabilities, so
its threshold could not be applied to a calibrated output. The model that is
evaluated here is the model that is saved and served.
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from app.ml.explainability import MONTHS_OBSERVED, month_label
from app.ml.feature_engineering import (
    AGE_BANDS,
    AGE_LABELS,
    EDUCATION_LABELS,
    MARRIAGE_LABELS,
    SEX_LABELS,
    build_splits,
    describe_pay_status,
    project_root,
)

# Averaging a few seeds removes run-to-run variance from row and column subsampling.
SEEDS = (42, 202, 777)

# Selected by randomized search over 70 configurations, scored by 5-fold
# cross-validation on the training split only. Best CV AUC 0.7915 (sd 0.0095);
# the full search spanned just 0.7865 to 0.7915, so this surface is flat and the
# exact configuration matters little.
HYPERPARAMETERS = dict(
    n_estimators=473,  # best_iteration from early stopping, plus one
    learning_rate=0.01,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.6413611870890207,
    subsample_freq=1,
    colsample_bytree=0.793822286088856,
    reg_alpha=0.004812512539835178,
    reg_lambda=50.17420195002703,
)

MODEL_VERSION = "lightgbm_ensemble_v1"

# Everything a training run writes. Anything else in artifacts/ is stale (§16).
WRITES = (
    "model_ensemble.joblib",
    "calibrator_isotonic.joblib",
    "scoring_config.json",
    "threshold_sweep.json",
    "feature_importance.json",
    "cost_curve.json",
    "fairness_detail.json",
    "registry.json",
    "officer_queue.json",
)

# Written by app.ml.segmentation, which is a separate step with its own run.
# Without this, a retrain would delete a segmentation that is still correct.
PRESERVES = (
    "segmenter.joblib",
    "segments.json",
)

LOW_QUANTILE = 0.50
MEDIUM_QUANTILE = 0.80

# Groups below either floor have rates too noisy to publish, so the fairness
# audit reports them as hidden rather than quoting a number.
MIN_GROUP_SIZE = 50
MIN_GROUP_DEFAULTS = 10

# Cost ratios swept for the dashboard's cost curve: a missed default costs this
# many times a wrongly refused customer.
COST_RATIOS = range(1, 11)

DECILES = 10

# Reasons stored per queued customer. Five fills the detail panel without
# inflating the artifact across thousands of rows.
QUEUE_REASONS = 5
CALIBRATION_BINS = 10


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_stale(path: Path) -> list[str]:
    """Delete anything in artifacts/ this run does not write.

    Stale files contradict the served model: `model_lightgbm.joblib` and
    `model_metadata.json` are notebook leftovers describing a different champion
    and a different threshold.
    """
    removed = []
    for existing in path.iterdir():
        if not existing.is_file():
            continue
        if existing.name in WRITES or existing.name in PRESERVES:
            continue
        existing.unlink()
        removed.append(existing.name)
    return removed


def fit_ensemble(X, y, seeds=SEEDS):
    return [
        LGBMClassifier(**HYPERPARAMETERS, n_jobs=-1, random_state=seed, verbose=-1).fit(X, y)
        for seed in seeds
    ]


def ensemble_predict(models, X) -> np.ndarray:
    return np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0)


# --- dashboard panels -----------------------------------------------------
# Computed here because train.py is the only writer to artifacts/. The UI reads
# these files and never recomputes, so the screen cannot disagree with the
# evaluation.


def decile_table(y, probabilities, bins: int = DECILES) -> list[dict]:
    """Actual default rate per predicted-risk decile, safest group first."""
    ranks = pd.qcut(pd.Series(probabilities).rank(method="first"), bins, labels=False)
    frame = pd.DataFrame({"decile": ranks + 1, "y": np.asarray(y)})
    grouped = frame.groupby("decile")["y"].agg(["count", "mean"])
    return [
        {
            "decile": int(decile),
            "customers": int(row["count"]),
            "actual_default_rate": float(row["mean"]),
        }
        for decile, row in grouped.iterrows()
    ]


def calibration_table(y, probabilities, bins: int = CALIBRATION_BINS) -> list[dict]:
    """Observed default rate against mean predicted probability, in equal-count bins."""
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(probabilities, edges[1:-1]), 0, len(edges) - 2)
    frame = pd.DataFrame({"bin": which, "p": probabilities, "y": np.asarray(y)})
    grouped = frame.groupby("bin").agg(
        mean_predicted=("p", "mean"), observed=("y", "mean"), customers=("y", "size")
    )
    return [
        {
            "mean_predicted": float(row["mean_predicted"]),
            "observed_rate": float(row["observed"]),
            "customers": int(row["customers"]),
        }
        for _, row in grouped.iterrows()
    ]


def cost_curves(sweep: list[dict]) -> dict:
    """Cost-minimising threshold per false-negative to false-positive ratio.

    Cost is in units of one wrongly refused customer, so the ratio is the only
    business input the dashboard needs.
    """
    curves = {}
    for ratio in COST_RATIOS:
        costs = [row["fn"] * ratio + row["fp"] for row in sweep]
        best = int(np.argmin(costs))
        curves[str(ratio)] = {
            "best_threshold": sweep[best]["threshold"],
            "cost": float(costs[best]),
            "curve": [
                {"threshold": row["threshold"], "cost": float(cost)}
                for row, cost in zip(sweep, costs)
            ],
        }
    return curves


def group_labels(audit: pd.DataFrame) -> dict:
    """Readable group names per protected attribute, matching the codebook."""
    return {
        "SEX": audit["SEX"].map(SEX_LABELS),
        "AGE": pd.cut(audit["AGE"], AGE_BANDS, labels=AGE_LABELS).astype(str),
        "EDUCATION": audit["EDUCATION"].map(EDUCATION_LABELS),
        "MARRIAGE": audit["MARRIAGE"].map(MARRIAGE_LABELS),
    }


def fairness_audit(y, probabilities, audit: pd.DataFrame, threshold: float) -> dict:
    """Disparate impact, equalized odds and per-group detail.

    Disparate impact is stable across splits. The equalized-odds gaps are not, and
    the dashboard labels them as such; they are reported for completeness.
    """
    y = np.asarray(y)
    predicted = (probabilities >= threshold).astype(int)
    summary = {}

    for attribute, labels in group_labels(audit).items():
        groups, approval_rates = [], []

        for name, index in labels.groupby(labels).groups.items():
            mask = labels.index.isin(index)
            group_y, group_p = y[mask], probabilities[mask]
            group_predicted = predicted[mask]
            size, defaults = int(mask.sum()), int(group_y.sum())
            reliable = size >= MIN_GROUP_SIZE and defaults >= MIN_GROUP_DEFAULTS
            approval_rate = float((group_predicted == 0).mean())

            groups.append(
                {
                    "group": str(name),
                    "customers": size,
                    "defaults": defaults,
                    "actual_default_rate": float(group_y.mean()),
                    "approval_rate": approval_rate,
                    "tpr": float(group_predicted[group_y == 1].mean()) if defaults else None,
                    "fpr": (
                        float(group_predicted[group_y == 0].mean())
                        if size - defaults else None
                    ),
                    "auc": (
                        float(roc_auc_score(group_y, group_p))
                        if reliable and 0 < defaults < size else None
                    ),
                    "reliable": reliable,
                }
            )
            if reliable:
                approval_rates.append(approval_rate)

        usable = [group for group in groups if group["reliable"]]
        tprs = [g["tpr"] for g in usable if g["tpr"] is not None]
        fprs = [g["fpr"] for g in usable if g["fpr"] is not None]
        aucs = [g["auc"] for g in usable if g["auc"] is not None]
        ratio = (
            float(min(approval_rates) / max(approval_rates)) if approval_rates else None
        )

        summary[attribute] = {
            "disparate_impact_ratio": ratio,
            "passes_four_fifths": bool(ratio >= 0.80) if ratio is not None else None,
            "equalized_odds_TPR_gap": float(max(tprs) - min(tprs)) if tprs else None,
            "equalized_odds_FPR_gap": float(max(fprs) - min(fprs)) if fprs else None,
            "auc_spread": float(max(aucs) - min(aucs)) if aucs else None,
            "groups": sorted(groups, key=lambda group: -group["customers"]),
            "hidden_groups": [g["group"] for g in groups if not g["reliable"]],
        }
    return summary


def month_history(row: pd.Series) -> list[dict]:
    """The six observed months, most recent first, as a drawable trajectory.

    Written here rather than derived in the UI because train.py is the only
    writer (§3). Each entry carries the raw status alongside its reading, so a
    screen can sort or colour on the number without re-parsing the words.
    """
    history = []
    for position in range(1, MONTHS_OBSERVED + 1):
        bill = float(row[f"BILL_AMT{position}"])
        limit = float(row["LIMIT_BAL"])
        status = int(row[f"PAY_{position}"])
        history.append(
            {
                "month": position,
                "label": month_label(position),
                "payment_status": status,
                "payment_status_reading": describe_pay_status(status),
                "bill_amount": bill,
                "amount_paid": float(row[f"PAY_AMT{position}"]),
                # None rather than zero on a zero limit: no utilisation exists to
                # report, and 0.0 would read as "used nothing".
                "utilisation": round(bill / limit, 4) if limit else None,
            }
        )
    return history


def test_segments(features: pd.DataFrame) -> pd.DataFrame | None:
    """Behavioural segment per customer, or None if segmentation has not run.

    Segmentation is a separate step with its own artifacts, so the queue is built
    with or without it rather than training failing because it is absent.
    """
    from app.ml.segmentation import ArtifactsMissing, Segmenter

    try:
        return Segmenter().assign(features)
    except ArtifactsMissing:
        print("  segmentation artifacts absent; queue rows will carry no segment")
        return None


def officer_queue(models, features: pd.DataFrame, probabilities, threshold: float,
                  band_of, reasons_shown: int = QUEUE_REASONS,
                  segments: pd.DataFrame | None = None) -> list[dict]:
    """The officer's worklist: every held-out customer, scored and explained.

    Precomputed rather than scored on demand, so the officer screen reads a file
    like every other screen (§3) and a queue of thousands opens instantly.

    Protected attributes are absent: build_splits(blind=True) has already dropped
    them, and an officer must not see demographics beside a risk score.

    The realised outcome is deliberately excluded. The officer is deciding, and
    showing next month's truth would make the queue a reveal rather than a task.
    """
    from app.ml.explainability import describe


    # One pass over the ensemble; per-row calls would be thousands of predicts.
    contributions = np.mean(
        [model.predict_proba(features, pred_contrib=True) for model in models], axis=0
    )
    effects = np.asarray(contributions)[:, :-1]
    columns = list(features.columns)
    strongest = np.argsort(-np.abs(effects), axis=1)[:, :reasons_shown]

    def segment_of(index) -> dict | None:
        """The customer's behavioural segment, when segmentation has been run.

        None rather than a placeholder: a screen must be able to tell "not
        segmented" from "segmented as something", and segmentation is a separate
        step with its own artifacts.
        """
        if segments is None or index not in segments.index:
            return None
        assigned = segments.loc[index]
        return {
            "label": int(assigned["segment"]),
            "name": str(assigned["segment_name"]),
        }

    queue = []
    for position, (index, row) in enumerate(features.iterrows()):
        probability = float(probabilities[position])
        queue.append(
            {
                "customer_id": f"C-{int(index):05d}",
                "probability": probability,
                "risk_band": band_of(probability),
                "flagged": bool(probability >= threshold),
                "credit_limit": float(row["LIMIT_BAL"]),
                "current_balance": float(row["BILL_AMT1"]),
                "last_payment": float(row["PAY_AMT1"]),
                "months_in_arrears": int(row["pay_months_late"]),
                "segment": segment_of(index),
                "history": month_history(row),
                "reasons": [
                    {
                        "feature": columns[column],
                        "effect": float(effects[position, column]),
                        "description": describe(columns[column], float(row[columns[column]])),
                    }
                    for column in strongest[position]
                ],
            }
        )
    queue.sort(key=lambda customer: -customer["probability"])
    return queue


def main() -> dict:
    frames, y, audits = build_splits(seed=42, blind=True)

    # Calibrator and operating points are fitted on validation, which the model has
    # not seen. Fitting them on training predictions would make them over-confident.
    calibration_models = fit_ensemble(frames["train"], y["train"])
    validation_raw = ensemble_predict(calibration_models, frames["validation"])

    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        validation_raw, y["validation"]
    )
    validation_calibrated = calibrator.predict(validation_raw)

    grid = np.unique(np.quantile(validation_calibrated, np.linspace(0.01, 0.99, 197)))
    f1_scores = [
        f1_score(y["validation"], (validation_calibrated >= t).astype(int)) for t in grid
    ]
    threshold = float(grid[int(np.argmax(f1_scores))])

    band_edges = {
        "low_max": float(np.quantile(validation_calibrated, LOW_QUANTILE)),
        "medium_max": float(np.quantile(validation_calibrated, MEDIUM_QUANTILE)),
    }

    # Final model is refit on train + validation; selection is finished, so holding
    # 15% back from the deployed model would only waste it.
    X_dev = pd.concat([frames["train"], frames["validation"]])
    y_dev = pd.concat([y["train"], y["validation"]])
    final_models = fit_ensemble(X_dev, y_dev)

    test_calibrated = calibrator.predict(ensemble_predict(final_models, frames["test"]))
    predicted = (test_calibrated >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y["test"], predicted).ravel()

    bands = pd.cut(
        test_calibrated,
        [-np.inf, band_edges["low_max"], band_edges["medium_max"], np.inf],
        labels=["Low", "Medium", "High"],
    )
    band_report = (
        pd.DataFrame({"band": bands, "y": y["test"].to_numpy()})
        .groupby("band", observed=True)["y"]
        .agg(["count", "mean"])
    )

    # Full metric suite. The dashboard reads these rather than recomputing, so the
    # screen always agrees with what was actually evaluated.
    fpr, tpr_curve, _ = roc_curve(y["test"], test_calibrated)
    auc = float(roc_auc_score(y["test"], test_calibrated))
    config = {
        "scale": "calibrated probability of default next month",
        "model_version": MODEL_VERSION,
        "trained": date.today().isoformat(),
        "training_rows": int(len(y["train"]) + len(y["validation"])),
        "threshold_f1_optimal": threshold,
        "band_edges": band_edges,
        "seeds": list(SEEDS),
        "hyperparameters": HYPERPARAMETERS,
        "feature_names": list(frames["train"].columns),
        "n_features": frames["train"].shape[1],
        "test_metrics": {
            "roc_auc": auc,
            "gini": 2 * auc - 1,
            "ks": float(np.max(tpr_curve - fpr)),
            "pr_auc": float(average_precision_score(y["test"], test_calibrated)),
            "brier": float(brier_score_loss(y["test"], test_calibrated)),
            "balanced_accuracy": float(balanced_accuracy_score(y["test"], predicted)),
            "mcc": float(matthews_corrcoef(y["test"], predicted)),
            "cohen_kappa": float(cohen_kappa_score(y["test"], predicted)),
            "f1_macro": float(f1_score(y["test"], predicted, average="macro")),
            "f1_weighted": float(f1_score(y["test"], predicted, average="weighted")),
            "f1": float(2 * tp / (2 * tp + fp + fn)),
            "precision": float(tp / (tp + fp)),
            "recall": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "split_sizes": {name: int(len(labels)) for name, labels in y.items()},
        "test_rows": int(len(y["test"])),
        "test_default_rate": float(y["test"].mean()),
        "band_report": {
            str(band): {
                "customers": int(row["count"]),
                "actual_default_rate": float(row["mean"]),
            }
            for band, row in band_report.iterrows()
        },
        "deciles": decile_table(y["test"], test_calibrated),
        "calibration": calibration_table(y["test"], test_calibrated),
    }

    # --- threshold sweep, for the dashboard's interactive slider ----------
    sweep = []
    for t in np.round(np.arange(0.05, 0.901, 0.01), 3):
        flags = (test_calibrated >= t).astype(int)
        if flags.sum() == 0:
            continue
        s_tn, s_fp, s_fn, s_tp = confusion_matrix(y["test"], flags, labels=[0, 1]).ravel()
        sweep.append({
            "threshold": float(t),
            "flagged_share": float(flags.mean()),
            "precision": float(s_tp / (s_tp + s_fp)) if (s_tp + s_fp) else 0.0,
            "recall": float(s_tp / (s_tp + s_fn)),
            "f1": float(2 * s_tp / (2 * s_tp + s_fp + s_fn)),
            "tn": int(s_tn), "fp": int(s_fp), "fn": int(s_fn), "tp": int(s_tp),
        })

    # --- permutation importance, model agnostic and not biased by cardinality ---
    permutation = permutation_importance(
        final_models[0], frames["test"], y["test"],
        scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1,
    )
    importance = sorted(
        (
            {
                "feature": name,
                "auc_drop": float(mean),
                "std": float(sd),
            }
            for name, mean, sd in zip(
                frames["test"].columns,
                permutation.importances_mean,
                permutation.importances_std,
            )
        ),
        key=lambda row: row["auc_drop"],
        reverse=True,
    )

    def band_of(probability: float) -> str:
        if probability < band_edges["low_max"]:
            return "Low"
        if probability < band_edges["medium_max"]:
            return "Medium"
        return "High"

    queue = officer_queue(
        final_models, frames["test"], test_calibrated, threshold, band_of,
        segments=test_segments(frames["test"]),
    )

    fairness = fairness_audit(y["test"], test_calibrated, audits["test"], threshold)
    passed = sum(1 for values in fairness.values() if values["passes_four_fifths"])

    # One row for now. The dashboard's registry tab fills as versions accumulate.
    registry = [
        {
            "version": MODEL_VERSION,
            "trained": config["trained"],
            "roc_auc": config["test_metrics"]["roc_auc"],
            "f1": config["test_metrics"]["f1"],
            "fairness_passed": passed,
            "fairness_total": len(fairness),
            "status": "live",
        }
    ]

    # A silent divergence here means the published metrics describe a model the
    # code no longer builds.
    assert config["hyperparameters"] == HYPERPARAMETERS

    out = artifacts_dir()
    remove_stale(out)
    joblib.dump(final_models, out / "model_ensemble.joblib")
    joblib.dump(calibrator, out / "calibrator_isotonic.joblib")
    with open(out / "scoring_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    with open(out / "threshold_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(sweep, fh, indent=2)
    with open(out / "feature_importance.json", "w", encoding="utf-8") as fh:
        json.dump(importance, fh, indent=2)
    with open(out / "cost_curve.json", "w", encoding="utf-8") as fh:
        json.dump(cost_curves(sweep), fh, indent=2)
    with open(out / "fairness_detail.json", "w", encoding="utf-8") as fh:
        json.dump(fairness, fh, indent=2)
    with open(out / "registry.json", "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    with open(out / "officer_queue.json", "w", encoding="utf-8") as fh:
        json.dump(queue, fh)

    print(f"Artifacts written to {out}")
    print(f"  test AUC   {config['test_metrics']['roc_auc']:.4f}")
    print(f"  test F1    {config['test_metrics']['f1']:.4f}")
    print(f"  test Brier {config['test_metrics']['brier']:.4f}")
    print(f"  threshold  {threshold:.4f}")
    print(f"  fairness   {passed}/{len(fairness)} attributes pass the four-fifths rule")
    print(f"  queue      {len(queue):,} customers, "
          f"{sum(c['flagged'] for c in queue):,} flagged")
    return config


if __name__ == "__main__":
    main()
