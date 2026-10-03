"""Fit the deployable model and write every artifact the scoring API needs.

Run this to regenerate artifacts after changing features or hyperparameters:

    python -m app.ml.train        (from the backend/ directory)

Deliberately serves a single calibrated model rather than the notebook's rank blend.
The blend ranked marginally worse on test and its scores were not probabilities, so
its threshold could not be applied to a calibrated output. The model that is
evaluated here is the model that is saved and served.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from app.ml.feature_engineering import build_splits, project_root

# Averaging a few seeds removes run-to-run variance from row and column subsampling.
SEEDS = (42, 202, 777)

HYPERPARAMETERS = dict(
    n_estimators=166,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)

LOW_QUANTILE = 0.50
MEDIUM_QUANTILE = 0.80


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_ensemble(X, y, seeds=SEEDS):
    return [
        LGBMClassifier(**HYPERPARAMETERS, n_jobs=-1, random_state=seed, verbose=-1).fit(X, y)
        for seed in seeds
    ]


def ensemble_predict(models, X) -> np.ndarray:
    return np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0)


def main() -> dict:
    frames, y, _ = build_splits(seed=42, blind=True)

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

    config = {
        "scale": "calibrated probability of default next month",
        "threshold_f1_optimal": threshold,
        "band_edges": band_edges,
        "seeds": list(SEEDS),
        "hyperparameters": HYPERPARAMETERS,
        "feature_names": list(frames["train"].columns),
        "n_features": frames["train"].shape[1],
        "test_metrics": {
            "roc_auc": float(roc_auc_score(y["test"], test_calibrated)),
            "brier": float(brier_score_loss(y["test"], test_calibrated)),
            "f1": float(2 * tp / (2 * tp + fp + fn)),
            "precision": float(tp / (tp + fp)),
            "recall": float(tp / (tp + fn)),
        },
        "band_report": {
            str(band): {
                "customers": int(row["count"]),
                "actual_default_rate": float(row["mean"]),
            }
            for band, row in band_report.iterrows()
        },
    }

    out = artifacts_dir()
    joblib.dump(final_models, out / "model_ensemble.joblib")
    joblib.dump(calibrator, out / "calibrator_isotonic.joblib")
    with open(out / "scoring_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"Artifacts written to {out}")
    print(f"  test AUC   {config['test_metrics']['roc_auc']:.4f}")
    print(f"  test F1    {config['test_metrics']['f1']:.4f}")
    print(f"  test Brier {config['test_metrics']['brier']:.4f}")
    print(f"  threshold  {threshold:.4f}")
    return config


if __name__ == "__main__":
    main()
