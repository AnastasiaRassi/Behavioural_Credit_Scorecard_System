"""Population stability between the training data and customers scored since.

PSI compares the distribution of each feature in a recent population against the
distribution the model was trained on. The reference side is reproducible today,
because `build_splits(seed=42)` is deterministic; the live side comes from stored
predictions, so it only exists once the system has scored customers.

Conventional bands: below 0.10 is stable, 0.10 to 0.25 is worth watching, above
0.25 is a significant shift.
"""
import numpy as np
import pandas as pd

from app.ml.feature_engineering import build_splits, engineer, clean

WATCH = 0.10
SIGNIFICANT = 0.25

BINS = 10

# Below this, PSI is dominated by sampling noise and should not be reported as a
# finding. Ten bins over fewer than 100 rows leaves single-digit bin counts.
MIN_LIVE_ROWS = 100

# Keeps the log finite when a bin is empty on one side.
FLOOR = 1e-6


def _bin_edges(reference: pd.Series) -> np.ndarray:
    """Quantile edges from the reference distribution, open at both ends."""
    quantiles = np.linspace(0, 1, BINS + 1)
    edges = np.unique(np.nanquantile(reference.to_numpy(dtype=float), quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values.dropna().to_numpy(dtype=float), bins=edges)[0]
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), FLOOR)
    return np.maximum(counts / total, FLOOR)


def population_stability_index(reference: pd.Series, live: pd.Series) -> float:
    """PSI for one feature. Zero means identical distributions."""
    edges = _bin_edges(reference)
    expected = _proportions(reference, edges)
    actual = _proportions(live, edges)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def band(psi: float) -> str:
    if psi >= SIGNIFICANT:
        return "significant"
    if psi >= WATCH:
        return "watch"
    return "stable"


def reference_features(seed: int = 42) -> pd.DataFrame:
    """The engineered training split, which is what the model learned from."""
    features, _, _ = build_splits(seed=seed)
    return features["train"]


def compare(live_raw: pd.DataFrame, seed: int = 42) -> dict:
    """PSI per feature between the training split and a live population.

    `live_raw` holds raw submitted columns, not engineered features, so it is
    engineered here with the same function training used. Returns an explicit
    `sufficient` flag rather than quietly reporting noise from a handful of rows.
    """
    if len(live_raw) < MIN_LIVE_ROWS:
        return {
            "sufficient": False,
            "live_rows": len(live_raw),
            "minimum_rows": MIN_LIVE_ROWS,
            "reason": (
                f"{len(live_raw)} scored customers is too few for a stable PSI; "
                f"at least {MIN_LIVE_ROWS} are needed."
            ),
            "features": [],
        }

    reference = reference_features(seed=seed)
    live = engineer(clean(live_raw))

    rows = []
    for feature in reference.columns:
        if feature not in live.columns:
            continue
        psi = population_stability_index(reference[feature], live[feature])
        rows.append({"feature": feature, "psi": round(psi, 4), "band": band(psi)})

    rows.sort(key=lambda row: row["psi"], reverse=True)
    return {
        "sufficient": True,
        "live_rows": len(live_raw),
        "reference_rows": len(reference),
        "watch_threshold": WATCH,
        "significant_threshold": SIGNIFICANT,
        "features": rows,
    }
