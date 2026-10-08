"""Cached readers for the files `train.py` writes.

Everything the API reports is read from artifacts/ rather than recomputed, so the
served numbers always agree with what was actually evaluated. Loaded once per
process, like the model itself.
"""
import json
from functools import lru_cache
from pathlib import Path

from app.ml.feature_engineering import project_root


class ArtifactsMissing(RuntimeError):
    pass


def artifacts_dir() -> Path:
    return project_root() / "backend" / "artifacts"


@lru_cache(maxsize=None)
def read(name: str):
    path = artifacts_dir() / name
    if not path.exists():
        raise ArtifactsMissing(
            f"Missing artifact {name} in {artifacts_dir()}. "
            "Run: python -m app.ml.train"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def scoring_config() -> dict:
    return read("scoring_config.json")


def threshold_sweep() -> list[dict]:
    return read("threshold_sweep.json")


def cost_curve() -> dict:
    return read("cost_curve.json")


def fairness_detail() -> dict:
    return read("fairness_detail.json")


def feature_importance() -> list[dict]:
    return read("feature_importance.json")


def registry() -> list[dict]:
    return read("registry.json")
