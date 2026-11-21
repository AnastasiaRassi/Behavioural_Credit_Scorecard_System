"""Errors from the model layer: artifacts, features, scoring.

The built-in bases (`RuntimeError`, `ValueError`) are kept so existing handlers
still catch these.
"""
from app.core.exceptions import AppError, InfraError


class ArtifactsMissing(InfraError, RuntimeError):
    """A file `train.py` should have written is not there. The one class for every reader."""

    code = "ARTIFACTS_MISSING"
    user_message = "The scoring model is not available right now."


class ArtifactCorrupt(InfraError, RuntimeError):
    """An artifact exists but cannot be read or does not hold what it should."""

    code = "ARTIFACT_CORRUPT"
    user_message = "The scoring model is not available right now."


class MissingColumns(AppError, ValueError):
    """The input lacks columns the model needs: the caller's mistake, not ours."""

    code = "MISSING_COLUMNS"
    status = 422
    user_message = "Some required fields are missing."


class FeatureSchemaMismatch(AppError, ValueError):
    """Engineered features differ from the trained model's: code and artifacts are out of step."""

    code = "FEATURE_SCHEMA_MISMATCH"
    user_message = "The scoring model is out of date with the service."


class ScoringFailed(AppError):
    """The model raised while scoring. The original error is chained as `__cause__`."""

    code = "SCORING_FAILED"
    user_message = "We could not score this account."


class UnknownBand(AppError, ValueError):
    code = "UNKNOWN_BAND"
    status = 422
    user_message = "That risk band does not exist."


class ExplanationFailed(AppError, ValueError):
    """An explanation was asked of the wrong input shape."""

    code = "EXPLANATION_FAILED"
    user_message = "We could not explain this score."


class DatasetUnavailable(InfraError, FileNotFoundError):
    """The raw dataset needed to build features is not on disk."""

    code = "DATASET_UNAVAILABLE"
