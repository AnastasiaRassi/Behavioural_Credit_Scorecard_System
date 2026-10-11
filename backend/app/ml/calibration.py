"""Bounds on a calibrated probability.

Isotonic regression is a step function fitted on the validation split, and at the
ends it has nowhere left to step: above a raw score of roughly 0.78 it returns
exactly 1.0, and at the bottom exactly 0.0. That is an artefact of the fit, not a
statement about the customer. The model itself still separates those cases, the
calibrator simply cannot express the difference.

A probability of 1.0 in a credit file says the customer will certainly default,
which no model knows and no lender should record. It also breaks anything built
on top: an expected loss computed from a certainty is not an expectation, and a
log-loss over it is infinite.

So calibrated output is clipped. The bound is 0.01, which is also roughly the
resolution the validation split can support: finer distinctions at the extremes
would be reading noise. Clipping is monotone, so it changes no ranking metric -
AUC, Gini and KS are untouched - and only the tails of Brier move.

Lives in its own module because `train.py` evaluates with it and `scorer.py`
serves with it, and a bound applied in one but not the other would mean the
published metrics described a model the API does not run.
"""
import numpy as np

CERTAINTY_FLOOR = 0.01
CERTAINTY_CEILING = 1.0 - CERTAINTY_FLOOR


def calibrate(calibrator, raw_scores) -> np.ndarray:
    """Calibrated probabilities, bounded away from certainty."""
    return np.clip(
        calibrator.predict(raw_scores), CERTAINTY_FLOOR, CERTAINTY_CEILING
    )
