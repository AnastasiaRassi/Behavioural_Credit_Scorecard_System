"""How a raw model score becomes the probability the system publishes.

**The model's own probabilities are used directly.** LightGBM is fitted with a
log-loss objective, so its output is already a probability rather than a score
needing a mapping, and measured on the test split it is better calibrated than
anything fitted on top of it:

    calibrator            brier     log loss   ECE       AUC       distinct
    raw (this)            0.134228  0.427689   0.008821  0.783033  4366
    isotonic              0.134935  0.443197   0.018100  0.781117  66
    platt (sigmoid)       0.134795  0.430364   0.023275  0.783033  4338

Isotonic regression was used until it was measured. It was worse on every count:
it doubled the calibration error it was supposed to reduce, it cost AUC by
collapsing 4,500 customers onto 66 distinct values and creating ties, and being
a step function it saturated, returning exactly 1.0 above a raw score of about
0.78. Fifty-one held-out customers were scored as certainties.

Two things made it fail. It had only the validation split to learn from, which
is too little to fit a step function that then has to generalise; and in the
pipeline it was fitted against one model's scores (trained on train) and applied
to another's (refitted on train plus validation), so the mapping never saw the
model it was correcting.

`BOUNDS` stays as a safety rail. It does not bind today - the model's range on
test is roughly 0.035 to 0.804 - but a probability of exactly 0 or 1 must never
reach a credit file, and a future model or calibrator could produce one.

Lives in its own module because `train.py` evaluates with it and `scorer.py`
serves with it, and a difference between the two would mean the published
metrics described a model the API does not run.
"""
import numpy as np

CERTAINTY_FLOOR = 0.01
CERTAINTY_CEILING = 1.0 - CERTAINTY_FLOOR


def calibrate(raw_scores) -> np.ndarray:
    """The published probability for a raw model score.

    An identity apart from the safety rail, which is the finding above rather
    than an omission.
    """
    return np.clip(np.asarray(raw_scores, dtype=float), CERTAINTY_FLOOR, CERTAINTY_CEILING)
