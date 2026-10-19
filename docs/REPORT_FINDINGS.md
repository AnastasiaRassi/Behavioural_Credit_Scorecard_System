# Report findings

Measured results worth carrying into the FYP report, each with how it was produced. Claims
without a number, or with a number nobody can regenerate, do not belong here.

Author: Anastasia Al Rassi · TM471, Arab Open University Lebanon
Last updated: 2026-10-19

**Provenance.** Findings marked **[artifact]** come from `backend/artifacts/`, regenerable with
the commands in the README. **[notebook]** comes from a notebook cell that can be re-run.
**[script]** means it was measured by an experiment that is *not yet in the repository*, so the
number cannot currently be reproduced by an examiner — those need moving into
`backend/app/ml/experiments.py` before they are cited. See [Open items](#open-items).

---

## 1. Headline performance  [artifact]

Test split of 4,500 customers, evaluated once. Threshold 0.2389, the F1 optimum chosen on
validation. Model: three LightGBM fits over seeds 42, 202, 777, refitted on train plus
validation.

| Metric | Value |
|---|---|
| ROC-AUC | 0.7830 |
| Gini | 0.5661 |
| KS | 43.3 |
| PR-AUC | 0.5584 |
| Brier | 0.1342 |
| F1 | 0.5405 |
| Precision / Recall | 0.4777 / 0.6225 |
| Specificity | 0.8065 |

Confusion matrix: tn 2,826, fp 678, fn 376, tp 620. Deciles run 4.0% to 69.1%, a 17.3× lift.
Risk bands: Low 2,224 at 8.9%, Medium 1,338 at 20.3%, High 938 at 56.3%.

**The prediction is a forecast, not a restatement of the inputs.** 48.3% of customers who
default next month were not late in the current month, and 49.7% of those late this month
recover. Half of all defaults come from customers who currently look healthy.

**Accuracy is never a headline.** Predicting that nobody defaults scores 77.9% on this data.

## 2. Protected attributes buy a small, consistent gain  [script]

The one lever measured to beat the baseline on every split. Full deployed pipeline, five
independent split seeds, paired by split.

| Arm | Test AUC | Test F1 | Recall @20% | AUC delta |
|---|---|---|---|---|
| Blind (deployed) | 0.7870 | 0.5447 | 0.5169 | — |
| + AGE | 0.7876 | 0.5452 | 0.5177 | +0.0005, better on 5/5 |
| + all four | 0.7885 | 0.5476 | 0.5193 | +0.0014 (sd 0.0008), 5/5 |

Logistic regression behaves the same way: +0.0020 AUC on 5/5 splits. F1 gains +0.0029 and
recall at a 20% alert rate +0.0024.

**For the report:** the project's earlier figure of 0.0007 came from a single split and
understates it. The defensible claim is that **fairness through unawareness costs about 0.0014
AUC** — a real effect, roughly twice the ensemble's own seed spread, and immaterial against the
0.78 ceiling. Say it costs something and is still worth paying, rather than claiming it is
free: using sex or marital status in a credit decision is prohibited and age is restricted, so
the gain cannot be spent at any size.

## 3. Engineered features do not move the model  [notebook] [script]

Twenty-four columns in twelve groups were engineered beside the 46 served features and tested
on training rows only (notebook 01, sections 3 and 4).

**Individually several are strong.** Univariate AUC: `delinquency_run_current` 0.698,
`pay_status_volatility` 0.659, `status_entropy` 0.633, `status_changes` 0.625, against 0.731 for
the served `pay_months_late`. The open delinquency run separates a 13.7% default rate at zero
months from 69.8% at six.

**None of it pays.** Validation AUC, deployed hyperparameters, three seeds:

| Arm | Validation AUC | Delta |
|---|---|---|
| Served only (46) | 0.7831 | — |
| + predictability | 0.7834 | +0.0004 |
| + recent vs historical | 0.7834 | +0.0003 |
| + roll-rate transitions | 0.7833 | +0.0003 |
| + every addition (24) | 0.7816 | −0.0015 |

Seed spread is 0.0003, so nothing clears noise, and adding everything at once is actively
worse. A five-split repeat through the full deployed pipeline reached the same conclusion on
AUC, F1 and recall at a 20% alert rate. Permutation importance inside the combined model puts
every addition below `util_2`.

**Distinctness was necessary and not sufficient.** Four variants were dropped before testing
for restating a served feature: a longest-arrears run (0.998 rank correlation with
`pay_months_late`), an absolute utilisation slope (0.96 with `util_trend`), a utilisation shift
(0.91), and the payment-to-bill ratio, which `repay_ratio_*` already computes by the same
formula. What remains is independent — the closest is `months_cleared_bill` at 0.85,
`payment_shift_ratio` at 0.30 — and still adds nothing. A feature can be unlike anything
present and carry no information the model lacks.

**For the report:** this is the strongest single piece of ceiling evidence, because it rules out
the most common reviewer suggestion ("engineer better features") with a measurement rather than
an assertion.

## 4. The 0.78 ceiling  [artifact]

| Evidence | Result |
|---|---|
| Hyperparameter search, 70 configurations | 0.7865 to 0.7915 CV AUC |
| Plain logistic regression | 0.760 |
| Resampling, 7 strategies | none beats the 0.7829 baseline by more than noise; best Tomek links 0.7833; SMOTE 0.7799, ADASYN 0.7777 |
| Ensembling | single tuned LightGBM 0.7831 beats soft voting (0.7803–0.7820), stacking (0.7824), hard voting (0.6455) |
| Model families | GradientBoosting 0.7807, ExtraTrees 0.7734, HistGradientBoosting 0.7724, AdaBoost 0.7704, KNN 0.7691, linear SVM 0.7531 |
| Neural networks, 6 architectures | best 0.7798 (PyTorch, four layers); sklearn MLPs 0.7677–0.7747 |
| Feature engineering, 24 columns | no gain (section 3 above) |

**Higher published figures on this dataset are usually artifacts.** Applying SMOTE *before* the
split was reproduced here and inflated test AUC to 0.9233. The other common artifact is
accuracy quoted as if it were AUC — easy to do when the majority class is 77.9%.

**For the report:** all sophistication is worth about 0.02 AUC over logistic regression. The
binding constraint is the information in 18 numbers per customer, not the model.

## 5. Calibrating a log-loss booster made it worse  [artifact]

| Calibrator | Brier | ECE | AUC | Distinct values |
|---|---|---|---|---|
| Raw model output | 0.1342 | 0.0088 | 0.7830 | 4,366 |
| Isotonic | 0.1349 | 0.0181 | 0.7811 | 66 |
| Platt | 0.1348 | 0.0233 | 0.7830 | 4,338 |

Isotonic doubled the calibration error it was meant to reduce, collapsed 4,500 customers onto
66 distinct values, and being a step function returned exactly 1.0 for the riskiest customers —
51 held-out customers scored as certainties. It was removed.

**For the report:** this contradicts the usual "always calibrate" advice, with the reason.
LightGBM optimises log loss, so its output is already a probability; and the calibrator had only
the validation split to learn from, which is too little to fit a step function that must then
generalise.

## 6. Equally accurate models disagree about individuals  [artifact]

Six configurations within 0.0088 AUC of each other, explained by one model-agnostic occlusion
method so model disagreement is not confounded with method disagreement.

| Question | Result |
|---|---|
| Customers flagged by some models, cleared by others | 8.5% |
| Decisions reversed by swapping the deployed model | 4.3% |
| Customers receiving the same top three drivers from all six | 0% |
| Customers whose single top driver is unanimous | 12.4% |
| Mean pairwise overlap of top-three reasons | 0.49 |
| Concentration of ambiguity in the Distressed segment | 1.96× |

**For the report:** this is the research contribution. Under a right to explanation, an
arbitrary choice between two defensible models changes who is declined and what they are told,
so the choice of model is a policy decision rather than a technical one — and the arbitrariness
falls hardest on the customers with the most at stake.

## 7. Fairness  [artifact]

| Attribute | DIR | Four-fifths | Per-group AUC spread |
|---|---|---|---|
| SEX | 0.938 | Pass | 0.044 (women 0.802, men 0.757) |
| AGE | 0.938 | Pass | 0.024 |
| EDUCATION | 0.880 | Pass | 0.050 (university 0.795, high school 0.744) |
| MARRIAGE | 0.990 | Pass | 0.001 |

Groups too small to measure are suppressed rather than reported: age 60+ (35 customers),
education "other" (68), marriage "other" (54).

**Equalized-odds gaps are not reliable on this sample.** Disparate impact is stable across
splits; the TPR, FPR and AUC spread columns move substantially and sometimes reverse sign. They
must not be used to claim bias against a specific group.

## 8. Behavioural segments  [artifact]

K-means on an interpretable subset, k = 3 by silhouette (0.284).

| Segment | Customers | Share | Default rate |
|---|---|---|---|
| Moderate user | 2,757 | 61% | 11.7% |
| Distressed | 1,008 | 22% | 48.0% |
| Dormant | 735 | 16% | 25.7% |

Used to report performance and fairness per archetype rather than only in aggregate, and to
locate where multiplicity concentrates (section 6).

## Open items

- **Two columns now breach the distinctness rule.** The latest run puts `spend_to_payment` at
  0.94 against `util_trend` and `months_below_minimum` at 0.905 against `repay_ratio_mean`,
  above the 0.90 threshold section 3 of notebook 01 states. They should be dropped for the same
  reason the other four were.
- **Three [script] findings are not reproducible from the repository**: the demographics
  comparison (section 2), the five-split repeats (sections 2 and 3). They belong in
  `experiments.py` as a study writing to `experiment_results.json`, per CODE_CONVENTIONS §16.
- **Running:** greedy group selection plus hyperparameter re-tuning on the expanded feature set,
  to test whether the added features were judged under a configuration chosen without them.
- **Untested:** a sequence model (GRU or 1D-CNN) over the raw 6×3 monthly matrix, stacked with
  the booster. Nothing has read these months as a sequence; every model so far saw flat
  summaries. Also untested: a systematic feature bank (`tsfresh`, `catch22`) with selection
  inside cross-validation, which would replace hand-picked features with a searched space.
- **The minimum-payment constant is an assumption.** `MINIMUM_SHARE = 0.10` in notebook 01 needs
  checking against the 2005 Taiwanese regulation before any figure derived from it is published.
