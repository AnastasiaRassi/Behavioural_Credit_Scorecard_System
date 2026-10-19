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

## 2. Protected attributes buy a small, consistent gain  [artifact]

Full deployed pipeline, five independent split seeds, paired by split. Regenerable with
`python -m app.ml.feature_experiments`.

| Arm | Test AUC | AUC delta | Better on | Test F1 | Recall @20% |
|---|---|---|---|---|---|
| Blind (deployed) | 0.7870 | — | — | 0.5451 | 0.5159 |
| + AGE | 0.7875 | +0.0004 | 4/5 | 0.5454 | 0.5185 |
| + EDUCATION | 0.7873 | +0.0002 | 4/5 | 0.5455 | 0.5159 |
| + SEX | 0.7873 | +0.0003 | 4/5 | 0.5438 | 0.5157 |
| + all four | 0.7882 | **+0.0012** (spread 0.0006) | **5/5** | 0.5463 | 0.5187 |

**For the report:** the project's earlier figure of 0.0007 came from a single split and
understates it. The defensible claim is that **fairness through unawareness costs about 0.0012
AUC**, consistent in sign on every split, plus 0.0011 F1 and 0.0028 recall at a 20% alert rate.
Say it costs something and is still worth paying, rather than claiming it is free: using sex or
marital status in a credit decision is prohibited and age is restricted, so the gain cannot be
spent at any size. Age contributes most of it, which is the one of the four with any
behavioural reading.

## 3. The extra features give a small, consistent gain  [artifact] [notebook]

Twenty-four columns in twelve groups engineered beside the 46 served features (notebook 01,
section 3) and tested by `app.ml.feature_experiments`.

**Individually several are strong.** Univariate AUC on training rows: `delinquency_run_current`
0.698, `pay_status_volatility` 0.659, `status_entropy` 0.633, `status_changes` 0.625, against
0.731 for the served `pay_months_late`. The open delinquency run separates a 13.7% default rate
at zero months from 69.8% at six.

**In the model, paired across five splits through the full pipeline:**

| Arm | Test AUC | AUC delta | Better on | Test F1 delta | Recall @20% delta |
|---|---|---|---|---|---|
| Served only (46) | 0.7870 | — | — | — | — |
| + inferred spending | 0.7873 | +0.0003 | 5/5 | +0.0013 | +0.0006 |
| + utilisation slope | 0.7873 | +0.0003 | 4/5 | +0.0008 | +0.0004 |
| + roll-rate transitions | 0.7869 | −0.0001 | 1/5 | +0.0034 | −0.0006 |
| + limit pressure | 0.7870 | −0.0000 | 1/5 | +0.0033 | +0.0002 |
| **+ every addition (70)** | **0.7876** | **+0.0006** (spread 0.0002) | **5/5** | **+0.0026** | **+0.0012** |

**A measurement lesson worth reporting.** The same comparison on a single validation split,
with the model fitted on train only, gave −0.0015 for the full set and supported the opposite
conclusion. The paired five-split version of the deployed pipeline — refit on train plus
validation, threshold from validation, then test — gives +0.0006 on 5 of 5 splits with a paired
spread of 0.0002. The effect is roughly three times that spread and a 5/5 sign run is p ≈ 0.03,
so it is small but not noise. A single split could not resolve it in either direction.

**Distinctness was necessary and not sufficient.** Four variants were dropped before testing
for restating a served feature: a longest-arrears run (0.998 rank correlation with
`pay_months_late`), an absolute utilisation slope (0.96 with `util_trend`), a utilisation shift
(0.91), and the payment-to-bill ratio, which `repay_ratio_*` already computes by the same
formula. What remains is independent, the closest being `months_cleared_bill` at 0.85.

**For the report:** the extra features are worth about +0.0006 AUC and +0.0026 F1 — real,
consistent, and far smaller than the effort that produced them. Whether to promote them into
`feature_engineering.py` is a judgement call rather than an obvious win, and the two groups
carrying most of it are the best motivated: inferred spending from the balance identity, and
the six-month utilisation slope.

## 4. What each modelling choice is worth  [artifact]

| Evidence | Result |
|---|---|
| Hyperparameter search, 70 configurations | 0.7865 to 0.7915 CV AUC |
| Plain logistic regression | 0.760 |
| Resampling, 7 strategies | none beats the 0.7829 baseline by more than noise; best Tomek links 0.7833; SMOTE 0.7799, ADASYN 0.7777 |
| Ensembling | single tuned LightGBM 0.7831 beats soft voting (0.7803–0.7820), stacking (0.7824), hard voting (0.6455) |
| Model families | GradientBoosting 0.7807, ExtraTrees 0.7734, HistGradientBoosting 0.7724, AdaBoost 0.7704, KNN 0.7691, linear SVM 0.7531 |
| Neural networks, 6 architectures | best 0.7798 (PyTorch, four layers); sklearn MLPs 0.7677–0.7747 |
| Feature engineering, 24 columns | +0.0006 AUC, consistent on 5/5 splits (section 3) |

**Re-tuning does not change the picture.** The objection that the extra features were judged
under hyperparameters searched for the old 46-feature set was tested directly: 40 random
configurations re-tuned by 3-fold CV on the expanded set. In a first run the deployed
configuration scored 0.7913 CV and the best random one also 0.7913, with 0.7911 and 0.7909
next. A greedy forward selection over the twelve feature groups kept exactly one,
`predictability`, at +0.0004 on validation, and stopped. The version inside
`feature_experiments.py` is re-running to make these figures regenerable. **[script]**

**On Tomek links**, the best of the resampling arms. A Tomek link is a pair of nearest
neighbours from opposite classes, which can only sit on the class boundary; the method deletes
the majority-class member of each pair. It is boundary cleaning rather than rebalancing —
nothing is synthesised and the ratio barely moves, 21,000 rows at 22.1% default becoming 19,782
at 23%. That is why it alone kept its calibration (Brier 0.1357 against the baseline's 0.1351,
where `class_weight=balanced` reached 0.178). But it bought +0.0004 AUC by deleting exactly the
ambiguous cases, and lost PR-AUC (0.5540) and best F1 (0.5414) doing it. For a lender that
trade is backwards: the ambiguous region is where the decisions are.

**Resampling improves only the cutoff-dependent metric, and for the wrong reason.** Balanced
accuracy rises for every reweighting arm (0.6482 to 0.7157 for `class_weight=balanced`) because
it is measured at the 0.5 cutoff and resampling inflates the predicted probabilities. The Brier
column shows the cost. The deployed model obtains the same effect by setting its threshold to
0.2389 and keeps honest probabilities, which is why threshold tuning is the right lever.

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

## 9. The engineered features and the raw sequence carry the same information  [script]

Every other model in the project reads flat per-customer statistics. This one reads the monthly
matrix in order: 6 timesteps by 3 channels (repayment status, utilisation, payment as a share of
the limit), with the log credit limit as a static input. A GRU, 48 hidden units, three seeds,
early stopped on validation AUC.

| Model | Validation AUC |
|---|---|
| Deployed booster, 46 engineered features | 0.7832 |
| GRU, raw sequence only, three-seed ensemble | 0.7752 |
| Rank-scale blend, weight 0.30 chosen on validation | 0.7839 |

Rank correlation between the GRU's scores and the booster's: **0.917**.

**For the report:** this is the clearest evidence that the dataset is the binding constraint. A network given
no feature engineering at all — only the raw months and the limit — lands 0.008 behind a tuned
booster on 46 engineered features, and ranks customers almost identically to it. The conclusion
is not that the engineered features are bad but that the information in those eighteen numbers
is already extracted, and two very different model classes converge on the same ordering. It
also corroborates Gunnarsson et al. (2021) on deep learning for credit scoring, and Grinsztajn
et al. (2022) on tree models still leading on tabular data.

**The blend does not survive an honest test.** On validation it gained +0.0008 across a plateau
(weights 0.20 to 0.35 all giving 0.7839) rather than a single spike, which looked like a small
real effect. With the weight then fixed at 0.30 and applied to the test split, both arms fitted
on train only:

| Arm | Test AUC |
|---|---|
| Booster alone | 0.7824 |
| GRU alone | 0.7685 |
| Blend at 0.30 | 0.7810 (−0.0014) |

**For the report:** this is a clean demonstration of selection bias worth a paragraph of its
own. Choosing the best of fourteen blend weights on validation produced a gain that reversed
sign on data that had not chosen it, and the plateau — which I had read as evidence against
noise — did not save it. Every tuning decision in the project is exposed to this, which is why
the test split is evaluated once and why operating points are chosen on validation.

**Caveats to state rather than hide:** one split, one architecture, no hyperparameter search for
the network, and an input encoding chosen by hand. The measured claim is about this GRU, not
about sequence models in general.

## Open items

- **Two columns now breach the distinctness rule.** The latest run puts `spend_to_payment` at
  0.94 against `util_trend` and `months_below_minimum` at 0.905 against `repay_ratio_mean`,
  above the 0.90 threshold section 3 of notebook 01 states. They should be dropped for the same
  reason the other four were.
- **Three [script] findings are not reproducible from the repository**: the demographics
  comparison (section 2), the five-split repeats (sections 2 and 3). They belong in
  `experiments.py` as a study writing to `experiment_results.json`, per CODE_CONVENTIONS §16.
- **Running:** the test evaluation of the sequence blend, weight fixed at 0.30 (section 9).
- **Untested:** a systematic feature bank (`tsfresh`, `catch22`) with selection inside
  cross-validation, which would replace hand-picked features with a searched space. Neither
  package is installed yet. Also untested on the network side: an input encoding that treats
  `PAY_n` as categorical rather than scaled, attention over the six months, a hyperparameter
  search for the network, and feeding the learned representation into the booster as features
  rather than blending the two sets of probabilities.
- **The minimum-payment constant is an assumption.** `MINIMUM_SHARE = 0.10` in notebook 01 needs
  checking against the 2005 Taiwanese regulation before any figure derived from it is published.
