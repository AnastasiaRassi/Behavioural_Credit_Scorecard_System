"""Metric captions and 💭 explanations shown in the dashboard.

Covers the dashboard's own vocabulary: what a Brier score is, why a number is
trustworthy. Per-customer and per-feature wording lives in
`app.ml.explainability`, not here, so the model's explanations stay in one place.

Kept in one file so the wording can be reviewed as prose rather than hunted down
across a dozen `help=` arguments. CAPTIONS are the one-liners always visible under
a number; DEEP are the longer plain-language explanations behind the 💭 icon.

Numbers are deliberately absent from most strings. Anything that would go stale
when the model is retrained is formatted in at render time from the artifacts.
"""

CAPTIONS = {
    "roc_auc": "ranks risk correctly {pct}% of the time",
    "f1_weighted": "overall quality across both classes",
    "balanced_accuracy": "treats both classes as equally important",
    "gini": "banking convention for the same thing",
    "f1_macro": "both classes counted equally",
    "f1": "performance on defaulters alone",
    "pr_auc": "precision vs recall on defaulters",
    "ks": "biggest separation between the groups",
    "brier": "are the probabilities honest",
    "mcc": "correlation with reality",
    "cohen_kappa": "agreement beyond luck",
}

DEEP = {
    "roc_auc": """
Take one customer who defaulted and one who paid. The model gives the defaulter a
higher risk score about {pct} times out of 100. A coin flip would get 50.

This is the main number to judge the model by, and it does not change if the
proportion of defaulters changes.
""",
    "f1_weighted": """
Combines how well we identify defaulters and how well we identify good payers,
counting each group by its size. This is what most papers mean when they report a
single F1 figure.
""",
    "balanced_accuracy": """
The average of two things: the share of defaulters we catch ({recall}%) and the
share of good payers we correctly leave alone ({specificity}%).

Because it averages them, a model that ignored defaulters entirely would score 50,
not {naive}.
""",
    "gini": """
Exactly the same information as AUC, rescaled: **Gini = 2 × AUC − 1**. Credit teams
quote this out of habit. 0 is random, 1 is perfect.
""",
    "f1_macro": """
Like weighted F1, but small and large groups count the same. Lower than weighted F1
because the minority class is harder.
""",
    "f1": """
The hardest single number we report, and the one we do not hide. It looks low, but on
this dataset a coin flip scores about 0.31 and a naive "was he late last month" rule
scores about 0.50.

Only {default_rate}% of customers default, which caps how high this can go.
""",
    "pr_auc": """
Similar to AUC but focused entirely on the defaulters. The no-skill baseline is
{default_rate_frac}, the default rate, not 0.5.
""",
    "ks": """
The widest gap between the score distributions of defaulters and payers. Credit
scorecards are often judged on this. Above 0.40 is considered strong.
""",
    "brier": """
Checks whether the numbers mean what they say. If the model says 30%, do about 30%
actually default? Lower is better, 0 is perfect.

Ours roughly halved after calibration.
""",
    "mcc": """
Treats the prediction as one variable and the truth as another and measures
correlation. 0 is random, 1 is perfect. Hard to fool with imbalanced data.
""",
    "cohen_kappa": """
How much better than guessing, after subtracting the agreement you would get by
chance.
""",
    "confusion_matrix": """
The two mistakes cost different amounts.

The **{fn}** in the bottom left are customers who defaulted and we did not flag,
which costs the outstanding balance. The **{fp}** in the top right are customers we
flagged who would have paid, which costs the interest we would have earned.

A lender usually treats the first as several times more expensive.
""",
    "bands": """
Band edges come from the validation set, never from the test set. The default rates
shown are what actually happened on held-out customers the model had never seen.
""",
    "deciles": """
This is the clearest evidence the model works. Customers are sorted by predicted risk
and split into ten groups.

If the model were guessing, every bar would sit at {average}%. Instead the safest
group defaults at {lowest}% and the riskiest at {highest}%, a {spread}× spread.
""",
    # --- fairness ---------------------------------------------------------
    "fairness_verdict": """
The model never sees sex, age, education or marital status when making a decision.
They are kept only to check the results afterwards.

Removing them cost almost nothing in accuracy, so there is no performance argument
for putting them back.
""",
    "disparate_impact": """
**Disparate impact ratio.** Do we approve every group at a similar rate? Take the
group approved least often and divide by the group approved most often.

1.00 means identical treatment. Regulators commonly treat anything below 0.80 as
evidence of adverse impact, which is where the "four fifths" name comes from.
""",
    "tpr_gap": """
**TPR gap.** Do we catch real defaulters equally well in every group? A gap of 0.080
means one group has its defaulters spotted 8 in 100 more often than another.
""",
    "fpr_gap": """
**FPR gap.** Do we wrongly flag good customers equally rarely in every group? A gap
of 0.079 means one group gets wrongly flagged about 8 in 100 more often.
""",
    "auc_spread": """
**AUC spread.** Is the model equally good at ranking risk inside each group? Near
zero means the model is equally capable for every group.
""",
    "equalized_odds_warning": """
These three columns come from a **single data split**. Repeating the analysis across
five independent splits showed the gaps move substantially and sometimes reverse
sign.

Treat them as **indicative only**. Do not conclude the model is unfair to a specific
group from these numbers alone. Disparate impact above is stable; these are not.
""",
    "equalized_odds_why": """
Why it matters: it would be easy to read the education row and announce the model
discriminates by education. We tested that and it did not hold up.

Honest dashboards say which of their own numbers are shaky.
""",
    "per_group": """
Groups with fewer than 50 members or 10 defaults are hidden, since their rates are
too noisy to read.
""",
    # --- explainability ---------------------------------------------------
    "feature_importance": """
Each feature is scrambled in turn and we measure how much the model gets worse. The
bigger the drop, the more the model relied on it.

A few features carry almost all the signal: has the customer missed a payment, how
badly, and how much room is left on the card.
""",
    "feature_importance_caveat": """
The six monthly payment columns are strongly correlated with each other, so this
method **understates each one individually**. The honest reading is that the
delinquency group of features dominates, not that exactly three features matter.
""",
    "single_prediction": """
Each row is a feature contribution: how much that single fact pushed this customer's
risk up or down, starting from the average customer.

The rows add up to the final score, so the explanation is complete rather than a
summary.
""",
    # --- threshold --------------------------------------------------------
    "threshold": """
The threshold is the cut-off where a score becomes a decision.

Raising it flags fewer customers, so the ones flagged are more likely to really
default (higher precision) but more defaulters slip through (lower recall).

You cannot improve both at once by moving this slider, only trade them.
""",
    "cost_curve": """
The two mistakes are not equally expensive. Missing a default costs the balance owed.
Refusing a good customer costs the interest you would have earned.

Set the ratio your business believes and the chart shows where to put the threshold.
""",
    "calibration": """
If the model says 30% and 30% of those customers actually default, the dot sits on
the diagonal. Points below the line mean the model is too pessimistic, above means
too optimistic.

This matters because the risk bands and any expected-loss calculation depend on the
probabilities being truthful, not just correctly ordered.
""",
    # --- registry and drift -----------------------------------------------
    "registry": """
Every model that has ever been trained is recorded with the numbers it achieved and
the data it saw, so a decision made six months ago can be traced back to the exact
model that made it.
""",
    "drift": """
PSI compares the customers arriving now against the customers the model was trained
on. A high value means the incoming population has shifted and the model may no
longer fit it, and this can be detected before any defaults are even observed.
""",
}

TRUST_STABLE = "▲ stable"
TRUST_UNSTABLE = "⚠ unstable"
