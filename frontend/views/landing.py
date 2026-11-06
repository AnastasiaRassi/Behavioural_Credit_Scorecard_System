"""What the sign-in page says below the fold, to someone who has never seen it.

Each section makes one claim and shows the product making good on it. The rule
the page is built to is that nothing here is mocked up: the scores, the reasons
and the fairness ratios are this project's own output, and every section whose
feature is not actually in the app was left out rather than illustrated.

One was left out for that reason:

* An application pipeline. The system scores existing card accounts from their
  payment history; there is no application, no approval step and no repayment
  schedule to track, so there is no Applied → Repaying story to tell.

The officer's actions below are the four the app really offers, not the
approve / decline / refer of a loan desk this system is not.
"""
from ui import components

# --- what the model actually said -----------------------------------------

# One real account from the held-out evaluation set, scored by the live model.
# The dataset's customers are anonymous — an id and a payment history, no name
# — so showing one invents nothing and identifies nobody. Reasons are in the
# model's own sign convention: positive pushes towards default.
SAMPLE_CUSTOMER = {
    "name": "Sample customer",
    "probability": 0.3093,
    "band": "Medium",
    "reasons": [
        ("Worst month in the six: 2 months in arrears", 0.494),
        ("Last month: paid in full", -0.232),
        ("Worst month's share of the bill paid at 0%", 0.230),
        ("195,429 of credit still available", -0.109),
        ("Share of bill paid, month 3 at 0%", 0.105),
    ],
}

# One real account, run through GET /cardholders/{id}/recourse. Its limit is
# 20,000 and it is carrying 18,710, which is why clearing the card is what the
# search lands on: it is the smallest move that reaches the next band, not a
# suggestion anyone wrote.
#
# Honest about the other half: for roughly half the accounts tried, nothing
# within the customer's own reach gets them a band better, because six months
# of repayment history outweighs anything this month can do. The endpoint says
# so rather than inventing a step, and the section's words say so too.
RECOURSE_TODAY = "High"
RECOURSE_OUTCOME = "Medium"
RECOURSE_STEPS = [
    "Bring the account up to date, so no month is in arrears",
    "Pay 19,132 into the card this month, +17,732 on now",
]

# From the fairness audit train.py writes, read back through GET /fairness.
# The ratio is the least-approved group's approval rate over the most-approved
# group's, per attribute; the audit accepts 0.80 and above (the four-fifths
# rule). All four pass on the current model.
FAIRNESS_FLOOR = 0.80
FAIRNESS_ROWS = [
    ("Marital status", 0.99, "single, married, other"),
    ("Sex", 0.94, "female, male"),
    ("Age", 0.94, "five age bands, 21 to 60+"),
    ("Education", 0.88, "university, graduate school, high school, other"),
]

# Five rows from the live queue, worst first, as the officer screen orders them.
QUEUE_ROWS = [
    ("20722", "20,000", "High", "Last month: 3 months in arrears"),
    ("2855", "50,000", "High", "Last month: 2 months in arrears"),
    ("19429", "200,000", "Medium", "Worst month in the six: 2 months in arrears"),
    ("19278", "80,000", "Medium", "Worst month's share of the bill paid at 0%"),
    ("261", "510,000", "Low", "Worst month in the six: revolving credit"),
]

# The officer screen's own four actions.
OFFICER_ACTIONS = ["No action", "Monitor", "Contact customer", "Reduce limit"]

# Wording an officer would type. The only invented text on the page, and it is
# a note rather than a result.
SAMPLE_NOTE = (
    "Two months behind at the worst point, but last month was paid in full and "
    "most of the limit is still free. Calling to check whether the missed "
    "months were a one-off before touching the limit."
)

# --- the small sketches in the three-view section --------------------------

_PREVIEW_OFFICER = """
<svg viewBox="0 0 160 96" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="12" width="60" height="5" rx="2.5" fill="#C6D1DB"/>
<rect x="10" y="30" width="90" height="6" rx="3" fill="#E3E8EE"/>
<rect x="10" y="30" width="38" height="6" rx="3" fill="#A6442F"/>
<rect x="10" y="46" width="90" height="6" rx="3" fill="#E3E8EE"/>
<rect x="10" y="46" width="52" height="6" rx="3" fill="#B8742A"/>
<rect x="10" y="62" width="90" height="6" rx="3" fill="#E3E8EE"/>
<rect x="10" y="62" width="22" height="6" rx="3" fill="#2F7A52"/>
<rect x="114" y="28" width="36" height="42" rx="6" fill="#EEF3F7"/>
<rect x="120" y="36" width="24" height="4" rx="2" fill="#C6D1DB"/>
<rect x="120" y="45" width="18" height="4" rx="2" fill="#C6D1DB"/>
<rect x="120" y="56" width="24" height="7" rx="3.5" fill="#2F6DA3"/>
</svg>
"""

_PREVIEW_ENGINEER = """
<svg viewBox="0 0 160 96" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="12" width="42" height="26" rx="5" fill="#EEF3F7"/>
<rect x="59" y="12" width="42" height="26" rx="5" fill="#EEF3F7"/>
<rect x="108" y="12" width="42" height="26" rx="5" fill="#EEF3F7"/>
<rect x="16" y="20" width="22" height="6" rx="3" fill="#2F6DA3"/>
<rect x="65" y="20" width="18" height="6" rx="3" fill="#2F6DA3"/>
<rect x="114" y="20" width="26" height="6" rx="3" fill="#2F6DA3"/>
<path d="M12 84 C 38 82 52 62 72 58 C 92 54 112 66 148 50"
stroke="#2F6DA3" stroke-width="2.4" stroke-linecap="round"/>
<line x1="10" y1="88" x2="150" y2="88" stroke="#E3E8EE" stroke-width="1.6"/>
<circle cx="72" cy="58" r="3" fill="#D9A462"/>
</svg>
"""

_PREVIEW_CARDHOLDER = """
<svg viewBox="0 0 160 96" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="22" y="14" width="116" height="68" rx="8" fill="#EEF3F7"/>
<rect x="34" y="26" width="34" height="5" rx="2.5" fill="#C6D1DB"/>
<rect x="34" y="38" width="46" height="14" rx="4" fill="#2F7A52"/>
<rect x="34" y="60" width="92" height="6" rx="3" fill="#FFFFFF"/>
<rect x="34" y="60" width="28" height="6" rx="3" fill="#2F6DA3"/>
<circle cx="118" cy="34" r="9" fill="#FFFFFF"/>
<circle cx="118" cy="34" r="3.2" fill="#D9A462"/>
</svg>
"""

VIEWS = [
    (
        "Loan officer",
        _PREVIEW_OFFICER,
        "The queue of accounts to look at, and the reasons behind each score.",
    ),
    (
        "ML engineer",
        _PREVIEW_ENGINEER,
        "How the model is holding up: accuracy, the fairness audit, and drift.",
    ),
    (
        "Cardholder",
        _PREVIEW_CARDHOLDER,
        "Your own account, and whether the next payment looks at risk.",
    ),
]


def sections() -> str:
    """Everything between the hero and the closing band."""
    # The anchor is its own empty target rather than the id of .story: a
    # container taller than the window only gets scrolled far enough to be
    # partly visible, which left the link stopping halfway.
    return components.flatten(
        '<div class="anchor" id="how-it-works"></div>'
        '<div class="story-page"><div class="story">'
        + _row(
            "See why, not just what",
            "Every score comes with the reasons behind it, written in plain "
            "language rather than feature names.",
            components.score_card(**SAMPLE_CUSTOMER),
            visual_first=False,
        )
        + _row(
            "Know what would change the outcome",
            "For every account, CrediTrait works out the smallest change that "
            "would move it to a better band — and says so plainly when nothing "
            "the customer can do on their own would be enough.",
            components.recourse_cards(
                today=RECOURSE_TODAY,
                steps=RECOURSE_STEPS,
                outcome=RECOURSE_OUTCOME,
            ),
            visual_first=True,
        )
        + _row(
            "Checked for fairness",
            "Before a model goes live, CrediTrait compares its decisions across "
            "customer groups and flags any gap. The audit runs on four "
            "attributes the model never sees when it scores.",
            components.fairness_plot(rows=FAIRNESS_ROWS, floor=FAIRNESS_FLOOR),
            visual_first=False,
        )
        + _row(
            "A person makes the final call",
            "The model scores and explains. A loan officer reads it, chooses "
            "what to do, and records the reason in their own words.",
            components.decision_panel(
                customer="Sample customer",
                band="Medium",
                actions=OFFICER_ACTIONS,
                chosen="Contact customer",
                note=SAMPLE_NOTE,
                recorded="Recorded by a loan officer at 14:02",
            ),
            visual_first=True,
        )
        + _row(
            "Every account in one queue",
            "Accounts are ordered by what needs attention first, each with its "
            "risk band and the one thing driving it.",
            components.worklist(rows=QUEUE_ROWS),
            visual_first=False,
        )
        + _views()
        + "</div></div>"
    )


def _row(claim: str, body: str, visual: str, *, visual_first: bool) -> str:
    """One section: a short claim on one side, a product surface on the other."""
    side = "story-row--flip" if visual_first else ""
    return (
        f'<section class="story-row {side}">'
        f'<div class="story-text"><h2>{claim}</h2><p>{body}</p></div>'
        f'<div class="story-visual">{visual}</div>'
        f"</section>"
    )


def _views() -> str:
    return (
        '<section class="story-views">'
        "<h2>One product, three views</h2>"
        "<p>The same scores and the same reasons, shown to each person in the "
        "terms their job is in.</p>"
        f"{components.view_previews(views=VIEWS)}"
        "</section>"
    )


def closing_band() -> str:
    """The final call to action: one solid card, one button back to the form."""
    return components.flatten("""
<div class="closing-card">
<h2 class="closing-title">Ready to look inside?</h2>
<p class="closing-lead">Sign in to see the real screens, on the real model.</p>
<a class="closing-button" href="#top">Sign in</a>
</div>
""")
