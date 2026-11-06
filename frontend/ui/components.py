"""Product surfaces, drawn as HTML so the same piece serves two callers.

The landing page shows these filled with one real, anonymous customer and the
real fairness audit. A signed-in screen can pass its own rows to the same
function and get the same surface back, which is the point of them living here
rather than inside the sign-in view: the page that sells the product and the
product cannot drift apart.

Every function returns one line of markup, because Streamlit renders through
Markdown and an indented line after a blank one becomes a code fence.

Signs follow the model's own convention: a positive effect pushes a customer
towards default, so it is drawn as a reason that hurts.
"""
from html import escape

# The two directions a reason can pull. Not the risk-band traffic light: these
# say which way a number moved a score, not how risky the customer is.
HELPS = "#2F6DA3"
HURTS = "#B8742A"

BAND_TONE = {"Low": "#2F7A52", "Medium": "#B8742A", "High": "#A6442F"}


def flatten(markup: str) -> str:
    """One line, no indentation, so Markdown leaves the HTML alone."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def surface(body: str, *, title: str = "", classes: str = "") -> str:
    """A piece of the product floating on the page: white, bordered, raised."""
    head = f'<div class="surface-title">{escape(title)}</div>' if title else ""
    return flatten(f"""
<div class="surface {classes}" aria-hidden="true">{head}{body}</div>
""")


def score_card(
    *,
    name: str,
    probability: float,
    band: str,
    reasons: list[tuple[str, float]],
) -> str:
    """A customer's score and the reasons behind it, as bars from a centre line.

    `reasons` is (plain-language description, effect) in the model's own sign
    convention, strongest first. The widest bar sets the scale, so the lengths
    compare within one customer and never across two.
    """
    widest = max((abs(effect) for _, effect in reasons), default=1.0) or 1.0
    rows = "".join(_reason_row(text, effect, widest) for text, effect in reasons)

    return surface(
        flatten(f"""
<div class="score-head">
<div>
<div class="score-name">{escape(name)}</div>
<div class="score-sub">Chance of missing the next payment</div>
</div>
<div class="score-figure">
<span class="score-value">{probability:.0%}</span>
<span class="score-band" style="color:{BAND_TONE.get(band, HURTS)}">{escape(band)} risk</span>
</div>
</div>
<div class="reason-head"><span>Raises the risk</span><span>Lowers it</span></div>
<div class="reason-list">{rows}</div>
"""),
        classes="surface--score",
    )


def _reason_row(text: str, effect: float, widest: float) -> str:
    """One reason: the words, then a bar either side of the centre line.

    Right helps the customer, left hurts them. That is the reverse of the
    model's sign, where a positive effect pushes towards default, and the
    reverse is the one a reader expects: better news to the right.
    """
    share = min(abs(effect) / widest, 1.0) * 50
    if effect > 0:
        bar = (
            f'<span class="reason-bar reason-bar--hurts" '
            f'style="right:50%;width:{share:.1f}%"></span>'
        )
    else:
        bar = (
            f'<span class="reason-bar reason-bar--helps" '
            f'style="left:50%;width:{share:.1f}%"></span>'
        )
    return (
        f'<div class="reason-row"><span class="reason-text">{escape(text)}</span>'
        f'<span class="reason-track">{bar}</span></div>'
    )


def recourse_cards(
    *,
    today: str,
    steps: list[str],
    outcome: str,
) -> str:
    """Where the account stands, what would move it, and where that lands it.

    Two states joined by a connector, with the actions on the join: the shape
    of a counterfactual, which is "this, if you do that".
    """
    moves = "".join(f"<li>{escape(step)}</li>" for step in steps)
    return flatten(f"""
<div class="recourse" aria-hidden="true">
<div class="recourse-state">
<span class="recourse-when">Today</span>
<span class="recourse-band" style="color:{BAND_TONE.get(today, HURTS)}">{escape(today)} risk</span>
</div>
<div class="recourse-join"><ul class="recourse-steps">{moves}</ul></div>
<div class="recourse-state recourse-state--after">
<span class="recourse-when">After those</span>
<span class="recourse-band" style="color:{BAND_TONE.get(outcome, HELPS)}">{escape(outcome)} risk</span>
</div>
</div>
""")


def fairness_plot(
    *,
    rows: list[tuple[str, float, str]],
    floor: float,
    parity: float = 1.0,
) -> str:
    """One dot per audited attribute, against the band the audit accepts.

    `rows` is (attribute, ratio, groups compared). The project computes its
    fairness ratio per attribute rather than per group — it is the lowest group
    approval rate over the highest — so an attribute is what a dot can stand
    for without inventing a number.
    """
    # The axis starts a little below the floor so a failing dot has somewhere to
    # sit, and ends just past parity.
    low, high = floor - 0.2, 1.1
    span = high - low

    def position(value: float) -> float:
        return (value - low) / span * 100

    band_left = position(floor)
    dots = "".join(
        f'<div class="fair-row">'
        f'<span class="fair-name">{escape(name)}</span>'
        f'<span class="fair-track">'
        f'<span class="fair-band" style="left:{band_left:.1f}%;'
        f'width:{position(high) - band_left:.1f}%"></span>'
        f'<span class="fair-line" style="left:{position(parity):.1f}%"></span>'
        f'<span class="fair-dot" style="left:{position(ratio):.1f}%"></span>'
        f'<span class="fair-value">{ratio:.2f}</span>'
        f"</span>"
        f'<span class="fair-groups">{escape(groups)}</span>'
        f"</div>"
        for name, ratio, groups in rows
    )

    return surface(
        flatten(f"""
<div class="fair-head">
<span>Approval rate of the least-approved group over the most-approved</span>
</div>
<div class="fair-rows">{dots}</div>
<div class="fair-key">
<span><i class="key-band"></i>Accepted: {floor:.2f} and above</span>
<span><i class="key-line"></i>Equal treatment: {parity:.2f}</span>
</div>
"""),
        classes="surface--fair",
    )


def decision_panel(
    *,
    customer: str,
    band: str,
    actions: list[str],
    chosen: str,
    note: str,
    recorded: str,
) -> str:
    """What an officer does with a score: pick an action, say why, record it."""
    buttons = "".join(
        f'<span class="action {"action--on" if label == chosen else ""}">'
        f"{escape(label)}</span>"
        for label in actions
    )
    return surface(
        flatten(f"""
<div class="decide-head">
<span class="decide-customer">{escape(customer)}</span>
<span class="decide-band" style="color:{BAND_TONE.get(band, HURTS)}">{escape(band)} risk</span>
</div>
<div class="decide-label">Action</div>
<div class="action-row">{buttons}</div>
<div class="decide-label">Why</div>
<div class="decide-note">{escape(note)}</div>
<div class="decide-foot">{escape(recorded)}</div>
"""),
        classes="surface--decide",
    )


def worklist(*, rows: list[tuple[str, str, str, str]]) -> str:
    """The officer's queue: who to look at, how risky, and the headline reason."""
    body = "".join(
        f'<div class="queue-row">'
        f'<span class="queue-id">{escape(customer)}</span>'
        f'<span class="queue-limit">{escape(limit)}</span>'
        f'<span class="queue-band" style="color:{BAND_TONE.get(band, HURTS)}">'
        f"{escape(band)}</span>"
        f'<span class="queue-reason">{escape(reason)}</span>'
        f"</div>"
        for customer, limit, band, reason in rows
    )
    return surface(
        flatten(f"""
<div class="queue-row queue-row--head">
<span class="queue-id">Account</span><span class="queue-limit">Credit limit</span>
<span class="queue-band">Risk</span><span class="queue-reason">Top reason</span>
</div>
{body}
"""),
        classes="surface--queue",
    )


def view_previews(*, views: list[tuple[str, str, str]]) -> str:
    """Three framed sketches, one per role, each with a sentence under it."""
    frames = "".join(
        f'<figure class="preview">'
        f'<div class="preview-frame" aria-hidden="true">{sketch}</div>'
        f'<figcaption><strong>{escape(name)}</strong>{escape(line)}</figcaption>'
        f"</figure>"
        for name, sketch, line in views
    )
    return flatten(f'<div class="preview-row">{frames}</div>')
