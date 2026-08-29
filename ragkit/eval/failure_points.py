"""Barnett et al.'s seven failure points, and what maps onto which.

WHY THIS IS A MODULE AND NOT A DICT IN A SCRIPT. The histogram lived only in
`scripts/failure_histogram.py`, writing a markdown file -- so the strongest
single result this project has (all failures are FP2/FP3, none are FP4, which
rules out prompt engineering and reordering ON EVIDENCE) was reachable only by
opening a file in the repo. The Inspector is the screen built to show findings
and it did not show that one.

Exposing it through the API needed the mapping in a place both the script and
the web layer can import. Copying it would have been the A-16 corollary all over
again: a second copy agrees with the first until it does not, and the
disagreement is silent.
"""

from __future__ import annotations

# The taxonomy, in order. Names are Barnett's.
FP_NAMES: dict[str, str] = {
    "FP1": "Missing Content",
    "FP2": "Missed Top Ranked",
    "FP3": "Not In Context",
    "FP4": "Not Extracted",
    "FP5": "Wrong Format",
    "FP6": "Incorrect Specificity",
    "FP7": "Incomplete",
}

# A ZERO IS NOT THE SAME EVERYWHERE. FP1 cannot be observed at all here -- the
# golden set is generated FROM the corpus, so "the answer is not in the corpus"
# is structurally impossible to produce. FP6 and FP7 need a human judgement no
# automated check in this project performs. Only FP4 and FP5 are zeros that mean
# "measured, did not happen", and FP4's zero is the load-bearing one.
FP_OBSERVABLE: dict[str, str] = {
    "FP1": "structurally unobservable — the golden set is generated from the corpus",
    "FP2": "measured",
    "FP3": "measured",
    "FP4": "measured",
    "FP5": "measured",
    "FP6": "not measured — needs a human judgement",
    "FP7": "not measured — needs a human judgement",
}

CAUSE_TO_FP = {
    "evidence_absent": (
        "FP2",
        "the needle exists in the corpus and retrieval did not deliver it",
    ),
    "evidence_partial": (
        "FP3",
        "some needles arrived and the rest did not survive the budget fill -- "
        "retrieved, then consolidated away",
    ),
    "starved_by_budget": (
        "FP3",
        "candidates ranked correctly and NONE fit the token budget, so nothing "
        "reached the model",
    ),
    "no_candidates": ("FP2", "ranking returned nothing"),
    "retrieval_miss_answered": (
        "FP2",
        "the golden needle was not retrieved at this budget AND the model answered "
        "anyway -- from other context, so the answer may still be right; the "
        "measurement cannot tell",
    ),
    "unsupported_answer": (
        "FP4",
        "context was delivered and the answer asserted something it does not support",
    ),
    "failed_verdict": (
        "FP5",
        "the judge could not produce a parseable verdict for this item",
    ),
}
