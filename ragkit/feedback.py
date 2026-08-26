"""
Feedback capture. Guide module M15, minimal but structurally complete.

WHY THIS IS NOT A UI NICETY, and why it is worth thirty minutes now rather than
later: it is the ONLY channel by which ground truth enters from outside the
system.

Every measurement in this project so far is computed by the system about itself,
and that has gone wrong three times in ways no internal check could catch:

  - table-detector prevalence counted what the DETECTOR found, which is bounded
    below by the detector's blind spots (an unruled table is invisible to it)
  - the 18% headerless figure described a parser we do not use
  - the golden set is machine-generated and machine-verified; 8 fabricated
    needles were caught, but nobody has read a stratified slice

The fix for a circular measurement is always the same: something outside the
circle. Human labels are that thing, and a flag on a claim is the cheapest
possible way to collect one.

---------------------------------------------------------------------------
THE DESIGN DECISION THAT MAKES THIS USEFUL

Store the SYSTEM'S OWN VERDICT alongside the human's, at the moment of the flag.

A bare list of "user disliked this" is nearly worthless -- it tells you something
was wrong, not what. Storing what the system believed at that moment turns each
flag into one cell of a confusion matrix:

    system said            human said              what it means
    ----------------------------------------------------------------------
    quote verified         "isn't in the source"   the quote check passed and
                                                   should not have
    structure_inferred     "the table reads fine"  a FALSE POSITIVE from the
                                                   table-corruption detector --
                                                   the only way to learn this
    assistant_reading      "source doesn't say"    the caption misread the chart
    found_not_quoted       "the table reads fine"  we declined a table that was
                                                   fine, which is a cost we
                                                   otherwise cannot see

That third and fourth row are the ones that cannot be obtained any other way.
The detector's false-positive rate is not derivable from the detector.

The two flag reasons come straight from the two citation failures that were
separated earlier, because they have different fixes:
    NOT_IN_SOURCE     -- the cited chunk does not contain this (fabrication)
    SOURCE_DOESNT_SAY -- the chunk exists and is cited but does not support the
                         claim (unsupported)

APPEND-ONLY JSONL. No update, no delete. A feedback store that can be edited is
not evidence, and the whole point is that it is the one record the system did not
write about itself.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import config

Verdict = Literal[
    "not_in_source",        # the cited chunk does not contain this
    "source_doesnt_say",    # cited chunk exists but does not support the claim
    "source_is_fine",       # we flagged/declined a source the user says reads fine
    "helpful",              # positive signal, kept so the set is not all-negative
]

VERDICT_LABEL: dict[str, str] = {
    "not_in_source": "This isn't in the source",
    "source_doesnt_say": "The source doesn't say this",
    "source_is_fine": "This source reads fine to me",
    "helpful": "This was helpful",
}

PATH = config.DATA_EVAL / "feedback.jsonl"


@dataclass
class Feedback:
    verdict: Verdict
    conversation_id: str = ""
    turn_index: int = -1
    claim_index: int = -1
    claim_text: str = ""
    note: str = ""

    # THE SYSTEM'S BELIEF AT THE TIME. Without these a flag says "something was
    # wrong"; with them it says which component was wrong, which is the
    # difference between a complaint and a measurement.
    chunk_id: str = ""
    source_id: str = ""
    evidence_kind: str = ""       # quoted | structure_inferred | assistant_reading | ...
    quote_status: str = ""        # verified | absent | unquotable | no_quote
    text_source: str = ""         # markdown | page_text_clip | gemini_caption
    claim_source: str = ""        # documents | conversation
    table_header_missing: bool | None = None
    table_continuation_suspect: bool | None = None
    pipeline_fingerprint: str = ""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def record(fb: Feedback, path: Path | None = None) -> Path:
    """Append one flag. Append-only, never rewritten."""
    p = path or PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fb.to_json(), ensure_ascii=False) + "\n")
    return p


def load(path: Path | None = None) -> list[Feedback]:
    p = path or PATH
    if not p.exists():
        return []
    out: list[Feedback] = []
    for line in p.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Feedback(**json.loads(line)))
        except Exception:  # noqa: BLE001 -- one bad line must not lose the rest
            continue
    return out


def stats(path: Path | None = None) -> dict[str, Any]:
    """Agreement between what the system believed and what a human said.

    This is the confusion matrix that breaks the circle. Every rate here reports
    its own n, and refuses a percentage below the eval's floor -- a false-positive
    rate computed from three flags is an anecdote, and the withdrawn 18%
    headerless figure is the cautionary case.
    """
    from .eval.metrics import MIN_N_FOR_RATE

    fbs = load(path)
    n = len(fbs)
    by_verdict = Counter(f.verdict for f in fbs)

    # Cells that identify a specific component as wrong.
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for f in fbs:
        matrix[f.evidence_kind or "unknown"][f.verdict] += 1

    def rate(hits: int, total: int) -> dict[str, Any]:
        if total == 0:
            return {"n": 0, "hits": 0, "rate": None, "sufficient": False, "label": "no flags"}
        if total < MIN_N_FOR_RATE:
            return {"n": total, "hits": hits, "rate": None, "sufficient": False,
                    "label": f"{hits} of {total}"}
        return {"n": total, "hits": hits, "rate": round(hits / total, 3),
                "sufficient": True, "label": f"{hits}/{total} = {hits / total:.0%}"}

    # The two numbers nothing internal can produce.
    inferred = [f for f in fbs if f.evidence_kind == "structure_inferred"]
    declined = [f for f in fbs if f.evidence_kind == "found_not_quoted"]
    verified = [f for f in fbs if f.quote_status == "verified"]

    return {
        "n": n,
        "by_verdict": dict(by_verdict),
        "by_evidence_kind": {k: dict(v) for k, v in matrix.items()},
        "table_detector_false_positives": rate(
            sum(1 for f in inferred if f.verdict == "source_is_fine"), len(inferred)
        ),
        "declined_but_fine": rate(
            sum(1 for f in declined if f.verdict == "source_is_fine"), len(declined)
        ),
        "verified_but_wrong": rate(
            sum(1 for f in verified if f.verdict in ("not_in_source", "source_doesnt_say")),
            len(verified),
        ),
        "note": (
            "these three are the only measurements in the project that the system "
            "cannot compute about itself: a detector's false-positive rate is not "
            "derivable from the detector"
        ),
    }
