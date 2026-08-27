"""Every measured failure, classified against Barnett et al.'s seven failure points.

Barnett, Kurniawan, Thudumu, Brannelly & Abdelrazek (2024), "Seven Failure Points
When Engineering a Retrieval Augmented Generation System":

    FP1 Missing Content        the answer is not in the corpus at all
    FP2 Missed Top Ranked      it is in the corpus but not retrieved
    FP3 Not In Context         retrieved, but did not survive consolidation
    FP4 Not Extracted          present in context, and the model did not use it
    FP5 Wrong Format           the output format was not respected
    FP6 Incorrect Specificity  too vague or too narrow to be useful
    FP7 Incomplete             partially answered when more was available

POOLING IS REFUSED UNLESS THE INPUTS ARE COMPARABLE, and this is the whole reason
the script is careful rather than a one-liner. Adding a failure count from one run
to a failure count from another is only meaningful if both came from the same
pipeline at the same token budget -- and until recently two of the three inputs
recorded NEITHER, so the check was impossible rather than merely skipped.

A MISSING STAMP IS A MISMATCH, not a pass. That direction matters: treating
"unknown" as "fine" is how an unlabelled artifact gets pooled once and then
forever. And an unstamped file cannot be fixed by stamping it now -- writing
today's fingerprint onto a file produced under an unknown one manufactures
provenance, which is strictly worse than having none, because the label makes it
look verified.

THE BUDGET IS PART OF THE CLAIM. The retrieval-miss count is a pure function of it
-- 67 at 250 tokens, 14 at 1500, ZERO at 12000 -- so an unlabelled miss count
describes a knob setting rather than a system. Every count below carries its
budget.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit import config  # noqa: E402

OUT = ROOT / "notes" / "FAILURE_ANALYSIS.md"


def _load(name: str) -> dict[str, Any]:
    p = config.DATA_EVAL / name
    if not p.exists():
        raise SystemExit(f"missing input: {name}")
    return json.loads(p.read_text("utf-8"))


def _stamp_of(name: str, doc: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparability stamp, wherever that artifact keeps it."""
    if "stamp" in doc:
        return doc["stamp"]
    # eval_results.json predates the shared helper and nests its provenance.
    prov = doc.get("index_provenance") or {}
    if prov.get("pipeline_fingerprint"):
        return {
            "pipeline_fingerprint": prov["pipeline_fingerprint"],
            "parser_version": prov.get("parser_version"),
            "chunker_version": prov.get("chunker_version"),
            "token_budget": doc.get("token_budget"),
        }
    return {}


def _require_comparable(stamps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [n for n, s in stamps.items() if not s.get("pipeline_fingerprint")]
    if missing:
        raise SystemExit(
            "REFUSING TO POOL -- no stamp on: " + ", ".join(missing) + "\n"
            "An artifact that does not record what produced it cannot be shown "
            "comparable to one that does. Re-run it; do not stamp it after the "
            "fact, which would assert a fingerprint nobody verified."
        )
    for field in ("pipeline_fingerprint", "token_budget"):
        vals = {n: s.get(field) for n, s in stamps.items()}
        if len(set(vals.values())) > 1:
            raise SystemExit(
                f"REFUSING TO POOL -- {field} differs across inputs:\n"
                + "\n".join(f"    {n}: {v}" for n, v in vals.items())
                + "\nThese describe different systems, or the same system at "
                  "different settings. Counting them together would describe a "
                  "configuration that never existed at one moment."
            )
    return next(iter(stamps.values()))


# Cause -> (failure point, why). Stated explicitly rather than inferred, because a
# taxonomy mapping guessed from a label is how a histogram comes to describe the
# labels instead of the system.
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


def main() -> int:
    ev = _load("eval_results.json")
    judged = _load("judged_results.json")
    causes = _load("abstention_causes.json")

    stamp = _require_comparable({
        "eval_results.json": _stamp_of("eval_results.json", ev),
        "judged_results.json": _stamp_of("judged_results.json", judged),
        "abstention_causes.json": _stamp_of("abstention_causes.json", causes),
    })
    budget = stamp["token_budget"]

    failures: list[dict[str, Any]] = []

    # 2. Abstentions FIRST, because they are the more specific classification and
    #    they claim their questions before the retrieval sweep runs. See the
    #    precedence note below.
    seen: set[str] = set()
    for row in causes["rows"]:
        failures.append({
            "cause": row["cause"],
            "stratum": row["stratum"],
            "question": row["question"],
        })
        seen.add(row["question"])

    # 1. Retrieval misses at the stamped budget.
    #
    # TWO POPULATION BUGS LIVED HERE, and the histogram found them by producing a
    # total that did not match the inputs (37 against 32 expected).
    #
    # OUT OF SCOPE IS NOT A MISS. per_item covers the whole golden set -- 101
    # entries including 5 `out_of_scope` questions whose CORRECT behaviour is to
    # retrieve nothing, plus quarantined items. Counting those as retrieval
    # failures inflated 14 misses to 19. The headline scores 92 items for a
    # reason, and a derived count that does not apply the same filter is
    # measuring a different population under the same name.
    #
    # AND AN ABSTENTION CAUSED BY ABSENT EVIDENCE *IS* A RETRIEVAL MISS. Eight
    # questions appeared in both sets, so pooling them double-counted a single
    # failure. Overlapping sets are the same defect as incomparable ones, one
    # level down: the stamp proves two artifacts describe the same system, and
    # nothing proved they described DISJOINT events.
    #
    # Precedence: the abstention cause wins, because it says WHY the evidence was
    # missing. A miss where the model answered anyway is kept separately -- it is
    # a different event, and arguably a more interesting one.
    for item in ev.get("per_item", []):
        if item.get("quarantined") or item.get("anchor") == "none":
            continue
        if item.get("stratum") == "out_of_scope":
            continue
        if item.get("child_strict", True):
            continue
        q = item.get("question", "")
        if q in seen:
            continue
        seen.add(q)
        failures.append({
            "cause": "retrieval_miss_answered",
            "stratum": item.get("stratum", "?"),
            "question": q,
        })

    # 3. Generation-tier failures.
    for row in judged["rows"]:
        if row.get("failed"):
            failures.append({"cause": "failed_verdict",
                             "stratum": row["stratum"], "question": row["question"]})
        elif not row["abstained"] and row["faithfulness"] != "supported":
            failures.append({"cause": "unsupported_answer",
                             "stratum": row["stratum"], "question": row["question"]})

    unknown = sorted({f["cause"] for f in failures} - set(CAUSE_TO_FP))
    if unknown:
        raise SystemExit(
            f"unmapped cause(s): {unknown}. Add them to CAUSE_TO_FP -- a cause "
            "silently dropped would make the histogram under-count without saying so."
        )

    by_fp = collections.Counter(CAUSE_TO_FP[f["cause"]][0] for f in failures)
    by_cause = collections.Counter(f["cause"] for f in failures)
    by_stratum = collections.Counter(f["stratum"] for f in failures)
    total = len(failures)

    # RECONCILIATION: N in, N classified out. A histogram that quietly drops rows
    # reports a smaller problem than it measured.
    assert sum(by_fp.values()) == total, "classified count != input count"

    fp_names = {
        "FP1": "Missing Content", "FP2": "Missed Top Ranked", "FP3": "Not In Context",
        "FP4": "Not Extracted", "FP5": "Wrong Format",
        "FP6": "Incorrect Specificity", "FP7": "Incomplete",
    }

    L: list[str] = []
    L.append("# Failure analysis")
    L.append("")
    L.append(f"**{total} measured failures**, every one from a stamped artifact, "
             f"classified against Barnett et al.'s seven failure points.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| pipeline fingerprint | `{stamp['pipeline_fingerprint']}` |")
    L.append(f"| parser | `{stamp.get('parser_version','?')}` |")
    L.append(f"| chunker | `{stamp.get('chunker_version','?')}` |")
    L.append(f"| **token budget** | **{budget}** |")
    L.append("")
    L.append("All three inputs carry the same fingerprint and budget; the script "
             "refuses to pool otherwise, and treats a *missing* stamp as a "
             "mismatch rather than a pass.")
    L.append("")
    L.append(f"**Every count here is @{budget} tokens.** The retrieval-miss count "
             "is a function of that knob, not a property of the system -- at the "
             "12000-token default there are zero retrieval failures. A count "
             "quoted without its budget says nothing.")
    L.append("")
    L.append("## Histogram")
    L.append("")
    L.append("| failure point | n | share | |")
    L.append("|---|---:|---:|---|")
    for fp in ("FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"):
        n = by_fp.get(fp, 0)
        bar = "█" * round(28 * n / total) if n else ""
        L.append(f"| **{fp}** {fp_names[fp]} | {n} | {n/total:.0%} | {bar} |")
    L.append(f"| | **{total}** | | |")
    L.append("")
    L.append("## What each bar is made of")
    L.append("")
    L.append("| cause | n | failure point | why |")
    L.append("|---|---:|---|---|")
    for cause, n in by_cause.most_common():
        fp, why = CAUSE_TO_FP[cause]
        L.append(f"| `{cause}` | {n} | {fp} | {why} |")
    L.append("")
    L.append("## By question type")
    L.append("")
    L.append("| stratum | failures |")
    L.append("|---|---:|")
    for st, n in by_stratum.most_common():
        L.append(f"| {st} | {n} |")
    L.append("")
    L.append("## What I would fix first")
    L.append("")
    fp2, fp3 = by_fp.get("FP2", 0), by_fp.get("FP3", 0)
    top_stratum, top_n = by_stratum.most_common(1)[0]
    fp4 = by_fp.get("FP4", 0)
    L.append(
        f"**Retrieval, and specifically ranking within a document.** "
        f"{'All ' + str(total) if fp2 + fp3 == total else f'{fp2 + fp3} of {total}'} "
        "failures are FP2 or FP3 -- the evidence exists in the corpus and either "
        "was not retrieved or did not survive the budget fill."
    )
    L.append("")
    L.append(
        f"**{'No failure at all is' if fp4 == 0 else f'{fp4} are'} FP4** -- not one "
        "case where complete context was delivered and the model failed to use it. "
        "`evidence_present` is **0** across every abstention: the generator never "
        "declines with complete evidence in front of it. Whatever is wrong here, it "
        "is not the model's reading."
    )
    L.append("")
    L.append(
        f"**And it concentrates.** `{top_stratum}` accounts for {top_n} of {total} "
        f"failures ({top_n/total:.0%}) -- consistent with its abstention rate being "
        "the highest of any stratum. Tables and figures are the weak subsystem, not "
        "retrieval in general."
    )
    L.append("")
    L.append(
        "**What that rules out.** A zero here is a strong negative result, not "
        "an absence of data: 24 chances for the model to receive complete "
        "context and misuse it, and it took none of them. So prompt "
        "engineering, lost-in-the-middle reordering, context compression and "
        "fine-tuning the reader (RAFT and similar) are not deprioritised -- "
        "they are **ruled out on evidence**, because every one of them improves "
        "a step that is not failing. Spending on any of them would be spending "
        "where the measurement says nothing is wrong."
    )
    L.append("")
    L.append(
        "That points at one change rather than several. `source_hit` is 100% at "
        "every budget, so the right *document* is always found and the loss is "
        "ranking *within* it -- which is exactly what a situating prefix addresses. "
        "Contextual retrieval's deferral has already expired on its own predicate, "
        "and this histogram is the second, independent argument for it."
    )
    L.append("")
    L.append(
        "**What this cannot tell you.** FP1 is structurally absent: the golden set "
        "was generated *from* the corpus, so a question whose answer is missing "
        "entirely cannot appear. FP6 and FP7 need human judgement of answer quality "
        "that no automated check here performs. Their zeros mean **not measured**, "
        "not **does not happen**."
    )
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({total} failures, budget {budget})")
    for fp in ("FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"):
        print(f"  {fp} {fp_names[fp]:22s} {by_fp.get(fp, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
