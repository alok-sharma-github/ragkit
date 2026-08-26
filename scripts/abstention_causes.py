"""Why did each abstention abstain? Localise it to a tier, with a number.

An abstention has two causes that look identical in the output and have opposite
owners:

  retrieval miss   the evidence was not in the delivered context. Abstaining is
                   then CORRECT -- the honest response to absent evidence -- and
                   the defect belongs to retrieval.
  reading failure  the evidence WAS present and the generator declined anyway.
                   A generation defect, and invisible in both metric tiers:
                   retrieval scores it a hit, and faithfulness scores the
                   abstention "supported" because an answer asserting nothing
                   asserts nothing false.

Nothing else in the harness separates them, which is why this exists.

Written as a script rather than left in a scratch file because its output feeds
the failure histogram, and an input to a document shown to other people has to be
reproducible and STAMPED. The first version of this analysis carried neither a
pipeline fingerprint nor a token budget, so its 18 rows could not be pooled with
anything -- including the retrieval misses measured at a specific budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit import config, gemini as GM  # noqa: E402
from ragkit.eval import goldenset as G, judge as J, metrics as M  # noqa: E402
from ragkit.index.hybrid import HybridIndex  # noqa: E402

OUT = config.DATA_EVAL / "abstention_causes.json"


def main() -> int:
    judged_path = config.DATA_EVAL / "judged_results.json"
    if not judged_path.exists():
        print("no judged_results.json -- run `ragkit judge score` first")
        return 1
    judged = json.loads(judged_path.read_text("utf-8"))

    # REFUSE ON AN UNSTAMPED INPUT. Deriving a stamped artifact from an unstamped
    # one would manufacture provenance -- asserting a fingerprint this analysis
    # never verified. That is a worse failure than having no stamp, because the
    # label makes it look checked.
    stamp = judged.get("stamp")
    if not stamp or not stamp.get("pipeline_fingerprint"):
        print("judged_results.json carries no stamp. Re-run `ragkit judge score`")
        print("rather than inheriting a fingerprint that was never recorded.")
        return 1

    target = {r["question"] for r in judged["rows"] if r["abstained"] or r["failed"]}
    items = [i for i in G.load() if i.question in target]
    hyb = HybridIndex.load()
    budget = stamp["token_budget"]

    out = []
    for n, it in enumerate(items, 1):
        qv = GM.embed_query(it.question)
        r = hyb.retrieve(it.question, qv, mode="dense", token_budget=budget, unit="parent")
        parents = r.parents[:5]
        texts = [p.display_text for p in parents]
        covs = [M.cover_needle(nd, texts) for nd in it.needles]
        got = sum(1 for c, _ in covs if c == "contained")
        kind = (
            "starved_by_budget" if not parents and r.starved_by_budget
            else "no_candidates" if not parents
            else "evidence_present" if it.needles and got == len(it.needles)
            else "evidence_partial" if got
            else "evidence_absent"
        )
        out.append({
            "question": it.question,
            "stratum": it.stratum,
            "cause": kind,
            "n_parents": len(parents),
            "needles_found": got,
            "needles_total": len(it.needles),
            "sources": sorted({p.source_id for p in parents}),
        })
        print(f"  [{n}/{len(items)}] {kind:18s} {it.stratum:16s} {it.question[:44]}",
              flush=True)

    # The stamp is INHERITED from the judged run, not recomputed, because these
    # rows describe that run. Recomputing would silently re-label them if the
    # index changed in between -- which is exactly the drift the stamp exists to
    # catch.
    OUT.write_text(
        json.dumps({"stamp": stamp, "n": len(out), "rows": out}, indent=1),
        encoding="utf-8",
    )

    import collections

    print(f"\nwritten {OUT.name}  (budget {budget}, fingerprint "
          f"{stamp['pipeline_fingerprint']})")
    for k, v in collections.Counter(o["cause"] for o in out).most_common():
        print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
