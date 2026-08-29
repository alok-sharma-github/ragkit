"""Did the LLM-written prefix earn its cost? A before/after over one golden set.

WHY A SCRIPT AND NOT A NUMBER IN A COMMIT MESSAGE. The two indexes have
different pipeline fingerprints, so nothing else in this repo will compare them
-- the eval refuses to, on purpose, and the CI gate refuses to, on purpose.
Comparing them is a deliberate act with a stated reason, which is what this file
is.

THE ONE THING THIS MEASUREMENT HAS TO SEPARATE.

A contextual prefix does two opposite things at a fixed token budget:

  it improves RANKING           -- the chunk now names what it is about, so a
                                   query that never appears in the body can
                                   still reach it
  it worsens PACKING            -- `search_budget` charges a child its full
                                   embed_text, and the prefix is ~70 tokens on
                                   a ~300-token body. Roughly a fifth fewer
                                   children fit in the same budget.

A single headline mixes them and can move either way for either reason. So the
comparison runs the whole budget sweep, and reads it like this:

  at a LARGE budget   packing is not binding, so a change is RANKING
  at a SMALL budget   both are live, and the net is what a user would feel

Reporting only the headline would let a real ranking gain read as a loss, or a
packing loss read as "contextual retrieval does not work here". Both are wrong
conclusions from correct arithmetic, which is this project's recurring failure
and the reason the budget sweep exists at all.

    uv run python scripts/contextual_ab.py --a numpy_index --b numpy_index_ctx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.eval import run as R  # noqa: E402
from ragkit.index.numpy_index import NumpyIndex  # noqa: E402

BUDGETS = (250, 500, 1000, 1500, 3000, 6000, 12000)


def _describe(index_name: str) -> dict[str, object]:
    idx = NumpyIndex.load(index_name)
    return {
        "index": index_name,
        "n_children": idx.meta.get("n_children_indexed"),
        "contextualized": idx.meta.get("n_contextualized"),
        "uniform_contextualization": idx.meta.get("uniform_contextualization"),
        "fingerprint": idx.meta.get("pipeline_fingerprint"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="numpy_index", help="the baseline index")
    ap.add_argument("--b", default="numpy_index_ctx", help="the contextualised index")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    meta = {"a": _describe(args.a), "b": _describe(args.b)}
    if meta["a"]["fingerprint"] == meta["b"]["fingerprint"]:
        # Not a warning. If the fingerprints match, one of these indexes is not
        # what its name says, and every number below would be noise reported to
        # three decimal places.
        print("REFUSING: both indexes carry the same pipeline fingerprint, so "
              "they are the same system and there is nothing to compare.")
        return 2

    out: dict[str, object] = {"meta": meta, "by_budget": {}}
    for name, index_name in (("a", args.a), ("b", args.b)):
        for budget in BUDGETS:
            res = R.run(index_name=index_name, token_budget=budget,
                        sweep=False, verbose=False)
            head = res["metrics"]["headline"]
            row = out["by_budget"].setdefault(str(budget), {})
            row[name] = {
                "child_strict": head["child_strict"]["label"],
                "child_strict_rate": head["child_strict"]["rate"],
                "child_strict_hits": head["child_strict"]["hits"],
                "parent_strict": head["parent_strict"]["label"],
                "source_hit": head["source_hit"]["label"],
                "child_no_delivery": head.get("child_no_delivery"),
                # THE PACKING NUMBER. Mean tokens per delivered child, which is
                # where a ~70-token prefix on a ~300-token body shows up -- and
                # the reason a fixed budget holds fewer of them.
                "mean_child_tokens": head.get("mean_child_tokens"),
            }

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    a_ctx, b_ctx = meta["a"]["contextualized"], meta["b"]["contextualized"]
    print(f"A  {args.a:20s} {meta['a']['fingerprint']}  "
          f"{a_ctx} of {meta['a']['n_children']} children contextualised")
    print(f"B  {args.b:20s} {meta['b']['fingerprint']}  "
          f"{b_ctx} of {meta['b']['n_children']} children contextualised")
    print()
    print(f"{'budget':>7}  {'A child_strict':>16}  {'B child_strict':>16}  {'delta':>7}")
    for budget in BUDGETS:
        row = out["by_budget"][str(budget)]
        d = row["b"]["child_strict_hits"] - row["a"]["child_strict_hits"]
        print(f"{budget:>7}  {row['a']['child_strict']:>16}  "
              f"{row['b']['child_strict']:>16}  {d:>+7}")
    print()
    print("read the large-budget rows as RANKING and the small-budget rows as the "
          "net of ranking and packing -- see this file's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
