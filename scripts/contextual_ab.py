"""Did the LLM-written prefix earn its cost? Three measurements over one golden set.

WHY A SCRIPT AND NOT A NUMBER IN A COMMIT MESSAGE. The two indexes have
different pipeline fingerprints, so nothing else in this repo will compare them
-- the eval refuses to and the CI gate refuses to, both on purpose. Comparing
them is a deliberate act with a stated reason, which is what this file is.

---------------------------------------------------------------------------
THE THING THIS COMPARISON HAS TO SEPARATE, AND WHY IT MATTERS

A contextual prefix does two opposite things at a fixed token budget:

  RANKING improves  -- the chunk now names what it is about, so a query whose
                       words never appear in the body can still reach it
  PACKING worsens   -- `search_budget` charges a child its whole `embed_text`,
                       and the prefix measures ~66 tokens against a ~300-token
                       body. Roughly a fifth fewer children fit in the same
                       budget.

A single headline mixes them and moves either way for either reason. So this
runs three measurements, each isolating something the others cannot:

  1. FIXED TOKEN BUDGET, cost = embed_text    the shipped metric; both live
  2. FIXED k                                  no budget at all, so PURE RANKING
  3. FIXED TOKEN BUDGET, cost = display_text  charges a child what it DELIVERS

Measurement 3 exists because measurement 1 has an inconsistency older than
contextual retrieval, which only became visible when the prefixes got long
enough to matter: a PARENT is charged `display_text` (what it delivers) and a
CHILD is charged `embed_text` (what it indexes). The child unit pays for its own
breadcrumb and prefix; the parent unit pays for nothing equivalent -- in the one
comparison budget normalisation exists to make fair.

The consequence is not subtle. Same index, same golden set, same technique, and
the conclusion INVERTS on the choice of cost function: under (1) contextual
retrieval loses, under (3) it wins clearly at tight budgets. Both are correct
arithmetic. Only one is over the right denominator, and this file exists so that
neither can be quoted without the other.

    uv run python scripts/contextual_ab.py
    uv run python scripts/contextual_ab.py --json > data/eval/contextual_ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit import gemini  # noqa: E402
from ragkit.eval import goldenset as G  # noqa: E402
from ragkit.eval import metrics as M  # noqa: E402
from ragkit.gemini import count_tokens  # noqa: E402
from ragkit.index.numpy_index import NumpyIndex  # noqa: E402

BUDGETS = (250, 500, 1000, 1500, 3000, 6000)
KS = (3, 5, 10, 20, 40)

CHARGE = {
    "embed_text": lambda c: c.embed_text or c.display_text,
    "display_text": lambda c: c.display_text,
}


def _describe(index_name: str) -> dict[str, object]:
    idx = NumpyIndex.load(index_name)
    kids = idx.children
    # THE FINGERPRINT IS READ OFF THE CHUNKS, not off a sidecar report.
    # index_report.json describes the LAST ingest, so with two indexes on disk
    # one of them is always described by the other one's report. The chunk
    # carries the fingerprint it was built under; that is the primary record and
    # the only one that cannot be about a different index.
    fps = {c.pipeline_fingerprint for c in kids}
    return {
        "index": index_name,
        "n_children": len(kids),
        "contextualized": sum(1 for c in kids if c.has_contextual_prefix),
        "fingerprint": sorted(fps)[0] if len(fps) == 1 else f"MIXED {sorted(fps)}",
    }


def _fill(idx: NumpyIndex, qv: np.ndarray, budget: int, charge) -> tuple[list, int]:
    """Strict budget fill, with the cost function as a parameter."""
    s = idx._scores(qv, None)
    order = [i for i in np.argsort(-s)[:400] if np.isfinite(s[i])]
    out, spent = [], 0
    for i in order:
        c = idx.children[i]
        cost = count_tokens(charge(c))
        if spent + cost > budget:
            break                      # STRICT: never exceed, even if that is zero
        out.append(c)
        spent += cost
    return out, spent


def _strict(items, texts_for) -> tuple[int, dict[str, list[int]]]:
    hits, per = 0, {}
    for it, texts in zip(items, texts_for):
        ok = all(M.cover_needle(n, texts)[0] == "contained" for n in it.needles)
        hits += ok
        s = per.setdefault(it.stratum, [0, 0])
        s[0] += ok
        s[1] += 1
    return hits, per


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
        # three significant figures.
        print("REFUSING: both indexes carry the same pipeline fingerprint, so they "
              "are the same system and there is nothing to compare.")
        return 2

    items = [it for it in G.load() if it.needles]
    vecs, _st = gemini.embed_texts([it.question for it in items], kind="query")
    n = len(items)
    idxs = {"a": NumpyIndex.load(args.a), "b": NumpyIndex.load(args.b)}

    out: dict[str, object] = {
        "meta": meta,
        "n_items": n,
        # STATED, because it is not the eval's 92. This script scores every
        # golden item carrying a needle; the eval additionally quarantines its
        # own fixture and drops out-of-scope items. Two defensible populations,
        # and quoting a number from one against a number from the other is the
        # exact mistake this file is about.
        "population_note": "every golden item with a needle; the eval's headline "
                           "population is smaller (quarantined fixture and "
                           "out-of-scope items excluded)",
        "fixed_k": {},
        "by_cost": {},
        "by_stratum_at_k10": {},
    }

    # -- 2. FIXED k: no budget, so no packing. What is left is ranking. --------
    for k in KS:
        row: dict[str, int] = {}
        for side, idx in idxs.items():
            texts = [[h.chunk.display_text for h in idx.search_k(qv, k=k)] for qv in vecs]
            hits, per = _strict(items, texts)
            row[side] = hits
            if k == 10:
                out["by_stratum_at_k10"][side] = dict(per)
        row["delta"] = row["b"] - row["a"]
        out["fixed_k"][str(k)] = row

    # -- 1 and 3. FIXED BUDGET, under each cost function -----------------------
    for cost_name, charge in CHARGE.items():
        out["by_cost"][cost_name] = {}
        for budget in BUDGETS:
            row = {}
            for side, idx in idxs.items():
                filled = [_fill(idx, qv, budget, charge) for qv in vecs]
                texts = [[c.display_text for c in got] for got, _sp in filled]
                hits, _per = _strict(items, texts)
                row[side] = hits
                row[f"{side}_mean_tokens"] = int(np.mean([sp for _g, sp in filled]))
                row[f"{side}_no_delivery"] = sum(1 for g, _sp in filled if not g)
            row["delta"] = row["b"] - row["a"]
            out["by_cost"][cost_name][str(budget)] = row

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"A  {args.a:16s} {meta['a']['fingerprint']}  "
          f"{meta['a']['contextualized']}/{meta['a']['n_children']} contextualised")
    print(f"B  {args.b:16s} {meta['b']['fingerprint']}  "
          f"{meta['b']['contextualized']}/{meta['b']['n_children']} contextualised")
    print(f"\nscored over {n} golden items with needles\n")

    print("PURE RANKING -- fixed k, no budget, so packing cannot confound it")
    print(f"  {'k':>4}  {'A':>9}  {'B':>9}  {'delta':>6}")
    for k in KS:
        r = out["fixed_k"][str(k)]
        print(f"  {k:>4}  {r['a']:>4}/{n:<4}  {r['b']:>4}/{n:<4}  {r['delta']:>+6}")

    for cost_name in CHARGE:
        tag = ("THE SHIPPED METRIC -- a child is charged what it INDEXES"
               if cost_name == "embed_text"
               else "ALTERNATIVE -- a child is charged what it DELIVERS")
        print(f"\n{tag}   (cost = {cost_name})")
        print(f"  {'budget':>7}  {'A':>9}  {'B':>9}  {'delta':>6}  {'A tok':>6} {'B tok':>6}")
        for budget in BUDGETS:
            r = out["by_cost"][cost_name][str(budget)]
            print(f"  {budget:>7}  {r['a']:>4}/{n:<4}  {r['b']:>4}/{n:<4}  {r['delta']:>+6}  "
                  f"{r['a_mean_tokens']:>6} {r['b_mean_tokens']:>6}")

    print("\nThe conclusion inverts between the two cost functions. Both are correct "
          "arithmetic; the docstring says which denominator is the right one, and "
          "why the asymmetry predates contextual retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
