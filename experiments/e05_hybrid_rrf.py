"""
Guide M2 Build task: dense vs BM25 vs RRF on the same golden set, same budget.

The experiment the eval pointed at. `source_hit = 100%` with `child_strict = 85%`
means every failure is ranking-within-document, which is what a lexical leg
reorders -- so this is the fix error analysis indicated, not the fix that sounded
good.

The `exact_identifier` stratum is the row to watch: those are the queries whose
answer hinges on a token embeddings compress away.
"""

from __future__ import annotations

import json

from ragkit import config, gemini
from ragkit.eval import goldenset as G
from ragkit.eval import metrics as M
from ragkit.index.hybrid import HybridIndex

BUDGETS = (500, 1500, 3000)
MODES = ("dense", "sparse", "rrf")


def main() -> None:
    items = [i for i in G.load() if i.stratum != "out_of_scope" and i.anchor != "none"]
    scoreable = [i for i in items if not i.quarantined]
    hyb = HybridIndex.load()
    print(f"index: {len(hyb.dense.children)} children | bm25: {json.dumps(hyb.sparse.stats())}")
    print(f"scoring {len(scoreable)} items ({len(items) - len(scoreable)} quarantined)\n")

    vecs, st = gemini.embed_texts([i.question for i in items], kind="query")
    print(" ", st.render(), "\n")
    qv = {id(i): v for i, v in zip(items, vecs)}

    rows: dict[tuple[str, int], dict] = {}
    leg_share: dict[int, dict] = {}
    for budget in BUDGETS:
        for mode in MODES:
            results = []
            uniq_sparse = 0
            for it in scoreable:
                r = hyb.retrieve(it.question, qv[id(it)], mode=mode,
                                 token_budget=budget, unit="child")
                res = M.ItemResult(question=it.question, stratum=it.stratum,
                                   source_id=it.source_id, anchor=it.anchor)
                texts = [c.display_text for c in r.children]
                res.n_children = len(r.children)
                res.child_tokens = r.child_tokens
                res.child_no_delivery = not r.children
                accept = set(it.accept_sources or [it.source_id])
                res.source_hit = any(c.source_id in accept for c in r.children)
                if it.anchor == "asset":
                    res.child_strict = res.child_partial = res.source_hit
                else:
                    cov = [M.cover_needle(n, texts)[0] for n in it.needles]
                    res.child_strict = bool(cov) and all(c == "contained" for c in cov)
                    res.child_partial = any(c in ("contained", "partial") for c in cov)
                res.parent_strict, res.parent_partial = res.child_strict, res.child_partial
                results.append(res)
                if mode == "rrf" and r.leg_stats:
                    uniq_sparse += r.leg_stats["sparse"]["only_this_leg"]
            agg = M.aggregate(results)
            rows[(mode, budget)] = agg
            if mode == "rrf":
                leg_share[budget] = {"sparse_unique_in_top50": uniq_sparse}

    print(f"{'budget':>7s} {'mode':>7s} {'strict':>14s} {'partial':>14s} "
          f"{'source_hit':>14s} {'tok':>6s} {'none':>5s}")
    for budget in BUDGETS:
        for mode in MODES:
            h = rows[(mode, budget)]["headline"]
            print(f"{budget:>7d} {mode:>7s} {h['child_strict']['label']:>14s} "
                  f"{h['child_partial']['label']:>14s} {h['source_hit']['label']:>14s} "
                  f"{h['mean_child_tokens']:>6d} {h['child_no_delivery']:>5d}")
        print()

    print("exact_identifier stratum (the queries dense retrieval loses):")
    print(f"{'budget':>7s} {'mode':>7s} {'strict':>14s}")
    for budget in BUDGETS:
        for mode in MODES:
            st_block = rows[(mode, budget)]["by_stratum"].get("exact_identifier")
            if st_block:
                print(f"{budget:>7d} {mode:>7s} {st_block['child_strict']['label']:>14s}")
        print()

    print("table_or_image stratum:")
    for budget in BUDGETS:
        for mode in MODES:
            b = rows[(mode, budget)]["by_stratum"].get("table_or_image")
            if b:
                print(f"{budget:>7d} {mode:>7s} {b['child_strict']['label']:>14s}")
        print()

    out = {
        "budgets": list(BUDGETS),
        "modes": list(MODES),
        "bm25": hyb.sparse.stats(),
        "rrf_k": config.RRF_K,
        "results": {f"{m}@{b}": rows[(m, b)] for (m, b) in rows},
        "sparse_unique": leg_share,
    }
    p = config.EXPERIMENTS_OUT / "e05_hybrid_rrf.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
