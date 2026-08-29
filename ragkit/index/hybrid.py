"""
Hybrid retrieval: three interchangeable retrievers over one corpus. Guide M2.

    dense   -- cosine over gemini-embedding-001 vectors
    sparse  -- hand-written BM25
    rrf     -- reciprocal rank fusion of both

THE POINT OF THE SHARED INTERFACE is that the eval can score all three on the
same golden set at the same token budget without any of them getting a different
deal. Every previous attempt to compare two things in this project went wrong
because the two things were not given identical treatment:

  - a survey script and the pipeline imported different parsers (5x difference)
  - the parent unit was allowed to exceed its budget while the child unit was not
    (a fake crossover in the recall sweep)

So the budget fill rule lives HERE, once, and every mode inherits it. A retriever
cannot be given more text than its competitor because there is only one loop.

---------------------------------------------------------------------------
WHAT FUSION HAPPENS OVER

Each leg produces a ranked list of CHILD ids. Fusion happens on children, then
parents are resolved from the fused order. Fusing parents instead would throw
away the ranking signal: two children of one parent are two independent pieces of
evidence for that parent, and collapsing them before fusion discards the second.

The dense leg is asked for a generous candidate pool rather than the final k,
because RRF can only reorder what it is given -- the ceiling law applies to each
leg separately, and a leg truncated to 10 cannot contribute an item it never
returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .. import config
from ..gemini import count_tokens
from ..ingest.document import Chunk
from .bm25 import BM25Index
from .fusion import FusedHit, leg_contribution, rrf
from .numpy_index import NumpyIndex

Mode = Literal["dense", "sparse", "rrf"]
Unit = Literal["child", "parent"]

# Candidate pool per leg before fusion. Stage 1 optimises RECALL (guide M3), and
# RRF cannot promote a document a leg never returned.
LEG_POOL = 200


@dataclass
class Retrieved:
    children: list[Chunk]
    parents: list[Chunk]
    child_tokens: int
    parent_tokens: int
    fused: list[FusedHit] | None = None
    leg_stats: dict[str, Any] | None = None
    # WHY AN EMPTY RESULT IS EMPTY.
    #
    # The strict fill returns nothing when the top-ranked unit alone exceeds the
    # budget, and that is deliberate -- see retrieve(). But "nothing was ranked"
    # and "everything ranked was too big" have OPPOSITE remedies:
    #
    #   nothing ranked   the query found no candidates. Look at the embedding,
    #                    the index, the corpus. Raising the budget does nothing.
    #   nothing fit      ranking worked; the budget is too small for this unit.
    #                    Raise the budget (or use the child unit) and it works.
    #                    Nothing about retrieval is broken.
    #
    # Downstream both arrived as an empty list, so the generator abstained with
    # the reason "retrieval returned nothing" -- pointing a reader at the one
    # subsystem that was working. The comment on the fill loop claimed "the
    # caller distinguishes nothing fit from ranking failed"; no caller could,
    # because the distinction was never returned. A documented contract with no
    # means of honouring it.
    n_ranked: int = 0
    min_unit_tokens: int | None = None   # set only when starved: the cheapest ranked unit

    @property
    def starved_by_budget(self) -> bool:
        """Candidates were ranked, and the budget admitted none of them."""
        return self.n_ranked > 0 and not self.children and not self.parents


class HybridIndex:
    def __init__(self, dense: NumpyIndex, sparse: BM25Index | None = None) -> None:
        self.dense = dense
        self.sparse = sparse or BM25Index(dense.children)
        self._by_id = {c.chunk_id: c for c in dense.children}

    @classmethod
    def load(cls, name: str = "numpy_index", **bm25_kw: Any) -> "HybridIndex":
        d = NumpyIndex.load(name)
        return cls(d, BM25Index(d.children, **bm25_kw))

    # -- ranked id lists per leg --------------------------------------------

    def _dense_ids(self, query_vec: np.ndarray, pool: int,
                   owner: str | None = None) -> list[str]:
        return [h.chunk.chunk_id for h in
                self.dense.search_k(query_vec, pool, owner=owner)]

    def _sparse_ids(self, query: str, pool: int,
                    owner: str | None = None) -> list[str]:
        return [h.chunk.chunk_id for h in
                self.sparse.search_k(query, pool, owner=owner)]

    def ranked_ids(
        self, query: str, query_vec: np.ndarray, *, mode: Mode,
        pool: int = LEG_POOL, owner: str | None = None,
    ) -> tuple[list[str], list[FusedHit] | None, dict[str, Any] | None]:
        # `owner` reaches BOTH legs. Filtering only the dense side would leave
        # sparse-only and RRF leaking, and the fused result would carry another
        # session's chunk while the dense half was provably clean.
        if mode == "dense":
            return self._dense_ids(query_vec, pool, owner), None, None
        if mode == "sparse":
            return self._sparse_ids(query, pool, owner), None, None
        legs = {
            "dense": self._dense_ids(query_vec, pool, owner),
            "sparse": self._sparse_ids(query, pool, owner),
        }
        fused = rrf(
            legs,
            k=config.RRF_K,
            weights={"dense": config.RRF_WEIGHT_DENSE, "sparse": config.RRF_WEIGHT_SPARSE},
        )
        return [h.key for h in fused], fused, leg_contribution(fused[:50], legs)

    # -- the single budget fill, shared by every mode -----------------------

    def retrieve(
        self,
        query: str,
        query_vec: np.ndarray,
        *,
        mode: Mode = "rrf",
        token_budget: int | None = None,
        unit: Unit = "child",
        max_items: int = 200,
        # WHO IS ASKING. None = the public corpus only. This is the
        # serving path -- the filter was on search_budget, which this
        # does not call, so retrieve() leaked while the isolation test
        # passed against a path the product never takes.
        owner: str | None = None,
    ) -> Retrieved:
        """Fill to the budget, strictly, in fused rank order.

        STRICT: never exceed the budget, even if that returns nothing. Returning
        an empty result at a tight budget is a real finding about the unit's
        granularity, not something to paper over by admitting one oversized item.
        """
        budget = token_budget or config.TOKENS_CONTEXT_BUDGET
        ids, fused, legs = self.ranked_ids(query, query_vec, mode=mode, owner=owner)

        children: list[Chunk] = []
        parents: list[Chunk] = []
        seen_parent: set[str] = set()
        spent = 0

        for cid in ids[:max_items]:
            child = self._by_id.get(cid)
            if child is None:
                continue
            if unit == "child":
                cost = count_tokens(child.embed_text or child.display_text)
                if spent + cost > budget:
                    break
                spent += cost
                children.append(child)
            else:
                target = self.dense.parents.get(child.parent_id or "") or child
                if target.chunk_id in seen_parent:
                    children.append(child)   # already paid for
                    continue
                cost = count_tokens(target.display_text)
                if spent + cost > budget:
                    break
                spent += cost
                seen_parent.add(target.chunk_id)
                parents.append(target)
                children.append(child)

        if unit == "child":
            # Parents are resolved for reporting even in child mode, so the two
            # units are described by the same object shape.
            for c in children:
                t = self.dense.parents.get(c.parent_id or "") or c
                if t.chunk_id not in seen_parent:
                    seen_parent.add(t.chunk_id)
                    parents.append(t)

        # Computed only on the starved path, so the normal path pays nothing for
        # a diagnostic it does not need. The number the caller actually wants is
        # not "how big was the thing that failed" but "what budget would work",
        # which is the cheapest ranked unit -- an actionable remedy rather than a
        # restatement of the failure.
        min_unit: int | None = None
        if ids and not children and not parents:
            costs: list[int] = []
            for cid in ids[:max_items]:
                c = self._by_id.get(cid)
                if c is None:
                    continue
                if unit == "child":
                    costs.append(count_tokens(c.embed_text or c.display_text))
                else:
                    t = self.dense.parents.get(c.parent_id or "") or c
                    costs.append(count_tokens(t.display_text))
            min_unit = min(costs) if costs else None

        return Retrieved(
            children=children,
            parents=parents,
            child_tokens=sum(count_tokens(c.embed_text or c.display_text) for c in children),
            parent_tokens=sum(count_tokens(p.display_text) for p in parents),
            fused=fused,
            leg_stats=legs,
            n_ranked=len(ids),
            min_unit_tokens=min_unit,
        )
