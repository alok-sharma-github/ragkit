"""
Exact cosine search over children, with parent resolution. Guide module M6.

WHY EXACT SEARCH IS BUILT FIRST, BEFORE ANY ANN INDEX

It is simultaneously the simplest thing that works at this corpus size AND the
only way to ever answer "what is your recall?". ANN recall is measured against
exact search; there is no other oracle. A project that starts with HNSW can
report latency and QPS forever and never once report recall, because it has
nothing to compare to. That is the guide's sharpest question (M6) and most
candidates cannot answer it -- not because it is hard, but because they never
built the baseline.

Same structure as everything else this session: you cannot validate a
measurement from inside it. ANN needs exact search. A detector needs hand
labels. A corrupt extractor needs a disagreeing extractor.

---------------------------------------------------------------------------
BUDGET-NORMALISED RETRIEVAL, which is here rather than in the eval

From your k-comparability argument, and it changes the API rather than a metric:

    fixed         27 chunks x 290 tokens -> top-10 = 2,900 tokens
    header_aware  42 chunks x 183 tokens -> top-10 = 1,830 tokens

Recall@10 hands `fixed` 60% more text to hide the answer in. It wins, or loses
by less than it should, for a reason unrelated to chunk quality. So the primary
retrieval call takes a TOKEN BUDGET, not a k -- which is also the honest
question, because the model downstream has a token budget
(config.TOKENS_CONTEXT_BUDGET) and has never once had a chunk budget.

`search_k()` still exists, because recall@k is worth reporting alongside. It is
just not what a decision should hang on.

And the budget applies to whichever unit is being measured:
  - child budget  -> measures RANKING (children differ in size too: 183 vs 290)
  - parent budget -> measures WHAT THE MODEL SAW
Two questions, two normalisations. The gap between them is what the
parent-document trick is buying, and it only exists as a number if both are
computed separately.

---------------------------------------------------------------------------
ZERO VECTORS ARE EXCLUDED, LOUDLY

A child whose embedding failed on free-tier quota is a zero row. Cosine against
zero is 0.0 for every query, so the chunk sits in the index permanently
unreachable -- present, counted, invisible. It is dropped at build time and
reported, because an index that silently contains unreachable rows makes recall
uninterpretable: you cannot tell a ranking failure from a chunk that was never
really there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from .. import config
from ..gemini import count_tokens
from ..ingest.document import Chunk, ChunkRole, TextProvenance

Unit = Literal["child", "parent"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class NumpyIndex:
    """Exact cosine over unit-normalised child vectors.

    Vectors arrive already L2-normalised (gemini.truncate does it), so cosine is
    a dot product and the whole search is one matmul. Kept deliberately dumb: its
    job is to be obviously correct so that something else can be measured
    against it.
    """

    def __init__(
        self,
        children: Sequence[Chunk],
        vectors: np.ndarray,
        parents: Sequence[Chunk] = (),
        *,
        meta: dict | None = None,
    ) -> None:
        if len(children) != vectors.shape[0]:
            raise ValueError(
                f"{len(children)} children but {vectors.shape[0]} vectors. Refusing to "
                "align mismatched lists -- that is the zip() truncation bug with a "
                "different shape."
            )
        self.children = list(children)
        self.vectors = vectors.astype(np.float32, copy=False)
        self.parents = {p.chunk_id: p for p in parents}
        self.meta = meta or {}

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        *,
        strategy: str = "unknown",
    ) -> tuple["NumpyIndex", dict]:
        """Children + their vectors, parents kept aside. Zero rows dropped."""
        kids = [c for c in chunks if c.role is ChunkRole.CHILD]
        parents = [c for c in chunks if c.role is ChunkRole.PARENT]
        if len(kids) != vectors.shape[0]:
            raise ValueError(
                f"build: {len(kids)} children vs {vectors.shape[0]} vectors -- the "
                "caller embedded a different set than it passed in."
            )

        norms = np.linalg.norm(vectors, axis=1)
        live = norms > 1e-6
        n_dropped = int((~live).sum())

        # An orphan child is a retrieval hit that cannot produce a parent to send
        # to the model. Counted, not tolerated silently.
        pids = {p.chunk_id for p in parents}
        orphans = [c for c, ok in zip(kids, live) if ok and c.parent_id and c.parent_id not in pids]

        report = {
            "strategy": strategy,
            "n_children_in": len(kids),
            "n_children_indexed": int(live.sum()),
            "n_children_dropped_zero_vector": n_dropped,
            "n_parents": len(parents),
            "n_orphan_children": len(orphans),
            "dim": int(vectors.shape[1]) if vectors.size else 0,
            # Provenance, so an index can refuse to be a CI baseline later.
            "n_prefixed": sum(1 for c in kids if c.text_provenance is TextProvenance.PREFIXED),
            "n_model_generated": sum(
                1 for c in kids if c.text_provenance is TextProvenance.MODEL_GENERATED
            ),
            "n_header_missing": sum(1 for c in kids if c.table_header_missing),
            "n_continuation_suspect": sum(1 for c in kids if c.table_continuation_suspect),
            # WHAT "UNIFORM" MEANS, corrected after this predicate hard-blocked
            # the first eval run as a false positive.
            #
            # The defect this flag exists to catch is a PARTIAL INGEST: quota died
            # midway, so some chunks carry contextual prefixes and some do not,
            # and the un-prefixed ones are systematically harder to retrieve for
            # reasons unrelated to relevance.
            #
            # My first version tested "do all children share one provenance
            # value". Once images were indexed that became permanently False,
            # because captions are MODEL_GENERATED and text is PREFIXED. But
            # captions are not damage -- they are a different KIND of content, and
            # they are supposed to be there.
            #
            # So uniformity is tested only over the population that could
            # legitimately be uniform: chunks derived from document text. The
            # model-generated population is reported alongside, separately,
            # rather than folded in. Same principle as reporting counts instead
            # of rates on a small slice: do not average two things that are not
            # the same thing.
            "uniform_provenance": len(
                {
                    c.text_provenance
                    for c in kids
                    if c.text_provenance is not TextProvenance.MODEL_GENERATED
                }
            ) <= 1,
            "provenance_populations": {
                p.value: sum(1 for c in kids if c.text_provenance is p)
                for p in TextProvenance
                if any(c.text_provenance is p for c in kids)
            },
        }
        if n_dropped:
            report["warning"] = (
                f"{n_dropped} children had zero vectors (embedding failed, likely "
                "free-tier quota) and were EXCLUDED. Recall over this index is "
                "bounded below by that exclusion, and the gap is a contiguous tail "
                "of ingest order rather than a random sample."
            )

        idx = cls(
            [c for c, ok in zip(kids, live) if ok],
            vectors[live],
            parents,
            meta=report,
        )
        return idx, report

    # -- search --------------------------------------------------------------

    def _scores(self, query_vec: np.ndarray) -> np.ndarray:
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        if q.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self.vectors.shape[1]}. "
                "A dimension mismatch here means the query and the corpus were "
                "embedded under different settings, and any score would be noise."
            )
        n = float(np.linalg.norm(q))
        if n < 1e-6:
            raise ValueError("query vector is zero -- embedding failed, likely quota")
        return self.vectors @ (q / n)

    def search_k(self, query_vec: np.ndarray, k: int = 10) -> list[Hit]:
        """Exact top-k. Reported alongside the budget number, never instead of it."""
        s = self._scores(query_vec)
        k = min(k, len(self.children))
        if k <= 0:
            return []
        # argpartition then sort the slice: O(n) instead of sorting everything.
        part = np.argpartition(-s, k - 1)[:k]
        order = part[np.argsort(-s[part])]
        return [Hit(self.children[i], float(s[i]), r) for r, i in enumerate(order)]

    def search_budget(
        self,
        query_vec: np.ndarray,
        *,
        token_budget: int | None = None,
        unit: Unit = "child",
        max_items: int = 200,
        # Optional out-dict for the two facts a caller needs when the result
        # is empty: whether anything was RANKED, and the cheapest unit that
        # would have fit. Returned this way rather than by changing the return
        # type, so every existing call site keeps working unchanged.
        stats: dict | None = None,
    ) -> list[Hit]:
        """Retrieve by descending score until the TOKEN BUDGET fills.

        unit="child"  -> budget counts child tokens; measures ranking.
        unit="parent" -> budget counts the tokens actually delivered to the model,
                         deduplicating parents (two children of one parent cost
                         that parent once, which is a real advantage of the
                         parent-document design and should be visible as one).
        """
        budget = token_budget or config.TOKENS_CONTEXT_BUDGET
        s = self._scores(query_vec)
        order = np.argsort(-s)[:max_items]

        hits: list[Hit] = []
        spent = 0
        seen_parents: set[str] = set()
        for rank, i in enumerate(order):
            child = self.children[i]
            if unit == "child":
                cost = count_tokens(child.embed_text or child.display_text)
            else:
                parent = self.parents.get(child.parent_id or "")
                target = parent or child
                if target.chunk_id in seen_parents:
                    cost = 0  # already paid for
                else:
                    cost = count_tokens(target.display_text)
            # STRICT: never exceed the budget, even if that returns nothing.
            #
            # This clause used to read `if spent + cost > budget and hits`. The
            # `and hits` was an "always return at least one result" guarantee, and
            # it silently destroyed the comparison this function exists to make:
            # at budget 250 the parent unit was always admitted one full ~1200
            # token parent (measured mean delivered: 378 tokens against a 250
            # budget, 51% over) while the child unit stayed under the line. Large
            # units systematically overshot, worst where the budget was smallest
            # -- which is exactly where parent recall appeared to "win".
            #
            # More rain in the bigger bucket, which is the very problem budget
            # normalisation was built to remove, reappearing at the boundary.
            #
            # Returning zero items at a tight budget is not a bug, it is the
            # finding: a 1200-token parent is undeliverable within 250 tokens.
            #
            # This comment used to end "the caller distinguishes 'nothing fit'
            # from 'ranking failed'" -- and no caller could, because only a bare
            # list was returned. One golden-set question hit it for real: the
            # top-ranked parent needs 1687 tokens against a 1500 budget, so the
            # generator received nothing, abstained, and the judge scored the
            # abstention "supported". A total retrieval failure recorded as a
            # success in both metric tiers. `stats` below makes the promised
            # distinction actually available.
            if spent + cost > budget:
                break
            spent += cost
            if unit == "parent":
                parent = self.parents.get(child.parent_id or "")
                seen_parents.add((parent or child).chunk_id)
            hits.append(Hit(child, float(s[i]), rank))

        if stats is not None:
            stats["n_ranked"] = int(len(order))
            stats["starved_by_budget"] = bool(len(order) and not hits)
            stats["min_unit_tokens"] = None
            if stats["starved_by_budget"]:
                costs: list[int] = []
                for i in order:
                    c = self.children[i]
                    if unit == "child":
                        costs.append(count_tokens(c.embed_text or c.display_text))
                    else:
                        t = self.parents.get(c.parent_id or "") or c
                        costs.append(count_tokens(t.display_text))
                stats["min_unit_tokens"] = min(costs) if costs else None
        return hits

    # -- parent resolution ---------------------------------------------------

    def parents_for(self, hits: Iterable[Hit]) -> list[Chunk]:
        """Children -> the parents that reach the model, deduped, rank order kept.

        Dedup matters for a reason beyond tokens: two children of one parent are
        two retrieval hits but ONE piece of evidence. Sending the parent twice
        would make the model see a repeated passage as corroboration.
        """
        out: list[Chunk] = []
        seen: set[str] = set()
        for h in hits:
            target = self.parents.get(h.chunk.parent_id or "") or h.chunk
            if target.chunk_id in seen:
                continue
            seen.add(target.chunk_id)
            out.append(target)
        return out

    # -- persistence ---------------------------------------------------------

    def save(self, name: str = "numpy_index") -> Path:
        d = config.DATA_INDEX / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "children.json").write_text(
            json.dumps([c.to_json() for c in self.children], indent=1), encoding="utf-8"
        )
        (d / "parents.json").write_text(
            json.dumps([p.to_json() for p in self.parents.values()], indent=1), encoding="utf-8"
        )
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, name: str = "numpy_index") -> "NumpyIndex":
        d = config.DATA_INDEX / name
        vectors = np.load(d / "vectors.npy")
        children = [Chunk.from_json(x) for x in json.loads((d / "children.json").read_text("utf-8"))]
        parents = [Chunk.from_json(x) for x in json.loads((d / "parents.json").read_text("utf-8"))]
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        return cls(children, vectors, parents, meta=meta)

    # -- the oracle role -----------------------------------------------------

    def ground_truth(self, query_vecs: np.ndarray, k: int = 10) -> list[list[str]]:
        """Exact top-k chunk ids per query. THIS is what ANN recall is measured against.

        Kept on the index rather than in the eval module on purpose: the oracle
        and the thing being tested must share the same corpus object, or you are
        back to comparing two different programs -- which is the bug that cost an
        hour earlier today when a survey script imported a different parser than
        the pipeline did.
        """
        return [[h.chunk.chunk_id for h in self.search_k(q, k)] for q in np.atleast_2d(query_vecs)]


def recall_at_k(retrieved: Sequence[str], truth: Sequence[str]) -> float:
    """Fraction of the exact top-k that an approximate search also returned."""
    if not truth:
        return 1.0
    return len(set(retrieved) & set(truth)) / len(set(truth))
