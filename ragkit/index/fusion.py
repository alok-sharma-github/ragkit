"""
Reciprocal Rank Fusion. Guide module M2.

    score(d) = sum over legs of  weight_i / (k + rank_i(d))          k = 60

WHY RANK FUSION AND NOT SCORE FUSION -- and this corpus provides the evidence.

The two legs produce numbers on incomparable scales. Measured here:

    dense (cosine)  range 0.503 .. 0.769   over the whole corpus, one query
    BM25            range 0      .. ~27    unbounded above, corpus-dependent

Cosine sits in a narrow band whose floor is high, because two arbitrary passages
of English about retrieval are already fairly similar. BM25 is unbounded and its
magnitude depends on IDF, document length and query length. There is no fixed
transform between them: normalising per query would make the dense leg's 0.769
"1.0" and its 0.503 "0.0", inventing spread that does not exist.

Ranks have none of that problem. Rank 1 means the same thing in both legs.

k = 60 IS A SMOOTHING CONSTANT, not magic (Cormack et al., 2009). It damps how
much the very top rank dominates: at k=0 rank 1 scores 1.0 and rank 2 scores
0.5, so a single leg's first place outweighs almost anything; at k=60 they score
0.0164 and 0.0161, so agreement across legs matters more than one leg's
confidence. Small k trusts individual legs; large k trusts consensus.

THE WEAKNESS, stated because a technique without a stated cost is a slogan: RRF
throws away score MAGNITUDE. A runaway-confident single hit is flattened to
"rank 1" like any other rank 1. When one leg is genuinely certain and the other
is noise, RRF dilutes the certainty -- which is the case where a tuned weighted
sum beats it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Sequence

from .. import config


@dataclass
class FusedHit:
    key: str
    score: float
    rank: int
    # Per-leg rank, kept for diagnosis. Without it "why did this rank third" is
    # unanswerable, and a fusion you cannot explain is a fusion you cannot tune.
    ranks: dict[str, int] = field(default_factory=dict)
    legs: list[str] = field(default_factory=list)

    @property
    def n_legs(self) -> int:
        return len(self.legs)


def rrf(
    legs: Mapping[str, Sequence[str]],
    *,
    k: int | None = None,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked id lists. `legs` maps a leg name to ids in rank order.

    Ranks are taken from POSITION in the list, so a caller cannot accidentally
    pass scores and have them silently treated as ranks -- the type makes the
    rank-based nature of the algorithm structural rather than documented.
    """
    kk = config.RRF_K if k is None else k
    w = dict(weights or {})
    acc: dict[str, FusedHit] = {}

    for leg, ids in legs.items():
        weight = w.get(leg, 1.0)
        seen: set[str] = set()
        for pos, cid in enumerate(ids):
            # A duplicate id within one leg would otherwise be counted twice,
            # inflating that document's score with its own repetition.
            if cid in seen:
                continue
            seen.add(cid)
            rank = pos + 1
            hit = acc.get(cid)
            if hit is None:
                hit = acc[cid] = FusedHit(key=cid, score=0.0, rank=0)
            hit.score += weight / (kk + rank)
            hit.ranks[leg] = rank
            hit.legs.append(leg)

    # Tie-break on best single-leg rank, then id, so the output is deterministic.
    # Without a deterministic tie-break, two eval runs over the same index can
    # differ, and a metric that moves without a cause is worse than no metric.
    out = sorted(
        acc.values(),
        key=lambda h: (-h.score, min(h.ranks.values()), h.key),
    )
    for i, h in enumerate(out):
        h.rank = i
    return out[:limit] if limit else out


def explain(hits: Sequence[FusedHit], *, top: int = 5) -> str:
    """Per-leg ranks for the top results. Diagnosis, not decoration."""
    lines = [f"{'rank':>4s} {'score':>8s} {'legs':>5s}  per-leg ranks"]
    for h in hits[:top]:
        legs = ", ".join(f"{leg}#{r}" for leg, r in sorted(h.ranks.items()))
        lines.append(f"{h.rank:>4d} {h.score:8.5f} {h.n_legs:>5d}  {legs}")
    return "\n".join(lines)


def leg_contribution(hits: Sequence[FusedHit], legs: Iterable[str]) -> dict[str, Any]:
    """How much did each leg actually contribute to the fused top-k?

    The number that answers "is the sparse leg earning its keep". A leg that
    never supplies a result the other leg missed is pure cost -- and an index
    that must be kept in sync with the corpus is a recurring operational cost,
    not a one-off.
    """
    legs = list(legs)
    total = len(hits)
    out: dict[str, Any] = {"n_fused": total}
    for leg in legs:
        present = [h for h in hits if leg in h.ranks]
        unique = [h for h in present if h.n_legs == 1]
        out[leg] = {
            "in_fused": len(present),
            "only_this_leg": len(unique),
            "share_unique": round(len(unique) / max(total, 1), 3),
        }
    out["both_legs"] = sum(1 for h in hits if h.n_legs == len(legs))
    return out
