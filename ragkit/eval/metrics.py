"""
Retrieval-tier metrics. Guide module M9, first tier.

TWO TIERS, MEASURED SEPARATELY, and this file is only the first. If you measure
only end to end you cannot localise a regression: a drop could be retrieval or
generation and the number does not say which.

---------------------------------------------------------------------------
RECALL IS OVERLAP, NOT MATCHING

With chunk-id labels, recall was "is the labelled id in the top k" -- clean
yes/no, and useless across chunking strategies, because 27 chunks under `fixed`
and 42 under `header_aware` mean id #14 denotes two unrelated things.

With needles it becomes "does any retrieved chunk CONTAIN this string". Fuzzier,
and strategy-independent, which is the entire point.

---------------------------------------------------------------------------
BUDGET NORMALISATION, TWICE

Recall@k is not comparable across strategies:

    fixed         27 chunks x 290 tokens -> top-10 = 2,900 tokens
    header_aware  42 chunks x 183 tokens -> top-10 = 1,830 tokens

Ten buckets catch more rain than ten cups. So the primary number is measured at
an equal TOKEN BUDGET, and at two different units, because they answer two
different questions:

    child budget  -> RANKING quality (children differ in size too: 183 vs 290)
    parent budget -> what the MODEL ACTUALLY SAW

The gap between them is what parent-document retrieval is buying. Measured
separately or it does not exist as a number. Recall@k is still reported, because
it is what everyone else quotes -- just not what a decision hangs on.

---------------------------------------------------------------------------
THREE OUTCOMES, NOT TWO

A needle can straddle a chunk boundary: under `fixed` the header lands in chunk 7
and the value in chunk 8, and neither contains the whole string. Collapsing that
into "missed" throws away the most diagnostic number available, because avoiding
exactly that is what header-aware chunking is FOR. So: contained / partial /
absent, with partial counted and reported apart.

And for multi-needle items (multi_hop, aggregative -- which cannot be answered
from one span by definition), STRICT means every needle was covered and PARTIAL
means some were. The generator needs all of them, so strict is the number that
predicts answer quality; partial is the diagnostic.

---------------------------------------------------------------------------
SMALL DENOMINATORS GET COUNTS, NOT RATES

Corpus-wide there are 3 `page_text_clip` chunks and 1 continuation flag. Slicing
120 golden items by a field whose category has three members is not a
comparison; it is an anecdote with error bars wider than the effect.

So a slice below MIN_N_FOR_RATE reports "2 of 3", never "67%". A percentage
implies a precision that is not there, and it is exactly the kind of number that
gets quoted later without its sample size -- which is how the withdrawn 18%
headerless figure happened. The number was not wrong; the confidence attached to
it was.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Sequence

from .. import config
from ..gemini import count_tokens
from ..index.numpy_index import Hit, NumpyIndex
from ..ingest.document import Chunk, ChunkRole
from ..ingest.loaders import _squash
from .goldenset import GoldenItem

Coverage = Literal["contained", "partial", "absent"]

# A needle counts as partially covered when this fraction of it is present in one
# retrieved chunk. Declared, not tuned.
PARTIAL_MIN = 0.5

# Below this many items, a slice reports counts instead of a rate.
MIN_N_FOR_RATE = 10


# --------------------------------------------------------------------------
# Needle coverage
# --------------------------------------------------------------------------


def cover_needle(needle: str, texts: Sequence[str]) -> tuple[Coverage, float]:
    """Is this needle in any of these retrieved texts?

    Squash-insensitive, using the SAME normaliser as the citation checker and the
    golden-set locator. Three normalisers for one job is how they end up
    disagreeing, which already happened once between table-cell text and page
    text.
    """
    n = _squash(needle)
    if not n:
        return "absent", 0.0
    best = 0.0
    for t in texts:
        sq = _squash(t)
        if n in sq:
            return "contained", 1.0
        # Longest prefix of the needle present, as a cheap straddle detector: a
        # needle split across a boundary leaves a long head in one chunk.
        lo, hi = 0, len(n)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if n[:mid] in sq:
                lo = mid
            else:
                hi = mid - 1
        best = max(best, lo / len(n))
    return ("partial" if best >= PARTIAL_MIN else "absent"), best


@dataclass
class ItemResult:
    question: str
    stratum: str
    source_id: str
    anchor: str
    quarantined: bool = False

    # per-unit outcomes
    child_strict: bool = False
    child_partial: bool = False
    parent_strict: bool = False
    parent_partial: bool = False

    needle_coverage: dict[str, str] = field(default_factory=dict)
    needle_fraction: float = 0.0        # fraction of needles contained (child unit)
    source_hit: bool = False            # any accepted source retrieved at all
    rank_first_hit: int | None = None   # rank of the first accepted-source chunk

    n_children: int = 0
    n_parents: int = 0
    # True when the unit could deliver NOTHING within the budget. Distinct from
    # a ranking failure: a 1200-token parent is undeliverable in 250 tokens no
    # matter how well it ranked. Scored as a miss (the model really did receive
    # nothing) but counted separately so the two causes stay legible.
    child_no_delivery: bool = False
    parent_no_delivery: bool = False
    child_tokens: int = 0
    parent_tokens: int = 0
    text_sources: list[str] = field(default_factory=list)
    skipped: str = ""                   # why this item could not be scored

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def score_item(
    item: GoldenItem,
    index: NumpyIndex,
    query_vec: Any,
    *,
    token_budget: int | None = None,
) -> ItemResult:
    """One golden item against one index, at both units.

    `out_of_scope` items are NOT scored here. Their correct behaviour is
    abstention, which is a generation-tier property -- retrieval returning
    something for an unanswerable question is not a retrieval failure. Scoring
    them as recall=0 would penalise the retriever for the eval's own design.
    """
    res = ItemResult(
        question=item.question, stratum=item.stratum, source_id=item.source_id,
        anchor=item.anchor, quarantined=item.quarantined,
    )
    if item.stratum == "out_of_scope" or item.anchor == "none":
        res.skipped = "out_of_scope: scored in the generation tier, not here"
        return res

    budget = token_budget or config.TOKENS_CONTEXT_BUDGET
    accept = set(item.accept_sources or [item.source_id])

    child_hits: list[Hit] = index.search_budget(query_vec, token_budget=budget, unit="child")
    parent_hits: list[Hit] = index.search_budget(query_vec, token_budget=budget, unit="parent")
    parents = index.parents_for(parent_hits)

    child_texts = [h.chunk.display_text for h in child_hits]
    parent_texts = [p.display_text for p in parents]

    res.n_children = len(child_hits)
    res.n_parents = len(parents)
    res.child_tokens = sum(count_tokens(t) for t in child_texts)
    res.parent_tokens = sum(count_tokens(t) for t in parent_texts)
    res.text_sources = sorted({h.chunk.text_source for h in child_hits})
    res.child_no_delivery = not child_hits
    res.parent_no_delivery = not parents

    # INVARIANT: a budget-normalised retrieval must respect its budget. This is
    # the assertion that would have caught the `and hits` overshoot immediately
    # instead of it surfacing as a fake crossover in the sweep. Same move as
    # partial >= strict: make the impossible loud.
    for unit, delivered in (("child", res.child_tokens), ("parent", res.parent_tokens)):
        if delivered > budget:
            raise AssertionError(
                f"{unit} unit delivered {delivered} tokens against a budget of "
                f"{budget} ({delivered / budget - 1:+.0%}). A budget-normalised "
                "comparison in which one unit exceeds the budget is not a "
                "comparison -- the larger unit is simply given more text."
            )

    # Did the right DOCUMENT surface at all? Separate from needle coverage,
    # because "wrong document" and "right document, wrong passage" are different
    # failures with different fixes (Barnett #1/#2 vs #3).
    for h in child_hits:
        if h.chunk.source_id in accept:
            res.source_hit = True
            res.rank_first_hit = h.rank
            break

    if item.anchor == "asset":
        # No text to match; the label is "did an accepted source surface".
        # PARTIAL IS SET TOO, and forgetting it was a real bug: 3 asset items with
        # strict=True and partial=False produced child_partial (89) < child_strict
        # (92), which is logically impossible -- partial is a superset of strict.
        # The impossible ordering is what made the bug visible, which is why the
        # invariant is now asserted in aggregate() rather than left to be noticed.
        res.child_strict = res.parent_strict = res.source_hit
        res.child_partial = res.parent_partial = res.source_hit
        res.needle_fraction = 1.0 if res.source_hit else 0.0
        return res

    child_cov = {n: cover_needle(n, child_texts) for n in item.needles}
    parent_cov = {n: cover_needle(n, parent_texts) for n in item.needles}
    res.needle_coverage = {n: c for n, (c, _f) in child_cov.items()}

    c_states = [c for c, _ in child_cov.values()]
    p_states = [c for c, _ in parent_cov.values()]
    res.child_strict = bool(c_states) and all(s == "contained" for s in c_states)
    res.child_partial = any(s in ("contained", "partial") for s in c_states)
    res.parent_strict = bool(p_states) and all(s == "contained" for s in p_states)
    res.parent_partial = any(s in ("contained", "partial") for s in p_states)
    res.needle_fraction = (
        sum(1 for s in c_states if s == "contained") / len(c_states) if c_states else 0.0
    )
    return res


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _rate(hits: int, n: int) -> dict[str, Any]:
    """A rate, or an honest refusal to compute one.

    `sufficient=False` is the machine-readable form of "this is an anecdote".
    Reporting `rate` as None below the threshold means a consumer cannot
    accidentally chart three data points as a percentage.
    """
    if n == 0:
        return {"n": 0, "hits": 0, "rate": None, "sufficient": False, "label": "no items"}
    if n < MIN_N_FOR_RATE:
        return {"n": n, "hits": hits, "rate": None, "sufficient": False,
                "label": f"{hits} of {n}"}
    p = hits / n
    # Wilson interval, so a reported rate carries its own uncertainty rather than
    # looking like a point measurement.
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {
        "n": n, "hits": hits, "rate": round(p, 3), "sufficient": True,
        "ci95": [round(max(0.0, centre - halfwidth), 3), round(min(1.0, centre + halfwidth), 3)],
        "label": f"{hits}/{n} = {p:.0%}",
    }


def aggregate(results: Sequence[ItemResult]) -> dict[str, Any]:
    """Headline numbers plus every slice, each carrying its own sample size."""
    scored = [r for r in results if not r.skipped and not r.quarantined]
    quarantined = [r for r in results if r.quarantined]
    skipped = [r for r in results if r.skipped]

    def block(rs: Sequence[ItemResult]) -> dict[str, Any]:
        return {
            "child_strict": _rate(sum(r.child_strict for r in rs), len(rs)),
            "child_partial": _rate(sum(r.child_partial for r in rs), len(rs)),
            "parent_strict": _rate(sum(r.parent_strict for r in rs), len(rs)),
            "parent_partial": _rate(sum(r.parent_partial for r in rs), len(rs)),
            "source_hit": _rate(sum(r.source_hit for r in rs), len(rs)),
            "mean_child_tokens": round(sum(r.child_tokens for r in rs) / max(len(rs), 1)),
            "mean_parent_tokens": round(sum(r.parent_tokens for r in rs) / max(len(rs), 1)),
            "child_no_delivery": sum(r.child_no_delivery for r in rs),
            "parent_no_delivery": sum(r.parent_no_delivery for r in rs),
        }

    def slice_by(key) -> dict[str, Any]:
        buckets: dict[str, list[ItemResult]] = {}
        for r in scored:
            buckets.setdefault(str(key(r)), []).append(r)
        return {k: block(v) for k, v in sorted(buckets.items())}

    # INVARIANT: partial is a superset of strict, at every unit. If this fails the
    # scorer is wrong, not the system -- so it raises rather than reporting a
    # number that cannot be true. A metric that can report impossible values will
    # eventually report a merely-wrong one without the tell.
    for u in ("child", "parent"):
        n_strict = sum(getattr(r, f"{u}_strict") for r in scored)
        n_partial = sum(getattr(r, f"{u}_partial") for r in scored)
        if n_partial < n_strict:
            raise AssertionError(
                f"{u}: partial ({n_partial}) < strict ({n_strict}). Partial is a "
                "superset of strict by construction, so this is a scorer bug. "
                "Refusing to emit metrics that cannot be true."
            )

    head = block(scored)
    # The gap that parent-document retrieval buys, stated as a number or refused.
    cs, ps = head["child_strict"], head["parent_strict"]
    if cs["sufficient"] and ps["sufficient"]:
        gap: Any = round((ps["rate"] or 0) - (cs["rate"] or 0), 3)
    else:
        gap = f"insufficient n ({cs['n']})"

    # THE NUMBER DOCUMENTS ITS OWN SCOPE.
    #
    # `aggregative` and `ambiguous` came back empty from generation and
    # `multi_hop` has 2 items, so a bare "85%" describes this system on EASY,
    # single-passage questions and silently excludes the categories where top-k
    # retrieval structurally struggles. Aggregative questions ("what are the main
    # themes across these papers") are the ones no amount of reranking fixes --
    # so their absence hides the one problem the obvious next fixes will not
    # solve.
    #
    # Same discipline as `sufficient: False` on a small slice: report the scope
    # with the number, so it cannot be quoted without it.
    present = {r.stratum for r in scored} | {r.stratum for r in skipped}
    declared = set(config.GOLDEN_SET_STRATA)
    missing = sorted(declared - present)
    coverage = {
        "declared": sorted(declared),
        "present": sorted(present & declared),
        "missing": missing,
        "thin": sorted(
            st for st in (present & declared)
            if 0 < sum(1 for r in scored if r.stratum == st) < MIN_N_FOR_RATE
        ),
    }

    return {
        "n_items": len(results),
        "stratum_coverage": coverage,
        "n_scored": len(scored),
        "n_quarantined": len(quarantined),
        "n_skipped_out_of_scope": len(skipped),
        "min_n_for_rate": MIN_N_FOR_RATE,
        "headline": head,
        "parent_minus_child_strict": gap,
        "by_stratum": slice_by(lambda r: r.stratum),
        "by_anchor": slice_by(lambda r: r.anchor),
        "by_source": slice_by(lambda r: r.source_id or "(none)"),
        "quarantined_regression_tests": {
            r.question[:70]: {"child_strict": r.child_strict, "parent_strict": r.parent_strict}
            for r in quarantined
        },
    }


def scope_label(agg: dict[str, Any]) -> str:
    """The honest form of the headline: '85% on 4 of 7 strata'."""
    cov = agg["stratum_coverage"]
    h = agg["headline"]["child_strict"]
    n_present, n_declared = len(cov["present"]), len(cov["declared"])
    core = h["label"]
    if n_present < n_declared:
        core += f" on {n_present} of {n_declared} strata"
    if cov["thin"]:
        core += f" ({', '.join(cov['thin'])} thin)"
    return core


def render(agg: dict[str, Any]) -> str:
    lines = [
        f"scored {agg['n_scored']} items "
        f"({agg['n_quarantined']} quarantined, {agg['n_skipped_out_of_scope']} out-of-scope)",
        "",
        f"HEADLINE: child_strict {scope_label(agg)}",
    ]
    cov = agg["stratum_coverage"]
    if cov["missing"]:
        lines.append(f"  NOT MEASURED: {', '.join(cov['missing'])} -- no items generated. "
                     "Aggregative questions in particular are the ones ranking fixes cannot help.")
    lines += ["", "RETRIEVAL (budget-normalised, both units):"]
    for k in ("child_strict", "child_partial", "parent_strict", "parent_partial", "source_hit"):
        lines.append(f"  {k:16s} {agg['headline'][k]['label']}")
    lines.append(f"  mean tokens      child={agg['headline']['mean_child_tokens']} "
                 f"parent={agg['headline']['mean_parent_tokens']}")
    lines.append(f"  parent - child (strict): {agg['parent_minus_child_strict']}")
    for name in ("by_stratum", "by_anchor"):
        lines.append("")
        lines.append(f"{name}:")
        for k, v in agg[name].items():
            flag = "" if v["child_strict"]["sufficient"] else "   [counts only, small n]"
            lines.append(f"  {k:20s} child_strict {v['child_strict']['label']:14s}"
                         f" parent_strict {v['parent_strict']['label']}{flag}")
    if agg["quarantined_regression_tests"]:
        lines.append("")
        lines.append("regression tests (excluded from every average):")
        for q, v in agg["quarantined_regression_tests"].items():
            ok = "PASS" if v["parent_strict"] else "FAIL"
            lines.append(f"  [{ok}] {q}")
    return "\n".join(lines)
