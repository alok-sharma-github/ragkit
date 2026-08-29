"""
Deferred decisions, with the condition that expires them.

WHY THIS FILE EXISTS. "Reranking deferred -- not much to gain" and "reranking
deferred until multi-hop questions exist" read the same in a document and behave
completely differently over time. The first sounds settled and never gets
revisited. The second expires on its own the moment the precondition changes.

So a deferral is stored as a PREDICATE over the current artifacts, not as prose.
`review()` re-evaluates every one against the eval results on disk and reports
which ones have expired. The ADR then quotes a live check rather than a claim
that was true once.

The specific case that prompted this: reranking was deferred because the eval
showed `source_hit = 100%` at every budget, so the right document always
surfaced and the remaining headroom looked small. True -- and measured on a
golden set with ZERO aggregative questions, ZERO ambiguous ones, and TWO
multi-hop. It is a strong result on the easy questions and silent on the hard
ones. The deferral is therefore conditional on that silence, and says so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import config
from .eval import metrics as M


@dataclass
class Deferral:
    name: str
    guide_module: str
    decision: str
    because: str
    # The condition under which this deferral STOPS being valid, in words and as
    # a predicate over the artifacts. Both, because the words go in the ADR and
    # the predicate is what actually gets checked.
    revisit_when: str
    cost_if_wrong: str
    # Symbols this deferral legitimately leaves unreachable, BY NAME.
    #
    # Was fuzzy text matching, and the trap fired on the first run: `Chunk.citation`
    # was excused as "explained by a deferral" because the word "citation" appears
    # in the entailment deferral's prose. A forgotten orphan waved through on a
    # keyword is exactly the failure mode of eyeballing a list where most entries
    # are legitimate. An explicit name cannot match by accident.
    orphans: tuple[str, ...] = ()
    expired: bool = False
    evidence: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _eval() -> dict[str, Any] | None:
    p = config.DATA_EVAL / "eval_results.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def review() -> dict[str, Any]:
    ev = _eval()
    cov = (ev or {}).get("metrics", {}).get("stratum_coverage", {})
    by_stratum = (ev or {}).get("metrics", {}).get("by_stratum", {})
    head = (ev or {}).get("metrics", {}).get("headline", {})
    n_children = None
    ip = config.DATA_EVAL / "index_report.json"
    if ip.exists():
        n_children = json.loads(ip.read_text("utf-8")).get("n_children_indexed")

    def stratum_n(name: str) -> int:
        b = by_stratum.get(name)
        return int(b["child_strict"]["n"]) if b else 0

    # HOW MANY QUESTIONS LOST THEIR DOCUMENT ENTIRELY, in the stratum where a
    # rare unique token is the whole query. This is the observable consequence of
    # the dilution described in A-14: a contextual prefix gives every
    # bibliography in the corpus the same generic description, and a lookup whose
    # only signal is one rare string then loses to fifteen passages that match
    # that description well.
    #
    # It is close to the property but not identical to it -- an item can lose its
    # document for unrelated reasons, a bad chunk boundary among them. So this
    # predicate is a PROMPT TO LOOK, not a proof, and the threshold says why: one
    # is a rounding error against 92, two is a pattern.
    rows = [r for r in (ev or {}).get("per_item", []) if not r.get("skipped")]
    identifier_lost = sum(
        1 for r in rows
        if r.get("stratum") == "exact_identifier" and r.get("source_hit") is False
    )

    hard_strata_measured = (
        not cov.get("missing")
        and stratum_n("multi_hop") >= M.MIN_N_FOR_RATE
    )

    items: list[Deferral] = [
        Deferral(
            name="reranking",
            guide_module="M3",
            decision="No cross-encoder or LLM reranker.",
            because=(
                "the eval reports source_hit = 100% at every budget, so the correct "
                "document always surfaces and every failure is ranking WITHIN a "
                "document; dense at 3000 tokens is already 96%"
            ),
            revisit_when=(
                "the golden set contains aggregative and ambiguous strata and at "
                f"least {M.MIN_N_FOR_RATE} multi-hop items -- reranking is exactly "
                "what reorders candidates for multi-facet questions, and the current "
                "measurement is silent on them"
            ),
            cost_if_wrong="~200-400ms per query inside TTFT, plus a model to host",
            orphans=(),
            expired=hard_strata_measured,
            evidence=(
                f"missing strata: {cov.get('missing') or 'none'}; "
                f"multi_hop n={stratum_n('multi_hop')}"
            ),
        ),
        Deferral(
            name="qdrant",
            guide_module="M6",
            decision="numpy exact cosine, no ANN index.",
            because=(
                "at this corpus size exact search IS the right answer: it is faster "
                "to operate, has no recall/latency dial to mistune, and it is the "
                "oracle any future ANN recall must be measured against"
            ),
            revisit_when=(
                "the corpus exceeds ~100k children, or p95 retrieval latency exceeds "
                "100ms -- currently retrieval is 5-20ms"
            ),
            cost_if_wrong="a second index to keep in sync, and an unmeasured recall dial",
            # The exact-search oracle and its metric exist only to measure an ANN
            # index that does not exist yet. Dead by design, named so.
            orphans=("NumpyIndex.ground_truth", "recall_at_k", "verify_truncation"),
            expired=bool(n_children and n_children > 100_000),
            evidence=f"n_children={n_children}",
        ),
        Deferral(
            name="entailment_verification",
            guide_module="M7",
            decision="Two deterministic citation checks only; no NLI pass.",
            because=(
                "entailment needs a judge, and an unvalidated judge is not a weaker "
                "signal but an unknown one. Character overlap provably cannot "
                "separate an honest rewording from an invented quote (measured: the "
                "fabrication scored 0.53, the rewording 0.48)"
            ),
            revisit_when="the LLM judge passes its kappa gate against hand labels",
            cost_if_wrong="unsupported-but-real-looking claims pass the free checks",
            orphans=("judge_unvalidated",),
            expired=bool((ev or {}).get("golden_set", {}).get("human_verified")),
            evidence=f"human_verified={(ev or {}).get('golden_set', {}).get('human_verified', 0)}",
        ),
        Deferral(
            name="parent_charged_for_index_text",
            guide_module="M9",
            decision=(
                "A child is charged the text it DELIVERS, not the text it was "
                "indexed under. Both bases stay selectable."
            ),
            because=(
                "search_budget charged a parent its display_text and a child its "
                "embed_text, so the small unit paid for its own heading trail and "
                "contextual prefix while the large unit paid for nothing "
                "equivalent -- in the one comparison budget normalisation exists "
                "to make fair. Harmless at a ~9% breadcrumb, decisive at a "
                "66-token prefix: the same index read -10 or +8 at a 250-token "
                "budget on the choice of basis alone"
            ),
            revisit_when=(
                "a retrieval unit is delivered to the model WITH its enrichment "
                "attached -- e.g. if a child ever became the delivered unit and "
                "the situating sentence were shown as context. Then the prefix "
                "IS delivered text and charging it becomes correct again"
            ),
            cost_if_wrong=(
                "every figure published before the correction was measured under "
                "the old basis. Kept re-runnable rather than restated: "
                "RAGKIT_CHILD_COST_BASIS=indexed still works, the eval records "
                "which basis produced a number, and the CI gate refuses to "
                "compare across them"
            ),
            orphans=(),
            # No predicate: nothing in data/eval can observe "the delivered unit
            # changed". Left False and said so, rather than wired to a proxy.
            expired=False,
            evidence=(
                "2x2 in data/eval/contextual_ab.json: child_strict at 250 tokens "
                "reads 25 / 30 / 15 / 38 across breadcrumb-indexed, "
                "breadcrumb-delivered, contextual-indexed, contextual-delivered"
            ),
        ),
        Deferral(
            name="suppress_prefix_on_rare_token_passages",
            guide_module="M5",
            decision=(
                "No rule that skips the contextual prefix for reference lists or "
                "other passages whose value is a rare unique token."
            ),
            because=(
                "the fix is a PREDICATE, and this project has now been wrong about "
                "a predicate repeatedly. A rule matching 'looks like a "
                "bibliography' would correlate with the target and fail at the "
                "edges -- a related-work section, a table of citations, an "
                "appendix of sources -- and because it SUPPRESSES a feature, its "
                "errors are silent: a wrongly-skipped passage just retrieves "
                "slightly worse forever, with nothing to notice. Worse, it would "
                "be tuned against n=1: one document, one question, one corpus of "
                "academic papers, with no way to tell whether it generalises. The "
                "honest version of the rule is 'skip passages whose value is a "
                "rare unique token', and that is not detectable from the passage "
                "-- it depends on what people search for"
            ),
            revisit_when=(
                "a SECOND exact_identifier question loses its document entirely, "
                "or a corpus arrives whose queries are mostly rare tokens -- "
                "contracts, invoices, part catalogues, policy schedules. Either "
                "turns n=1 into evidence. The second would make it urgent within "
                "a day, and is the case to check before enabling prefixes on any "
                "non-academic corpus"
            ),
            cost_if_wrong=(
                "citation-style lookups retrieve slightly worse, silently. "
                "Currently one item of 92 against a +3 headline; the risk is that "
                "the ratio is a property of THIS corpus rather than of the "
                "technique"
            ),
            orphans=(),
            expired=identifier_lost >= 2,
            evidence=(
                f"exact_identifier questions that lost their document entirely: "
                f"{identifier_lost}. See A-14 -- the measured case retrieved 19 "
                "children and none from the right document, against 6 and 2 "
                "before contextualisation"
            ),
        ),
        Deferral(
            name="sweep_on_upload_couples_visitors",
            guide_module="M14",
            decision=(
                "Expired sessions are purged at the start of an UPLOAD, not on a "
                "timer and not from the sweep endpoint."
            ),
            because=(
                "purge_expired() had exactly one caller -- POST "
                "/api/sessions/sweep -- which a demo refuses categorically, so on "
                "the one deployment where strangers upload the TTL never ran "
                "while the UI promised deletion. A timer was rejected when the "
                "sweep was designed, and the reason still holds: a thread that "
                "deletes documents has no request to attribute it to and nowhere "
                "for its failures to surface. An upload is the only event that "
                "makes the session set grow, so it is where the bound belongs"
            ),
            revisit_when=(
                "the demo carries enough traffic that upload latency matters. "
                "THE PROPERTY TO KNOW: one visitor's upload now triggers deletion "
                "of somebody else's documents, so a slow or failing purge shows up "
                "as latency on an unrelated person's request. Bounded at demo "
                "traffic -- a purge is a manifest read plus a rebuild -- and not "
                "bounded at real load. At that point it becomes a queue, or a "
                "scheduled task with somewhere to report to"
            ),
            cost_if_wrong=(
                "an uploader waits for someone else's cleanup. Also: on a quiet "
                "day an expired session sits until the next upload arrives, which "
                "is why the visitor-facing text states the property rather than "
                "the schedule -- 'deleted when the next visitor uploads', not "
                "'deleted after an hour'"
            ),
            orphans=(),
            # No predicate: nothing in data/eval observes request latency or
            # concurrent uploads. Left False and said so, rather than wired to a
            # proxy that merely correlates.
            expired=False,
            evidence=(
                "verified end to end at a 25s TTL: purged_documents "
                "['zarnak-bulletin.pdf'], failed [], sessions_remaining 0. The "
                "first run failed loudly -- remove_source called ingest() without "
                "the owner argument D-18 made required -- which is the sweep "
                "reporting what it could not do rather than retrying in silence"
            ),
        ),
        Deferral(
            name="generation_tier_ci_gating",
            guide_module="M9",
            decision=(
                "CI gates on the RETRIEVAL tier only (>3% NDCG@5 regression). "
                "Generation-tier metrics are reported, never gated."
            ),
            because=(
                "the two tiers differ in a way that decides this. Retrieval is "
                "deterministic -- cached embeddings, hand-written BM25, RRF -- so "
                "the same commit against the same index gives the same NDCG, and a "
                "3% drop is a real change. Generation is SAMPLED, so a single run "
                "is n=1 and a one-run comparison cannot separate a regression from "
                "noise. Gating on it would fail builds for no reason, and a gate "
                "that fires without cause gets muted, which costs the retrieval "
                "gate its credibility too"
            ),
            revisit_when=(
                "generation-tier gating is actually wanted, and EITHER several runs "
                "per commit are affordable (each full pass is ~$0.86 and ~18 min) OR "
                "a threshold has been calibrated against measured run-to-run "
                "variance -- which requires repeated runs on an unchanged index, "
                "something never done here"
            ),
            cost_if_wrong=(
                "a faithfulness regression ships unnoticed between eval runs. "
                "Mitigated, not solved: every generation failure so far has been "
                "FP2/FP3 -- retrieval, which IS gated -- and zero were FP4, so the "
                "ungated tier is currently the one where nothing has ever gone wrong"
            ),
            orphans=(),
            # No predicate, and for the same reason as durable_spend_ledger: the
            # condition is "someone measured the variance", and nothing in
            # data/eval can observe whether that happened. A proxy -- say, two
            # judged artifacts existing -- would fire on two runs made for
            # unrelated reasons, which is a correlation standing in for a
            # condition. The pattern this file exists to refuse.
            expired=False,
            evidence=(
                "observed ONCE, and a single difference is not a variance estimate: "
                "re-running the generation tier on an unchanged index at an "
                "unchanged budget moved abstentions 17 -> 18 of 92 and flipped one "
                "judge verdict from failed to succeeded. Direction and magnitude "
                "unknown from n=2 runs"
            ),
        ),
        Deferral(
            name="durable_spend_ledger",
            guide_module="M14",
            decision=(
                "Daily spend counted in a file; only the per-operation ceiling is "
                "restart-proof."
            ),
            because=(
                "the layer that has to be right is the per-operation cap, and it is "
                "computed from the work in front of it rather than from remembered "
                "state -- so it holds across restarts by construction. The daily "
                "counter is a second, softer layer, and giving it a durable store "
                "today would mean standing up a database for a number that only "
                "catches slow drift"
            ),
            revisit_when=(
                "there is a durable store to count in (Phase 2 brings Postgres for "
                "conversations, jobs and quota), or a deployment starts restarting "
                "often enough that the daily counter is materially under-reporting"
            ),
            cost_if_wrong=(
                "on a container that scales to zero the daily total resets, so "
                "cumulative spend is UNDER-counted and the daily ceiling fails open "
                "across restarts. It cannot fail open per operation, which is the "
                "case that matters"
            ),
            orphans=(),
            # Not a predicate over artifacts: nothing in data/eval can observe
            # whether the filesystem is durable. Left False deliberately and
            # stated as such, rather than wired to a proxy signal that would
            # merely correlate with the condition -- that is the fail-open check
            # pattern this file exists to avoid.
            expired=False,
            evidence=(
                f"ledger={config.SPEND_LEDGER_PATH.name}; "
                f"per-operation cap={config.GEMINI_MAX_OPERATION_TOKENS:,} tok "
                "(restart-proof); daily cap is not"
            ),
        ),
        Deferral(
            name="opentelemetry_tracing",
            guide_module="M14",
            decision="Per-stage timings in the response; no OTel exporter.",
            because=(
                "the numbers that inform a decision (embed / retrieve / generate ms, "
                "token counts) are already collected and returned per request. OTel "
                "adds a collector and a backend, which is infrastructure rather than "
                "insight at one user and one process"
            ),
            revisit_when="more than one process or service needs to correlate a trace",
            cost_if_wrong="no distributed trace when there is something to distribute",
        ),
    ]
    return {
        "deferrals": [d.to_json() for d in items],
        "expired": [d.name for d in items if d.expired],
        "note": "a deferral stored as a predicate expires by itself; one stored as "
                "prose sounds settled forever",
    }


def render(rev: dict[str, Any]) -> str:
    lines = ["deferred decisions (each with the condition that expires it):", ""]
    for d in rev["deferrals"]:
        mark = "EXPIRED -- revisit" if d["expired"] else "still valid"
        lines.append(f"  [{mark}] {d['name']} ({d['guide_module']})")
        lines.append(f"      decision : {d['decision']}")
        lines.append(f"      because  : {d['because']}")
        lines.append(f"      revisit  : {d['revisit_when']}")
        lines.append(f"      evidence : {d['evidence']}")
    return "\n".join(lines)
