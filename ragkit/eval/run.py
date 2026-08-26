"""
The eval harness entry point. Guide module M9.

    uv run python -m ragkit.eval.run                      # retrieval tier
    uv run python -m ragkit.eval.run --baseline           # write the CI baseline
    uv run python -m ragkit.eval.run --gate               # fail on regression

WHAT THIS REFUSES TO DO, and why each refusal is the point:

1. IT REFUSES TO SCORE A MIXED-PROVENANCE INDEX as one population. If the index
   holds both contextualised and un-contextualised chunks, an A/B over it
   measures a diluted effect of unknown size -- which makes a real technique look
   not-worth-the-cost, a wrong conclusion from correct arithmetic.
   (config.EVAL_REFUSE_MIXED_PROVENANCE)

2. IT REFUSES TO LET A MIXED INDEX BECOME THE CI BASELINE. A baseline is a
   promise about what was measured, and a mixed index cannot keep it.
   (config.CI_BASELINE_REQUIRES_UNIFORM_PROVENANCE)

3. IT REFUSES TO AVERAGE THE QUARANTINED FIXTURE. `benchmark-report.pdf` is a
   file I authored containing the failure I was designing against. It runs as a
   REGRESSION TEST -- "this known failure must not return" -- and is reported
   separately from every rate.

4. IT REFUSES TO EMIT A RATE BELOW MIN_N_FOR_RATE. Slices report "2 of 3".

5. IT REFUSES TO COMPARE ACROSS PIPELINE FINGERPRINTS without saying so. A
   baseline recorded under a different parser, chunker, embedding model or
   dimensionality describes a different system. The gate reports the mismatch
   rather than quietly comparing.

Rule 5 exists because of a measured incident: a survey script that imported a
different module measured a different parser and reported table counts differing
5x on identical bytes. Neither number was wrong. They were answers about
different programs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .. import config, gemini, limits
from ..index.numpy_index import NumpyIndex
from . import goldenset as G
from . import metrics as M

# THE HEADLINE BUDGET IS NOT THE GENERATION BUDGET, and the first eval run is why.
#
# Scored at config.TOKENS_CONTEXT_BUDGET (12,000) the harness reported
# child_strict = 100%, with mean retrieved child tokens of 11,324 -- about 60
# chunks, roughly 8% of the entire 802-chunk corpus, for every single query. A
# recall figure obtained by returning a twelfth of the corpus is not a
# measurement; it is arithmetic that cannot fail.
#
# So the headline is scored at a budget resembling what actually reaches a model
# after reranking, and the real artifact is the SWEEP -- recall against budget.
# A single point hides where the curve bends, and where it bends is the only
# thing that tells you whether more retrieval would help.
HEADLINE_BUDGET = 1500
SWEEP_BUDGETS = (250, 500, 1000, 1500, 3000, 6000, 12000)

BASELINE = config.DATA_EVAL / "baseline.json"
RESULTS = config.DATA_EVAL / "eval_results.json"


def _index_provenance(idx: NumpyIndex) -> dict[str, Any]:
    rep_path = config.DATA_EVAL / "index_report.json"
    rep = json.loads(rep_path.read_text("utf-8")) if rep_path.exists() else {}
    return {
        "parser_version": rep.get("parser_version"),
        "chunker_version": rep.get("chunker_version"),
        "pipeline_fingerprint": rep.get("pipeline_fingerprint"),
        "strategy": idx.meta.get("strategy"),
        "dim": idx.meta.get("dim"),
        "n_children": idx.meta.get("n_children_indexed"),
        "uniform_provenance": idx.meta.get("uniform_provenance"),
        "child_text_source": rep.get("child_text_source"),
        "n_header_missing": rep.get("n_header_missing"),
        "resolved_models": gemini.resolve_models(),
    }


def run(
    *,
    index_name: str = "numpy_index",
    token_budget: int | None = None,
    limit: int | None = None,
    sweep: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    items = G.load()
    if not items:
        raise RuntimeError(
            "no golden set at data/eval/golden.jsonl -- build it first:\n"
            "  uv run python -c \"from ragkit.eval import goldenset as G; "
            "G.save(G.build()[0])\""
        )
    if limit:
        items = items[:limit]

    idx = NumpyIndex.load(index_name)
    prov = _index_provenance(idx)

    # Refusal 1.
    if config.EVAL_REFUSE_MIXED_PROVENANCE and prov.get("uniform_provenance") is False:
        raise RuntimeError(
            "index has MIXED provenance (some chunks contextualised, some not). "
            "Scoring it as one population produces a diluted effect size of unknown "
            "magnitude. Re-run ingest to completion, or set "
            "EVAL_REFUSE_MIXED_PROVENANCE=False and accept an uninterpretable number."
        )

    budget = token_budget or HEADLINE_BUDGET
    t0 = time.time()
    results: list[M.ItemResult] = []
    with limits.collect() as log:
        # Queries are embedded in one batched call rather than per item: 120
        # separate embed calls against a free-tier per-minute limit is the
        # difference between a runnable suite and a rate-limit endurance test.
        scoreable = [i for i in items if i.stratum != "out_of_scope" and i.anchor != "none"]
        vecs, st = gemini.embed_texts(
            [i.question for i in scoreable], kind="query", stage="eval_embed_queries"
        )
        if verbose:
            print(f"  queries: {st.render()}")
        qv = {id(i): v for i, v in zip(scoreable, vecs)}

        for i, item in enumerate(items, 1):
            if item.stratum == "out_of_scope" or item.anchor == "none":
                results.append(M.score_item(item, idx, np.zeros(1), token_budget=budget))
                continue
            results.append(M.score_item(item, idx, qv[id(item)], token_budget=budget))
            if verbose and i % 25 == 0:
                print(f"  scored {i}/{len(items)}", flush=True)

        agg = M.aggregate(results)

        # The sweep reuses the SAME query vectors, so it costs zero extra API
        # calls -- the expensive part was embedding 96 questions once.
        curve: dict[str, Any] = {}
        if sweep:
            for b in SWEEP_BUDGETS:
                if b == budget:
                    curve[str(b)] = agg["headline"]
                    continue
                rs = [
                    M.score_item(it, idx, qv[id(it)], token_budget=b)
                    for it in scoreable
                    if not it.quarantined
                ]
                curve[str(b)] = M.aggregate(rs)["headline"]
            if verbose:
                print()
                print(f"{'budget':>8s} {'child_strict':>14s} {'parent_strict':>14s} "
                      f"{'child_tok':>10s} {'parent_tok':>11s}")
                for b, h in curve.items():
                    print(f"{b:>8s} {h['child_strict']['label']:>14s} "
                          f"{h['parent_strict']['label']:>14s} "
                          f"{h['mean_child_tokens']:>10d} {h['mean_parent_tokens']:>11d}")

        # REGRESSION TESTS CARRY THEIR BUDGET AS PART OF THEIR IDENTITY.
        #
        # The efConstruction test passed at 12,000 and failed at 1,500. That is
        # not a broken build, it is a budget-sensitive test -- the value lives in
        # Table 4 on the page whose OTHER table is corrupted, and at a realistic
        # budget it does not make the cut. Reported as bare pass/fail, "fails" and
        # "fails at this budget" are indistinguishable and someone chases a code
        # fix for a knob turn.
        #
        # So each test reports the SMALLEST swept budget at which it passes.
        # None means it never passes, which is a real regression.
        quarantined = [i for i in items if i.quarantined and i.anchor != "none"]
        regressions: dict[str, Any] = {}
        for it in quarantined:
            passes_at = None
            for b in sorted(SWEEP_BUDGETS):
                r = M.score_item(it, idx, qv[id(it)], token_budget=b)
                if r.parent_strict:
                    passes_at = b
                    break
            regressions[it.question[:70]] = {
                "passes_at_budget": passes_at,
                "passes_at_headline": (passes_at is not None and passes_at <= budget),
                "needles": len(it.needles),
            }
        if verbose and regressions:
            print()
            print(f"regression tests (headline budget {budget}):")
            for q, v in regressions.items():
                at = v["passes_at_budget"]
                verdict = ("PASS" if v["passes_at_headline"]
                           else (f"passes only at >={at}" if at else "NEVER PASSES"))
                print(f"  [{verdict}] {q}")

        payload = {
            "seconds": round(time.time() - t0, 1),
            "regression_tests": regressions,
            "token_budget": budget,
            "budget_sweep": curve,
            "golden_set": G.summarise(items),
            "index_provenance": prov,
            "metrics": agg,
            "per_item": [r.to_json() for r in results],
            "degradations": log.to_dicts(),
        }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------
# CI gate
# --------------------------------------------------------------------------


def write_baseline(payload: dict[str, Any]) -> Path:
    prov = payload["index_provenance"]
    # Refusal 2.
    if config.CI_BASELINE_REQUIRES_UNIFORM_PROVENANCE and prov.get("uniform_provenance") is False:
        raise RuntimeError(
            "refusing to write a CI baseline from a mixed-provenance index. A "
            "baseline is a promise about what was measured."
        )
    BASELINE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BASELINE


def gate(payload: dict[str, Any], *, tolerance: float | None = None) -> tuple[bool, list[str]]:
    """Compare against the stored baseline. Returns (ok, messages)."""
    tol = config.CI_NDCG_REGRESSION_TOLERANCE if tolerance is None else tolerance
    msgs: list[str] = []
    if not BASELINE.exists():
        return True, ["no baseline stored; nothing to compare (use --baseline to create one)"]

    base = json.loads(BASELINE.read_text("utf-8"))

    # Refusal 5: say so rather than quietly comparing across systems.
    for k in ("parser_version", "chunker_version", "pipeline_fingerprint"):
        a, b = base["index_provenance"].get(k), payload["index_provenance"].get(k)
        if a != b:
            msgs.append(f"PROVENANCE CHANGED {k}: baseline={a} now={b}")
    if base["golden_set"].get("evaluable") != payload["golden_set"].get("evaluable"):
        msgs.append(
            f"GOLDEN SET CHANGED: baseline had {base['golden_set'].get('evaluable')} "
            f"evaluable items, now {payload['golden_set'].get('evaluable')} -- "
            "a different suite is not a comparison"
        )

    ok = True
    for key in ("child_strict", "parent_strict", "source_hit"):
        b = base["metrics"]["headline"][key]
        n = payload["metrics"]["headline"][key]
        if not (b["sufficient"] and n["sufficient"]):
            msgs.append(f"{key}: insufficient n to gate ({b['n']} -> {n['n']})")
            continue
        delta = (n["rate"] or 0) - (b["rate"] or 0)
        verdict = "REGRESSION" if delta < -tol else ("improved" if delta > tol else "flat")
        if delta < -tol:
            ok = False
        msgs.append(f"{key}: {b['rate']:.3f} -> {n['rate']:.3f} ({delta:+.3f}) {verdict}")

    # Regression tests are pass/fail, never averaged -- and a test that merely
    # needs a bigger budget is reported as that, not as a failure.
    for q, v in (payload.get("regression_tests") or {}).items():
        at = v["passes_at_budget"]
        if at is None:
            ok = False
            msgs.append(f"REGRESSION TEST NEVER PASSES: {q}")
        elif not v["passes_at_headline"]:
            msgs.append(f"budget-sensitive (passes at >={at}, headline is "
                        f"{payload['token_budget']}): {q}")
    return ok, msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ragkit.eval.run", description=__doc__)
    ap.add_argument("--index", default="numpy_index")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline", action="store_true", help="store this run as the CI baseline")
    ap.add_argument("--gate", action="store_true", help="fail (exit 1) on regression")
    ap.add_argument("--no-sweep", action="store_true", help="skip the recall-vs-budget curve")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    payload = run(index_name=args.index, token_budget=args.budget, limit=args.limit,
                  sweep=not args.no_sweep)
    if args.json:
        print(json.dumps(payload["metrics"], indent=2))
    else:
        print()
        print(M.render(payload["metrics"]))
        print()
        print("golden set:", json.dumps(payload["golden_set"]))
        print(f"wrote {RESULTS}  ({payload['seconds']}s)")

    if args.baseline:
        print(f"baseline -> {write_baseline(payload)}")

    if args.gate:
        ok, msgs = gate(payload)
        print("\nCI GATE:")
        for m in msgs:
            print("  " + m)
        print("  " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
