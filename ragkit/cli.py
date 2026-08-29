"""
Command line entry point.

    uv run python -m ragkit.cli models
    uv run python -m ragkit.cli ingest [--no-images] [--strategy S] [--file F]
    uv run python -m ragkit.cli ask "question" [--budget N] [--sources N] [--json]
    uv run python -m ragkit.cli chat
    uv run python -m ragkit.cli status

Everything routes through ragkit.pipeline rather than re-assembling the stages,
so what a reviewer runs and what the eval measures are the same program. That is
not tidiness: a survey script that reached around the pipeline once reported
table counts differing 5x from the pipeline's, because it imported a different
module and therefore a different parser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, gemini, limits, pipeline
from .index.numpy_index import NumpyIndex
from .ingest.document import PUBLIC_OWNER, Manifest


def _fix_console() -> None:
    """Force UTF-8 output.

    The default Windows console codepage is cp1252, which cannot encode an em
    dash, a box-drawing character, or an accented author name. During this build
    it produced a UnicodeEncodeError while PRINTING A DEGRADATION WARNING -- the
    error path taking out the message it existed to deliver -- and separately made
    a correctly-stored 'Herve Jegou' look like corrupted data three times.
    Rendering is the last place a diagnostic should be able to fail.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    resolved = gemini.resolve_models(force=args.refresh, verbose=True)
    if args.all:
        print("\nreachable with this key:")
        for m in gemini.available_models():
            print("  ", m)
    print(f"\nwritten to {config.RESOLVED_MODELS_PATH}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    files = [Path(f) for f in args.file] if args.file else None
    # The CLI builds the SHARED corpus. Stated explicitly because a default
    # would make forgetting indistinguishable from intending.
    r = pipeline.ingest(
        owner=PUBLIC_OWNER,
        origin="corpus",
        strategy=args.strategy,
        caption_images=not args.no_images,
        files=files,
        verbose=True,
    )
    print()
    print(r.render())
    # A failed provenance cross-check is a non-zero exit, not a printed note.
    # The index is still written -- refusing to write it would lose work -- but
    # anything scripted around this must not treat the run as clean.
    return 0 if r.provenance_ok else 2


def cmd_ask(args: argparse.Namespace) -> int:
    r = pipeline.ask(
        args.question,
        token_budget=args.budget,
        max_sources=args.sources,
    )
    if args.json:
        print(json.dumps(
            {
                "question": r.question,
                "answer": r.answer.text,
                "abstained": r.answer.abstained,
                "abstain_reason": r.answer.abstain_reason,
                "grounded": r.answer.grounded,
                "reasons": r.answer.reasons,
                "reconciliation": r.answer.reconciliation,
                "citations": [
                    {
                        "label": c.label, "chunk_id": c.chunk_id,
                        "quote_status": c.quote_status, "overlap": round(c.overlap, 3),
                        "fabricated": c.fabricated, "detail": c.detail,
                    }
                    for c in r.answer.checks
                ],
                "sources": [
                    {"label": i, "source_id": p.source_id, "page": p.page,
                     "kind": p.kind.value, "text_source": p.text_source}
                    for i, p in enumerate(r.parents, 1)
                ],
                "timings_ms": {k: round(v) for k, v in r.timings_ms.items()},
                "degradations": r.degradations,
            },
            indent=2,
        ))
    else:
        print(r.render())
    # An ungrounded answer exits non-zero. Fail closed at the process boundary
    # too: a caller piping this somewhere should not have to parse prose to find
    # out that the citations did not verify.
    return 0 if (r.answer.grounded and not r.answer.abstained) else 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Multi-turn, with the honest caveat printed up front.

    History is passed to generation but NOT used to rewrite the query, so a
    follow-up like "what about the other one?" will retrieve badly. That is the
    guide's M1 condensation gap, and saying so beats letting it look like a bug.
    """
    idx = NumpyIndex.load(args.index)
    print(f"chat over {len(idx.children)} chunks. blank line or 'exit' to quit.")
    print("NOTE: query condensation is not implemented yet, so pronoun/ellipsis")
    print("follow-ups ('what about the other one?') will retrieve poorly.\n")
    history: list[str] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q or q.lower() in {"exit", "quit"}:
            return 0
        r = pipeline.ask(q, history="\n".join(history[-6:]), index=idx,
                         token_budget=args.budget, max_sources=args.sources)
        print()
        print(r.render(show_sources=not args.quiet))
        print()
        history += [f"user: {q}", f"assistant: {r.answer.text or r.answer.abstain_reason}"]


def cmd_judge(args: argparse.Namespace) -> int:
    """Tier-two eval. The order of these actions is the whole discipline.

        sample   build a class-balanced sample that oversamples hard cases
        label    hand-label it BLIND (never shows the judge's verdicts)
        run      judge the same sample, stored separately from the labels
        validate join them, compute kappa, write the gate
        score    judged metrics over the golden set -- withheld unless gated

    `label` before `run` is not a suggestion. Labelling after reading the judge's
    output measures agreement with something already seen; kappa comes out high
    and means nothing. If verdicts exist on disk when labelling happens, the
    kappa is recorded as possibly-anchored.
    """
    import json as _json
    from .eval import judge as J

    action = args.action
    if action == "sample":
        items = J.validation_sample(target=args.n)
        path = J.save_sample(items)
        from collections import Counter as _C
        counts = _C(i.constructed_truth or "organic" for i in items)
        print(f"{len(items)} items -> {path}")
        # All three classes, because the previous line printed only "supported"
        # and "unsupported" and therefore reported 4 negatives when there were 15
        # across two classes -- a display that hid the imbalance it should have
        # shown.
        print(f"  class balance: {dict(counts)}")
        worst = min(counts.values()) if counts else 0
        if worst < 5:
            print(f"  WARNING: smallest class has {worst} item(s); kappa over it "
                  "will swing on a single relabel")
        print(f"  by text_source: "
              f"{_json.dumps({t: sum(1 for i in items if i.text_source == t) for t in sorted({i.text_source for i in items})})}")
        print("  NOTE: this is a DISCRIMINATION sample (balanced, hard-case weighted).")
        print("  The kappa it yields is not a population faithfulness statistic.")
        return 0
    if action == "label":
        J.label_blind()
        return 0
    if action == "run":
        print(J.run_judge_on_sample() and "verdicts written")
        return 0
    if action == "validate":
        r = J.validate()
        print(_json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))
        return 0 if r.get("passes_gate") else 1
    if action == "gate":
        print(_json.dumps(J.gate(), indent=2))
        return 0 if J.gate()["may_emit_judged_metrics"] else 1
    if action == "score":
        r = J.score(limit=args.n if args.n else None)
        if r.get("withheld"):
            print("JUDGED METRICS WITHHELD")
            print("  " + (r["gate"]["withheld_because"] or ""))
            print("  " + r["note"])
            return 1
        # The denominator is printed FIRST and on its own line, because the
        # previous version printed "supported 90/91 = 99%" over a population in
        # which 17 rows were abstentions scored as supported. A rate whose
        # denominator is not visible is an invitation to misread it, and this one
        # got misread by the person who wrote it.
        print(f"  answered           {r['n_answered']} of "
              f"{r['n_answered'] + r['n_abstained'] + r['n_starved']}"
              f"   (abstained {r['n_abstained']}"
              + (f", starved {r['n_starved']}" if r["n_starved"] else "") + ")")
        print("  -- faithfulness, over ANSWERS only --")
        for k in ("supported", "partly", "unsupported", "answers_question"):
            print(f"    {k:16s} {r[k]['label']}")
        print("  -- coverage: questions the pipeline declined --")
        print(f"    {'overall':16s} {r['abstention_rate']['label']}")
        for st, v in sorted(r["abstention_by_stratum"].items(),
                            key=lambda kv: -(kv[1].get("rate") or 0)):
            print(f"    {st:16s} {v['label']}")
        print("  by text_source (answers only):")
        for ts, v in r["by_text_source"].items():
            print(f"    {ts or '(no sources)':22s} supported {v['supported']['label']}")
        return 0
    print(f"unknown action: {action}")
    return 2


def cmd_feedback(args: argparse.Namespace) -> int:
    """Agreement between the system's belief and human judgment."""
    import json as _json
    from . import feedback as F
    print(_json.dumps(F.stats(), indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Static audits: is every definition reached, and is every guard complete?

    Two scripts, one command, because they answer different questions and nobody
    should have to remember the second one exists:

      reachability  is this called from an entry point at all?
      guards        is this guard on EVERY route that needs it?

    The second exists because the first cannot ask it, and the gap between them
    cost ~2,500 tokens of unmetered spend: the ceiling was wired to one of three
    paid routes and reported itself as protection.
    """
    import subprocess
    import sys as _sys

    worst = 0
    for script in ("audit_reachability.py", "audit_guards.py"):
        print(f"\n===== {script} =====")
        rc = subprocess.call([_sys.executable, str(config.ROOT / "scripts" / script)])
        worst = max(worst, rc)
    return worst


def cmd_capabilities(args: argparse.Namespace) -> int:
    """What the API can actually do, measured. Config states intent; this states fact."""
    import json as _json
    print(_json.dumps(gemini.capabilities(), indent=2))
    return 0


def cmd_spend(args: argparse.Namespace) -> int:
    """The spend ceiling: what it is, what has been spent, and how durable that is.

    Exists because the quota banner and the budget refusals both tell the reader
    to look here. A message naming a command that does not exist is the same
    defect as a message naming the wrong cause -- it is confidently unhelpful.
    """
    from . import budget

    snap = budget.snapshot()
    print(f"deployment kind      {snap.deployment_kind}")
    if config.DEPLOYMENT_LABEL:
        print(f"deployment label     {config.DEPLOYMENT_LABEL}")
    print()
    print("ceilings")
    print(f"  per operation      {snap.per_operation_cap:,} tokens"
          "   <- checked BEFORE any paid call; restart-proof")
    print(f"  per day            {snap.daily_cap:,} tokens")
    print()
    print("today")
    print(f"  spent              {snap.spent_today:,} tokens")
    print(f"  remaining          {snap.remaining_today:,} tokens")

    ledger = budget._load()
    today = ledger.get(budget._today(), {})
    if today.get("by_stage"):
        print("  by stage")
        for stage, n in sorted(today["by_stage"].items(), key=lambda kv: -kv[1]):
            print(f"    {stage:20s} {n:>10,}")
        print(f"  calls              {today.get('calls', 0)}")

    if not snap.durable:
        print()
        print("NOTE: the daily counter is a file and RESETS ON RESTART, so on a")
        print("container that scales to zero it under-counts. The per-operation")
        print("ceiling does not depend on it -- that one is computed from the work")
        print("in front of it, which is why it is the layer that actually protects")
        print("you. See `ragkit deferred` for the condition that closes this.")
    return 0


def cmd_deferred(args: argparse.Namespace) -> int:
    """Deferred decisions and whether their preconditions still hold."""
    from . import deferred
    rev = deferred.review()
    print(deferred.render(rev))
    if rev["expired"]:
        print()
        print("EXPIRED -- these deferrals no longer hold: " + ", ".join(rev["expired"]))
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    """Invariants over the artifacts on disk."""
    from .eval import reconcile as R
    rec = R.reconcile()
    print(R.render(rec))
    return 0 if rec["trusted"] else 2


def cmd_status(args: argparse.Namespace) -> int:
    print(f"corpus dir : {config.DATA_RAW}")
    files = pipeline.L.corpus_files()
    print(f"corpus     : {len(files)} loadable files")
    m = Manifest()
    print(m.summary())
    if m.mixed_provenance():
        print("  MIXED PROVENANCE: contextual prefixes are partial, so A/B results")
        print("  over this index are not interpretable (config.EVAL_REFUSE_MIXED_PROVENANCE)")
    rep_path = config.DATA_EVAL / "index_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        print(f"\nindex      : {rep.get('n_children_indexed')} children, "
              f"{rep.get('n_parents')} parents, dim {rep.get('dim')}")
        print(f"  by kind      : {rep.get('child_kind')}")
        print(f"  by text src  : {rep.get('child_text_source')}")
        print(f"  parser       : {rep.get('parser_version')}")
        print(f"  chunker      : {rep.get('chunker_version')}")
        print(f"  provenance ok: {rep.get('provenance_cross_check_ok')}")
        for d in rep.get("degradations") or []:
            print(f"  DEGRADED [{d['stage']}] {d['impact']}")
    else:
        print("\nno index yet -- run: uv run python -m ragkit.cli ingest")
    return 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _fix_console()
    ap = argparse.ArgumentParser(prog="ragkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("models", help="resolve and print the model IDs your key can reach")
    p.add_argument("--refresh", action="store_true", help="re-resolve instead of reading cache")
    p.add_argument("--all", action="store_true", help="also list every reachable model")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("ingest", help="parse, chunk, embed and index the corpus")
    p.add_argument("--strategy", default="header_aware_parent",
                   choices=sorted(pipeline.S.CHUNKER_VERSIONS))
    p.add_argument("--no-images", action="store_true",
                   help="skip Gemini image captioning (images become unretrievable)")
    p.add_argument("--file", action="append", help="ingest only this file (repeatable)")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="answer one question with verified citations")
    p.add_argument("question")
    p.add_argument("--budget", type=int, default=None, help="context token budget")
    p.add_argument("--sources", type=int, default=6, help="max sources sent to the model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("chat", help="multi-turn session")
    p.add_argument("--index", default="numpy_index")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--sources", type=int, default=6)
    p.add_argument("--quiet", action="store_true", help="hide the retrieved source list")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("status", help="corpus, manifest and index state")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("capabilities", help="what the API can do, probed not configured")
    p.set_defaults(fn=cmd_capabilities)

    p = sub.add_parser("spend", help="the spend ceiling and what has been spent today")
    p.set_defaults(fn=cmd_spend)

    p = sub.add_parser("deferred", help="deferred decisions, and which have expired")
    p.set_defaults(fn=cmd_deferred)

    p = sub.add_parser("reconcile", help="invariant checks over the artifacts")
    p.set_defaults(fn=cmd_reconcile)

    p = sub.add_parser("judge", help="tier-two eval: sample, label, run, validate, score")
    p.add_argument("action",
                   choices=["sample", "label", "run", "validate", "gate", "score"])
    p.add_argument("-n", type=int, default=0, help="sample size / score limit")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("feedback", help="human-vs-system agreement from flags")
    p.set_defaults(fn=cmd_feedback)

    p = sub.add_parser("audit", help="reachability audit: name the caller")
    p.set_defaults(fn=cmd_audit)

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except limits.QuotaExhausted as exc:
        print(f"\nSTOPPED: {exc}")
        # Was an unconditional "This is a free-tier Gemini limit". True when
        # the only key this could run against was free; false the moment a card
        # is attached -- and a banner naming the wrong limit sends the reader to
        # the wrong page. Fifth instance of that bug class in this codebase.
        if config.DEPLOYMENT_KIND == "demo":
            print("This is a free-tier Gemini limit, not a bug in the pipeline.")
            print("Per-minute limits clear in about a minute; a daily limit needs")
            print("tomorrow or billing enabled.")
        else:
            print("This is a Gemini API rate limit, not a bug in the pipeline.")
            print("Per-minute limits clear in about a minute. If instead this was a")
            print("ceiling you set, see `ragkit spend`.")
        print("Finished work is cached, so a re-run resumes.")
        return 3
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
