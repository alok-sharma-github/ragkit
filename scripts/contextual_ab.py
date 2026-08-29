"""What the accounting correction and the contextual prefix each did, separately.

TWO CHANGES LANDED TOGETHER, so this file exists to keep them separable. One
corrected how a budget is spent (ADR A-13); the other turned on an LLM-written
situating prefix (D-6). Shipping them in one commit was right -- each is bad on
its own, see the ADR -- but a reader must still be able to ask "which of these
moved the number", and a single before/after cannot answer that.

So the comparison is a 2x2 over the same golden set, run through the eval itself
rather than a private reimplementation, so every cell is a published figure:

                  charged INDEXED          charged DELIVERED
                  (embed_text)             (display_text)
  breadcrumb      the old published        accounting fixed, feature off
                  numbers
  contextual      feature on, old          what ships now
                  accounting

Read the rows for the FEATURE and the columns for the ACCOUNTING. The single
number that used to be quoted is the top-left cell; the one quoted now is the
bottom-right. Nothing about that is obvious from the pair alone, which is the
reason for the other two.

---------------------------------------------------------------------------
WHY THE ACCOUNTING WAS WRONG, IN ONE PARAGRAPH

`search_budget` charges a PARENT its `display_text` -- what the model reads --
and used to charge a CHILD its `embed_text`, which is the body plus a heading
trail plus, once prefixes shipped, a model-written sentence. None of that
reaches the model. So the small unit paid for its own index-time enrichment and
the large unit paid for nothing equivalent, in the one comparison budget
normalisation exists to make fair. Harmless at a ~9% breadcrumb. At a 66-token
prefix on a ~300-token body it inverted a conclusion: contextual retrieval read
-11 at a 250-token budget under the old basis and +9 under the corrected one, on
the same index and the same questions.

    uv run python scripts/contextual_ab.py
    uv run python scripts/contextual_ab.py --json > data/eval/contextual_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.index.numpy_index import NumpyIndex  # noqa: E402

BUDGETS = (250, 500, 1000, 1500, 3000, 6000)
BASES = ("indexed", "delivered")


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


def _eval(index_name: str, basis: str, budget: int) -> dict[str, object]:
    """One eval run, in a SUBPROCESS, because the cost basis is read at import.

    Reloading config inside one process to flip a module-level constant is the
    kind of shortcut that produces a table where half the cells silently used
    the other setting. A process boundary cannot be half-applied.
    """
    env = {**os.environ, "RAGKIT_CHILD_COST_BASIS": basis, "PYTHONIOENCODING": "utf-8"}
    # Calls run() rather than the CLI, because the CLI interleaves progress with
    # its JSON on stdout. Parsing a stream that mixes the two is a way to be
    # wrong quietly, which is the failure this whole file is about.
    code = (
        "import json;from ragkit.eval import run as R;"
        f"print('<<<'+json.dumps(R.run(index_name={index_name!r},"
        f"token_budget={budget},sweep=False,verbose=False))+'>>>')"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, env=env, cwd=str(ROOT))
    if out.returncode != 0:
        raise RuntimeError(f"eval failed for {index_name}/{basis}/{budget}:\n"
                           f"{out.stderr[-1500:]}")
    payload = json.loads(out.stdout.split("<<<", 1)[1].rsplit(">>>", 1)[0])
    got = payload["index_provenance"].get("child_cost_basis")
    # The run REPORTS the basis it used; check it rather than trusting the env
    # var to have arrived. A table whose cells disagree with their own labels is
    # worse than no table.
    if got != basis:
        raise RuntimeError(f"asked for basis={basis}, run reported {got}")
    head = payload["metrics"]["headline"]
    return {
        "child_strict": head["child_strict"]["hits"],
        "child_strict_label": head["child_strict"]["label"],
        "parent_strict": head["parent_strict"]["hits"],
        "parent_strict_label": head["parent_strict"]["label"],
        "source_hit": head["source_hit"]["label"],
        "n": head["child_strict"]["n"],
        "mean_child_tokens": head.get("mean_child_tokens"),
        "child_no_delivery": head.get("child_no_delivery"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shipped", default="numpy_index",
                    help="the index that ships (contextual prefixes on)")
    ap.add_argument("--alternative", default="numpy_index_breadcrumb",
                    help="the index it is compared against (breadcrumbs only)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    arms = {"contextual": args.shipped, "breadcrumb": args.alternative}
    meta = {k: _describe(v) for k, v in arms.items()}
    if meta["contextual"]["fingerprint"] == meta["breadcrumb"]["fingerprint"]:
        # Not a warning. If the fingerprints match, one of these indexes is not
        # what its name says, and every number below would be noise reported to
        # three significant figures.
        print("REFUSING: both indexes carry the same pipeline fingerprint, so they "
              "are the same system and there is nothing to compare.")
        return 2

    # EVERY RUN BELOW OVERWRITES data/eval/eval_results.json, because run() writes
    # its artifact -- so this script would leave the primary eval artifact
    # describing whichever cell happened to go last (a 250-token run against the
    # wrong index), and the failure histogram would then refuse to pool. Same
    # shape as the manifest defect this comparison already exposed: a
    # side-by-side must not damage the thing it is compared against. Snapshot
    # and restore, in a finally, so a crash mid-table cannot leave it damaged
    # either.
    artifacts = [ROOT / "data" / "eval" / "eval_results.json"]
    saved = {a: a.read_bytes() for a in artifacts if a.exists()}
    cells: dict[str, dict[str, dict[str, object]]] = {}
    try:
        for arm, index_name in arms.items():
            cells[arm] = {}
            for basis in BASES:
                cells[arm][basis] = {str(b): _eval(index_name, basis, b) for b in BUDGETS}
    finally:
        for a, blob in saved.items():
            a.write_bytes(blob)

    out = {
        "meta": meta,
        "note": ("2x2: rows are the FEATURE (breadcrumb vs contextual prefix), "
                 "columns are the ACCOUNTING (a child charged its index text vs "
                 "the text it delivers). Old published figures are "
                 "breadcrumb/indexed; what ships now is contextual/delivered."),
        "cells": cells,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    for arm in arms:
        m = meta[arm]
        print(f"{arm:12s} {m['index']:24s} {m['fingerprint']}  "
              f"{m['contextualized']}/{m['n_children']} contextualised")

    n = cells["breadcrumb"]["indexed"]["1500"]["n"]
    print(f"\nchild_strict, {n} evaluable golden items\n")
    print(f"{'budget':>7} | {'breadcrumb':>21} | {'contextual':>21} |")
    print(f"{'':>7} | {'indexed':>10} {'delivered':>10} | "
          f"{'indexed':>10} {'delivered':>10} | {'what ships':>11}")
    print("-" * 74)
    for b in BUDGETS:
        bi = cells["breadcrumb"]["indexed"][str(b)]["child_strict"]
        bd = cells["breadcrumb"]["delivered"][str(b)]["child_strict"]
        ci = cells["contextual"]["indexed"][str(b)]["child_strict"]
        cd = cells["contextual"]["delivered"][str(b)]["child_strict"]
        # "what ships" is the only number a reader should quote, and it is
        # stated against the only number they used to quote.
        print(f"{b:>7} | {bi:>10} {bd:>10} | {ci:>10} {cd:>10} | {cd - bi:>+11}")

    print(f"\nparent_strict, same runs (unchanged by the accounting -- a parent was "
          f"always charged what it delivers)\n")
    print(f"{'budget':>7} | {'breadcrumb':>10} | {'contextual':>10} |")
    print("-" * 36)
    for b in BUDGETS:
        bp = cells["breadcrumb"]["delivered"][str(b)]["parent_strict"]
        cp = cells["contextual"]["delivered"][str(b)]["parent_strict"]
        print(f"{b:>7} | {bp:>10} | {cp:>10} |")

    print("\nRows are the feature, columns are the accounting. The left-most column "
          "is what this project published before the correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
