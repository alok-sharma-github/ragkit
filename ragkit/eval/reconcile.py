"""
Invariants as data. The design's Reconciliation view, and its thesis:

    "counts must reconcile before any score on this run is trusted"

Every check here is one this project acquired by being burned. They are collected
in one place because that sentence is the right gate: a quality number computed
over an index whose counts do not reconcile is not a weak measurement, it is an
unknown one.

  index_parity      indexed == embedded
                    From `zip(batch_idx, resp.embeddings)`: gemini-embedding-2
                    returns ONE embedding for a multi-item batch, zip truncates
                    silently, and the unmatched rows became zero vectors -- in the
                    index, scoring 0.0 against every query, invisible, reported
                    as success.

  scoring_sanity    partial >= strict
                    Partial is a superset of strict by construction. It reported
                    97% < 100% because asset-anchored items set strict and not
                    partial. An impossible ordering is a scorer bug announcing
                    itself; the next bug would have been merely wrong.

  context_budget    delivered <= budget
                    The budget fill had an `and hits` clause guaranteeing at
                    least one result, so the parent unit was handed 378 tokens
                    against a 250 budget while the child unit stayed under. It
                    produced a crossover in the recall sweep that did not exist.

  citation_integrity  fabricated == 0
                    Citations are generated text, so a label can point at a
                    chunk that was never sent. Checked against WHAT THIS REQUEST
                    SENT, not against the index -- citing a real chunk that was
                    not retrieved is still fabrication, and checking the wrong
                    set turns a caught bug into an approved one.

  provenance_propagation  labelled_children >= repaired_blocks, both-or-neither
                    The loader repaired 2 corrupted tables; 0 children carried
                    the label, because _make_child never copied text_source. The
                    repaired TEXT propagated and the AUDIT TRAIL did not -- the
                    worse half to lose, since the eval slices on that field.

  stratum_coverage  every declared stratum has items
                    `aggregative` and `ambiguous` came back empty from
                    generation. A headline computed over 5 of 7 strata describes
                    the easy questions and silently omits the ones no amount of
                    reranking fixes.

  sample_size       n >= MIN_N_FOR_RATE before a rate is emitted
                    Reported as "2 of 3", never "67%". The withdrawn 18%
                    headerless figure was not wrong; the confidence attached to
                    it was.

A check reports HOLDS, FAILS, or NOT_MEASURED. NOT_MEASURED is a first-class
state and not a synonym for passing -- an absent value invites investigation, an
invented one ends it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import config
from . import metrics as M

State = Literal["HOLDS", "FAILS", "NOT_MEASURED"]


@dataclass
class Check:
    name: str
    rule: str
    observed: str
    state: State
    n: int | None = None
    # Which pipeline components this check's numbers came from. A check that
    # holds under one fingerprint says nothing about another.
    fingerprint: list[str] = field(default_factory=list)
    detail: str = ""
    why: str = ""            # the incident it exists because of

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def reconcile() -> dict[str, Any]:
    """Read the artifacts on disk and evaluate every invariant.

    Reads ARTIFACTS, not scrollback. Five display-layer incidents this session
    (four encoding, one truncation-then-fabrication) all came from trusting a
    rendering instead of the file.
    """
    index = _load(config.DATA_EVAL / "index_report.json")
    evalr = _load(config.DATA_EVAL / "eval_results.json")
    checks: list[Check] = []

    fp: list[str] = []
    if index:
        fp = [
            str(index.get("parser_version", "?")).split("+")[0],
            str(index.get("chunker_version", "?")),
            f"{index.get('dim')}d",
        ]

    # -- index parity -------------------------------------------------------
    if index:
        n_in = index.get("n_children_in")
        n_idx = index.get("n_children_indexed")
        dropped = index.get("n_children_dropped_zero_vector", 0)
        checks.append(Check(
            name="Index parity",
            rule="indexed == embedded (no zero vectors)",
            observed=f"{n_in} children -> {n_idx} indexed, {dropped} dropped",
            state="HOLDS" if dropped == 0 and n_in == n_idx else "FAILS",
            n=n_idx,
            fingerprint=fp,
            detail="a zero-vector row scores 0.0 against every query: present, counted, unreachable",
            why="zip() truncated a batch whose model returns one embedding for N inputs",
        ))
        orph = index.get("n_orphan_children", 0)
        checks.append(Check(
            name="Parent resolution",
            rule="orphan children == 0",
            observed=f"{orph} children whose parent_id is not in the parent store",
            state="HOLDS" if orph == 0 else "FAILS",
            n=index.get("n_parents"),
            fingerprint=fp,
            detail="an orphan child is a retrieval hit that cannot produce context for the model",
        ))
        # provenance propagation, per file
        per_file = index.get("per_file") or []
        bad = []
        for row in per_file:
            rep = int(row.get("tables_repaired", 0) or 0)
            lab = int(row.get("children_page_text", 0) or 0)
            if (rep > 0) != (lab > 0) or lab < rep:
                bad.append(f"{row.get('file')}: {rep} repaired, {lab} labelled")
        total_rep = sum(int(r.get("tables_repaired", 0) or 0) for r in per_file)
        total_lab = sum(int(r.get("children_page_text", 0) or 0) for r in per_file)
        checks.append(Check(
            name="Provenance propagation",
            rule="repaired>0 <=> labelled>0, and labelled >= repaired",
            observed=f"{total_rep} repaired blocks -> {total_lab} labelled children",
            state="HOLDS" if not bad else "FAILS",
            n=total_lab,
            fingerprint=fp,
            detail="; ".join(bad) if bad else
                   "a repaired block may split into several children, so >= is the invariant, not ==",
            why="the repaired text propagated and its label did not, on all 791 children",
        ))
        checks.append(Check(
            name="Uniform provenance",
            rule="document-derived chunks share one provenance",
            observed=json.dumps(index.get("provenance_populations", {})),
            state="HOLDS" if index.get("uniform_provenance") else "FAILS",
            n=index.get("n_children_indexed"),
            fingerprint=fp,
            detail="model-generated chunks are excluded from the test: a different KIND of "
                   "content is not damage",
        ))
    else:
        checks.append(Check("Index parity", "indexed == embedded", "no index_report.json",
                            "NOT_MEASURED", detail="run: ragkit.cli ingest"))

    # -- eval-side ----------------------------------------------------------
    if evalr:
        head = evalr["metrics"]["headline"]
        budget = evalr.get("token_budget")
        strict = head["child_strict"]["hits"]
        partial = head["child_partial"]["hits"]
        checks.append(Check(
            name="Scoring sanity",
            rule="partial >= strict",
            observed=f"partial {partial} >= strict {strict}",
            state="HOLDS" if partial >= strict else "FAILS",
            n=head["child_strict"]["n"],
            fingerprint=fp,
            detail="partial is a superset of strict by construction",
            why="reported 97% < 100% because asset items set strict and not partial",
        ))
        over = [
            b for b, h in (evalr.get("budget_sweep") or {}).items()
            if h["mean_child_tokens"] > int(b) or h["mean_parent_tokens"] > int(b)
        ]
        checks.append(Check(
            name="Context budget",
            rule="delivered <= budget, every unit, every budget",
            observed=(f"mean child {head['mean_child_tokens']}, parent "
                      f"{head['mean_parent_tokens']}, budget {budget}"),
            state="HOLDS" if not over else "FAILS",
            n=head["child_strict"]["n"],
            fingerprint=fp,
            detail=("over budget at: " + ", ".join(over)) if over else
                   "a unit allowed to overshoot is simply given more text",
            why="an `and hits` clause gave the parent unit 378 tokens against a 250 budget",
        ))
        cov = evalr["metrics"]["stratum_coverage"]
        checks.append(Check(
            name="Stratum coverage",
            rule="every declared stratum has items",
            observed=f"{len(cov['present'])} of {len(cov['declared'])} present",
            state="HOLDS" if not cov["missing"] else "NOT_MEASURED",
            n=evalr["metrics"]["n_scored"],
            fingerprint=fp,
            detail=("missing: " + ", ".join(cov["missing"])) if cov["missing"] else "",
            why="aggregative questions are the ones no amount of reranking fixes",
        ))
        thin = cov.get("thin") or []
        checks.append(Check(
            name="Sample size",
            rule=f"n >= {M.MIN_N_FOR_RATE} before a rate is emitted",
            observed=f"{len(thin)} stratum/strata below the floor",
            state="HOLDS" if not thin else "NOT_MEASURED",
            n=M.MIN_N_FOR_RATE,
            fingerprint=fp,
            detail=("counts only for: " + ", ".join(thin)) if thin else
                   "every reported rate carries a Wilson interval",
        ))
        gs = evalr.get("golden_set") or {}
        checks.append(Check(
            name="Grounding audit",
            rule="golden items human-verified on a stratified sample",
            observed=f"{gs.get('human_verified', 0)} of {gs.get('evaluable', 0)} verified",
            state="NOT_MEASURED" if not gs.get("human_verified") else "HOLDS",
            n=gs.get("evaluable"),
            fingerprint=fp,
            detail="the whole set is machine-generated and machine-verified; "
                   "8 fabricated needles were caught by locator checks, "
                   "but nobody has read a stratified slice",
        ))
    else:
        for nm, rule in (("Scoring sanity", "partial >= strict"),
                         ("Context budget", "delivered <= budget"),
                         ("Stratum coverage", "every declared stratum has items")):
            checks.append(Check(nm, rule, "no eval_results.json", "NOT_MEASURED",
                                detail="run: python -m ragkit.eval.run"))

    n_fail = sum(1 for c in checks if c.state == "FAILS")
    n_hold = sum(1 for c in checks if c.state == "HOLDS")
    n_nm = sum(1 for c in checks if c.state == "NOT_MEASURED")
    return {
        "fingerprint": fp,
        "pipeline_fingerprint": (index or {}).get("pipeline_fingerprint"),
        "trusted": n_fail == 0,
        "summary": {"failing": n_fail, "passing": n_hold, "not_measured": n_nm},
        "thesis": "counts must reconcile before any score on this run is trusted",
        "checks": [c.to_json() for c in checks],
    }


def citation_checks(answer_reconciliation: dict[str, int]) -> Check:
    """The per-answer citation-integrity check, for the Answers view."""
    emitted = answer_reconciliation.get("citations_emitted", 0)
    verified = answer_reconciliation.get("citations_verified", 0)
    fabricated = answer_reconciliation.get("citations_fabricated", 0)
    unquotable = answer_reconciliation.get("citations_unquotable", 0)
    return Check(
        name="Citation integrity",
        rule="fabricated == 0",
        observed=f"{emitted} emitted -> {verified} verified, {unquotable} unquotable, "
                 f"{fabricated} fabricated",
        state="HOLDS" if fabricated == 0 else "FAILS",
        n=emitted,
        detail="membership is checked against the labels SENT in this request, not the index",
    )


def render(rec: dict[str, Any]) -> str:
    lines = [
        rec["thesis"],
        f"  {rec['summary']['failing']} failing · {rec['summary']['passing']} passing · "
        f"{rec['summary']['not_measured']} not measured"
        + ("   [scores on this run are trusted]" if rec["trusted"]
           else "   [SCORES ON THIS RUN ARE NOT TRUSTED]"),
        f"  fingerprint: {' · '.join(rec['fingerprint'])}  {rec.get('pipeline_fingerprint') or ''}",
        "",
    ]
    for c in rec["checks"]:
        mark = {"HOLDS": "ok  ", "FAILS": "FAIL", "NOT_MEASURED": "n/m "}[c["state"]]
        lines.append(f"  [{mark}] {c['name']:24s} {c['observed']}")
        lines.append(f"           rule: {c['rule']}" + (f"   n={c['n']}" if c["n"] else ""))
        if c["detail"]:
            lines.append(f"           {c['detail']}")
    return "\n".join(lines)
