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
import os
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


def _isolation_check(fp: str) -> list[Check]:
    """Can one session retrieve another's upload? Answered by TRYING it.

    Not "we implemented ownership" -- an executed retrieval, scored, in the same
    harness as every other invariant. The claim a customer needs is a number.

    TWO CASES, because they fail differently:

      cross-session   session A queries with session B's OWN vector -- the
                      strongest possible pull toward B's chunk -- and must not
                      receive it.
      public visitor  a caller with NO uploads must not receive ANY upload. This
                      catches the public-sentinel bug: an uploaded chunk carrying
                      owner="" is visible to everyone, and a check comparing only
                      sessions can be written so that reads as legitimate.

    A CONTROL CASE is included deliberately: B must still retrieve its OWN chunk.
    Without it a filter that returns nothing at all passes both isolation cases
    perfectly, and "isolated" becomes indistinguishable from "broken".
    """
    import numpy as np

    from ..index.hybrid import HybridIndex
    from ..ingest.document import Chunk, ChunkRole

    try:
        # THE SERVING PATH, not NumpyIndex.search_budget.
        #
        # This test used to call search_budget directly and passed while
        # HybridIndex.retrieve -- the method the API actually calls -- leaked,
        # because retrieve goes through ranked_ids -> search_k and the filter was
        # only on search_budget. A test that asserts the property on a path the
        # product does not take is worse than no test: it reports the guarantee
        # holding while the guarantee is absent where it matters.
        hyb = HybridIndex.load()
        ix = hyb.dense
    except Exception:  # noqa: BLE001 -- no index is NOT_MEASURED, not a failure
        return [Check(
            name="Upload isolation",
            rule="a session cannot retrieve another session's upload",
            observed="no index available", state="NOT_MEASURED", n=None,
            fingerprint=fp,
            detail="isolation is a property of a live index; there is none to query",
        )]

    rng = np.random.default_rng(20260829)
    dim = ix.vectors.shape[1]
    probes: dict[str, "np.ndarray"] = {}
    added: list[Chunk] = []
    for owner in ("iso-session-A", "iso-session-B"):
        v = rng.normal(size=dim).astype("float32")
        v /= np.linalg.norm(v)
        probes[owner] = v
        added.append(Chunk(
            chunk_id=f"{owner}-probe", source_id=f"{owner}.pdf", ordinal=0,
            role=ChunkRole.CHILD, owner=owner, origin="upload",
            embed_text=f"probe belonging to {owner}",
            display_text=f"probe belonging to {owner}",
        ))

    # In-memory copy only. Nothing is written to disk, so running the reconciler
    # cannot contaminate the index it is reporting on.
    ix.children = list(ix.children) + added
    ix.vectors = np.vstack([ix.vectors] + [probes[c.owner][None, :] for c in added])
    ix._owners = None

    # Rebuild the sparse leg so it sees the probes too: filtering only the dense
    # side would leave RRF and sparse-only mode leaking.
    from ..index.bm25 import BM25Index

    hyb.sparse = BM25Index(ix.children)
    hyb._by_id = {c.chunk_id: c for c in ix.children}

    def seen(owner):
        """What the SERVING path returns, in every retrieval mode."""
        out: set[str] = set()
        for mode in ("dense", "sparse", "rrf"):
            r = hyb.retrieve("probe belonging to iso-session-B",
                             probes["iso-session-B"], mode=mode,
                             token_budget=4000, unit="child", owner=owner)
            out |= {c.chunk_id for c in r.children}
        return out

    cross = "iso-session-B-probe" in seen("iso-session-A")
    public = bool({"iso-session-A-probe", "iso-session-B-probe"} & seen(None))
    own = "iso-session-B-probe" in seen("iso-session-B")

    return [
        Check(
            name="Upload isolation",
            rule="a session cannot retrieve another session's upload",
            observed=("A RETRIEVED B's chunk" if cross
                      else "A queried B's own vector, received nothing of B's"),
            state="FAILS" if cross else "HOLDS",
            n=2, fingerprint=fp,
            detail="the query used B's own embedding, the strongest possible pull",
        ),
        Check(
            name="Upload isolation (public)",
            rule="a caller with no uploads retrieves no upload at all",
            observed=("a public visitor RETRIEVED an upload" if public
                      else "public visitor received no uploaded chunk"),
            state="FAILS" if public else "HOLDS",
            n=2, fingerprint=fp,
            detail='catches an upload carrying owner="" -- the one value whose bug '
                   "is a leak rather than a miss",
        ),
        Check(
            name="Upload retrievability",
            rule="a session CAN retrieve its own upload",
            observed=("B retrieved its own chunk" if own
                      else "B could NOT retrieve its own chunk"),
            state="HOLDS" if own else "FAILS",
            n=1, fingerprint=fp,
            detail="the control: without it, a filter returning nothing would pass "
                   "both isolation checks and be useless",
        ),
    ]


def _contextual_prefix_check(fp: list[str]) -> list[Check]:
    """Can a model-written situating sentence reach a quotation? Answered by TRYING.

    THE PROPERTY THIS PROTECTS. A contextual prefix is written by a model, sits
    directly against document prose, and travels with the chunk's real page
    number. If it ever reached `verbatim_text`, the product would display
    invented text as a quotation from a source, with a citation that opens the
    right page -- indistinguishable from a real quote to anyone who has not read
    the document. That is the image-caption failure, arriving through a new door.

    WHY THIS IS EXECUTED AND NOT ASSERTED IN A COMMENT. The separation lives in
    one line of `_make_child`, and this project has now shipped four things that
    were correct, tested, and not on the request path. A property held in prose
    is a property nobody will notice losing.

    AND WHY THE POSITIONAL CHECK IS HERE TOO. `splitters` documents the
    prefix-then-slice bug at length: enrichment applied before slicing lands on
    child 0 alone, section openers become systematically more findable, nothing
    errors, and aggregate recall rises. The contextual prefix is the second
    enrichment to travel this path, so it gets the same check rather than the
    same comment.
    """
    from ..chunking import splitters as S
    from ..ingest.document import Block, ChunkKind, ChunkRole, Source, TextProvenance

    MARK = "RECONCILE-SITUATING-PROBE"
    body = " ".join(f"sentence{i} concerning the described method." for i in range(220))
    src = Source(source_id="reconcile-probe", uri="probe.pdf", doc_type="pdf",
                 content_hash="probe", title="Probe Document")
    blocks = [Block(kind=ChunkKind.TEXT, text="# Results\n\n" + body, page=1,
                    heading_path=("Results",))]

    chunks = S.build_chunks(src, blocks, owner="", origin="corpus",
                            contextualizer=lambda parent, piece: f"{MARK} {piece[:20]}")
    kids = [c for c in chunks if c.role is ChunkRole.CHILD]
    parents = [c for c in chunks if c.role is ChunkRole.PARENT]
    if len(kids) < 2:
        return [Check("Contextual prefix containment",
                      "a model-written prefix never reaches quotable text",
                      "probe produced fewer than 2 children", "NOT_MEASURED",
                      fingerprint=fp)]

    quotable = [MARK in ((c.verbatim_text or "") + c.display_text) for c in kids]
    quotable += [MARK in ((c.verbatim_text or "") + c.display_text + c.embed_text)
                 for c in parents]
    leaked = sum(quotable)
    with_prefix = sum(MARK in c.embed_text for c in kids)
    prov_ok = all(c.text_provenance is TextProvenance.PREFIXED for c in kids)

    plain = S.build_chunks(src, blocks, owner="", origin="corpus")
    fp_moved = plain[0].pipeline_fingerprint != chunks[0].pipeline_fingerprint

    return [
        Check(
            name="Contextual prefix containment",
            rule="a model-written prefix reaches embed_text and nothing quotable",
            observed=(f"{leaked} of {len(kids) + len(parents)} records carry it in "
                      "quotable text" if leaked else
                      "prefix present in embed_text only; quote and display are the body"),
            state="FAILS" if leaked else "HOLDS",
            n=len(kids) + len(parents), fingerprint=fp,
            detail=("provenance is PREFIXED on every child"
                    if prov_ok else "a child claimed VERBATIM while carrying a prefix"),
            why="model-written text against document prose with a real page number "
                "is a fabricated quotation that survives inspection",
        ),
        Check(
            name="Contextual prefix has no positional bias",
            rule="every child of a section gets its own prefix, not just child 0",
            observed=f"{with_prefix} of {len(kids)} children carry a prefix",
            state="HOLDS" if with_prefix == len(kids) else "FAILS",
            n=len(kids), fingerprint=fp,
            detail="prefix-then-slice would give child 0 alone the context and make "
                   "section openers findable for reasons unrelated to relevance -- "
                   "recall would RISE and the bias would read as a win",
        ),
        Check(
            name="Contextual prefix moves the fingerprint",
            rule="contextualised and plain indexes are not comparable, and say so",
            observed=("fingerprint differs with and without the contextualiser"
                      if fp_moved else "SAME fingerprint for two different indexes"),
            state="HOLDS" if fp_moved else "FAILS",
            n=2, fingerprint=fp,
            detail="without this the gate would compare a contextualised run against "
                   "a breadcrumb baseline and attribute the difference to the commit",
        ),
    ]


# Imported rather than restated: a second copy of 1500 would drift, and this
# check exists to catch exactly that class of disagreement.
from .run import HEADLINE_BUDGET as _HEADLINE_BUDGET  # noqa: E402


def _primary_artifact_check(fp: list[str]) -> list[Check]:
    """Does the headline artifact describe the index that actually ships?

    THE HAZARD THIS EXISTS FOR, twice observed. An experiment writes to a shared
    file and the record of the shipped system quietly becomes a record of the
    experiment. First the ingest manifest, when building a second index for an
    A/B repointed the shipped index's provenance at a different pipeline. Then
    eval_results.json, when the A/B's 24 runs left it describing a 250-token run
    against the wrong index. Both times nothing failed: the demo served, the UI
    rendered, and a document said something true about a system that was not the
    one running.

    Both are now structurally prevented -- index-scoped manifests, and an eval
    that writes only where it is told. This is the check that would have caught
    either WITHOUT knowing about it, which is the point: the prevention is
    specific to two causes and this is specific to the property.

    It compares the artifact against the INDEX ON DISK rather than against
    another artifact, because a sidecar describing the wrong index is exactly
    the failure being tested for.
    """
    from ..index.numpy_index import NumpyIndex

    ev = _load(config.DATA_EVAL / "eval_results.json")
    if not ev:
        return [Check("Headline artifact describes the shipped index",
                      "eval_results.json provenance == data/index/numpy_index",
                      "no eval_results.json", "NOT_MEASURED", fingerprint=fp)]
    try:
        kids = NumpyIndex.load("numpy_index").children
    except Exception as exc:  # noqa: BLE001
        return [Check("Headline artifact describes the shipped index",
                      "eval_results.json provenance == data/index/numpy_index",
                      f"index unreadable: {type(exc).__name__}", "NOT_MEASURED",
                      fingerprint=fp)]

    on_disk = sorted({c.pipeline_fingerprint for c in kids})
    claimed = (ev.get("index_provenance") or {}).get("pipeline_fingerprint")
    matches = len(on_disk) == 1 and on_disk[0] == claimed
    basis = (ev.get("index_provenance") or {}).get("child_cost_basis")
    budget = ev.get("token_budget")

    return [
        Check(
            name="Headline artifact describes the shipped index",
            rule="eval_results.json provenance == the chunks in data/index/numpy_index",
            observed=(f"both {claimed}" if matches
                      else f"artifact says {claimed}, index carries {on_disk}"),
            state="HOLDS" if matches else "FAILS",
            n=len(kids), fingerprint=fp,
            detail=f"budget {budget}, child cost basis {basis!r}",
            why="an experiment writing to a shared file has twice turned the record "
                "of the shipped system into a record of the experiment, silently",
        ),
        Check(
            name="Headline artifact was measured at the headline budget",
            rule=f"token_budget == {_HEADLINE_BUDGET}",
            observed=f"{budget}",
            state=("HOLDS" if budget == _HEADLINE_BUDGET
                   else "FAILS"),
            n=1, fingerprint=fp,
            detail="a stray run at another budget leaves a headline nobody quoted "
                   "on purpose",
        ),
        Check(
            name="Headline artifact is complete",
            rule="the headline artifact carries every section the UI renders",
            observed=(f"budget_sweep has {len(ev.get('budget_sweep') or {})} points, "
                      f"per_item {len(ev.get('per_item') or [])} rows"),
            state=("HOLDS" if (ev.get("budget_sweep") and ev.get("per_item"))
                   else "FAILS"),
            n=1, fingerprint=fp,
            detail="a --no-sweep run writes a VALID artifact with an empty sweep, so "
                   "the recall-vs-budget chart renders blank while every other check "
                   "on this page passes -- found by reading the live page, not the file",
            why="the first version of this invariant checked the artifact's identity "
                "and not its completeness, and passed an artifact whose chart was empty",
        ),
        Check(
            name="Headline artifact used the current cost basis",
            rule=f"child_cost_basis == {config.CHILD_COST_BASIS!r}",
            observed=f"{basis!r}",
            state="HOLDS" if basis == config.CHILD_COST_BASIS else "FAILS",
            n=1, fingerprint=fp,
            detail="two bases answer different questions under the same field "
                   "names -- see A-13",
        ),
    ]


def _refuse_vacuous_passes(checks: list[Check]) -> list[Check]:
    """A check that passed over an empty population did not pass. It abstained.

    THE PASS THAT PRODUCED THIS. Every invariant here was written from a specific
    incident, and each therefore tests what that incident violated. The two
    "headline artifact" checks asserted the artifact's IDENTITY -- right index,
    right budget, right cost basis -- because identity was what the two known
    incidents had corrupted. An artifact with an empty budget sweep satisfied all
    of them, and the Inspector's recall chart rendered blank while every check
    was green.

    A check written from the last bug tests the last bug. So this is one pass
    over the whole list asking a different question: *what would a valid-but-
    useless version of this look like?* The answers were uncomfortable:

      Index parity            0 indexed == 0 embedded, 0 dropped. An EMPTY INDEX
                              passes perfectly.
      Parent resolution       0 orphans, because there are no children.
      Uniform provenance      one provenance among zero chunks is one provenance.
      Scoring sanity          partial 0 >= strict 0. An eval that scored nothing.
      Context budget          delivered 0 <= budget. **A retrieval system that
                              returns nothing satisfies this at every budget.**

    That last one is the isolation control case, one file over and not applied:
    `Upload retrievability` exists precisely because a filter returning nothing
    passes both isolation tests, and the same reasoning had never been pointed at
    the budget invariant.

    The fix is not to make these FAIL. An empty index is not a broken index, it
    is an absent one, and NOT_MEASURED already means exactly that -- "an absent
    value invites investigation, an invented one ends it". So a HOLDS over a
    population of zero is downgraded, with the population named, and the
    reconciler's summary stops counting it as evidence.
    """
    out: list[Check] = []
    for c in checks:
        if c.state == "HOLDS" and not c.n:
            c.state = "NOT_MEASURED"
            c.detail = (f"population is {c.n!r} -- this passed over nothing. "
                        f"Downgraded from HOLDS: a check with no subjects "
                        f"abstained, it did not hold. " + c.detail)
        out.append(c)
    return out


def _upload_additivity_check(fp: list[str]) -> list[Check]:
    """Can adding one session's file remove somebody else's? Answered by trying it.

    WRITTEN BEFORE THE CODE IT GUARDS, because the failure mode was known in
    advance and is catastrophic rather than degraded.

    `ingest(files=[one_upload])` looks exactly like the right call and would
    DELETE THE CORPUS. `Manifest.plan` computes removals as
    `set(records) - present_ids`, so naming one file declares every other
    document absent and purges it -- dense index, sparse index, parent store,
    assets, caches. The convenience argument and the delete detector share one
    notion of "what is present"; for a full corpus walk that is correct, and for
    a subset it is a corpus-wiping bug in the shape of a filter.

    So the invariant is not "ingest_upload works". It is that the destructive
    path is **not reachable from it**: the function must never consult the
    planner, and the corpus walker must never see the uploads directory.
    """
    import inspect

    from .. import pipeline as PL
    from ..ingest import loaders as L

    src = inspect.getsource(PL.ingest_upload)
    touches_planner = any(t in src for t in (".plan(", "purge_source", "deleted_sources"))

    # The other half, and the reason it is a separate assertion: even a perfect
    # ingest_upload does not help if a CORPUS ingest walks into the uploads
    # directory, because that run DOES delete and DOES own everything publicly.
    walked = L.corpus_files()
    up = config.DATA_UPLOADS.resolve()
    leaked = [str(f) for f in walked if str(f.resolve()).startswith(str(up))]

    # The manifest's own view of who may KNOW a document exists, which is a
    # different question from who may retrieve its text.
    from ..ingest.document import Manifest

    # Does anything the demo permits actually run the TTL purge?
    import inspect as _insp

    from .. import sessions as _sess  # noqa: F401  (named for the reader)

    ttl_callers = set()
    try:
        import app.api as _api
        for name in ("upload",):
            fn = getattr(_api, name, None)
            if fn and "purge_expired" in _insp.getsource(fn):
                ttl_callers.add(f"/api/documents ({name})")
    except Exception:  # noqa: BLE001 -- no web layer installed is NOT_MEASURED-ish
        ttl_callers = {"(web layer not importable)"}

    # THE ROUND TRIP, asserted rather than remembered. Build a record with every
    # field set to a distinguishable value, save, load, compare.
    import dataclasses as _dc
    import tempfile as _tmp

    from ..ingest.document import DocType, Source, SourceRecord

    rt_fields = [f.name for f in _dc.fields(SourceRecord) if f.name != "source"]
    probe = SourceRecord(
        source=Source(source_id="rt-probe", uri="rt.pdf", doc_type=DocType.PDF,
                      content_hash="h", title="rt"),
        chunk_ids=["c1"], parent_ids=["p1"], asset_paths=["a1"],
        cache_keys=["k1"], n_uncontextualized=3, owner="rt-owner",
    )
    with _tmp.TemporaryDirectory() as td:
        path = Path(td) / "m.json"
        m = Manifest(path)
        m.records["rt-probe"] = probe
        m.save()
        back = Manifest(path).records.get("rt-probe")
    dropped = [] if back is None else [
        f for f in rt_fields if getattr(probe, f) != getattr(back, f, None)
    ]
    if back is None:
        dropped = ["<record did not survive at all>"]

    recs = Manifest().records
    n_upload = sum(1 for r in recs.values() if (getattr(r, "owner", "") or ""))
    n_public = len(recs) - n_upload
    mislabelled = [
        sid for sid, r in recs.items()
        if r.source.doc_type and not (getattr(r, "owner", "") or "")
        and str(getattr(r.source, "uri", "")).replace("\\", "/").find("/uploads/") >= 0
    ]

    return [
        Check(
            name="Session ingest cannot delete",
            rule="ingest_upload never reaches the manifest planner",
            observed=("it references the planner" if touches_planner
                      else "no plan(), purge_source() or deleted_sources() on this path"),
            state="FAILS" if touches_planner else "HOLDS",
            n=1, fingerprint=fp,
            detail="Manifest.plan derives deletions from absence, so a subset "
                   "ingest declares the rest of the corpus absent",
            why="the obvious one-line fix for the upload dead-end would have "
                "purged every document in the index",
        ),
        Check(
            name="The TTL promise has an enforcer",
            rule="purge_expired() is reached from a route the demo permits",
            observed=(f"called from {', '.join(sorted(ttl_callers))}" if ttl_callers
                      else "purge_expired() is only reachable from a DENIED route"),
            state="HOLDS" if ttl_callers else "FAILS",
            n=len(ttl_callers), fingerprint=fp,
            detail="/api/status tells every visitor their upload is deleted after "
                   f"{config.UPLOAD_TTL_SECONDS // 3600}h; the only caller used to "
                   "be POST /api/sessions/sweep, which a demo refuses outright",
            why="a stated guarantee with no enforcer is worse than an absent "
                "feature -- the absence is visible and the broken promise is not",
        ),
        Check(
            name="Manifest round-trips every field",
            rule="_load(save(x)) preserves every field of SourceRecord",
            observed=(f"DROPPED on save/load: {sorted(dropped)}" if dropped
                      else f"all {len(rt_fields)} fields survive"),
            state="FAILS" if dropped else "HOLDS",
            n=len(rt_fields), fingerprint=fp,
            detail="save() and _load() are a hand-written pair, so a field added "
                   "to the dataclass round-trips only if BOTH are edited",
            why="`owner` was set correctly at ingest and silently dropped on save, "
                "so the sidebar filter read \"\" for every record and showed every "
                "visitor every upload -- while looking like a working filter",
        ),
        Check(
            name="Manifest records who may see a document",
            rule="every upload-owned source record carries a non-public owner",
            observed=(f"{len(mislabelled)} upload records with a public owner: "
                      f"{mislabelled[:2]}" if mislabelled else
                      f"{n_upload} upload record(s), all session-owned; "
                      f"{n_public} public"),
            state="FAILS" if mislabelled else "HOLDS",
            n=n_upload + n_public, fingerprint=fp,
            detail="the documents sidebar reads the manifest, not the index, so "
                   "chunk-level ownership does not reach it -- a filename is "
                   "content, and `Q3-redundancies.pdf` discloses its subject "
                   "without a byte of its text",
            why="every visitor saw the filename, title and chunk count of every "
                "other visitor's upload",
        ),
        Check(
            name="Corpus walker cannot see uploads",
            rule="corpus_files() returns nothing under the uploads directory",
            observed=(f"LEAKED {len(leaked)}: {leaked[:2]}" if leaked
                      else f"{len(walked)} corpus files, none under {up.name}/"),
            state="FAILS" if leaked else "HOLDS",
            n=len(walked), fingerprint=fp,
            detail="a corpus ingest deletes by absence and owns publicly, so a "
                   "stranger's file inside its walk becomes public and permanent",
        ),
    ]


def _upload_reachability_check(fp: list[str]) -> list[Check]:
    """Can an uploaded file be read WITHOUT retrieval? Answered by trying it.

    WRITTEN BEFORE THE FIX, and it failed on the first run -- which is the only
    way to know a check is load-bearing rather than decorative.

    THE GAP IT EXISTS FOR. Ownership was enforced where retrieval happens: the
    `-inf` mask inside `_scores`, verified three ways by the isolation
    invariants above. All three ask the same question -- can another session
    RETRIEVE this? None asks whether another session can simply ASK FOR THE FILE.

    Measured, before the fix: a visitor with no cookie, who had never uploaded
    anything, fetched another session's PDF in full with
    `GET /api/asset?path=data/raw/<name>` -- 200, every byte. The containment
    check resolved a relative path against the PROCESS WORKING DIRECTORY rather
    than against the corpus root, so it refused the app's own asset URLs
    (`assets/x.png` -> 403) while admitting paths that happened to be relative to
    the CWD. Wrong in both directions at once.

    So the guarantee was real inside the index and absent at the door, and a day
    of work on the index could not see that -- the same shape as an invariant
    that passes on a path the product does not take, one layer further out.

    Two things are asserted, and the second is the one that generalises:
      1. the containment check accepts what the UI generates
      2. NO PATH under the uploads directory is reachable through it at all --
         because uploads no longer live where this endpoint can see them
    """
    # THE ENDPOINT'S OWN FUNCTION, imported. The first draft of this check
    # re-implemented the containment logic and passed against its own copy while
    # the endpoint was broken in two directions -- this project's oldest failure,
    # very nearly shipped inside a check written to catch another instance of it.
    #
    # Imported from `ragkit`, not from `app.api`: the first version of this line
    # reached into the web layer and made the eval harness require FastAPI, which
    # crashed reconcile in any environment installed without the server.
    from ..corpus_paths import resolve_asset

    root = config.DATA_RAW.resolve()

    def reachable(raw: str) -> bool:
        return resolve_asset(raw) is not None

    ui_form = reachable("assets/hnsw_p2.png")
    escapes = [r for r in ("../../.env", "../../../etc/passwd", "/etc/passwd")
               if reachable(r)]
    up = config.DATA_UPLOADS.resolve()
    # The uploads directory must not be INSIDE the servable root. Containment
    # then refuses it by construction rather than by a rule someone maintains --
    # the same reasoning as putting the owner filter in _scores().
    uploads_outside = not str(up).startswith(str(root) + os.sep) and up != root

    return [
        Check(
            name="Asset endpoint serves what the UI asks for",
            rule="a relative asset path resolves under the corpus root",
            observed=("'assets/hnsw_p2.png' resolves inside the root" if ui_form
                      else "the app's own asset URL is REFUSED by its own guard"),
            state="HOLDS" if ui_form else "FAILS",
            n=1, fingerprint=fp,
            detail="it resolved relative paths against the process working "
                   "directory, so it refused legitimate URLs and admitted "
                   "whatever happened to sit under the CWD",
        ),
        Check(
            name="Asset endpoint refuses traversal",
            rule="no path escapes the corpus root",
            observed=(f"ESCAPED: {escapes}" if escapes else
                      "traversal attempts refused"),
            state="FAILS" if escapes else "HOLDS",
            n=3, fingerprint=fp,
            detail="a path parameter that reaches the filesystem is untrusted input",
        ),
        Check(
            name="Uploads are unreachable by file path",
            rule="the uploads directory is not inside the servable corpus root",
            observed=(f"uploads at {up.name}/, servable root is {root.name}/"
                      if uploads_outside else
                      "UPLOADS LIVE INSIDE THE SERVABLE ROOT"),
            state="HOLDS" if uploads_outside else "FAILS",
            n=1, fingerprint=fp,
            detail="closing the route rather than filtering it: an endpoint that "
                   "cannot address a directory needs no rule about who may read it",
            why="a visitor with no cookie fetched another session's PDF in full "
                "with GET /api/asset?path=data/raw/<name>",
        ),
    ]


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
            # NOT_MEASURED when the loader repaired nothing: a biconditional
            # over two zeros is satisfied and has tested nothing. The label this
            # exists to protect only exists when there is a repair to label.
            state=("FAILS" if bad else "HOLDS" if total_rep else "NOT_MEASURED"),
            n=total_lab,
            fingerprint=fp,
            detail="; ".join(bad) if bad else
                   ("a repaired block may split into several children, so >= is the "
                    "invariant, not ==" if total_rep else
                    "nothing was repaired in this ingest, so there was no label to "
                    "propagate and nothing to check"),
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
        sweep = evalr.get("budget_sweep") or {}
        over = [
            b for b, h in sweep.items()
            if h["mean_child_tokens"] > int(b) or h["mean_parent_tokens"] > int(b)
        ]
        checks.append(Check(
            name="Context budget",
            rule="delivered <= budget, every unit, every budget",
            observed=(f"mean child {head['mean_child_tokens']}, parent "
                      f"{head['mean_parent_tokens']}, budget {budget}"
                      + (f", over {len(sweep)} budgets" if sweep else
                         " -- NO SWEEP TO CHECK")),
            # An empty sweep is not a pass. The list comprehension above is empty
            # when the sweep is, and `not over` was reading that as compliance --
            # the same defect as the headline artifact rendering a blank chart
            # while reporting green, in a different check, found in the same pass.
            state=("FAILS" if over else "HOLDS" if sweep else "NOT_MEASURED"),
            n=head["child_strict"]["n"] if sweep else 0,
            fingerprint=fp,
            detail=("over budget at: " + ", ".join(over)) if over else
                   ("a unit allowed to overshoot is simply given more text" if sweep
                    else "the artifact carries no budget sweep, so nothing was checked"),
            why="an `and hits` clause gave the parent unit 378 tokens against a 250 budget",
        ))
        # THE CONTROL CASE, and its absence was the sharper half of the finding.
        #
        # "delivered <= budget" is satisfied perfectly by delivering NOTHING, at
        # every budget, forever. Exactly the reasoning behind `Upload
        # retrievability` -- which exists because a filter returning nothing
        # passes both isolation tests -- never pointed at this invariant.
        #
        # Strict fill legitimately returns nothing at a tight budget, so the
        # control has to be stated where that is not a defence: at the LARGEST
        # budget measured, both units must deliver something.
        top = max(sweep, key=int) if sweep else None
        if top:
            th = sweep[top]
            delivered = (th["mean_child_tokens"], th["mean_parent_tokens"])
            checks.append(Check(
                name="Context budget delivers something",
                rule=f"at the largest measured budget ({top}), both units deliver > 0 tokens",
                observed=f"child {delivered[0]}, parent {delivered[1]} tokens at {top}",
                state="HOLDS" if all(d > 0 for d in delivered) else "FAILS",
                n=int(top), fingerprint=fp,
                detail="the control for the check above: a retriever that returns "
                       "nothing never exceeds its budget, so 'delivered <= budget' "
                       "alone cannot distinguish compliance from silence",
                why="the same gap that made Upload retrievability necessary, "
                    "unapplied here until a pass over every check went looking for it",
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

    # Isolation is executed, not asserted: two probe uploads are added to an
    # in-memory copy of the index and actually queried. Nothing is written to disk.
    checks.extend(_isolation_check(fp))
    checks.extend(_contextual_prefix_check(fp))
    checks.extend(_primary_artifact_check(fp))
    checks.extend(_upload_reachability_check(fp))
    checks.extend(_upload_additivity_check(fp))
    checks = _refuse_vacuous_passes(checks)

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
