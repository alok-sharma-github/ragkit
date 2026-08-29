"""
FastAPI layer. Thin on purpose: no retrieval, chunking or verification logic
lives here.

    uv run uvicorn app.api:app --reload --port 8000

WHY THIN. Everything goes through `ragkit.pipeline`, which is the same entry
point the CLI and the eval harness use. A web layer that reassembled the stages
itself would be a THIRD program that could disagree with the other two -- and this
project has already paid for that once, when a survey script imported a different
parser than the pipeline and reported table counts differing 5x on identical
bytes. One entry point is not tidiness, it is the fix.

WHAT THE UI NEEDS THAT THE CLI DOES NOT: per-claim citation provenance shaped for
rendering. The design's three colours map onto fields we already store:

    blue   EXACT QUOTE                             verified against verbatim_text
    stone  DOCUMENT TEXT, STRUCTURE INFERRED       text_source=page_text_clip
    amber  THE ASSISTANT'S READING OF A CHART      text_provenance=MODEL_GENERATED
    grey   SAID EARLIER IN THIS CONVERSATION       not implemented -- see below

`grey` is deliberately absent rather than faked. It requires per-claim
attribution of documents vs conversation, which the generator does not produce
yet. Rendering an unattributed claim in the "from your documents" colour would be
the exact failure this whole citation layer exists to prevent, so the API reports
`conversation_attribution: "unsupported"` and the UI must not draw that state.
"""

from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal

import re
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragkit import (budget, citations_index, config, conversations,
                    feedback as fb, gemini, jobs, pipeline, sessions,
                    upload_guard)
from ragkit.eval import reconcile as R
from ragkit.generate import answer as A
from ragkit.index.fusion import explain as explain_fusion
from ragkit.index.hybrid import HybridIndex
from ragkit.retrieve.query import condense
from ragkit.ingest.document import (PUBLIC_OWNER, Chunk, ChunkRole, Manifest,
                                    Source, TextProvenance)

app = FastAPI(title="RAGkit", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    # The dev origins stay unconditionally, so configuring a deployment can never
    # break local work -- a CORS failure presents as an opaque browser error with
    # a perfectly healthy backend behind it, and that is an expensive hour.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   *config.CORS_EXTRA_ORIGINS],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# PUBLIC DEMO GUARD
#
# DENY BY METHOD, ALLOW BY EXPLICIT PATTERN. The alternative -- a check at the
# top of each write endpoint -- was rejected because it fails OPEN: the next
# endpoint anyone adds is unprotected until its author remembers, and the author
# who forgets is the one who did not read this comment. Here a new POST or DELETE
# is refused until someone deliberately adds it below, which is the same
# fail-closed rule the eval harness holds itself to.
#
# What stays open is exactly what the demo IS: asking questions, and the feedback
# capture that makes a wrong answer reportable. Both are metered. What closes is
# anything that spends quota in bulk or destroys state -- see config.DEMO_MODE
# for why those two are not merely untidy but destructive.
_DEMO_ALLOWED_WRITES = (
    re.compile(r"^/api/ask$"),
    re.compile(r"^/api/conversations$"),
    re.compile(r"^/api/conversations/[^/]+/ask$"),
    re.compile(r"^/api/feedback$"),
    # UPLOAD IS PRODUCT USE, NOT CORPUS MUTATION -- the same distinction that
    # caused a live outage when this middleware gated /api/ask by method and
    # returned 401 to every visitor asking a question.
    #
    # Without this line, flipping DEMO_MODE off would not have opened uploads to
    # a visitor: it would have moved them from "read-only demo" (403, honest) to
    # "write password required" (401, a door they cannot open). Every piece
    # behind the flag was tested; the SEQUENCE was not, and the sequence is what
    # a visitor experiences.
    #
    # Anchored exactly. `^/api/documents$` matches the POST and cannot match
    # `/api/documents/{source_id}`, so DELETE stays refused -- which is the whole
    # reason this is a widened exemption rather than DEMO_MODE=0.
    *((re.compile(r"^/api/documents$"),) if config.DEMO_ALLOW_UPLOADS else ()),
)
# Only the paths that call Gemini are metered. Creating a conversation writes a
# few bytes; generating an answer spends a shared, unpublished free-tier budget.
_METERED = (
    re.compile(r"^/api/ask$"),
    re.compile(r"^/api/conversations/[^/]+/ask$"),
    # An upload spends embeddings and image captions, so it is metered like an
    # answer. Left unmetered it would be the one paid route a visitor could call
    # without limit -- the guard-coverage failure, arriving through the door
    # that was just opened.
    re.compile(r"^/api/documents$"),
)
_hits: dict[str, deque] = defaultdict(deque)


def _client(request: Request) -> str:
    """Best-effort client identity behind a platform proxy.

    X-Forwarded-For is spoofable, so this is a courtesy meter and NOT a security
    boundary -- it stops one enthusiastic visitor from draining the day's quota,
    which is the actual failure being prevented. Anything stronger needs
    authentication, and a login wall on a portfolio demo defeats its purpose.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


WRITE_COOKIE = "ragkit_write"
# The unlock route itself must be reachable without the token, or there is no way
# to present one. Everything else is gated.
_UNLOCK_PATH = re.compile(r"^/api/unlock$")


def _write_token_ok(request: Request) -> bool:
    """Does this request carry the write password?

    Header or cookie. The header is for scripts, the cookie for a browser that
    unlocked once.

    compare_digest, not `==`. String equality returns as soon as it finds a
    differing byte, so the time it takes leaks how much of a guess was correct --
    enough to recover a secret one character at a time. The cost of doing this
    right is one import.
    """
    import secrets as _secrets

    want = config.WRITE_TOKEN
    if not want:
        return True
    got = request.headers.get("x-ragkit-write-token") or request.cookies.get(WRITE_COOKIE) or ""
    return _secrets.compare_digest(got, want)


@app.middleware("http")
async def demo_guard(request: Request, call_next):
    path, method = request.url.path, request.method.upper()
    is_write = method in ("POST", "PUT", "PATCH", "DELETE")

    # THE WRITE PASSWORD, checked before DEMO_MODE returns early.
    #
    # Ordering matters and the obvious order is wrong. `if not config.DEMO_MODE:
    # return await call_next(...)` used to be the first line, so every non-demo
    # deployment skipped the whole middleware -- and a customer deployment is
    # non-demo by definition. Adding the token check after that early return
    # would have gated exactly the deployments that already deny writes and none
    # of the ones that accept them.
    #
    # That is the "guarded on some of the traffic" failure again, and it would
    # have been invisible: the demo would still refuse writes, the local dev box
    # would still accept them, and only a customer deployment -- the one case
    # nobody tests before there is a customer -- would be open.
    # WHAT THE PASSWORD PROTECTS: corpus MUTATION, not product USE.
    #
    # This read `if is_write and ...`, and every product action is a POST --
    # so setting a write token on the public demo returned 401 for /api/ask and
    # the demo could not answer a question at all. Shipped, and caught by the
    # post-deploy check rather than by me.
    #
    # The local test is why it got through: I asserted the paths that must be
    # DENIED (ingest, upload, delete) and never asserted that the ALLOWED ones
    # still work once a token exists. Testing the deny path is half a guard.
    #
    # The allowlist already names product use exactly -- ask, feedback,
    # conversations. Everything else that writes is mutation. So both mechanisms
    # now key off the same set instead of two overlapping notions of "write".
    is_mutation = is_write and not any(p.match(path) for p in _DEMO_ALLOWED_WRITES)

    # THE CATEGORICAL REFUSAL WINS. On a demo, mutation is denied outright -- no
    # password exists that would permit it -- so answering 401 "write password
    # required" would tell a visitor to go and find a key to a door that is
    # bricked up. Checked before the token so the more fundamental reason is the
    # one reported.
    if is_mutation and config.DEMO_MODE:
        return JSONResponse(
            status_code=403,
            content={
                "detail": "read-only demo",
                "why": "this deployment shares one Gemini key, so uploads, "
                       "re-ingest and deletion are disabled -- a single ingest "
                       "would exhaust the day's budget for everyone else",
                "what_you_can_do": "ask questions of the indexed documents, or "
                                   "clone the repo and run it against your own "
                                   "key to use the full pipeline",
            },
        )

    if is_mutation and not _UNLOCK_PATH.match(path) and not _write_token_ok(request):
        return JSONResponse(
            status_code=401,
            content={
                "detail": "write password required",
                "why": "this deployment requires a shared write password for "
                       "uploads, re-ingest and deletion",
                "what_you_can_do": "POST /api/unlock with {\"token\": \"...\"} to "
                                   "unlock this browser, or send the "
                                   "X-RAGkit-Write-Token header",
            },
        )

    if not config.DEMO_MODE:
        return await call_next(request)

    if is_write:
        if is_mutation:
            # 403 with the REASON, not a bare status. A demo that silently lacks
            # upload reads as unfinished; one that says why it is read-only reads
            # as operated. Same principle as every degradation notice here.
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "read-only demo",
                    "why": "this deployment shares one free-tier Gemini key, so "
                           "uploads, re-ingest and deletion are disabled -- a "
                           "single ingest would exhaust the day's quota for "
                           "everyone else",
                    "what_you_can_do": "ask questions of the indexed documents, or "
                                       "clone the repo and run it against your own "
                                       "key to use the full pipeline",
                },
            )

        if any(p.match(path) for p in _METERED):
            # Counted BEFORE the handler runs, so a malformed request still spends
            # one of the visitor's allowance. Slightly unfair and deliberately the
            # conservative direction: metering only successful generations means a
            # flood of requests that error inside the handler -- after the Gemini
            # call -- costs real quota while the meter reads zero.
            now = time.time()
            window = config.DEMO_RATE_LIMIT_WINDOW_SECONDS
            q = _hits[_client(request)]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= config.DEMO_RATE_LIMIT_REQUESTS:
                retry = int(window - (now - q[0])) + 1
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                    content={
                        "detail": "demo rate limit reached",
                        "why": f"{config.DEMO_RATE_LIMIT_REQUESTS} questions per "
                               f"{window // 60} minutes per visitor, so one "
                               "session cannot spend the shared free-tier quota",
                        # Ceiling, not floor-plus-one. `retry // 60 + 1` turned a
                        # full 3600s window into "about 61 minutes", which is the
                        # kind of detail that makes a reader distrust every other
                        # number on the page.
                        "what_you_can_do": f"try again in about {(retry + 59) // 60} "
                                           "minute(s), or run it locally with your "
                                           "own key",
                    },
                )
            q.append(now)

    return await call_next(request)

# One index in memory. Loading it per request would add ~100ms and, worse, let
# two requests disagree about the corpus mid-conversation.
_HYBRID: HybridIndex | None = None


def hybrid() -> HybridIndex:
    global _HYBRID
    if _HYBRID is None:
        try:
            _HYBRID = HybridIndex.load()
        except FileNotFoundError as exc:
            raise HTTPException(503, "no index yet -- run: ragkit.cli ingest") from exc
    return _HYBRID


# --------------------------------------------------------------------------
# Evidence kinds -- the design's citation taxonomy, derived not asserted
# --------------------------------------------------------------------------

EvidenceKind = Literal["quoted", "structure_inferred", "assistant_reading",
                       "found_not_quoted", "conversation"]


def evidence_kind(chunk: Chunk, quote_status: str) -> EvidenceKind:
    """Which of the design's four evidence states this citation is.

    Derived from stored provenance rather than guessed from the text, which is
    what makes the colour trustworthy: `amber` means MODEL_GENERATED, full stop.
    """
    if chunk.text_provenance is TextProvenance.MODEL_GENERATED:
        return "assistant_reading"
    if "page_text_clip" in (chunk.text_source or ""):
        return "structure_inferred"
    if quote_status == "verified":
        return "quoted"
    # In the source and retrieved, but the quote did not verify. The design's
    # "FOUND -- NOT QUOTED: it's in your documents. I won't quote it." Declining
    # is the honest state; presenting it as a quotation would not be.
    return "found_not_quoted"


EVIDENCE_LABEL: dict[str, str] = {
    "quoted": "Quoted exactly from your documents",
    "structure_inferred": "Document text, structure inferred by the assistant",
    "assistant_reading": "Read from a chart by the assistant -- not a quotation",
    "found_not_quoted": "Found in your documents, but not quoted",
    "conversation": "Said earlier in this conversation -- not from your documents",
}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    budget: int | None = Field(default=None, ge=200, le=32000)
    sources: int = Field(default=6, ge=1, le=20)
    mode: Literal["dense", "sparse", "rrf"] = "dense"
    history: str = ""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


class Unlock(BaseModel):
    token: str = Field(min_length=1, max_length=512)


@app.post("/api/unlock")
def unlock(body: Unlock) -> JSONResponse:
    """Exchange the write password for a cookie, so a browser asks once.

    Deliberately NOT a login: there is one password, no identity, and no session
    record. See config.WRITE_TOKEN for why that is the right shape for a
    single-customer deployment and the wrong shape the moment two tenants share
    one.

    Returns the same 401 body as the middleware on a wrong token, so this endpoint
    cannot be used to distinguish "wrong password" from "no password configured".
    """
    import secrets as _secrets

    want = config.WRITE_TOKEN
    if not want:
        # No gate configured. Saying so is not a leak -- every write already
        # succeeds without a token, so a caller learns nothing they could not
        # learn by trying one.
        return JSONResponse(
            status_code=200,
            content={"unlocked": True, "note": "no write password is configured "
                                               "on this deployment"},
        )
    if not _secrets.compare_digest(body.token, want):
        return JSONResponse(status_code=401, content={"detail": "write password required"})

    r = JSONResponse(status_code=200, content={"unlocked": True})
    r.set_cookie(
        WRITE_COOKIE, want,
        # httponly: script on the page cannot read it, so an XSS cannot exfiltrate
        # the password. samesite=strict: it is not sent on cross-site requests, so
        # another origin cannot drive a write with the visitor's cookie.
        httponly=True, samesite="strict", secure=True, max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return r


def session_owner(request: Request) -> str | None:
    """The owner to retrieve as, from an untrusted cookie. None = public only.

    `sessions.resolve` does the validating: shape, plus an identity check against
    the public sentinel. Anything that fails becomes None, which means "public
    corpus only" -- the safe direction. A bad cookie degrades a visitor to the
    shared demo; it never promotes them into somebody else's documents.
    """
    return sessions.resolve(request.cookies.get(sessions.COOKIE))


def _set_session_cookie(response: JSONResponse, session_id: str) -> None:
    """Issue the session cookie.

    HttpOnly is the load-bearing flag. The session id IS the secret -- 128 bits,
    unguessable -- so the only realistic way it leaks is a script on the page
    reading document.cookie. Unguessability protects against everything except a
    value someone can simply read.
    """
    response.set_cookie(
        sessions.COOKIE, session_id,
        httponly=True, samesite="lax", secure=True,
        max_age=config.UPLOAD_TTL_SECONDS, path="/",
    )


@app.get("/api/status")
def status() -> dict[str, Any]:
    rep_path = config.DATA_EVAL / "index_report.json"
    rep = json.loads(rep_path.read_text("utf-8")) if rep_path.exists() else {}
    m = Manifest()
    docs = []
    for sid, rec in sorted(m.records.items()):
        docs.append({
            "source_id": sid,
            "title": rec.source.title,
            "doc_type": rec.source.doc_type.value,
            "pages": rec.n_pages,
            "chunks": len(rec.chunk_ids),
            "tables": rec.n_tables_detected,
            "continuation_suspects": rec.n_continuation_suspects,
            # The design's states. Our ingest is synchronous, so nothing is ever
            # PREPARING or QUEUED -- reporting those would be inventing a
            # capability. INCOMPLETE is real, and it now has two causes: a
            # skipped image caption (that part of the document is unretrievable
            # at all) or a skipped contextual prefix (that part is indexed with
            # weaker retrieval text than its neighbours). The second is milder
            # and still worth saying, because a document where only SOME
            # children got the paid treatment is not comparable with one where
            # all of them did.
            "state": "SEARCHABLE_INCOMPLETE" if rec.n_uncontextualized else "READY",
        })
    return {
        "documents": docs,
        "tombstones": m.tombstones,
        "index": {
            "children": rep.get("n_children_indexed"),
            "parents": rep.get("n_parents"),
            "dim": rep.get("dim"),
            "by_kind": rep.get("child_kind"),
            "by_text_source": rep.get("child_text_source"),
            "provenance_populations": rep.get("provenance_populations"),
            "parser_version": rep.get("parser_version"),
            "chunker_version": rep.get("chunker_version"),
            # WHICH ENRICHMENT BUILT THIS INDEX: "breadcrumb-only" or
            # "llm-prefix@1". Surfaced because two indexes with the same parser
            # and chunker are still different systems if one of them paid for a
            # situating sentence per chunk, and the inspector's whole job is to
            # let a reader check a number against the thing that produced it.
            "contextualizer": rep.get("contextualizer"),
            "n_contextualized": rep.get("n_contextualized"),
            "contextualization": rep.get("contextualization"),
            "pipeline_fingerprint": rep.get("pipeline_fingerprint"),
        },
        "models": gemini.resolve_models(),
        # PROBED, not configured. Built as `capabilities()` and then wired
        # nowhere -- the fifth instance of "written, correct, not in the traffic".
        # It belongs here because the UI should be able to say WHY a tier behaves
        # the way it does (e.g. thinking cannot be disabled on the cheap model).
        "capabilities": gemini.capabilities(),
        "degradations": rep.get("degradations") or [],
        # The empty-state promise from the design, stated by the API so the UI
        # cannot overclaim: there is no general knowledge here.
        "scope_note": "Answers come only from the indexed documents. "
                      "If it is not in them, the answer is 'not in your documents'.",
        # Reported so the UI can DISABLE the upload affordance and explain it,
        # rather than offering a control that 403s. A button that fails when
        # pressed is worse than a button that is absent with a reason beside it.
        # The ceiling, reported BEFORE it refuses anything. A limit a caller only
        # discovers by hitting it is indistinguishable from a bug, and on a
        # customer deployment this is the number they need to plan an upload
        # against.
        "budget": budget.snapshot().to_json(),
        "demo": {
            "read_only": config.DEMO_MODE,
            # SEPARATE FROM read_only, because one boolean cannot answer two
            # questions. The UI gated BOTH the upload control and the delete
            # control on `read_only`, so opening uploads by flipping that flag
            # would have re-enabled deletion in the same motion. Uploads are on;
            # deleting the shared corpus is still refused.
            "uploads_enabled": config.DEMO_ALLOW_UPLOADS,
            "upload_limits": (
                f"PDF only, up to {config.MAX_UPLOAD_PAGES} pages and "
                f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Your upload is "
                f"visible only to you and is deleted after "
                f"{config.UPLOAD_TTL_SECONDS // 3600} hours."
            ) if config.DEMO_ALLOW_UPLOADS else "",
            "why": ("this deployment shares one free-tier Gemini key, so "
                    "re-ingest and delete are disabled" if config.DEMO_MODE else ""),
            "rate_limit": (f"{config.DEMO_RATE_LIMIT_REQUESTS} questions per "
                           f"{config.DEMO_RATE_LIMIT_WINDOW_SECONDS // 60} minutes"
                           if config.DEMO_MODE else ""),
        },
    }


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> dict[str, Any]:
    hyb = hybrid()
    budget = req.budget or config.TOKENS_CONTEXT_BUDGET

    import time
    t: dict[str, float] = {}

    # CONDENSE FIRST. Everything before generation sits inside TTFT, and this is
    # the layer that decides whether retrieval runs at all -- so it has to run
    # before the embed, not alongside it.
    t0 = time.time()
    cq = condense(req.question, req.history)
    t["condense"] = (time.time() - t0) * 1000

    parents: list[Chunk] = []
    retrieved = None
    if cq.needs_retrieval:
        t0 = time.time()
        qv = gemini.embed_query(cq.search_query)
        t["embed"] = (time.time() - t0) * 1000

        t0 = time.time()
        # WHO IS ASKING reaches retrieval. Without this the filter exists and is
        # never given anyone to filter for, which is the same "correct but not on
        # the request path" failure that let retrieve() leak in the first place.
        retrieved = hyb.retrieve(cq.search_query, qv, mode=req.mode,
                                 token_budget=budget, unit="parent",
                                 owner=session_owner(request))
        parents = retrieved.parents[: req.sources]
        t["retrieve"] = (time.time() - t0) * 1000
    # else: route is conversation_only or ask_fresh. Retrieval is SKIPPED, not
    # run-and-ignored -- the guide's retrieval-necessity classification saves the
    # latency AND removes the irrelevant-context degradation that comes from
    # retrieving for "summarise that".

    t0 = time.time()
    if cq.route == "ask_fresh":
        # No answer attempted. The design's honest refusal: the conversation
        # cannot answer it, and a standalone question can. Fabricating an answer
        # from unrelated retrieved passages would be worse than saying so.
        ans = A.Answer(
            text="", abstained=True,
            abstain_reason=(cq.read_as or "this needs a standalone question") +
                           (f" Try asking: {cq.suggested_fresh_question}"
                            if cq.suggested_fresh_question else ""),
            reconciliation={"chunks_sent": 0},
        )
    elif not cq.needs_retrieval:
        # conversation_only: answer from what was said, labelled as such.
        ans = A.answer_from_conversation(cq.original, req.history)
    else:
        ans = A.answer(
            cq.original, parents, history=req.history,
            starved=bool(retrieved and retrieved.starved_by_budget),
            min_unit_tokens=retrieved.min_unit_tokens if retrieved else None,
            context_budget=budget,
        )
    t["generate"] = (time.time() - t0) * 1000

    claims: list[dict[str, Any]] = []
    checks_by_label: dict[int, list[A.CitationCheck]] = {}
    for c in ans.checks:
        checks_by_label.setdefault(c.label, []).append(c)

    for claim in ans.claims:
        claim_source = (claim.get("source") or "documents").strip() or "documents"
        cits = []
        for cit in claim.get("citations") or []:
            label = cit.get("label")
            chunk = ans.sources.get(label) if isinstance(label, int) else None
            check = next(
                (c for c in ans.checks
                 if c.label == label and c.quote == (cit.get("quote") or "")),
                None,
            )
            kind = evidence_kind(chunk, check.quote_status) if chunk and check else None
            cits.append({
                "label": label,
                "quote": cit.get("quote"),
                "chunk_id": chunk.chunk_id if chunk else None,
                "evidence_kind": kind,
                "evidence_label": EVIDENCE_LABEL.get(kind or "", ""),
                "quote_status": check.quote_status if check else None,
                "fabricated": check.fabricated if check else True,
                "overlap": round(check.overlap, 3) if check else 0.0,
                "detail": check.detail if check else "citation not found in the response",
                "locator": A.locator(chunk) if chunk else None,
            })
        claims.append({
            "text": claim.get("text", ""),
            "source": claim_source,
            # The design's grey state, now earned rather than guessed: the model
            # declared this claim as conversation-derived, so it renders grey and
            # carries no document colour.
            "evidence_kind": "conversation" if claim_source == "conversation" else None,
            "citations": cits,
        })

    mix = {k: 0 for k in EVIDENCE_LABEL}
    for cl in claims:
        if cl["source"] == "conversation":
            mix["conversation"] += 1
        for c in cl["citations"]:
            if c["evidence_kind"]:
                mix[c["evidence_kind"]] += 1

    return {
        "question": req.question,
        # The design shows the interpretation before the answer ("READ AS ...").
        # Surfacing the rewrite is what makes a bad retrieval diagnosable by the
        # user rather than mysterious: they can see the question we actually asked.
        "understanding": cq.to_json(),
        "answer_markdown": ans.text,
        "abstained": ans.abstained,
        "abstain_reason": ans.abstain_reason,
        # Prefer the answer layer's suggestion (it looked and found nothing) and
        # fall back to the condenser's, which in practice is never populated
        # because the ask_fresh route does not fire.
        "suggested_question": ans.suggested_question or cq.suggested_fresh_question,
        "grounded": ans.grounded,
        "reasons": ans.reasons,
        "claims": claims,
        "evidence_mix": mix,
        "reconciliation": ans.reconciliation,
        "citation_integrity": R.citation_checks(ans.reconciliation).to_json(),
        # Now supported: the model declares per claim whether it came from the
        # documents or from the conversation, so the grey state is a declaration
        # rather than an inference.
        "conversation_attribution": "per_claim",
        # Built from Chunk.citation() rather than reassembled by hand. The hand
        # version duplicated fields the model already knows how to emit, which is
        # two places to drift -- and Chunk.citation() had no caller at all, so the
        # highlightable/quote logic it carries was dead.
        "sources": [
            {"label": i, **p.citation(), "text_source": p.text_source,
             "provenance": p.text_provenance.value,
             "heading_path": list(p.heading_path)}
            for i, p in enumerate(parents, 1)
        ],
        "retrieval": {
            "mode": req.mode,
            "budget": budget,
            "ran": retrieved is not None,
            "skipped_because": None if retrieved is not None else cq.route,
            "children_considered": len(retrieved.children) if retrieved else 0,
            "parents_delivered": len(parents),
            "child_tokens": retrieved.child_tokens if retrieved else 0,
            "parent_tokens": retrieved.parent_tokens if retrieved else 0,
            "leg_stats": retrieved.leg_stats if retrieved else None,
            # "a fusion you cannot explain is a fusion you cannot tune" -- which
            # was written in fusion.py and then never surfaced anywhere. Per-leg
            # ranks for the top results, so a surprising order is diagnosable.
            "fusion_explain": (
                explain_fusion(retrieved.fused, top=6)
                if retrieved and retrieved.fused else None
            ),
        },
        "usage": {"prompt_tokens": ans.usage.prompt_tokens,
                  "output_tokens": ans.usage.output_tokens,
                  "cached_tokens": ans.usage.cached_tokens},
        "timings_ms": {k: round(v) for k, v in t.items()},
    }


@app.get("/api/source/{chunk_id}")
def source(chunk_id: str, quote: str = Query(default="")) -> dict[str, Any]:
    """One source, with the cited span located for highlighting."""
    hyb = hybrid()
    chunk = hyb.dense.parents.get(chunk_id) or hyb._by_id.get(chunk_id)
    if chunk is None:
        # A citation can outlive its document. 410 GONE with the tombstone is a
        # different answer from 404 NOT FOUND, and the difference matters: one
        # says "this was removed", the other says "this never existed". Collapsing
        # them would make a correct old answer look like a fabricated citation.
        tomb = Manifest().tombstone_for_chunk(chunk_id)
        if tomb:
            raise HTTPException(
                410,
                {
                    "reason": "source_removed",
                    "source_id": tomb.get("source_id"),
                    "deleted_at": tomb.get("deleted_at"),
                    "message": "The answer stays as it was given. This citation can "
                               "no longer be opened or verified -- its source was removed.",
                },
            )
        raise HTTPException(404, f"no chunk {chunk_id}")

    span = A.locate_span(quote, chunk.display_text) if quote else None
    kind = evidence_kind(chunk, "verified" if span else "absent")
    note = ""
    if kind == "structure_inferred":
        note = ("This is the page's own text. The table's row and column structure "
                "was lost in extraction and inferred by the assistant, so "
                "value-to-label pairing is not guaranteed.")
    elif kind == "assistant_reading":
        note = ("This is the assistant's description of an image, not text from "
                "the document. Nothing here is quotable.")

    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "locator": A.locator(chunk),
        "page": chunk.page,
        "page_end": chunk.page_end,
        "kind": chunk.kind.value,
        "text_source": chunk.text_source,
        "provenance": chunk.text_provenance.value,
        "evidence_kind": kind,
        "evidence_label": EVIDENCE_LABEL.get(kind, ""),
        "note": note,
        "heading_path": list(chunk.heading_path),
        "text": chunk.display_text,
        "verbatim": chunk.verbatim_text,
        "highlight": {"start": span[0], "end": span[1]} if span else None,
        "asset_url": (f"/api/asset?path={chunk.asset_path}" if chunk.asset_path else None),
        "table_header_missing": chunk.table_header_missing,
        "table_continuation_suspect": chunk.table_continuation_suspect,
    }


@app.get("/api/asset")
def asset(path: str) -> FileResponse:
    """Serve an extracted image, confined to the corpus directory.

    resolve() then check containment: without it, `path=../../.env` is a file
    read. A path parameter that reaches the filesystem is untrusted input.
    """
    root = config.DATA_RAW.resolve()
    p = (root / Path(path).name if Path(path).is_absolute() else Path(path)).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(403, "outside the corpus directory") from None
    if not p.exists():
        raise HTTPException(404, str(path))
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")


@app.get("/api/inspector")
def inspector() -> dict[str, Any]:
    """Everything the Inspector view renders, read from artifacts on disk."""
    def load(name: str) -> Any:
        p = config.DATA_EVAL / name
        return json.loads(p.read_text("utf-8")) if p.exists() else None

    hybrid_cmp = None
    p = config.EXPERIMENTS_OUT / "e05_hybrid_rrf.json"
    if p.exists():
        hybrid_cmp = json.loads(p.read_text("utf-8"))

    evalr = load("eval_results.json")
    baseline = load("baseline.json")
    from ragkit.eval import judge as J
    from ragkit import deferred as D

    judged = None
    jp = config.DATA_EVAL / "judged_results.json"
    if jp.exists():
        jd = json.loads(jp.read_text("utf-8"))
        judged = {k: v for k, v in jd.items() if k != "rows"}

    return {
        "reconciliation": R.reconcile(),
        # The tier-two gate, always reported -- including when it withholds.
        # "no judged numbers yet, and here is exactly why" is information; a
        # missing section would read as an oversight.
        "judge_gate": J.gate(),
        "judged": judged,
        "deferred": D.review(),
        "eval": None if not evalr else {
            "token_budget": evalr["token_budget"],
            "headline": evalr["metrics"]["headline"],
            "scope_label": __import__("ragkit.eval.metrics", fromlist=["x"]).scope_label(
                evalr["metrics"]
            ),
            "stratum_coverage": evalr["metrics"]["stratum_coverage"],
            "by_stratum": evalr["metrics"]["by_stratum"],
            "by_anchor": evalr["metrics"]["by_anchor"],
            "budget_sweep": evalr.get("budget_sweep"),
            "regression_tests": evalr.get("regression_tests"),
            "golden_set": evalr["golden_set"],
            "index_provenance": evalr["index_provenance"],
            "seconds": evalr["seconds"],
        },
        "baseline": None if not baseline else {
            "headline": baseline["metrics"]["headline"],
            "index_provenance": baseline["index_provenance"],
        },
        "hybrid_comparison": hybrid_cmp,
        "goldenset_report": load("goldenset_report.json"),
    }


# --------------------------------------------------------------------------
# Documents: upload, ingest, remove
# --------------------------------------------------------------------------

# Extensions with a loader. Anything else is rejected at upload rather than
# accepted and silently never indexed -- a file that appears in the corpus and
# answers nothing is worse than a refusal, because the user believes it is there.
ALLOWED = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _invalidate_index() -> None:
    """Drop the cached index so the next request reloads it.

    Called after any ingest or removal. Without this the process keeps answering
    from the pre-change index while /api/status reports the new one -- two
    representations disagreeing, which is this project's recurring failure.
    """
    global _HYBRID
    _HYBRID = None


@app.post("/api/documents")
async def upload(request: Request,
                 files: list[UploadFile] = File(...)) -> JSONResponse:
    """Accept files into the corpus. Does NOT index them -- call /api/ingest.

    Two steps rather than one, deliberately: upload is fast and per-file, ingest
    is slow and corpus-wide (it rebuilds the index). Fusing them would make a
    5-file upload run five full rebuilds.
    """
    session = sessions.STORE.get_or_create(session_owner(request))
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        name = Path(f.filename or "").name          # strip any path from the client
        ext = Path(name).suffix.lower()
        if not name or ext not in ALLOWED:
            rejected.append({"name": name or "(unnamed)", "reason":
                             f"no loader for '{ext}' -- supported: {sorted(ALLOWED)}"})
            continue
        data = await f.read()

        # QUARANTINE FIRST, THEN INSPECT, THEN ADMIT.
        #
        # The file has to exist on disk to be inspected -- the probe is a separate
        # process and takes a path -- but it must not land in DATA_RAW before it
        # is judged, because anything in DATA_RAW is a candidate for the next
        # ingest. Writing straight to the corpus and deleting on rejection means
        # there is a window in which a refused file is indexable, and windows like
        # that are how a rejected document ends up searchable.
        quarantine = config.DATA_RAW.parent / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        staged = quarantine / f"{uuid.uuid4().hex[:12]}{ext}"
        staged.write_bytes(data)

        try:
            verdict = upload_guard.check_upload(staged)
            if not verdict.ok:
                # The visitor-facing sentence, not the code. See upload_guard for
                # why: this reader cannot raise a page cap, so the message has to
                # name the limit AND what would work instead.
                rejected.append({
                    "name": name,
                    "reason": verdict.visitor_message,
                    "code": verdict.code,
                    "pages": verdict.pages,
                })
                continue
            dest = config.DATA_RAW / name
            dest.write_bytes(data)
        finally:
            # The quarantine copy goes whether the file was admitted or refused.
            staged.unlink(missing_ok=True)
        saved.append({"name": name, "bytes": len(data),
                      "source_id": Source.normalize_id(dest)})
    # THE UPLOAD BELONGS TO A SESSION, NOT THE CORPUS.
    #
    # This is the line that closes the gap the upload guard did not: the guard
    # bounds the PARSER, ownership bounds WHO CAN SEE THE RESULT. Until an
    # uploaded file was ingested under a session id, a perfectly safe PDF from a
    # stranger still landed in the shared public corpus -- validated, and visible
    # to the next visitor.
    if saved:
        for row in saved:
            sessions.STORE.note_document(session.session_id, row["source_id"])

    body = {"saved": saved, "rejected": rejected,
            "session": session.to_json(),
            "next": "POST /api/ingest to make these searchable"}

    resp = JSONResponse(content=body)
    # Issued on upload rather than on first visit: a visitor who only asks
    # questions of the public corpus needs no session at all, and handing an
    # identifier to someone who does not need one is a cost with no benefit.
    _set_session_cookie(resp, session.session_id)
    return resp


@app.post("/api/sessions/sweep")
def sweep_sessions() -> dict[str, Any]:
    """Purge expired upload sessions and everything they added.

    An endpoint rather than a background thread, deliberately: a thread that
    deletes documents is a thing running with no request to attribute it to and
    no place for its failures to surface. This returns what it removed and what
    it could not, so a failed purge is visible rather than retried silently
    forever.

    The deletion itself is `pipeline.remove_source` -- the same path a manual
    delete takes. Two ways to delete is how "gone from one store, still in
    another" happens.
    """
    return sessions.purge_expired()


@app.get("/api/sessions/me")
def my_session(request: Request) -> dict[str, Any]:
    """What this browser's session holds, if it has one.

    Reports the PUBLIC state for a visitor with no session rather than 404ing:
    "you have uploaded nothing" is the true answer, and it is what the UI needs
    to decide whether to offer a purge control.
    """
    owner = session_owner(request)
    s = sessions.STORE.get(owner) if owner else None
    return {
        "has_session": s is not None,
        "session": s.to_json() if s else None,
        "ttl_seconds": config.UPLOAD_TTL_SECONDS,
        "corpus": "public documents only" if s is None
                  else "public documents plus your uploads",
    }


@app.post("/api/ingest")
def start_ingest(caption_images: bool = Query(default=True)) -> dict[str, Any]:
    """Queue a corpus ingest. Returns immediately with a job id."""
    def work(job: jobs.Job) -> dict[str, Any]:
        def progress(stage: str, cur: int, total: int, detail: str) -> None:
            jobs.STORE.update(job, stage=stage, current=cur, total=total, detail=detail)
        res = pipeline.ingest(caption_images=caption_images, verbose=False,
                              owner=PUBLIC_OWNER, origin="corpus",
                              on_progress=progress)
        _invalidate_index()
        return {
            "sources": res.n_sources, "children": res.n_children,
            "parents": res.n_parents, "seconds": round(res.seconds, 1),
            "provenance_ok": res.provenance_ok,
            "provenance_problems": res.provenance_problems,
            "degradations": res.degradations,
            "per_file": res.per_file,
            "plan": {k: (len(v) if isinstance(v, list) else v)
                     for k, v in res.plan.items()},
        }

    resp = JSONResponse(content=body)
    # Issued on upload rather than on first visit: a visitor who only asks
    # questions of the public corpus needs no session, and handing out an
    # identifier to someone who does not need one is a cost with no benefit.
    _set_session_cookie(resp, session.session_id)
    return resp

@app.get("/api/documents/{source_id:path}/impact")
def removal_impact(source_id: str) -> dict[str, Any]:
    """What removing this document would affect. Changes nothing.

    THE CONFIRM SCREEN NEEDS THIS AND COULD NOT USE DELETE. That endpoint returns
    a `will_remove` preview *and queues the removal in the same call*, so its
    "preview" describes what it has already started -- useless for a dialog whose
    entire purpose is to be shown before the user commits.

    A GET, deliberately: it is safe under the read-only demo guard, so the
    consequences of a delete are visible on the public deployment even though the
    delete itself is refused. Showing what a destructive action would do is not a
    destructive action.
    """
    m = Manifest()
    if source_id not in m.records:
        raise HTTPException(404, f"no indexed document '{source_id}'")
    rec = m.records[source_id]
    return {
        "source_id": source_id,
        "will_remove": {
            "chunks": len(rec.chunk_ids),
            "parents": len(rec.parent_ids),
            "assets": len(rec.asset_paths),
            "cache_entries": len(rec.cache_keys),
        },
        "cited_in": citations_index.impact(source_id).to_json(
            current_fingerprint=_current_fingerprint()
        ),
    }


@app.delete("/api/documents/{source_id:path}")
def remove(source_id: str, purge_cache: bool = Query(default=True)) -> dict[str, Any]:
    """Remove a document and everything derived from it. Queued, because it reindexes."""
    m = Manifest()
    if source_id not in m.records:
        raise HTTPException(404, f"no indexed document '{source_id}'")
    rec = m.records[source_id]
    # WHAT ALREADY-ANSWERED QUESTIONS THIS TOUCHES.
    #
    # The preview counted chunks, parents, assets and cache entries -- everything
    # derived from the document -- and said nothing about the conversations that
    # had already cited it. Those are the part a user actually recognises: "you
    # asked about this in three turns" is a reason to hesitate, where "1,247 cache
    # entries" is not.
    #
    # Reported per TURN as well as per conversation, because Turn carries its own
    # pipeline_fingerprint and a conversation can span fingerprint changes. Turns
    # cited under a pipeline no longer in use are counted separately rather than
    # pooled -- they are real citations and a weaker claim about the current index,
    # and collapsing the two would overstate the blast radius.
    imp = citations_index.impact(source_id)
    fp = _current_fingerprint()
    preview = {
        "source_id": source_id,
        "chunks": len(rec.chunk_ids),
        "parents": len(rec.parent_ids),
        "assets": len(rec.asset_paths),
        "cache_entries": len(rec.cache_keys),
        "cited_in": imp.to_json(current_fingerprint=fp),
    }

    def work(job: jobs.Job) -> dict[str, Any]:
        jobs.STORE.update(job, stage="removing", current=0, total=1, detail=source_id)
        out = pipeline.remove_source(source_id, purge_cache=purge_cache, verbose=False)
        _invalidate_index()
        return out

    job = jobs.STORE.submit("remove", work, source_id=source_id)
    return {"job": job.to_json(), "will_remove": preview,
            "note": "answers already given stay as they were; citations into this "
                    "document will report that its source was removed"}


@app.get("/api/jobs")
def list_jobs(limit: int = Query(default=25, ge=1, le=200)) -> dict[str, Any]:
    return {
        "active": (j.to_json() if (j := jobs.STORE.active()) else None),
        "queued": len(jobs.STORE.queued()),
        "jobs": [j.to_json() for j in jobs.STORE.list(limit=limit)],
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.STORE.get(job_id)
    if job is None:
        raise HTTPException(404, job_id)
    return job.to_json()


@app.get("/api/tombstones")
def tombstones() -> dict[str, Any]:
    """Removed documents, and what they took with them."""
    return {"tombstones": Manifest().tombstones}


# --------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------


class ConvAsk(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    budget: int | None = Field(default=None, ge=200, le=32000)
    sources: int = Field(default=6, ge=1, le=20)
    mode: Literal["dense", "sparse", "rrf"] = "dense"


@app.get("/api/conversations")
def list_conversations(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    return {"conversations": conversations.STORE.list(limit=limit)}


@app.post("/api/conversations")
def create_conversation(title: str = Query(default="")) -> dict[str, Any]:
    return conversations.STORE.create(title).to_json()


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str) -> dict[str, Any]:
    c = conversations.STORE.get(cid)
    if c is None:
        raise HTTPException(404, cid)
    return c.to_json()


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str) -> dict[str, Any]:
    return {"deleted": conversations.STORE.delete(cid)}


@app.post("/api/conversations/{cid}/ask")
def conversation_ask(cid: str, req: ConvAsk) -> dict[str, Any]:
    """Ask within a conversation: history is supplied and the turn is persisted.

    The history comes from the STORE, not the client. A client-supplied history
    can be edited, and an answer condensed against a history nobody recorded is
    not reproducible -- the turn would be unexplainable afterwards.
    """
    c = conversations.STORE.get(cid)
    if c is None:
        raise HTTPException(404, cid)

    payload = ask(AskRequest(
        question=req.question, budget=req.budget, sources=req.sources,
        mode=req.mode, history=c.history(),
    ))

    u = payload["understanding"]
    turn = conversations.Turn(
        question=req.question,
        answer_markdown=payload["answer_markdown"],
        route=u["route"], header=u["header"], read_as=u["read_as"],
        search_query=u["search_query"], was_rewritten=u["was_rewritten"],
        retrieval_ran=payload["retrieval"]["ran"],
        abstained=payload["abstained"], grounded=payload["grounded"],
        claims=payload["claims"], sources=payload["sources"],
        reconciliation=payload["reconciliation"],
        evidence_mix=payload["evidence_mix"],
        timings_ms=payload["timings_ms"],
        pipeline_fingerprint=(
            hybrid().dense.meta.get("pipeline_fingerprint") or ""
        ),
    )
    c = conversations.STORE.append(cid, turn)
    payload["conversation"] = {
        "id": c.id, "title": c.title, "n_turns": len(c.turns),
        # The drift signal, per conversation. A rising share of
        # conversation_only turns means the system is increasingly answering from
        # itself, where no citation check can reach it.
        "drift": c.drift(),
    }
    return payload


# --------------------------------------------------------------------------
# Feedback -- the only channel by which ground truth enters from outside
# --------------------------------------------------------------------------


class FeedbackIn(BaseModel):
    verdict: Literal["not_in_source", "source_doesnt_say", "source_is_fine", "helpful"]
    conversation_id: str = ""
    turn_index: int = -1
    claim_index: int = -1
    claim_text: str = ""
    chunk_id: str = ""
    note: str = Field(default="", max_length=2000)


@app.post("/api/feedback")
def submit_feedback(item: FeedbackIn) -> dict[str, Any]:
    """Record a flag, together with what the SYSTEM believed at that moment.

    The client sends only the verdict and what it is about. Everything describing
    the system's own belief -- evidence kind, quote status, text source, table
    flags -- is looked up SERVER-SIDE from the chunk. A client-supplied verdict
    about the system's state could be stale or edited, and the whole value of this
    record is that it pairs an independent human judgment with an authoritative
    machine one.
    """
    rec = fb.Feedback(
        verdict=item.verdict,
        conversation_id=item.conversation_id,
        turn_index=item.turn_index,
        claim_index=item.claim_index,
        claim_text=item.claim_text[:500],
        note=item.note,
        chunk_id=item.chunk_id,
    )

    if item.chunk_id:
        hyb = hybrid()
        chunk = hyb.dense.parents.get(item.chunk_id) or hyb._by_id.get(item.chunk_id)
        if chunk is not None:
            rec.source_id = chunk.source_id
            rec.text_source = chunk.text_source
            rec.evidence_kind = evidence_kind(chunk, "verified")
            rec.table_header_missing = chunk.table_header_missing
            rec.table_continuation_suspect = chunk.table_continuation_suspect
        else:
            # The chunk is gone -- likely a removed document. Recorded as such
            # rather than dropped: a flag on a tombstoned source is still a
            # signal, and silently discarding it would bias the feedback set
            # toward documents that happen to still exist.
            tomb = Manifest().tombstone_for_chunk(item.chunk_id)
            rec.source_id = (tomb or {}).get("source_id", "")
            rec.evidence_kind = "source_removed"

    # The turn's recorded state is authoritative for quote_status and claim
    # source, because that is what the user actually saw.
    if item.conversation_id and item.turn_index >= 0:
        conv = conversations.STORE.get(item.conversation_id)
        if conv and 0 <= item.turn_index < len(conv.turns):
            turn = conv.turns[item.turn_index]
            rec.pipeline_fingerprint = turn.pipeline_fingerprint
            if 0 <= item.claim_index < len(turn.claims):
                claim = turn.claims[item.claim_index]
                rec.claim_source = claim.get("source", "")
                for c in claim.get("citations") or []:
                    if c.get("chunk_id") == item.chunk_id or not item.chunk_id:
                        rec.quote_status = c.get("quote_status") or ""
                        rec.evidence_kind = c.get("evidence_kind") or rec.evidence_kind
                        break

    path = fb.record(rec)
    return {"recorded": rec.to_json(), "path": str(path),
            "label": fb.VERDICT_LABEL[item.verdict]}


@app.get("/api/feedback")
def feedback_stats() -> dict[str, Any]:
    """Agreement between the system's belief and human judgment."""
    return fb.stats()


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})

# ---------------------------------------------------------------------------
# THE BUILT FRONTEND, SERVED BY THIS PROCESS
#
# Mounted LAST, deliberately. A catch-all at "/" registered before the API routes
# would swallow them, and the failure looks like the backend is broken rather
# than like a routing order mistake -- so the mount goes at the bottom of the
# file where nothing can be added after it by accident.
#
# One service rather than two. A separate static host would mean two
# deployments, a cross-origin hop to configure, and a second thing that can be
# asleep when someone clicks the link. For a demo whose entire job is to work on
# the first click, that is two extra ways to fail for no benefit.
if config.WEB_DIST_DIR:
    _dist = Path(config.WEB_DIST_DIR)   # `Path` is what this module imports
    if not (_dist / "index.html").exists():
        # Loud, at startup, not a 404 per request. A missing bundle means the
        # frontend build step did not run -- a deploy-time mistake, and the
        # cheapest moment to learn about it is now.
        raise RuntimeError(
            f"RAGKIT_WEB_DIST={config.WEB_DIST_DIR} has no index.html. "
            "Run `npm run build` in app/web, or unset the variable to run "
            "API-only."
        )

    # html=True serves index.html at the mount root. It does NOT rewrite unknown
    # paths to index.html -- Starlette 404s those -- and that is correct here:
    # this UI has no client-side router (screens are `useState<Screen>`), so the
    # whole app lives at "/" and there are no deep links to preserve. Verified:
    # "/" is 200, "/inspector" is 404, and /api/* is untouched by the mount.
    #
    # If a router is ever added, this needs an explicit catch-all that returns
    # index.html for non-/api paths. Noting it here because the symptom then is a
    # 404 on a link that works in dev, which is a confusing hour otherwise.
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
