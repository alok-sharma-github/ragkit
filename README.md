# RAGkit

A multi-format RAG system (PDF / DOCX / images) built from scratch on Gemini, with
a **two-tier evaluation harness** and a **judge validated against hand labels**.

**Live demo:** _(URL added on deploy — read-only; see [Running it](#running-it))_

The pipeline is the easy half. What this repo is actually about is the part that is
usually missing: knowing whether the thing works, being able to say how you know,
and being honest about where it does not.

---

## What it measures, and what those numbers mean

Two metric tiers, kept separate so a regression can be localised to retrieval or
to generation rather than blamed on "the RAG."

### Retrieval

| metric | value |
|---|---|
| `child_strict` recall @1500-token budget | **78/92 = 85%** |
| `source_hit` (right document surfaced) @every budget | **92/92 = 100%** |
| NDCG@5 — dense vs RRF | 0.848 [0.761, 0.907] vs 0.880 [0.798, 0.932] |
| RRF on `exact_identifier` queries @500 tokens | 68% → **89%** |

`source_hit = 100%` with `child_strict = 85%` is the useful finding: the correct
document is *always* retrieved, so every remaining failure is ranking **within** a
document. That localises the next piece of work precisely, and it is why no
reranker was added — see [Decisions](#decisions-and-the-conditions-that-reverse-them).

Recall is reported **budget-normalised** rather than as recall@k. Comparing a
child unit against a parent unit at "k=10" compares 3,000 tokens of context
against 12,000; at a fixed token budget the comparison is fair. The fill rule is
strict — it never exceeds the budget even when that returns nothing.

Confidence intervals are Wilson, and any rate with n < 10 is printed as counts
(`1 of 2`) rather than a percentage.

### Generation

| metric | value |
|---|---|
| faithfulness, **over answers only** | 74/74 = 100% |
| questions the system declined to answer | 17 of 91 |
| abstention rate, `table_or_image` stratum | **10/37 = 27%** |
| judge agreement with hand labels (Cohen's kappa) | **0.897** (raw 0.933, chance 0.353, n=30) |

Read those first two rows together. **100% faithfulness is not a quality claim.**
The generator is constrained to the supplied sources and a deterministic citation
checker already fails closed on unverifiable quotes before the judge sees
anything, so this number mostly confirms the citation layer works. Its value is
as a regression detector.

The first version of this metric reported "99% supported" over a pooled
population — until it turned out **17 of the 91 rows were abstentions**, each
scored `supported` because an answer that asserts nothing asserts nothing false.
A faithfulness score that improves when the system answers fewer questions is
measuring the wrong direction. Splitting them out surfaced the real finding: a
**27% decline rate on table and figure questions.**

All 17 abstentions were then checked individually by re-retrieving and testing
needle presence. **Zero had their full evidence in the delivered context** (9
absent, 8 partial, 1 starved by the token budget). So the model abstains exactly
when evidence is incomplete and never when it is complete — which is the desired
behaviour, and localises all 17 to the retrieval tier.

### Is the judge trustworthy?

kappa = 0.897 against 30 hand-labelled items, on a **class-balanced** sample
(13 partly / 10 supported / 7 unsupported). That licenses *"this judge can tell
the three classes apart."* It does **not** license a precision claim on the real
population, which is nearly all-supported — the sample is a discrimination
estimate, and the code records it as such rather than quoting the number bare.

No judged metric is emitted at all until that gate passes. An unvalidated judge
is not a weaker signal, it is an unknown one.

The judge also runs a different model (`gemini-3.1-pro-preview`) than the
generator (`gemini-3.7-flash`), and the API refuses to report a score if those
collide — a model grading its own output is not evidence.

---

## What it does

- **Ingest** PDF, DOCX and images. Tables are detected rather than stitched;
  page-spanning tables are *flagged* as continuation suspects instead of being
  silently joined, because a wrong join is undetectable downstream while a flag
  is actionable.
- **Parent-document retrieval** — embeds a small precise child, returns the
  larger parent for generation.
- **Hybrid retrieval** — dense embeddings + BM25 written by hand (verified
  against `rank_bm25`) fused with Reciprocal Rank Fusion, with per-leg
  contribution explainable per query.
- **Cited answers** with per-claim citations, each carrying a verbatim quote that
  is checked against the source. Four outcomes are distinguished: verified,
  absent, **unquotable**, and fabricated — a fabricated citation and an
  unsupported claim are different failures with different fixes.
- **Grounded abstention.** "Not in your documents" is a correct answer and is
  scored as one.
- **Provenance per chunk.** Every chunk records whether its text is verbatim,
  prefixed, or model-generated, and the UI colours evidence accordingly — so a
  model-written image caption can never be presented as quoted source text.
- **Free-tier degradation channel.** Every capability lost to a quota limit is
  reported with five required fields: stage, cause, impact, fallback, remedy.
  Notices aggregate ("40 of 300"), never flood.
- **An inspector UI** exposing the reconciliation table, recall-vs-budget curves,
  per-stratum coverage with explicit `NOT_MEASURED` states, and the judge gate
  showing *why* it is withholding a number.

---

## Running it

### The hosted demo is read-only, on purpose

It shares a single free-tier Gemini key, so upload, re-ingest and delete return
403 with the reason stated, and questions are rate-limited per visitor. Without
that, one `curl` to `POST /api/ingest` exhausts the day's quota for everyone, and
`DELETE /api/documents/{id}` — which purges the embedding cache by default —
empties the demo permanently.

Concurrent visitors may still hit Gemini free-tier 429s. When that happens the
app says so, with the cause named.

### Locally, with the full pipeline

```bash
uv sync --extra web
cp .env.example .env          # paste a free key: https://aistudio.google.com/apikey

uv run python -m ragkit.cli models     # resolve model IDs against your key
uv run python -m ragkit.cli ingest     # parse, chunk, caption, embed, index
uv run python -m ragkit.cli ask "what is the RRF k parameter set to, and why?"
```

The prebuilt index is committed, so `ask` works immediately — `ingest` is only
needed if you change the corpus or the parser.

```bash
# the eval harness
uv run python -m ragkit.cli eval          # both metric tiers, per-stratum
uv run python -m ragkit.cli reconcile     # invariants: HOLDS / FAILS / NOT_MEASURED
uv run python -m ragkit.cli judge gate    # why judged metrics are or are not emitted
uv run python -m ragkit.cli deferred      # deferred decisions + expiry status

# the UI (two terminals, dev mode)
uv run uvicorn app.api:app --port 8000
cd app/web && npm install && npm run dev
```

`ask` exits non-zero when an answer is ungrounded or abstained, and `ingest` exits
non-zero when the provenance cross-check fails — so a caller never has to parse
prose to discover the citations did not verify.

**Note:** use `npm run build` to typecheck the frontend. `npx tsc --noEmit` exits
0 on code the build rejects (the root config uses project references).

### Docker

```bash
docker build -t ragkit .
docker run -p 8000:8000 -e GEMINI_API_KEY=... ragkit   # http://localhost:8000
```

The image builds the frontend and serves it from FastAPI — one service, one URL,
no CORS hop. `RAGKIT_DEMO_MODE=1` is the image default, so a deployment that
forgets to set it is still safe.

---

## Decisions, and the conditions that reverse them

`ragkit/deferred.py` stores each deferred decision as a **predicate that expires
itself** when its precondition changes. Run `ragkit deferred` for live status.

| decision | why | reverses when |
|---|---|---|
| No reranker | `source_hit` is already 100% at every budget; dense@3000 is 96% | the `aggregative` / `ambiguous` strata are measured and `multi_hop` reaches n≥10 |
| numpy exact cosine, no ANN | at 814 chunks exact search *is* correct, and it is the oracle any future ANN recall must be measured against | corpus > ~100k chunks, or p95 > 100ms (currently 5–20ms) |
| Two deterministic citation checks, no NLI | character overlap provably cannot separate an honest rewording from an invented quote (measured: fabrication 0.53, rewording 0.48) | the judge passes its kappa gate |
| Heading breadcrumbs, no LLM chunk prefixes | the free version has to be beaten, not an empty baseline | recall plateaus below 90% with `source_hit` at 1.0 — **this has now fired** |
| Per-stage timings, no OpenTelemetry | the numbers that inform a decision are already returned per request | more than one process needs to correlate a trace |

The fourth row is the interesting one: it expired on its own, without anyone
remembering to check.

---

## Known limitations

[**FUTURE_SCOPE.md**](FUTURE_SCOPE.md) is a full accounting — six verified unfixed
defects with costs, five deferred decisions with live expiry predicates, what was
planned and never built, and five places the harness is deliberately silent.

The headline ones:

- **23% of embedded text contains HTML markup** (`<sup>`, `<mark>`, `<br>`) that
  the PDF parser passes through. Measured, unfixed: the fix changes
  `PARSER_VERSION`, forcing a full re-embed and invalidating every number above.
  Fixing it and re-measuring must happen together or not at all.
- **Two of seven strata are unmeasured** (`aggregative`, `ambiguous`). So "85%
  recall" means 85% on five of seven strata, and the UI marks the other two
  `NOT_MEASURED` rather than rendering an ambiguous blank.
- **Table and figure retrieval is the weak subsystem** — 27% of those questions
  are declined for want of complete evidence.
- **One deferral predicate is wrong and fails open**, keeping a deferral alive
  past its own expiry.

---

## Repo map

```
ragkit/
  config.py          every knob, with the reasoning for its value
  gemini.py          the only place the API is called; capabilities are PROBED, not assumed
  limits.py          the degradation channel — five required fields per notice
  ingest/            loaders (PDF/DOCX/image), table detection, provenance, manifest
  chunking/          header-aware parent/child splitting + prefix-leak assertions
  index/             numpy exact cosine, hand-written BM25, RRF, hybrid budget fill
  retrieve/query.py  history-conditioned condensation + retrieval-necessity routing
  generate/answer.py cited generation, quote location, citation verification
  eval/              golden set, metrics, judge + kappa gate, invariant reconciler
  deferred.py        deferred decisions as self-expiring predicates
app/
  api.py             FastAPI, 19 endpoints, read-only demo guard
  web/               React + TypeScript + Tailwind inspector UI
```

---

## On how this was built

This started as a guided build against a study guide
([rag-learning-material.md](rag-learning-material.md) — module tags `M1`–`M16` in
the code point back to it), with design questions answered before each module was
written. Several of those answers changed the architecture materially: kappa as a
*gate* rather than a reported number, provenance tracked per chunk, flagging
page-spanning tables instead of stitching them, budget-normalised recall, and
auditing reachability from entry points rather than by counting references.

The code carries that reasoning inline. Comments explain why a value is what it
is and, where something was wrong first, what the wrong version did — because the
bug that keeps recurring in this codebase is not a crash. It is **the operation
succeeding, the output being well-formed, and something being wrong anyway**:
a check whose predicate merely correlates with its condition, a module that is
correct but off the request path, a reference standard that is itself mistaken.
Every instance was found by *running* something, never by reading it.
