# RAGkit — future scope

Build stopped 2026-08-26. This records what is **not** done, why, and what it would
cost — separated by kind, because "a bug we found and left" and "a feature we
decided against" are different things and mixing them makes both unreadable.

Every number here was produced by running something. Where a thing is unmeasured
it says so rather than estimating.

---

## 0. Where the build actually got to

Working end to end: ingest (PDF / DOCX / image) → header-aware parent-document
chunking → Gemini embeddings → hybrid retrieval (dense + hand-written BM25 + RRF)
→ cited generation with deterministic citation verification → FastAPI (19
endpoints) → React inspector UI. Plus a golden-set eval harness with two metric
tiers, a validated LLM judge, and an invariant reconciler.

**Measured today** — 814 passages from 15 sources:

| tier | metric | value |
|---|---|---|
| retrieval | `child_strict` @1500 tok | 78/92 = **85%** |
| retrieval | `source_hit` @every budget | 92/92 = **100%** |
| retrieval | dense vs RRF (NDCG) | 0.848 [0.761, 0.907] vs 0.880 [0.798, 0.932] |
| retrieval | RRF on `exact_identifier` @500 | 68% → **89%** |
| generation | faithfulness, **answers only** | 74/74 = 100% |
| generation | abstention rate | 17 of 91 |
| judge | Cohen's kappa vs hand labels | **0.897** (raw 0.933, chance 0.353, n=30) |
| invariants | `ragkit reconcile` | 0 failing / 6 passing / 3 not measured |

Read the two generation rows together. 100% faithfulness over answers is close to
tautological: the generator is constrained to the supplied sources and the
deterministic citation checker already fails closed on unverifiable quotes before
the judge sees anything. It confirms the citation layer works; it does not
discriminate quality. Its real use is as a regression detector.

---

## A. Known defects — verified, reproducible, unfixed

These are bugs. They are first because none of them is a design choice.

### A1. HTML markup is inside the embeddings — **highest priority**

`pymupdf4llm` passes formatting markup straight through into indexed text.

| field | affected | example |
|---|---|---|
| `embed_text` | 193/814 = **23%** | `**Darren Edge**<sup>**1†**</sup>` |
| `display_text` | 305/1256 = 24% | same — visible in the source panel |
| `heading_path` | 18/1256 | a PUA bullet followed by `<u>HOME SCREEN:</u>` |

Tag counts across the index: `br` 2386, `mark` 1612, `sup` 746, `u` 200, `p` 111.
The bullet is U+F0B7, a Private Use Area codepoint that Symbol/Wingdings fonts use
for a list marker — it renders as a tofu box in every font. `clean_heading()`
strips neither HTML tags nor the PUA range.

**Fix**: strip tags keeping inner text, `<br>` to a space, drop U+E000–U+F8FF.
Perhaps twenty lines in `ragkit/ingest/loaders.py`.

**Why it wasn't done**: the fix changes `PARSER_VERSION`, which forces a full
re-parse *and* a full re-embed of 814 children against a free-tier key, and
invalidates every number in the table above. ~20 min re-ingest + ~18 min to
re-run `judge score`, and all recorded results must be regenerated together or
not at all.

Deliberately **not** patched at the render layer. Stripping tags in the UI would
hide the defect while the embeddings stayed polluted — the worse outcome.

### A2. A deferral predicate that fails open

`ragkit/deferred.py`, `entailment_verification`:

    revisit_when = "the LLM judge passes its kappa gate against hand labels"
    expired      = bool(ev["golden_set"]["human_verified"])   # unrelated counter

The stated condition is about *judge label agreement*; the predicate reads a
count of hand-verified *golden-set items*. The gate has passed
(`validated: true`, kappa 0.897) and the deferral still reports "still valid."

This is the repo's dominant bug family — a check whose predicate correlates with
its condition instead of matching it — and this instance fails in the worse
direction: it silently keeps a deferral alive past its own expiry. Same shape as
the `not existing` truthiness bug and the `crumb in body` containment check.

The other four predicates in that file were spot-checked and do match their
stated conditions. That is a spot-check, not a proof; the file has no test.

### A3. One unexplained judge verdict failure

`judge score` item 50 returned `failed: true`. Correctly excluded from `ok`, so it
does not corrupt any reported number — but the cause was never looked at. One of
92. Could be a transient API error, could be a schema violation on a particular
answer shape; nothing distinguishes those today.

### A4. Parent chunks are nothing like their configured size

`CHUNK_PARENT_TOKENS = 1200`. Measured over 434 parents: **median 192**, p90 799,
max 2672. Some queries return 24 parents inside a 1500-token budget.

Not known to be wrong — the header-aware splitter cuts on real section
boundaries and this corpus has short sections, which is arguably correct
behaviour. But a knob set to 1200 that yields a median of 192 is not controlling
what its name says it controls, and the parent/child size ratio is the entire
premise of parent-document retrieval. Unexamined.

### A5. `npx tsc --noEmit` checks nothing

It exits 0 on code that `tsc -b` rejects (found the hard way: a real type error
in `Markdown.tsx` passed `--noEmit` and failed the build). The root config uses
project references. **Use `npm run build`.** Worth either fixing the config or
deleting the misleading path.

### A6. Stale eval artifact

`data/eval/judged_results.json` — the per-row verdicts are current, but the
summary block still carries the old pooled denominator (the "99%"). The corrected
split is implemented in `score()` and in the CLI printer; the file regenerates
only on the next full run (~18 min of API calls). Read the rows, not the summary.

---

## B. Deferred decisions — two are now due

`ragkit/deferred.py` stores each deferral as a predicate that expires itself.
Run `uv run python -m ragkit.cli deferred` for live status. As of the last run:

### B1. Contextual Retrieval (M5) — **EXPIRED, revisit**

Deferred in favour of a heading-breadcrumb baseline, on the reasoning that the
paid version must beat the free one rather than an empty one. Its own expiry
condition has now fired: `child_strict = 85%` (below the ~90% threshold) while
`source_hit = 100%`. Every remaining retrieval failure is ranking *within* the
right document — which is precisely what a situating prefix addresses.

This is the highest-value *feature* work remaining, and the Anthropic result
(35% / 49% / 67% reduction in failed retrievals) is the single strongest citation
in the study guide.

Cost: one LLM call per chunk at ingest (814 calls, cached by content hash), plus
a re-embed. Implicit context caching on the shared document prefix is what makes
it affordable on a free key — and `usage.cached_tokens` is already collected, so
the caching claim can be *proven* rather than asserted.

### B2. Entailment verification (M7) — due, blocked by A2

Revisit condition ("judge passes its kappa gate") is satisfied. The predicate
does not read it. Fix A2 first, then decide.

### B3. Reranking (M3) — still valid

No cross-encoder, no LLM listwise reranker. `source_hit = 100%` at every budget
means the right document always surfaces, and dense@3000 is already 96%.

Revisits when the `aggregative` and `ambiguous` strata are measured and
`multi_hop` has at least 10 items — reranking is what reorders candidates for
multi-facet questions, and the current measurement is silent on exactly those.
See D1.

Note: `RERANK_K` exists in `config.py` with no reader. Config with no consumer is
how a knob comes to look implemented.

### B4. Qdrant / ANN index (M6) — still valid

numpy exact cosine, 814 children, retrieval 5–20 ms. Exact search *is* the right
answer at this size, and it is the oracle any future ANN recall must be measured
against. Revisits above ~100k children or p95 > 100 ms.

### B5. OpenTelemetry (M14) — still valid

Per-stage timings and token counts are already returned per request. OTel adds a
collector and a backend, which is infrastructure rather than insight at one
process. Revisits when more than one service needs to correlate a trace.

---

## C. Planned and never built

From the original plan. Absent from the tree, not deferred with a predicate —
which makes these easier to forget than B, and worth listing for that reason.

| module | what it was for |
|---|---|
| `integrations/langchain_port.py` | The explicitly chosen "from scratch, **then** a LangChain wrapper" second half. Never started. Interview-relevant: being able to speak both. |
| `retrieve/rerank.py`, `retrieve/retriever.py` | see B3 |
| `chunking/contextualize.py` | see B1 |
| `index/qdrant_store.py` | see B4 |
| `generate/context.py` | Explicit token-budget allocation and lost-in-the-middle reordering. Budget lives in retrieval; the reorder was never built or measured. |
| `generate/verify.py` | Folded into `answer.py` as two deterministic checks. No NLI pass — see B2. |
| `ingest/tables.py` | Folded into `loaders.py`. Only one of three planned table strategies exists (markdown serialisation); no LLM table summary indexed alongside the raw table. |
| `trace.py` | Timings are inline in `pipeline.py` / `api.py`. No waterfall panel. |
| `notes/` | Directory never created — see E1. |
| `experiments/` | 1 of ~9 planned scripts (`e05_hybrid_rrf.py`). The other experiments were run ad hoc and their numbers live in the eval artifacts, not in re-runnable scripts. |

**Query understanding (M1)** is the thinnest layer relative to plan. Built:
`condense()` (history-conditioned), the retrieval-necessity router
(`needs_retrieval` / `route`), and `route_drift()`. Never built: multi-query
expansion + RRF, HyDE, step-back prompting, metadata filter extraction via
structured output. The guide argues this layer is where the leverage is, and it
is the one that got least attention.

**Scope filtering** was discussed and explicitly dropped. Not an oversight.

---

## D. Measurement gaps

Nothing here is broken. These are places the harness is honestly silent, and each
one bounds a claim that could otherwise be overread.

### D1. Two of seven strata are unmeasured

`aggregative` and `ambiguous` have no items. The UI marks them `NOT_MEASURED`
rather than rendering an em-dash, and `reconcile` reports 3 invariants as "not
measured" for the same reason. So "85% recall" means 85% **on five of seven
strata**. `multi_hop` has n=2 — reported as counts, not a rate, by the
`MIN_N_FOR_RATE = 10` floor.

Consequence: the two strata most likely to need reranking are the two with no
data, which is why B3 cannot yet be decided on evidence.

### D2. `table_or_image` declines 27% of questions

**10 of 37.** The single most actionable measured number in the repo, and it was
invisible until abstentions were split out of the faithfulness denominator.

All 17 abstentions were verified justified: re-retrieved each one and checked
needle presence — **zero had their full needle set in the delivered context**
(9 absent, 8 partial, 1 starved). So the generator abstains exactly when
evidence is incomplete and never when it is complete. That is the behaviour you
want, and it localises all 17 to the retrieval tier. Table and figure retrieval
is the weak subsystem; nothing was done about it.

Full per-item causes: `data/eval/abstention_causes.json`.

### D3. The judge's kappa is a discrimination estimate, not a population estimate

kappa = 0.897 was measured on a deliberately class-balanced sample (13 partly /
10 supported / 7 unsupported, `prevalence_warning: false`). That licenses "this
judge can tell the three classes apart." It does **not** license a precision
claim on the real population, which is ~all-supported. `estimate_type` records
this distinction; any future report must carry it.

Also recorded honestly: `labelled_before_verdicts: false`. The hand labels were
not provably made before verdicts existed (labelling ran across two sessions).
Not evidence of anchoring — but the flag exists so the caveat cannot be quietly
dropped, and re-labelling from scratch is the only way to clear it.

### D4. The `page_text_clip` prediction was never resolved

Prediction: repaired-table text (values whose column header was reconstructed)
would score visibly worse on faithfulness. Result: `markdown,page_text_clip
supported 3 of 3`. n=3 — counts, not a rate, so this neither confirms nor
refutes.

The earlier hand-test that *did* produce `unsupported` isolated a single repaired
chunk with no column header available anywhere; the real pipeline usually also
returns the neighbouring page, where the header is present. So that test measured
an artificial condition. Recorded in `sample_limitations.cannot_measure` — needs
a corpus with more repaired tables, not more analysis.

### D5. Failure-taxonomy histogram — never built

The plan called for labelling 50 real failures against Barnett's seven failure
points and producing the histogram. Not done. Together with the ADR (E1) this was
repeatedly identified as the highest-value *portfolio* artifact — the harness
that produces the failures exists, so this is labelling work, not engineering.

---

## E. Process debt

### E1. The ADR was never written — **highest-value remaining artifact**

Constraints, three options considered, the choice, the tradeoffs accepted. The
guide is blunt that this plus the eval harness *is* the deliverable, because
anyone can wire a pipeline.

The raw material is unusually good and already exists: `ragkit/deferred.py` holds
five real decisions with their reasoning, their cost-if-wrong, and the condition
that expires each one. The ADR is largely a matter of writing that up with the
measurements that justified it. `notes/` was never created, so none of it is
prose yet.

### E2. The teaching pass never happened

The original intent was tutorial-style: DRILL questions answered before each
module, corrected, then implemented. That held for the first several modules —
and those answers materially changed the design (kappa as a gate rather than a
report, provenance tracked per chunk, detect-don't-stitch for page-spanning
tables, budget normalisation, feedback as external ground truth, reachability
audited from entry points rather than reference counts). Then the build was
finished at engineer pace by explicit request, to be learned afterwards.

Consequence: roughly 6,000 lines carry heavy inline reasoning, and none of it has
been read back. For a job application the walk-through matters more than the
remaining features. **Recommended before anything in A–D.**

### E3. Drill answers were never written down

They exist only in conversation history. Their value is as interview rehearsal,
which requires them to be findable.

---

## Recommended order

1. **E2** — read back what exists. Nothing else improves the submission as much.
2. **E1** — the ADR, from `deferred.py`.
3. **A1** — the HTML leak, then re-ingest and re-run everything in one pass.
4. **A2** — the fail-open predicate. Cheap, and it is currently hiding B2.
5. **D5** — the failure histogram, from eval output that already exists.
6. **B1** — Contextual Retrieval. Its predicate has expired; measure against the
   breadcrumb baseline, not against nothing.
7. **D1** — write `aggregative` and `ambiguous` items, which unblocks deciding B3
   on evidence.
8. **C** — the LangChain port, if time remains.

A2 before B2, and A1 before any re-measurement, or the numbers will not agree.

---

## Standing rules earned during this build

Every one of these came from a bug that had already shipped. The dominant family,
~12 instances: **the operation succeeded, the output was well-formed, and
something was wrong.**

- Name the caller for every module. Code that is correct and off the request path
  is not working code (`n_uncontextualized` had four readers and no writer).
- A check must **fail closed**. A predicate that merely correlates with its
  condition passes exactly when it matters least (A2 is the live instance).
- **Read the artifact, not the terminal.** Scrollback has been wrong; the file
  has not. Three sweep-table rows were once fabricated from a truncated view.
- **Probe API capabilities at startup**; do not configure them from memory.
  Batching support, thinking-disable support and task-type schemes all differ by
  model and all three were wrong once.
- **Name the gap rather than filling it plausibly** — `NOT_MEASURED`,
  `unquotable`, `n_tables_undetected_manual: null`, "counts only" below n=10.
- A reference standard being wrong is the worst class, because nothing downstream
  catches it. The judge's ground truth was wrong three times; the hand labels
  caught the third.
- **Every diagnostic must name its cause, not its symptom.** Four instances, one
  of them corrected three times in a single sitting: a warning blamed deferred
  contextual prefixes for an image-caption failure; "retrieval returned nothing"
  blamed the retriever for a budget that admitted nothing; "structured output did
  not parse" blamed the schema for our own output ceiling; and the remedy quoted
  the config default instead of the value actually in use.
