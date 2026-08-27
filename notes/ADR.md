# RAGkit — architecture decision record

A multi-format RAG system (PDF / DOCX / images) on Gemini, with a two-tier
evaluation harness and an LLM judge gated on agreement with hand labels.

**What this document is.** Not a description of the system — the README does that.
This records the *decisions*, the reasoning that produced them, the tradeoffs
accepted, and the condition under which each one stops being right. Several
entries are decisions **not** to build something; those are the ones most worth
reading, because a system is defined as much by what it refuses.

**Why the arguments matter more than the decisions.** Part 2 collects the rules
this build earned by getting things wrong. Every one of them came from a bug that
had already shipped, and every one was found by *running* something rather than
by reading it. The decisions in Part 1 are contingent on a corpus size and a
deployment; the arguments in Part 2 are the transferable part.

**Live status of every deferral:** `uv run python -m ragkit.cli deferred` — each
is stored as a predicate that expires itself, so this document cannot silently go
stale relative to the code.

---

## Context and constraints

| | |
|---|---|
| Corpus | 15 documents, 814 child chunks, 442 parents |
| Provider | Gemini only. Free tier during the build; billed key from 2026-08-26 |
| Deployment | Single AWS Lightsail container, read-only, always-on |
| Users | One. No customer exists yet — see D-11 |

Two constraints shaped nearly everything. **The corpus is small** — small enough
that exact search is not merely adequate but *correct*, which removes an entire
category of infrastructure. And **the key was free**, which made every API call a
budgeted decision and forced caching, degradation reporting and refusal paths to
be first-class rather than afterthoughts.

---

# Part 1 — Decisions

## Retrieval

### D-1 · Parent-document retrieval: embed the child, return the parent

Embedding a 300-token child and returning its ~1,200-token parent decouples *what
is matched* from *what is read*. A chunk small enough to be precisely retrievable
is usually too small to answer from.

**Tradeoff accepted:** two units to keep in sync, and a parent that can overflow a
context budget (see D-3). Measured: parent sizes are median 190 / p90 793 / max
2,672 tokens, so the configured `CHUNK_PARENT_TOKENS = 1200` is not controlling
what its name suggests — recorded as an open question rather than defended.

### D-2 · Hybrid retrieval: dense + BM25 written by hand, fused with RRF

BM25 implemented directly (k1 saturation, b length normalisation) and cross-checked
against `rank_bm25`. RRF at k=60.

**Why by hand:** "explain BM25" becomes something built rather than configured.
This is also why Qdrant is refused in D-8 — it would turn the same thing back into
a server setting.

**Measured, honestly:** NDCG@5 dense `0.848 [0.761, 0.907]` vs RRF
`0.880 [0.798, 0.932]`. **The intervals overlap; the aggregate gain is inside the
noise.** The real result is in a slice — on `exact_identifier` queries at a
500-token budget, recall goes 68% → **89%**. Reporting the headline alone would
have claimed a win the data does not support.

### D-3 · Budget-normalised recall, and a strict fill that may return nothing

Recall is reported at a fixed **token budget**, not at recall@k. Comparing a child
unit against a parent unit at "k=10" compares 3,000 tokens of context against
12,000 — the bigger bucket catches more rain. The fill never exceeds the budget,
**even when that means returning zero results.**

**Tradeoff accepted:** a query whose top-ranked parent exceeds the budget gets
nothing. That is a true finding about unit granularity, not a failure to paper
over — measured, 7 of 442 parents individually exceed a 1,500-token budget.

**This decision bit back**, and the recovery is the point: one golden-set item
received zero context, the generator abstained, and the judge scored the
abstention *supported*. A total retrieval failure recorded as a success in both
metric tiers. The fill rule was right; what was missing was that `Retrieved` could
not say **why** it was empty. "Nothing was ranked" and "everything ranked was too
big" have opposite remedies and arrived as the same empty list.

### D-4 · numpy exact cosine, no ANN index

At 814 vectors, brute-force search is 4.3 ms. Exact search is not a placeholder
here — it is the correct answer, and it is simultaneously the **oracle** any
future ANN recall must be measured against.

**Reverses when:** a corpus passes ~100k chunks, or p95 retrieval exceeds 100 ms.

### D-5 · No reranker

`source_hit = 100%` at every budget: the correct *document* is always retrieved.
So every remaining failure is ranking **within** a document, and a reranker
reorders documents that are already correct. Dense at 3,000 tokens is already 96%.

**Reverses when:** the `aggregative` and `ambiguous` strata have items and
`multi_hop` reaches n ≥ 10 — the three question types a reranker would actually
help are exactly the ones currently unmeasured. **The decision is deferred on
evidence that does not exist yet, and says so.**

### D-6 · Heading breadcrumbs, not LLM-written chunk prefixes — **now expired**

Contextual Retrieval (the Anthropic result: 35% / 49% / 67% reduction in failed
retrievals) was deferred in favour of a free heading-breadcrumb baseline, on the
principle that the paid version must beat the free one rather than an empty one.

**Its own predicate has since fired**: `child_strict = 85%` (below the ~90%
threshold) with `source_hit` already at 1.0. It expired without anyone
remembering to check, which is the entire argument for storing deferrals as
predicates. Cost to act: ~$0.69.

## Ingestion

### D-7 · Detect page-spanning tables; do not stitch them

A table continuing across a page break is *flagged* as a continuation suspect
rather than silently joined.

**Reasoning:** a wrong join is undetectable downstream — the rows look plausible
and no later stage can tell. A flag is actionable. Roughly 80% of the value of
stitching for 20% of the risk, and the 20% that is lost is visible.

### D-8 · One datastore, when a datastore is needed. Not Qdrant

Deferred, and the reasoning is sharper than "not yet":

- **The oracle and the product would become two different programs.** The
  reference is numpy; production would be Qdrant. Every recall number would then
  compare two pieces of *software* rather than two settings. This project has
  already paid for that once: a survey script imported a different parser than the
  pipeline and reported table counts differing 5× on identical bytes, costing an
  hour and a withdrawn finding. Rebuilding that at the storage layer, permanently,
  in the component whose entire job is telling you whether retrieval works, is the
  worst available place for it.
- **It splits deletion.** `purge_source` already touches six stores plus two
  caches. Two databases make deletion non-atomic, so "gone from Postgres, still in
  Qdrant" becomes possible — a document returning real-looking hits until someone
  clicks and finds nothing. One datastore makes that *impossible* rather than
  merely *checked*.
- It undoes D-2 by turning hand-written BM25 into a server setting.

**When storage does become necessary** (Phase 2, ~5 customers) it is one Postgres,
arriving for conversations, jobs, documents, feedback, tombstones and quota —
**vectors ride along as one column.** There is no vector problem: 814 chunks ×
768 dims × 4 bytes ≈ 3.8 MB per tenant; a thousand tenants is under 4 GB. Choosing
a vector database at this scale is hiring a moving truck to shift three boxes.

### D-9 · No HNSW even after Postgres arrives

Two reasons, and the second is the load-bearing one:

- Filtered ANN fails in both directions — filter-first breaks the graph's
  reachability, filter-after under-returns.
- **It keeps the oracle and production the same program.** With exact search
  inside the tenant filter, HNSW can later be compared to exact search *in the
  same database, through the same planner, with the same filter* — so only the one
  thing that changed differs.

## Generation and evidence

### D-10 · Citations carry verbatim quotes, and failure is a taxonomy

Every claim cites, and every citation carries a quote checked against the source.
Four distinct outcomes: **verified**, **absent**, **unquotable**, **fabricated**.

**Why not entailment/NLI:** character overlap provably cannot separate an honest
rewording from an invented quote — measured on this corpus, the fabrication scored
0.53 and the rewording 0.48. A metric that ranks the fabrication *higher* is not a
weak signal, it is an inverted one. Entailment needs a judge, and an unvalidated
judge is not a weaker signal but an unknown one.

**Provenance is per chunk**, so `evidence_kind` is *derived* from stored fields
rather than guessed from text — which is what makes the colour trustworthy. A
model-written image caption can never render as quoted source text.

### D-11 · Grounded abstention is a correct answer, and is scored as one

**And it must not share a denominator with answers.** The first generation report
read "99% supported" — until 18 of the 92 rows turned out to be abstentions, each
correctly marked supported because an answer asserting nothing asserts nothing
false. A fifth of the headline was vacuous passes, and **the metric improved
whenever the system answered fewer questions.**

Split apart: **74/74 = 100% over answers**, with a **20% abstention rate**
reported alongside as coverage. That surfaced the finding hiding inside the good
news — **29% of table and figure questions are declined** — which is now the
weakest measured part of the system.

All 18 abstentions were then checked individually: **zero had their full evidence
in the delivered context.** The generator abstains exactly when evidence is
incomplete and never when it is complete, which localises all 18 to retrieval.

## Evaluation

### D-12 · The judge is gated on Cohen's kappa, and emits nothing until it passes

κ = **0.897** against 30 blind hand labels (raw agreement 0.933, chance 0.353).
No judged metric is produced at all until that gate passes.

**Two caveats the code records rather than the prose:**

- The sample is deliberately **class-balanced** (13 partly / 10 supported / 7
  unsupported), so it is a *discrimination* estimate. It licenses "this judge can
  tell the three classes apart" and **not** a precision claim on the real
  population, which is nearly all-supported.
- `labelled_before_verdicts: false` — the labels are not provably anchored.
  Recorded so the caveat cannot be quietly dropped.

The judge also runs a different model (`gemini-3.1-pro-preview`) than the
generator (`gemini-3.7-flash`), and the API **refuses to report a score** if they
ever collide. A model grading its own output is not evidence.

### D-13 · Rates below n = 10 are printed as counts

`1 of 2`, never "50%". A percentage implies a precision the sample does not have.
Confidence intervals are Wilson.

### D-14 · Unmeasured is a distinct state from zero

`aggregative` and `ambiguous` have no items, and the UI renders `NOT_MEASURED`
rather than a blank that could read as failure. The headline is therefore "85% on
**five of seven** strata". The reconciler reports `0 failing · 6 passing · 3 not
measured` — three-state, not two.

The most uncomfortable one is left on the page deliberately: **`Grounding audit:
0 of 97 verified`.** The golden set is machine-generated and machine-verified;
eight fabricated needles were caught automatically, but no human has read a
stratified slice.

## Operations

### D-15 · Read-only deployment, denying by method with an explicit allowlist

The API has nine unauthenticated write endpoints, two destructive:
`DELETE /api/documents` purges the embedding cache by default, and one
`POST /api/ingest` could exhaust a day of quota.

**Deny by method, allow by pattern** — so an endpoint added later is refused until
someone deliberately opens it. The alternative (a check per handler) fails *open*
on whoever forgets.

### D-16 · A spend ceiling in two layers that fail in opposite directions

The free tier failed **closed**: a 429 cost nothing. A billed key removes that
wall, so identical code fails **open**, into a bill.

| layer | defends against | restart-safe |
|---|---|---|
| **per-operation** | the 500-PDF upload — *one* operation, knowable in advance | **yes** |
| cumulative daily | slow drift within a process lifetime | **no** |

Counted over **cache misses only** — a re-ingest whose embeddings are all cached
costs nothing, and a ceiling that refused free work would train whoever hit it to
raise the cap for no reason.

### D-17 · `durable_spend_ledger` is deferred **with no predicate**, on purpose

Every other deferral in this project carries a machine-checkable condition that
expires it. This one carries none, and that is the decision:

> Nothing in `data/eval` can observe whether a filesystem is durable. A proxy
> signal that merely *correlates* with the condition is the fail-open pattern the
> deferral system exists to prevent. A deferral honestly marked "no automatic
> trigger, revisit at Phase 2" is stronger than one wired to evidence that would
> fire for the wrong reason.

Counterintuitive enough to need the reasoning attached: **correlation is
acceptable in a report and disqualifying in a gate.** The generalisation of the
`crumb in body` bug (see A-2), applied here to decline building something rather
than to fix something.

### D-18 · One deployment per customer before multi-tenancy

At 3.8 MB per tenant, a container per customer costs almost nothing and works
today. Multi-tenancy is months of work, produces nothing a customer can see, and
is **the least differentiated thing available** — anyone can build auth; the
evidence and provenance work is the part that is hard to copy. Ugly past ~5
customers, by which point the business either justifies the foundation or does
not.

**The part actually worth building when it comes:** the permission check inside
the search query — `WHERE tenant_id = $1` in the same statement that computes
distances, so no code path can produce another tenant's chunk and then discard it
— plus an **isolation test in the eval output**: tenant A asks a question whose
answer sits in tenant B's documents, and retrieval returns nothing. A measured
claim, not "we implemented permissions".

---

# Part 2 — Arguments

The rules, each earned from a shipped bug. This is the transferable half.

## A-1 · The dominant failure: the operation succeeded and something was wrong

~12 instances. Not crashes — **well-formed output that was wrong**. Three
sub-families:

1. Code built, correct, and **not on the request path**. Seven confirmed:
   `condense()` in its module while `/api/ask` never called it; `capabilities()`
   written, documented, wired nowhere; a provenance guard on the child while the
   model read the parent.
2. Checks whose predicate **correlates with** rather than **matches** the
   condition.
3. **A reference standard that is itself wrong** — the worst class, because
   nothing downstream can catch it. The judge's ground truth was wrong three
   times; the third was caught by a human labeller disagreeing.

**Every instance was found by running something.** None by reading code.

## A-2 · A check must fail closed, and correlation is disqualifying

Nine wrong predicates found. The archetype: verifying a breadcrumb prefix with
`crumb in body` — containment, where the property is `startswith`. It passes for
the right reason almost always, and silently accepts the defect it exists to
catch.

Corollary in D-17: when no honest predicate exists, **say so** rather than wiring
a plausible one.

## A-3 · Existence is not coverage — "built, correct, on *some* of the traffic"

**The newest variant, and worse than A-1, because it reports a pass.**

The spend ceiling was wired into `embed_texts()` only. `ragkit ask` under a
5-token cap sailed through and spent ~2,500 tokens: the one check in the codebase
correctly saw zero billable *embedding* work — the query embedding was cached —
and **nothing examined the generate call at all.** The most-called paid route
recorded its spend and enforced nothing.

Everything looked healthy. `ragkit spend` printed real numbers, the notice
rendered correctly, the unit test was green. **Nothing in any output said "one of
three."** A partial pass is the kind you stop looking at.

**Why the reachability audit structurally could not catch it.** That tool asks
*"is this reachable from an entry point?"* It cannot ask *"is this reached from
**every** entry point that needs it?"* — and for a guard those are different
questions with the same-looking answer. **For a guard, coverage is the invariant,
not existence.**

The fix is `scripts/audit_guards.py`, asserting `paid_routes ⊆ guarded_routes`
with `paid_routes` **derived** — every function reaching `client()` — rather than
listed. A hand-maintained roster is the same class of object as a hardcoded count
beside a derived one: correct the day it is written, silently wrong after.

## A-4 · An exemption is a claim about a subtree, not about a vertex

The audit's first pass filtered exempt *names* out of its results, and flagged
`cache_key`, `caption_cache_key` and `model_for` as unguarded paid routes — they
call `resolve_models()` to name a model in a cache key, and that reaches
`client()`.

Filtering the output would have required one exemption per caller, and per future
caller: **the hand-written list the derivation existed to avoid, reconstituted one
indirection away.**

Cutting the traversal says the right thing. *"Spends nothing metered"* is a claim
about everything downstream, not just that node. Filtering leaves treats a
subtree-claim as a vertex-claim, and the two agree only when the subtree happens
to be a leaf.

Same shape as asserting `partial ≥ strict`: **assert the property the operation
preserves, not the one that looked equal on your test case.**

## A-5 · Every diagnostic must name its cause, not its symptom

Five instances, one corrected three times in a single sitting:

- a warning blamed deferred contextual prefixes for an image-caption failure
- *"retrieval returned nothing"* blamed the retriever for a budget that admitted
  nothing
- *"structured output did not parse"* blamed the schema for our own output ceiling
- a remedy quoted `config.TOKENS_CONTEXT_BUDGET` instead of the budget in use
- *"free-tier Gemini limit"* survived the arrival of a billed key

**And widening a guard invalidates strings written while it was narrow.** Once the
ceiling covered three routes, `"the ingest limit"` was false on two of them — a
refused *answer* told the reader their ingest was too large. Now asserted:
messages may not name a single route.

**Corollary — operator advice only reaches operators.** Telling a public demo
visitor to raise `RAGKIT_MAX_OPERATION_TOKENS` is correct information delivered to
someone who cannot act on it. A message earns its place by whether *the reader*
can act on it.

## A-6 · A gate that always fails is ignored exactly like one that never does

Wiring the guard audit into `ragkit audit` made the command exit non-zero. The
temptation was to exempt and move on. Fixing instead exposed two blind spots in
the *other* audit:

- **`@app.middleware` was not a root.** `demo_guard` runs on every request, so the
  reachability audit had been reporting the demo's entire write protection as dead
  code. The audit's own bug class turned on itself — a predicate covering the
  entry shapes that existed when it was written.
- **Aliased imports broke the name trail.** `explain as explain_fusion`. The
  docstring had *recorded this as a known blind spot* and warned that the next
  aliasing would silently drop a function from coverage.

**Being accurate about a hole is not the same as not having one.** And the fix was
per-file, not a global alias map — a global map would mark a genuinely dead `foo`
reachable because some unrelated module aliased something to `foo`, **closing a
blind spot by weakening the check**, which is how a check stops checking.

## A-7 · Probe capabilities; do not configure them from memory

Three wrong at once: four asserted model IDs of which two were unverifiable;
`gemini-embedding-2` returning one embedding for N inputs (which `zip()` truncated
into invisible zero vectors); `gemini-3.5-flash-lite` rejecting `thinking_budget=0`
and failing the entire cheap tier on its first call.

Config states intent. Only a real call states fact.

## A-8 · Name the gap rather than filling it plausibly

`NOT_MEASURED`; `unquotable`; `n_tables_undetected_manual: null`;
`verification: not_applicable_no_sources`; counts instead of rates below n = 10;
`daily_counter_durable: false`.

## A-9 · Read the artifact, not the terminal

Three rows of a sweep table were once fabricated from a truncated scrollback view,
and a conclusion drawn from the invented rows. The artifact on disk was right the
whole time.

**This rule caught a violation of itself inside this document, on the day it was
written.** Re-checking the appendix against the artifacts found three numbers
recalled rather than read -- parent median 192 (190), p90 799 (793), and "7 of
434 parents" in a file whose own corpus table says **442**. The last was not just
wrong but internally inconsistent, and it survived writing, reading back, and a
proofread. Only running the measurement caught it.

## A-10 · Comparable is not the same as disjoint

Two artifacts can be provably about the same system and still not be addable.
**Same-system** and **disjoint-events** are independent properties, and the
stamping work bought the first while leaving the second entirely unguarded.

The failure analysis pools three sources. All three carry the same fingerprint and
the same token budget, so pooling was *comparable*. Eight questions were both a
retrieval miss and an abstention — one failure, appearing in two inputs — so
pooling was **not disjoint**, and a single event was counted twice in a document
whose entire output is a per-category count. Every proportion would have been
wrong by roughly a third.

**It surfaced only because the total refused to reconcile:** 37 classified against
32 the inputs implied. That is the same instrument that caught `partial < strict`
— two numbers that must agree, disagreeing — and it is the only reason this was
caught at all. **Had I expected 37, the histogram would have shipped.** No
inspection of the code would have found it; the sets look independent right up
until you list them.

So pooling now asserts both properties: the stamp for same-system, and `N in ==
N classified out` for disjointness, with an explicit precedence rule for events
that legitimately appear in two inputs.

### The sibling bug, in the same function

`per_item` holds all 101 golden items, including 5 `out_of_scope` questions whose
**correct** behaviour is retrieving nothing. Counting those as retrieval failures
turned 14 misses into 19.

That is the third instance of one shape: **a derived number computed over a
population that does not match the name on it.** The 18%-headerless figure was the
first and had to be withdrawn. The `text_source` slice that would have run over
zero repaired chunks was the second. Each time the tell was **arithmetic, not
inspection** — a count that did not add up, noticed because something else already
knew what the answer should be.

The defensive form is cheap: a derived count must apply the same population filter
as the headline it will sit beside, and should be checked against it.

## A-11 · An untestable path is where a trivial bug hides indefinitely

`Manifest.summary()` returns a `str`. The removal preview called
`.get("pipeline_fingerprint")` on it. That raises on **every** call — a bug a
single execution catches, and one that no amount of subtlety was required to
create.

It survived because `DELETE /api/documents/{id}` was its only caller, and **you
cannot exercise that endpoint without deleting a document.** Not
built-and-unreachable (A-1), not correct-on-some-of-the-traffic (A-3), but a third
thing: **reachable, and never reached, because reaching it costs something.**

That is a distinct hazard, and it selects for *trivial* bugs rather than subtle
ones. A subtle bug on a hot path gets found eventually because the path runs. A
trivial bug on a destructive path can sit indefinitely, because every reasonable
person avoids running it — and reviewing it is no help, since reviewing is exactly
what missed it.

### The consequence for design

**Separating "what would happen" from "do it" makes destructive paths
exercisable.** `GET /api/documents/{id}/impact` was added because the removal
confirm dialog needed to show consequences *before* the user committed — `DELETE`
returns its preview and queues the deletion in the same call, so its "preview"
describes what it has already started.

That was the first argument. The second is stronger and was not anticipated: the
dry-run is a **testable surface for a destructive operation**, and the bug it
would have caught was already sitting there. Every destructive endpoint deserves
one, and not only for the dialog.

A second property arrived from the HTTP verb rather than from a special case:
because the dry-run is a `GET`, it passes the read-only demo guard untouched. The
consequences of a delete stay visible on a deployment where the delete itself is
refused. **Showing what a destructive action would do is not a destructive
action** — and expressing that as a verb rather than an exception to the guard
means nothing had to be widened to allow it.

### The same reasoning, applied to a list

Un-ignoring `data/index/` so the prebuilt index ships also captured
`conversations/` and `jobs/` — eight ad-hoc debugging conversations and nine job
records, staged for a public repo. On a customer deployment those files are *their*
data. **Widening an exemption captured more than it named**, which is the same
shape as widening a guard and invalidating a string written while it was narrow
(A-5).

Listing the two offending paths in `.gitignore` would have restated the
distinction as a literal, and a literal drifts the moment a third runtime store
appears. So it is derived instead (`audit_guards.py` check 4): every path under
`data/index/` must be either declared build output *with a reason*, or ignored —
and one that is neither **fails**, so a new runtime store is caught the day it
appears.

The derivation unions **code literals with the filesystem**, because either alone
under-covers. `numpy_index` is composed as `DATA_INDEX / name` from a parameter
default, so no literal grep can see it — and a check that silently missed the most
important directory under audit would be A-3 all over again, inside the tool
written to prevent it.

---

# Part 3 — Deliberately not decided

| | expires when |
|---|---|
| Multi-tenancy and auth | per-customer deployments become painful (~5 customers) |
| HNSW / ANN | a tenant passes N chunks, or exact search passes M ms |
| Qdrant | in-DB hybrid fusion is needed **and** split-deletion is solved |
| Entailment / NLI | judge passes its kappa gate — **satisfied; predicate reads the wrong counter, see below** |
| Per-conversation scope | a real customer asks (design-brief-driven today) |
| xlsx + OCR detection | a prospect's corpus is materially scans or spreadsheets |
| Stripe billing | more than ~10 customers; invoice manually until then |
| OpenTelemetry | more than one process needs to correlate a trace |
| Generation-tier CI gating | multiple runs per commit become affordable, **or** a threshold is calibrated against measured variance — **no predicate**, same reasoning as D-17 |
| Durable spend ledger | Phase 2 — **no predicate, by decision D-17** |

**A known defect in this table.** The `entailment_verification` predicate reads
`golden_set.human_verified` while its stated condition is *"the judge passes its
kappa gate"*. The gate has passed; the deferral still reports valid. It is an
instance of A-2 in the very file built to prevent A-2 — a predicate that
correlates with its condition instead of matching it — and it **fails open**,
silently keeping a deferral alive past its expiry.

**And the direction matters more than the fact.** Every other wrong predicate in
this codebase failed *toward refusal*: `crumb in body` over-rejected, the strict
budget fill returned nothing, the guard audit went red. Those announce themselves.
This one is silent — it keeps a deferral alive rather than killing a live one. "My
predicate was wrong" and "my predicate was wrong in the direction that hides
things" are different severities, and only the second can sit undetected for
weeks.

Recorded here rather than quietly fixed, because a document that only lists
solved problems is a brochure.

---

## Appendix — the numbers

All produced by running something. Reproduce with `ragkit eval`,
`ragkit judge score`, `ragkit reconcile`, `ragkit audit`, `ragkit spend`.

**Which run.** Every generation-tier figure below comes from one stamped artifact:
fingerprint `6fd55e19a82a7c28`, parser
`pymupdf4llm@1.28+...`, budget
`1500`. Artifacts that record neither cannot be pooled with
each other, which is why the stamp exists -- and why the failure analysis refuses
an unstamped input rather than inheriting a fingerprint nobody verified.

**These numbers move between runs, and the variance is not quantified.** Re-running
the generation tier on an unchanged index and an unchanged budget moved abstentions
17 -> 18 and `table_or_image` 27% -> 29%, and made one judge verdict that had
failed succeed. Generation is sampled, so a single run is a sample of one. Nothing
here reports a confidence interval over repeated runs, and a difference of one or
two items between runs should not be read as a change in the system.

| | |
|---|---|
| `child_strict` @1500 tok | 78/92 = **85%** (5 of 7 strata) |
| `source_hit` @every budget | 92/92 = **100%** |
| Retrieval misses by budget | 67 @250 · 14 @1500 · 4 @3000 · **0 @12000** |
| NDCG@5 dense vs RRF | 0.848 [0.761, 0.907] vs 0.880 [0.798, 0.932] |
| RRF on `exact_identifier` @500 | 68% → **89%** |
| Faithfulness over **answers** | 74/74 = 100% |
| Abstention rate | 18/92 = **20%**; `table_or_image` **11/38 = 29%** |
| Judge κ vs hand labels | **0.897** (raw 0.933, chance 0.353, n=30) |
| Invariants | 0 failing · 6 passing · 3 not measured |
| Guard coverage | 8 paid routes · 4 guarded · 4 exempt · 0 unguarded |
| Reachability | 0 unexplained · 4 explained by deferrals |
| Container | 150 MiB / 512 · 22 s boot · ~7 s per answer |

**A number that must always carry its qualifier:** the retrieval-miss count is a
function of the budget knob, not a property of the system. At
`TOKENS_CONTEXT_BUDGET = 12000` there are **zero** retrieval failures. "14 misses"
means nothing without "@1500".
