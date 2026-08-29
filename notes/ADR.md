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

`source_hit` is **100% at 3,000 tokens and above, and 99% (91/92) at the headline
budget of 1,500** — so the correct *document* is retrieved for every question but
one, and nearly every remaining failure is ranking **within** a document, which a
reranker does not touch: it reorders documents that are already correct.

*That figure used to read "100% at every budget", and it moved when contextual
prefixes shipped.* The single loss is a bibliography line whose rare token is
drowned by a prefix every bibliography in the corpus now shares — see D-6. One
item is not a business case for a reranker, but the argument is now stated at the
strength the evidence supports rather than at the strength it had last month.

**Reverses when:** the `aggregative` and `ambiguous` strata have items and
`multi_hop` reaches n ≥ 10 — the three question types a reranker would actually
help are exactly the ones currently unmeasured. **The decision is deferred on
evidence that does not exist yet, and says so.**

### D-6 · Contextual prefixes are ON — shipped with the accounting fix that made them legible

Contextual Retrieval (the Anthropic result: 35% / 49% / 67% fewer failed
retrievals) was deferred behind a free heading-breadcrumb baseline, on the rule
that the paid version must beat the free one rather than an empty one. Its
predicate fired — `child_strict` below 90% with `source_hit` already 1.0 — so it
was built, run over the whole corpus, and measured.

**It was then held for one commit**, because on the first measurement it lost.
The reason it lost turned out to be the measurement (**A-13**), and the two
changes shipped together. Separately each is bad: correcting the accounting alone
moves every published figure with nothing to show for it; enabling the feature
alone ships a measured regression against this project's own headline.

**The 2×2 that separates them.** `child_strict`, 92 evaluable golden items, rows
are the feature and columns are the accounting:

| budget | breadcrumb / indexed | breadcrumb / delivered | contextual / indexed | contextual / delivered |
|---|---|---|---|---|
| 250 | 25 | 30 | 15 | **38** |
| 500 | 55 | 58 | 56 | **66** |
| 1000 | 73 | 76 | 74 | **77** |
| **1500** | **79** | 79 | 79 | **82** |
| 3000 | 88 | 88 | 84 | **87** |
| 6000 | 90 | 90 | 89 | **91** |

The left column is what this project published before; the right column is what
it publishes now. Neither is derivable from the other, which is why all four are
kept: at the headline budget the accounting alone changes **nothing** (79 → 79)
and the entire gain is the feature, while at 250 tokens the feature alone is a
**disaster** (25 → 15) and only the pair is an improvement.

**Parents moved too, and the parent branch never touched the accounting.** A
parent was always charged what it delivers, so `parent_strict` 71 → **75** at
1,500 is a clean ranking effect: better child ranking selects better parents.
Same at 500 (42 → 51) and 1000 (58 → 63).

*One qualifier, found by checking rather than assumed:* `parent_strict` is not
purely a parent-branch number. For the **three asset-anchored items** — images,
which have no text to match — the metric is defined as "did an accepted source
surface", and that is read off the **child** hits. So those three inherit the
child cost basis, and they are the entire reason breadcrumb `parent_strict` moves
21 → 22 at a 250-token budget between the two accountings. Every other budget is
identical. It is one item out of 92 and it changes nothing, but a metric that
silently borrows from another unit is exactly the kind of coupling that makes a
later comparison wrong for reasons nobody can reconstruct.

**Headline, restated.** `child_strict` 79/92 = 86% → **82/92 = 89%**;
`parent_strict` 71/92 = 77% → **75/92 = 82%**.

**Where the gain came from, and it is not spread evenly.** By stratum, at the
headline budget:

| stratum | breadcrumb | contextual | |
|---|---|---|---|
| `table_or_image` | 32/38 = 84% | **35/38 = 92%** | +3 |
| `multi_hop` | 1 of 2 | 2 of 2 | +1 |
| `exact_identifier` | 17/19 = 89% | 17/19 = 89% | 0 |
| `simple_factual` | 29/33 = 88% | 28/33 = 85% | **−1** |

Almost all of it is tables and figures — the weak subsystem, and the one where a
situating sentence obviously helps: a table torn from its caption gains most from
being told what it is a table *of*. Prose that already names its own subject
gains nothing and pays the ranking churn.

**The generation tier moved with it.** Re-run over the same golden set: answers
74 → **78** of 92, abstentions 18 → **14** (15%), and `table_or_image` declines
11/38 = 29% → **8/38 = 21%**. Faithfulness stays 100% over answers (78/78). More
questions answered with no loss of support is the shape you want; the reverse
would have meant the prefix was talking the model into claims.

**Measured failures fell from 23 to 15** — FP2 14 → 8, FP3 9 → 7, and **FP4 still
zero**. The negative result that ruled out prompt work, reordering and reader
fine-tuning is not weakened by this; it is now zero out of a smaller total, which
is the same conclusion with less room left for it to hide in.

### What it cost, beyond tokens: `source_hit` left 100%

`source_hit` at the headline budget went **92/92 → 91/92**, and this project has
been leading with "the right document is always found". It now is not, once.

The item is an `exact_identifier` question: *"what is the arXiv identifier for the
paper titled MultiHop-RAG..."*, answered by a line in another paper's
bibliography. Under breadcrumbs, retrieval returned 6 children, two of them from
the right document. Under contextual prefixes it returns 19 children — the
corrected accounting fits more — and **none** of them are from the right
document.

The mechanism is worth stating because it is the general limit of the technique.
Every reference section in every paper now carries a prefix that says
approximately the same thing: *"in the references section of this paper, these
citations document related work on..."*. That is genuine context, and it is
**shared vocabulary across the corpus**. A query whose entire discriminating
signal is one rare token — a specific paper title inside a citation — is
competing against a document-level description that every bibliography in the
index now matches. **The prefix supplied context where the passage needed to
remain unlike its neighbours.**

So the rule is not "contextual prefixes help retrieval". It is:

> A situating prefix helps a passage that cannot describe itself, and hurts a
> passage whose retrieval value is precisely that it is unlike its neighbours.

Tables and captions are the first kind, which is why they gained 8 points.
Bibliographies are the second. That is one item out of 92 against a +3 headline,
so it does not change the decision — but it is the failure mode to watch when
this meets a corpus of contracts, invoices, or anything where the rare token is
the whole point.

**It also weakens D-5's argument for having no reranker**, which rests on
`source_hit` being 1.0 at every budget. It is 1.0 at 3,000 tokens and above, and
0.99 at the headline. One item is not a reranker's business case; the entry now
says so rather than repeating a figure that has moved.

**What it cost.** 814 prefixes and 15 document synopses on the first build:
1,032,463 input and 55,660 output tokens, zero refusals, mean prefix 66 tokens
against a ~300-token body. Cached thereafter — rebuilding the index takes two
seconds and zero API calls. That is also why `index_report.json` reports zero
tokens for the contextualisation stage, and why those fields are now named
`this_run_*`: they answer "what did this run spend", not "what did this index
cost", and a field called `prompt_tokens` reporting 0 would eventually be read as
the second.

**Reverses when:** a corpus arrives where the prefix cannot say anything the body
does not already say — near-duplicate boilerplate, or single-topic documents
where every chunk situates identically. The prefix then costs budget and adds no
signal, and the 250-token column above shows what that looks like.

### D-6a · A per-document synopsis, not the whole document — a deviation from the published method

Anthropic's Contextual Retrieval puts the **entire document** in the prompt for
every chunk, and relies on prompt caching to make the repeated prefix nearly
free. This implementation sends a **per-document synopsis (15 calls) plus the
chunk's own parent section** instead. That is a deviation from the published
method and it was measured before it was chosen, not assumed:

| | input tokens |
|---|---|
| whole document per chunk | **15,791,052** |
| synopsis + parent per chunk | **1,032,463** |

11.3×, and the saving is not the whole argument. The discount that makes the
published version affordable is Gemini's implicit cache, which requires a
**4,096-token cacheable prefix** — and **six of this corpus's fifteen sources are
shorter than that**, so those documents would have paid full price per chunk no
matter what. Building a cost estimate on a discount that provably does not apply
to 40% of the corpus is the same error as gating on an uncalibrated threshold.

**What is given up:** cross-section context. A prefix written from the synopsis
plus the local section cannot say "this contradicts the result in §7". Whether
that matters is an empirical question, and on this corpus the measured gain
(+3 at the headline budget, +13 at 250) came without it.

**Reverses when:** a corpus has documents whose sections only make sense against
each other — a contract with cross-referenced clauses, a manual with
forward-references — or when a cache discount is *observed* rather than assumed,
at which point the full-document version becomes nearly free and the deviation
stops paying for itself.

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

Split apart: **100% over answers**, with an abstention rate
reported alongside as coverage. That surfaced the finding hiding inside the good
news — at the time, **29% of table and figure questions were declined**, the
weakest measured part of the system.

Every abstention was then checked individually: **zero had their full evidence in
the delivered context.** The generator abstains exactly when evidence is
incomplete and never when it is complete, which localises all of them to
retrieval.

*Current figures, after contextual prefixes shipped (D-6):* abstention
**14/92 = 15%** overall and **8/38 = 21%** on tables and figures, faithfulness
**78/78 = 100%**. The split is what made the improvement legible: pooled, this
change would have shown as "100% supported" before and after, and the four extra
questions now answered would have been invisible.

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
rather than a blank that could read as failure. The headline is therefore "86% on
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

### D-19 · Strangers may upload documents to the public demo — an accepted risk

**The decision.** The public demo accepts PDF uploads from anyone, ingests them
into a session-scoped sandbox, and purges them on a timer.

**Why, and it is not "because uploads are a nice feature."** This system's
differentiator is that it refuses to quote what it cannot verify. On *our* corpus
that behaviour is invisible: a visitor has no way to know the tables are broken,
so `FOUND — NOT QUOTED` reads as a limitation. On *their* document they know
exactly what is in it, and the identical behaviour reads as a catch. **The refusal
state is only impressive to someone who can check it.** An examiner uploading
their own file is also the customer research, done in ten minutes rather than five
conversations.

**What was measured before deciding**, because the original objection was cost and
it turned out to be the wrong constraint:

| per 50-page upload | |
|---|---|
| vectors | 0.8 MB |
| embedding | $0.0077 |
| ingest wall clock | ~8 s/page — **the binding constraint** |

Twenty concurrent sessions is 16 MB against 360 MB of headroom. A hundred uploads
a day is $1.50. Neither is close to binding, so the caps are sized by *how long a
visitor will wait*, not by memory or money.

**The mitigations, and what each one actually bounds:**

| mitigation | bounds |
|---|---|
| all parsing in a subprocess | the API process never touches hostile bytes |
| `RLIMIT_AS`, `RLIMIT_CPU` | runaway allocation and CPU inside the probe |
| wall-clock timeout in the parent | a hang the child's own limits cannot stop |
| scrubbed environment | what a compromised probe can reach — no API key, no AWS credentials |
| page / size / object caps | wait time, and complexity a page count cannot see |
| active-content refusal | `/JavaScript`, `/Launch`, `/EmbeddedFile`, `/RichMedia` |
| quarantine before admission | a refused file is never briefly indexable |
| session ownership + TTL purge | who can retrieve it, and for how long |

**What this does NOT cover, stated plainly.** PyMuPDF is a C library, and PDF
parsers have a long history of memory-safety bugs — a property of the format, not
of this library. **`RLIMIT_AS` bounds allocation; it does not bound memory
corruption.** If a malformed PDF achieves code execution inside the probe, the
only thing containing it is the process boundary and the scrubbed environment. It
is not sandboxed further: no separate user, no namespace, no seccomp filter, and
the probe can still read the container's filesystem.

That is an accepted risk with real mitigations, not a solved problem. Saying so is
stronger than implying otherwise, and it is the sentence to re-read before
widening what the demo accepts.

**NOT YET ENABLED, and the reason is a walk rather than a doubt.** Every
component was built and individually checked, so the remaining step looked like
flipping a flag. Walking the visitor's path end to end took fifteen minutes and
found that the path does not complete:

1. **The upload succeeds and then dead-ends.** The response says *"POST
   /api/ingest to make these searchable"*, and `/api/ingest` returns 403 on a
   demo — correctly, it is a corpus-wide rebuild. So the file lands, is owned by
   the session, and can never be asked about.
2. **The file is written to `data/raw`, the shared corpus directory.**
   `corpus_files()` rglobs it, so the next operator-run `ragkit ingest` would
   index a stranger's document with `owner=PUBLIC_OWNER`. Ownership is enforced
   *in the index*; the route around it through the filesystem was not closed.
3. **The obvious fix wipes the corpus.** `Manifest.plan` computes deletions as
   `set(records) - present_ids`, so ingesting only the uploaded file marks every
   other document absent and purges it.

None of these is a bug in a component. Each component is correct, and this
project has now produced the same shape often enough to name it: **the pieces
were tested and the sequence was not, and the sequence is what a visitor
experiences.** It is the same lesson as the isolation test that passed on a path
the product does not take, arriving at the level of a user journey rather than a
function call.

What ships in the meantime is the half that is unambiguously right: uploads are
opened by **widening one exemption** (`^/api/documents$`) rather than by turning
`DEMO_MODE` off, so re-ingest, delete and the session sweep stay refused; and the
UI's single `read_only` flag — which drove the upload control *and* the delete
control — is split, because opening uploads through it would have re-enabled
deletion in the same motion.

The remaining work is a session-scoped, **additive** ingest: chunk under the
session's owner, append to the live index, and never participate in delete
detection. The in-memory version of exactly that already exists — it is what the
isolation invariant does to inject its probes.

**Reverses when:** the demo carries anything worth stealing beyond a corpus of
public papers, or a customer's documents share a deployment with strangers'
uploads. Either makes the process boundary too thin, and the next step is a
separate unprivileged container for the probe.

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

### The shortest demonstration of this in the project

The session resolver validates an untrusted cookie two ways: a shape rule (22
urlsafe characters) and an identity check against `PUBLIC_OWNER`. The shape rule
already excludes `""`, so the identity check looks redundant.

It was tested rather than argued. `PUBLIC_OWNER` was changed to
`"public_corpus_shared__"` — 22 characters — and the resolver re-run:

```
regex accepts it : True     ← the shape check now passes the sentinel
resolve() returns: None     ← the identity check still rejects it
```

The regex **correlates** with "not the public sentinel". The identity check **is**
that property. With only the regex, a visitor could set `owner` to the public
sentinel and publish their own upload to every other visitor — and nothing would
have announced that the protection had lapsed, because the rule that lapsed was
still passing its own tests.

Correlation is acceptable in a report and disqualifying in a gate.

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

## A-12 · A provenance stamp only covers what it hashes

The retrieval headline moved 78/92 to 79/92 between two runs whose
`index_provenance` was **byte-identical** — same parser version, same chunker
version, same `pipeline_fingerprint`. The gate reported "flat" and was right to;
the fingerprint reported "same system" and was wrong.

Cause not established, and that is the point. The most likely explanation is that
an unrelated test queued real ingest jobs, which rewrote the index files from
cached embeddings and changed a tie-break in ranking. **But nothing recorded
enough to confirm it**, because the fingerprint covers the *pipeline* —
`parser_version`, `chunker_version`, embedding dimension — and not the *artifact*
the pipeline produced.

So a rebuilt index that differs in chunk ordering carries the same stamp as the
one before it. Two runs can then be compared as though they measured the same
index when they did not, which is the precise failure the stamp exists to prevent,
in the stamp itself.

Retrieval *is* deterministic — re-running immediately reproduced that run's
79/92 exactly, which is what justifies gating it in CI. (That figure is from the
breadcrumb index under the old cost basis; the headline today is 82/92. The point
is the reproduction, not the number.) The non-determinism is not in the search;
it is in what "the index" refers to across a rebuild.

**The fix is a content hash** of the serialised index alongside the pipeline
fingerprint: `same pipeline` and `same artifact` are different claims and deserve
different fields. Not yet done, and recorded here rather than quietly patched,
because it changes what every stored comparison means.

The general form, worth carrying: **a stamp is a claim about exactly the inputs it
hashes.** Everything else it appears to vouch for, it does not.


## A-13 · A budget is a claim about delivery. Charge every unit the resource the claim is about

The finding that came out of contextual retrieval, and worth more than the
feature was.

`search_budget` fills a token budget with retrieval units and stops. It charged

- a **parent** the tokens of its `display_text` — what the model will read
- a **child** the tokens of its `embed_text` — what the *index* holds

`embed_text` is the body plus a heading breadcrumb plus, once prefixes shipped, a
model-written situating sentence. None of that reaches the model: what a child
retrieval delivers is its body, or its parent. So the child unit paid for its own
index-time enrichment and the parent unit paid for nothing equivalent — **in the
one comparison budget normalisation exists to make fair.**

This is older than contextual retrieval. Breadcrumbs were charged the whole time;
they were simply short enough (~9% of a paper chunk) that nobody looked. A
66-token situating sentence on a ~300-token body is ~22%, and at that size the
asymmetry stopped being a rounding error and started deciding conclusions:

| budget | Δ under `indexed` | Δ under `delivered` |
|---|---|---|
| 250 | **−10** | **+8** |
| 500 | +1 | +8 |
| 1000 | +1 | +1 |
| 1500 | 0 | +3 |
| 3000 | −4 | −1 |

Same index. Same golden set. Same technique. **The verdict flips on the choice of
denominator, and both columns are correct arithmetic.**

The mechanism is visible in the fill. At 250 tokens under `indexed`, **46** of 92
questions received *nothing at all* against 31 for the breadcrumb baseline — a
single contextualised child no longer fit inside the budget, so the strict-fill
rule correctly returned an empty result, 46 times. Under `delivered` that number
is **21**, lower than the baseline's own 31.

### What this did to the result the project leads with

`parent_strict < child_strict` was measured under the same asymmetry, and
children carry the invisible overhead while parents do not. The direction of the
distortion was knowable in advance — it penalises children — but the size was
not, so the number was provisional until re-measured. **Re-measured on the
breadcrumb index, it holds and widens:**

| budget | gap under `indexed` | gap under `delivered` |
|---|---|---|
| 250 | +4 | **+8** |
| 500 | +13 | **+16** |
| 1000 | +15 | **+18** |
| 1500 | +8 | +8 |
| 3000 | +4 | +4 |
| 6000 | +3 | +3 |

Unchanged at and above the headline budget, wider below it. **The old accounting
was handicapping the winner**, so the published claim was conservative rather
than inflated — the better direction to have been wrong in, and not one that
could be assumed without checking.

### Why both accountings stay reachable

`RAGKIT_CHILD_COST_BASIS=indexed` still works, and `scripts/contextual_ab.py`
still runs the full 2×2. Every figure this project published before the
correction was measured under the old basis, and a comparison nobody can re-run
is an assertion. Same reasoning as keeping `parser_survey.INVALID.json`: the
record of how a number was found to be wrong is worth more than the corrected
number standing alone.

The eval now records `child_cost_basis` alongside the pipeline fingerprint, and
the CI gate refuses to compare across it — because a run under `indexed` and a
run under `delivered` answer different questions under the same field name, which
is exactly what the fingerprint rule already exists to prevent.

The general form, and it is A-12's shape one level up: **a normalised metric is a
claim about one specific resource. Charge every unit that resource, and nothing
else.** A cost function that quietly includes index-time text is measuring storage
while reporting delivery.



## A-14 · Context helps a passage that cannot describe itself, and hurts one whose value is being unlike its neighbours

The most transferable thing to come out of D-6, and it is a rule about *when* a
technique works rather than whether it does.

A situating prefix adds a sentence saying what a passage is about. For a table of
numbers whose headers were lost in extraction, that is transformative: the
passage previously carried nothing searchable and now states what it is a table
*of*. Measured, `table_or_image` recall went **32/38 → 35/38**, and generation
declines on the same stratum went **29% → 21%**. It is the weakest subsystem in
the project and it gained the most.

For a bibliography it is poison, and the mechanism is worth stating exactly.

A bibliography's entire retrieval value is that it contains a rare string nobody
else has. After contextualisation, **every** reference section in **every**
document carries a prefix saying approximately *"in the references section of
this paper, these citations document related work on…"*. That is accurate,
useful, document-level context — and it is **shared vocabulary across the
corpus**. A query whose only discriminating signal is one rare token now competes
against fifteen passages that all match the generic description well.

Measured, on the one item that lost its document:

| | children retrieved | from the right document |
|---|---|---|
| breadcrumb | 6 | 2 |
| contextual | 19 | **0** |

Nineteen passages were retrieved and none of them were the right one. The prefix
did not make the target worse; it made the *competition* better at looking like a
plausible answer to a citation lookup.

The analogy that makes it stick: labelling every box in the attic "OLD PAPERS".
Helpful for most of them, and disastrous for the one box you could previously
find because it was the only one with your passport in it.

**So the rule, and it is predictable from the content type in advance:**

> A situating prefix helps a passage that cannot describe itself, and hurts a
> passage whose retrieval value is precisely that it is unlike its neighbours.

**Why this matters more than the one lost item.** 1 of 92 is a rounding error
against a +3 headline. The mechanism is not, because it names a content type
where the technique backfires — and **contracts, invoices, part catalogues and
policy schedules are made of that content type.** A clause number, an invoice ID,
a part code: the rare token *is* the query. Any corpus of those is a corpus where
enabling this feature uncritically would cost retrievals, and the failure would
be silent — a slightly worse ranking, forever, on exactly the lookups the
customer cares most about.

That is the note to re-read before turning contextual prefixes on for the first
non-academic corpus. It is recorded as a deferral with a predicate rather than as
a fix, for reasons in Part 3.

## A-15 · An experiment writes to its own namespace, never to the primary artifact

Twice now, running an experiment has quietly rewritten the record of the shipped
system:

1. **The manifest.** Building a second index for an A/B repointed
   `data/index/manifest.json` at a different pipeline. The demo kept serving with
   a manifest claiming a fingerprint its own chunks did not carry.
2. **The eval results.** The 2×2 comparison ran 24 evals, each of which wrote
   `data/eval/eval_results.json`, leaving the primary artifact describing the
   last cell — a 250-token run against the wrong index.

Neither failed loudly. Both were caught by something refusing to combine
mismatched inputs — the failure histogram declining to pool across fingerprints,
which is a check built for a different reason entirely.

Two instances is a pattern, and the pattern has a shape: **a measurement is
supposed to be a read, and these were writes.** The fix is not to be careful.

**What was done, in the order it should have been done.** The first attempt was
snapshot-and-restore in the comparison script — correct, and at the wrong layer:
it defended one caller against a hazard that every caller had. So instead:

- `eval.run()` takes `artifact: Path | None = None` and writes **nowhere** by
  default. The CLI — the caller that speaks for the shipped system — is the one
  that names the primary path. The snapshot-and-restore was then deleted, which
  is the test of whether a fix is structural: the workaround becomes removable.
- Manifests and index reports are index-scoped, so a non-default index cannot
  write the default one's files.

**And one check that does not depend on knowing the cause.** The prevention above
is specific to two mechanisms; a third would find a third door. So the reconciler
now asserts the *property*: **the headline artifact describes the index that
actually ships** — fingerprint read off the chunks on disk, plus the budget and
the cost basis it was measured under.

**That check immediately found a hole inside the mechanism it was checking.** Fed
a deliberate reproduction of incident (2), it *held*. `_index_provenance` read
`pipeline_fingerprint` from `index_report.json` — a sidecar describing the last
ingest, not the index being scored — so an eval of index B produced a payload
stamped with index A's fingerprint. **The rule that exists to refuse comparisons
across systems would have reported a match between two different systems.** Same
defect as A-12 and one level further in: the stamp was being copied from
whichever file was nearest rather than read off the thing it describes.

Fixed by reading the fingerprint from the chunks, which travel with what is being
measured. Sidecar fields are now returned only when the sidecar provably
describes the same index, and `index_report_describes_this_index` says so
explicitly so a null reads as "that file was about something else" rather than
"not measured".

**Then the same rule caught a third instance, one level in.** The completeness
half of that invariant — added an hour later, after the Inspector's
recall-vs-budget chart was found rendering *blank* — failed immediately. Cause:
the CI gate runs `--no-sweep`, and the CLI wrote the primary artifact
unconditionally, so **every CI invocation replaced a complete artifact with one
whose budget sweep was empty**. The artifact stayed otherwise valid: right index,
right budget, right cost basis, every other check green. Only the chart was gone,
and only on the demo.

That is the third door, found exactly where the argument above predicted one
would be — and it says something about the first two checks. They asserted the
artifact's *identity* and never its *completeness*, because identity was the
property the two known incidents had violated. **A check written from the last
bug tests the last bug.**

A gate is a *read*: it compares a run against the stored baseline and has no
business rewriting the headline. The CLI now writes the primary artifact only
when it is producing the shipped measurement — never on `--gate`, `--no-sweep`,
or `--limit`.

The general form: **a measurement must not have side effects on the record of the
thing it measures; its provenance must be read from the subject rather than from
whatever file is closest; and a record can be wrong by being incomplete, not only
by being about something else.**


## A-16 · A check written from the last bug tests the last bug

Every invariant in this project was written from an incident, which means each
one tests what that incident violated. That is how they came to be good, and it
is also a systematic blind spot: **no check tests a failure mode nothing has
caused yet.**

The completeness bug made the shape visible. The two "headline artifact" checks
asserted the artifact's *identity* — right index, right budget, right cost basis
— because identity was what both known incidents had corrupted. An artifact whose
budget sweep was empty satisfied all of them, and the demo's recall chart
rendered blank while the reconciler reported green.

So rather than wait for instance four, one pass over all nineteen checks asking a
different question: **what would a valid-but-useless version of this look like?**
Five answers were uncomfortable:

| check | passes perfectly when |
|---|---|
| Index parity | the index is **empty** — 0 indexed == 0 embedded, 0 dropped |
| Parent resolution | there are **no children**, so no orphans |
| Uniform provenance | one provenance among **zero** chunks is one provenance |
| Scoring sanity | `partial 0 >= strict 0` — an eval that scored nothing |
| Context budget | **the retriever returns nothing**, at every budget, forever |

The last one is the sharpest, and it is a lesson this project had already learned
and not carried across. `Upload retrievability` exists *precisely* because a
filter that returns nothing passes both isolation tests — the control case was
written for exactly this reasoning, in a file one directory away, and was never
pointed at the budget invariant.

**Two fixes, of different kinds.**

A *generic* one: a HOLDS over a population of zero is downgraded to
`NOT_MEASURED`, with the population named. Not to FAILS — an empty index is not a
broken index, it is an absent one, and `NOT_MEASURED` already means exactly that.
A check with no subjects abstained; it did not hold.

And a *specific* one, because the budget case cannot be caught generically:
strict fill legitimately returns nothing at a tight budget, so the control has to
be stated where silence is not a defence. **At the largest measured budget, both
units must deliver more than zero tokens.** Verified by simulation: with every
delivery zeroed, `Context budget` still reports HOLDS — correctly, it is a true
statement — and the new control reports FAILS.

Same pass caught `Provenance propagation` satisfying a biconditional over two
zeros, and `Context budget` reading an empty sweep as compliance — the
completeness bug again, in a second check, found because the pass went looking
rather than waiting.

The general form: **an assertion that can only be violated by activity is not an
assertion about correctness, it is an assertion about activity.** Pair every
"nothing bad happened" invariant with a "something happened at all" control, or
silence will pass as compliance.

### A corollary, learned the embarrassing way

**A check that reimplements what it checks is testing itself.**

The first draft of the upload-reachability invariant copied `/api/asset`'s
containment logic into the check, wrote the *corrected* version of it there, and
reported HOLDS — while the endpoint it was supposedly guarding was broken in two
directions and serving strangers' documents. It was a green check over a live
vulnerability, and the greenness came from the check grading its own copy.

That is this project's oldest failure — the survey script that imported a
different parser than the pipeline and reported table counts differing 5× on
identical bytes — arriving a year of lessons later **inside a security check
written to catch a different instance of the same thing.** The pattern does not
get easier to see with practice; it gets easier to see *if you look*.

The fix is the only version that means anything: the containment lives in one
named function, the endpoint and the check both call it, and reverting the fix
makes the check fail. That last clause is the test of whether a check is
load-bearing — not that it passes, but that a specific known break makes it stop
passing.

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

**A second defect, fixed but with an unprovable blast radius.** Until A-15,
`_index_provenance` read `pipeline_fingerprint` from `index_report.json` — a file
describing the last *ingest*, not the index being *scored*. So **every eval run
against a non-default index, before today, carried the wrong fingerprint**: a
measurement of index B stamped with index A's identity, by the mechanism whose
entire job is refusing to compare across systems.

The fix is in. What cannot be established is the blast radius. Nothing published
appears to depend on one of those runs — the 2×2 in D-6 reads each arm's
fingerprint off its own chunks, and every headline figure came from the default
index, where the sidecar happened to be correct. But "appears to" is the accurate
verb. There is no log of which indexes were ever scored, so **it cannot be proven
that no earlier comparison was affected**, and a claim of "no impact" would be a
guess wearing the clothes of an audit.

Recorded at the strength it can be supported: one defect, one fix, one unprovable
scope. The same reason the 18% headerless figure was withdrawn rather than
qualified — the confidence, not the number, was the part that was wrong.

---

## Appendix — the numbers

All produced by running something. Reproduce with `ragkit eval`,
`ragkit judge score`, `ragkit reconcile`, `ragkit audit`, `ragkit spend`.

**Which run.** Every generation-tier figure below comes from one stamped artifact:
fingerprint `4d54ab24999d336e`, parser `pymupdf4llm@1.28+...`, budget `1500`,
child cost basis `delivered`. (The **was** column is fingerprint
`6fd55e19a82a7c28` under basis `indexed`.) Artifacts that record neither cannot be pooled with
each other, which is why the stamp exists -- and why the failure analysis refuses
an unstamped input rather than inheriting a fingerprint nobody verified.

**These numbers move between runs, and the variance is not quantified.** Re-running
the generation tier on an unchanged index and an unchanged budget once moved
abstentions 17 -> 18 and `table_or_image` 27% -> 29%, and made one judge verdict
that had failed succeed. Generation is sampled, so a single run is a sample of one. Nothing
here reports a confidence interval over repeated runs, and a difference of one or
two items between runs should not be read as a change in the system.

Measured under `contextualizer=llm-prefix@1`, `child_cost_basis=delivered`,
fingerprint `4d54ab24999d336e`. The **was** column is the same suite under
`breadcrumb-only` / `indexed` — kept because two of these moved for reasons that
are not visible from the new number alone (D-6, A-13).

| | | was |
|---|---|---|
| `child_strict` @1500 tok | 82/92 = **89%** (5 of 7 strata) | 79/92 = 86% |
| `source_hit` @1500 tok | 91/92 = **99%** (100% @≥3000) | 92/92 = 100% |
| Retrieval misses by budget | 54 @250 · 10 @1500 · 5 @3000 · **0 @12000** | 67 · 14 · 4 · 0 |
| NDCG@5 dense vs RRF | 0.848 [0.761, 0.907] vs 0.880 [0.798, 0.932] | — |
| RRF on `exact_identifier` @500 | 68% → **89%** | — |
| Faithfulness over **answers** | 78/78 = 100% | 74/74 = 100% |
| Abstention rate | 14/92 = **15%**; `table_or_image` **8/38 = 21%** | 18/92 = 20%; 11/38 = 29% |
| Measured failures | 15 — FP2 8 · FP3 7 · **FP4 0** | 23 — FP2 14 · FP3 9 · FP4 0 |
| Judge κ vs hand labels | **0.897** (raw 0.933, chance 0.353, n=30) | unchanged |
| Invariants | 0 failing · 17 passing · 3 not measured | 0 · 9 · 3 |
| Guard coverage | 8 paid routes · 4 guarded · 4 exempt · 0 unguarded | unchanged |
| Reachability | 0 unexplained · 3 explained by deferrals | 0 · 4 |
| Container | 150 MiB / 512 · 22 s boot · ~7 s per answer | unchanged |

**A number that must always carry its qualifier:** the retrieval-miss count is a
function of the budget knob, not a property of the system. At
`TOKENS_CONTEXT_BUDGET = 12000` there are **zero** retrieval failures. "10 misses"
means nothing without "@1500" — and, since A-13, without "charged delivery".
