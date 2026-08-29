# Glossary

Terms used in the README, the decision record, and the inspector screen.

**Every entry carries the qualifier that must travel with the term.** That is the
point of this file, not decoration. Several of these words are meaningless — or
worse, actively misleading — without a condition attached: a recall figure without
its token budget describes a setting rather than a system, and an agreement
statistic without its sample composition is a claim about one thing being read as
a claim about another. A glossary that defined the word and dropped the qualifier
would reintroduce the exact error the measurement work exists to prevent.

Where a number appears below it is the current measured value, and it is stated
with the condition it was measured under.

---

## Retrieval

**Chunk** — a passage cut from a document, small enough to be matched precisely
against a question. Documents are cut up because a whole PDF is too big to hand a
model, and too coarse to point at.

**Child / parent** — two sizes of the same passage. The small **child** (~300
words) is what gets matched against a question; the larger **parent** (~1,200
words) is what the model actually reads. Splitting these lets a passage be
precisely findable *and* usefully complete, which one size cannot be at once.
→ *Qualifier: parents here are much smaller than the 1,200-word target — median
190 words — because sections in this corpus are short. The knob does not control
what its name suggests.*

**Dense retrieval** — matching by meaning. Every passage and every question is
turned into a list of numbers positioned so that similar meanings land near each
other; the nearest passages win. Finds a paraphrase; misses an exact code.

**Sparse retrieval / BM25** — matching by words. Scores a passage on how often the
question's words appear in it, discounting words common across the whole corpus
and correcting for passage length. Finds an exact part number; misses a paraphrase.

**Hybrid retrieval / RRF** — running both and merging. The two methods score on
incompatible scales, so their *scores* cannot be averaged; their **rankings** can.
Reciprocal Rank Fusion combines the two ordered lists by position rather than
score.
→ *Qualifier: the aggregate gain here is inside the noise — NDCG@5 0.848 [0.761,
0.907] dense against 0.880 [0.798, 0.932] fused, intervals overlapping. The real
effect is in one slice: questions containing exact identifiers, **at a 500-token
budget**, go 68% → 89%. Quoting either number without the other claims a win the
data does not support.*

**Token budget** — the amount of text allowed into the model for one answer,
counted in tokens (roughly ¾ of a word each). Everything retrieved must fit.

**Budget-normalised recall** — measuring retrieval at a fixed amount of *text*
rather than a fixed *number* of passages. Comparing "ten children" against "ten
parents" compares 3,000 words against 12,000, so the larger unit wins by being
given more room. At a fixed budget the comparison is fair.
→ *Qualifier: **every recall figure in this project is meaningless without its
budget.** From one sweep over the same 92 questions — misses at 250 tokens: 67;
at 1,500: 14; at 12,000: zero. A miss count without its budget describes a knob
setting rather than a system. (That sweep's 1,500 figure is 78/92; the headline
below is 79/92 from a later run — see **pipeline fingerprint**.)*

**Strict fill** — the rule that retrieval never exceeds the token budget, even when
that means returning nothing at all.
→ *Qualifier: returning nothing is a finding, not a bug — 7 of 442 parents are
individually larger than a 1,500-token budget, so no amount of ranking helps.*

**Starved by budget** — the case where passages ranked correctly and none fit. A
distinct condition from "nothing was found", because the two have opposite fixes:
raise the budget, versus look at the retriever.

---

## The measurements

**`child_strict`** — the fraction of test questions where every fact needed to
answer was inside the retrieved children. **Currently 79/92 = 86%.**
→ *Qualifier: at a 1,500-token budget, on five of seven question types. Two types
have no test items at all.*

**`source_hit`** — the fraction where the correct *document* was retrieved,
regardless of which passage. **Currently 92/92 = 100%.**
→ *Qualifier: this is why no reranker was added. The right document always
surfaces, so every remaining failure is ranking within a document — and reranking
reorders documents that are already correct.*

**Faithfulness** — the fraction of answers whose every claim is supported by the
sources given to the model. **Currently 74/74 = 100%.**
→ *Qualifier: **over answers only.** Refusals are counted separately, and this is
close to tautological anyway — the model is confined to the supplied sources and a
citation checker already rejects unverifiable quotes before anything is scored. It
detects regressions; it is not a quality score.*

**Abstention rate** — how often the system declines to answer. **Currently 18/92 =
20%**, and **29%** on questions about tables and figures.
→ *Qualifier: this must never share a denominator with faithfulness. Pooled, the
first report read "99% supported" — because a refusal asserts nothing and so
asserts nothing false, and the score therefore improved whenever the system
answered fewer questions.*

**Cohen's kappa** — agreement between two graders, with agreement-by-luck
subtracted. Two graders who both mark everything "correct" agree 100% of the time
and have demonstrated nothing; kappa removes that. **Currently 0.897** (raw
agreement 0.933, chance 0.353).
→ *Qualifier: **n = 30, on a deliberately class-balanced sample** (13 partly / 10
supported / 7 unsupported). That makes it a *discrimination* estimate — evidence
the grader can tell the three verdicts apart. It is **not** a precision claim on
the real population, which is nearly all "supported".*

**Wilson interval** — the error bar on a percentage from a small sample. Two
figures whose intervals overlap are not distinguishable.

**Counts, not rates** — below ten samples this project prints "1 of 2" rather than
"50%", because a percentage implies a precision the sample does not have.

**NOT_MEASURED** — a third state alongside pass and fail, meaning no data exists.
Kept distinct because a blank cell reads as a zero, and "we did not test this" and
"this scored nothing" are opposite claims.

---

## Evidence and citations

**Citation** — a claim's pointer to the passage supporting it, carrying a
**verbatim quote** that is checked against the source before the answer is shown.

**The four citation states** — *verified* (the quote is in the cited passage);
*absent* (the passage exists but the quote is not in it); *fabricated* (the cited
passage was never sent to the model); *unquotable* — see below.

**FOUND — NOT QUOTED** — the system located the answer and declined to quote it,
because the text it holds cannot be quoted faithfully. Usually a table whose
column headers were reconstructed.
→ *Qualifier: this is the behaviour the product exists to demonstrate, and it is
invisible on our own corpus — a visitor cannot know the table was damaged. It only
reads as a catch to someone who knows what is in the document.*

**Provenance** — a per-passage record of where its text came from: copied
**verbatim**, **prefixed** with a heading trail, or **model-generated** (an image
description). Stored, not inferred, so a model-written caption can never be
displayed as a quotation from the document.

**Tombstone** — the marker left when a cited document is deleted, so an old answer
renders as "this source was removed" rather than as a broken reference.

---

## The test set

**Golden set** — the question-and-answer set the system is scored against.
→ *Qualifier: generated **from** the corpus, so a question whose answer is missing
entirely cannot appear. Nobody has hand-checked a stratified sample — the
inspector reports this as `0 of 97 verified`, deliberately.*

**Needle** — the exact phrase that must appear in retrieved text for a question to
count as answered. Anchoring on text rather than on passage identifiers means the
test survives re-chunking.

**Distractor** — a plausible wrong answer recorded alongside the right one, so a
wrong answer can be diagnosed rather than merely counted.

**Stratum** — a question type: *simple factual*, *exact identifier*, *table or
image*, *multi-hop*, *aggregative*, *ambiguous*, *out of scope*.
→ *Qualifier: aggregative and ambiguous have **no items**. Every headline is
therefore "on five of seven".*

**The seven failure points (FP1–FP7)** — a published taxonomy of how retrieval
systems fail: content missing entirely, retrieved but not ranked, ranked but not
fitted into the context, present but unused by the model, wrong format, wrong
specificity, incomplete.
→ *Qualifier: **all 23 measured failures here are FP2 or FP3** — evidence exists
and was not delivered — and **none are FP4**. Zero FP4 across 23 chances is a
strong negative result: it rules out prompt engineering, context reordering and
reader fine-tuning, because each improves a step that is not failing.*

---

## Operations

**Pipeline fingerprint** — a short code identifying the parser and chunker versions
that produced an index. Two sets of numbers can only be compared if their
fingerprints match.
→ *Qualifier: it covers the pipeline, **not the index files it produced.** A
rebuilt index that differs in passage ordering carries the same fingerprint — the
headline moved 78/92 to 79/92 between two runs whose fingerprints were identical.*

**Invariant / reconcile** — a check that must hold before any score is trusted,
executed rather than asserted. Currently 0 failing, 9 passing, 3 not measured.

**Deferral** — a decision *not* to build something, stored with the condition that
reverses it, so it expires itself rather than being remembered. One has already
fired on its own terms.

**Owner** — who may retrieve a passage. The shared demo corpus is public; an
uploaded document belongs to the session that uploaded it and to nobody else.
→ *Qualifier: the public value is the empty string, which makes it the one value
whose bug is a leak rather than a miss — everything else fails toward a passage
being unreachable, which is visible.*

**Session** — a temporary sandbox for one visitor's uploads, identified by an
unguessable random value in a cookie and purged on a timer.

**Spend ceiling** — two limits on paid API use: a per-operation cap checked before
the first request, and a daily total.
→ *Qualifier: only the per-operation cap survives a restart. The daily counter is a
file and resets, which is recorded rather than glossed.*
