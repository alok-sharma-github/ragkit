# Failure analysis

**23 measured failures**, every one from a stamped artifact, classified against Barnett et al.'s seven failure points.

## Provenance

| | |
|---|---|
| pipeline fingerprint | `6fd55e19a82a7c28` |
| parser | `pymupdf4llm@1.28+lines-detector@1+quadcorr-unset+cleanhdr@2+pagetextrepair@3+bboxflags@1` |
| chunker | `header_aware_parent@3` |
| **token budget** | **1500** |

All three inputs carry the same fingerprint and budget; the script refuses to pool otherwise, and treats a *missing* stamp as a mismatch rather than a pass.

**Every count here is @1500 tokens.** The retrieval-miss count is a function of that knob, not a property of the system -- at the 12000-token default there are zero retrieval failures. A count quoted without its budget says nothing.

## Histogram

| failure point | n | share | |
|---|---:|---:|---|
| **FP1** Missing Content | 0 | 0% |  |
| **FP2** Missed Top Ranked | 14 | 61% | █████████████████ |
| **FP3** Not In Context | 9 | 39% | ███████████ |
| **FP4** Not Extracted | 0 | 0% |  |
| **FP5** Wrong Format | 0 | 0% |  |
| **FP6** Incorrect Specificity | 0 | 0% |  |
| **FP7** Incomplete | 0 | 0% |  |
| | **23** | | |

## What each bar is made of

| cause | n | failure point | why |
|---|---:|---|---|
| `evidence_absent` | 9 | FP2 | the needle exists in the corpus and retrieval did not deliver it |
| `evidence_partial` | 8 | FP3 | some needles arrived and the rest did not survive the budget fill -- retrieved, then consolidated away |
| `retrieval_miss_answered` | 5 | FP2 | the golden needle was not retrieved at this budget AND the model answered anyway -- from other context, so the answer may still be right; the measurement cannot tell |
| `starved_by_budget` | 1 | FP3 | candidates ranked correctly and NONE fit the token budget, so nothing reached the model |

## By question type

| stratum | failures |
|---|---:|
| table_or_image | 14 |
| simple_factual | 5 |
| exact_identifier | 3 |
| multi_hop | 1 |

## What I would fix first

**Retrieval, and specifically ranking within a document.** All 23 failures are FP2 or FP3 -- the evidence exists in the corpus and either was not retrieved or did not survive the budget fill.

**No failure at all is FP4** -- not one case where complete context was delivered and the model failed to use it. `evidence_present` is **0** across every abstention: the generator never declines with complete evidence in front of it. Whatever is wrong here, it is not the model's reading.

**And it concentrates.** `table_or_image` accounts for 14 of 23 failures (61%) -- consistent with its abstention rate being the highest of any stratum. Tables and figures are the weak subsystem, not retrieval in general.

**What that rules out.** A zero here is a strong negative result, not an absence of data: 24 chances for the model to receive complete context and misuse it, and it took none of them. So prompt engineering, lost-in-the-middle reordering, context compression and fine-tuning the reader (RAFT and similar) are not deprioritised -- they are **ruled out on evidence**, because every one of them improves a step that is not failing. Spending on any of them would be spending where the measurement says nothing is wrong.

That points at one change rather than several. `source_hit` is 100% at every budget, so the right *document* is always found and the loss is ranking *within* it -- which is exactly what a situating prefix addresses. Contextual retrieval's deferral has already expired on its own predicate, and this histogram is the second, independent argument for it.

**What this cannot tell you.** FP1 is structurally absent: the golden set was generated *from* the corpus, so a question whose answer is missing entirely cannot appear. FP6 and FP7 need human judgement of answer quality that no automated check here performs. Their zeros mean **not measured**, not **does not happen**.
