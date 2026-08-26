"""
Every knob in the system lives here, in one place.

WHY THIS FILE EXISTS AT ALL (this is the lesson, not the code):

Your study guide's M16 says the framing that reads as senior is "every layer is
optional; start with the simplest thing and add layers only where error analysis
says the failures are." You cannot make that argument if your chunk size is
hardcoded in three files and your model ID is a string literal inside a function.

An eval harness (Session 10) works by changing ONE knob, re-running the suite,
and diffing the numbers. That only works if the knobs are addressable. So this
file is a prerequisite for measurement, not just tidiness.

Read the comments — several of them are the interview answer for that knob.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"          # the corpus
DATA_INDEX = ROOT / "data" / "index"      # serialised indexes
DATA_EVAL = ROOT / "data" / "eval"        # golden set + baselines
CACHE_DIR = ROOT / ".cache"               # embedding cache, LLM response cache
EXPERIMENTS_OUT = ROOT / "experiments" / "out"
NOTES = ROOT / "notes"


# --------------------------------------------------------------------------
# Gemini models
# --------------------------------------------------------------------------
#
# These IDs were read from Google's live docs, not from memory. Session 0's
# smoke test calls client.models.list() with YOUR key and prints what your
# project can actually reach — free-tier availability differs per project, so
# we verify rather than assume.
#
# Deliberate model-tiering, because free tier makes this a real constraint:
#
#   WORKHORSE  -> generation, contextual prefixes, image captioning, condensation
#   CHEAP      -> binary/classification decisions where a big model is waste
#   JUDGE      -> the eval harness only. Never in the request path.
#
# The guide's M14 calls this a "model cascade". Notice the shape of the
# argument: you do not pick one model, you allocate models to jobs by how much
# the job's error costs you.

# CORRECTION (yours, and you were right to check): I asserted all four of these
# came from live docs. Two are confirmed — `gemini-3.7-flash` (GA 2026-08-13) and
# `gemini-embedding-2` (multimodal). The other two I could not confirm, and this
# family churns fast: `gemini-3.1-flash-lite-preview` was already shut down in
# May 2026. Asserting an unverified ID in a config comment is exactly the kind of
# stale-prior error that costs an afternoon.
#
# So the shape changes: each role holds an ORDERED CANDIDATE LIST, and Session 0
# resolves it against client.models.list() with your key. First reachable
# candidate wins. Unavailable IDs are a normal outcome, not a crash — and the
# resolved choice gets written to data/index/resolved_models.json so every later
# run and every eval baseline records which model actually answered.

@dataclass(frozen=True)
class Models:
    # Generation, contextual prefixes, image captioning, query condensation.
    # Multimodal, so "supports images" needs no separate vision model.
    workhorse: tuple[str, ...] = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash")

    # Routing ("does this even need retrieval?"), relevance gating, format checks.
    # Near-free, and the task is nearly trivial. Ordered newest-first; the
    # flash-lite line is the one that churns most, hence three fallbacks.
    cheap: tuple[str, ...] = (
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3.7-flash",  # last resort: correct, just not cheap
    )

    # LLM-as-judge, eval harness ONLY — never in the request path.
    # Must be a DIFFERENT model from `workhorse`, not merely a stronger one.
    # See JUDGE_* below for why the fallback rule is "wait", not "swap".
    judge: tuple[str, ...] = ("gemini-3.1-pro-preview", "gemini-2.5-pro")

    # Embeddings. MEASURED, not assumed — see the batching note below, which is
    # why the older model is first. `embedding_multimodal` is kept reachable for
    # the Session 9 comparison and for embedding images directly.
    embedding: tuple[str, ...] = ("gemini-embedding-001", "gemini-embedding-2")
    embedding_multimodal: tuple[str, ...] = ("gemini-embedding-2",)


MODELS = Models()

# Written by Session 0's smoke test, read by everything else. Keeping the
# resolved IDs on disk rather than re-resolving per process means an eval run
# and the CI baseline it compares against provably used the same models.
RESOLVED_MODELS_PATH = DATA_INDEX / "resolved_models.json"


# --------------------------------------------------------------------------
# Embedding configuration
# --------------------------------------------------------------------------
#
# ASYMMETRIC EMBEDDING — the thing most tutorials get wrong.
#
# A question and the passage that answers it do not look alike. "What was Q3
# revenue?" and "Revenue for the third quarter was $4.2M" are different in
# surface form. If you embed both with the same instruction, you are asking the
# model to put questions near questions and passages near passages — which is
# the opposite of what retrieval needs.
#
# So: documents get indexed with a "search result" instruction, queries with a
# "search query" instruction. Same model, two different embedding spaces
# deliberately aligned to each other.
#
# This is also the mechanism behind the guide's M5 note that "question-to-question
# matching beats question-to-passage matching" — hypothetical-question indexing
# is the extreme version of the same idea.

# The two models express the SAME idea through different APIs, which is why the
# client has to know which one it is talking to:
#   gemini-embedding-2   -> a text prompt prefix (below)
#   gemini-embedding-001 -> a task_type parameter (RETRIEVAL_DOCUMENT / _QUERY)
# Whichever scheme was used is recorded in the embedding cache key, because a
# vector produced under one scheme is not interchangeable with one produced
# under the other even at identical model and dimension.
EMBED_PREFIX_DOCUMENT = "task: search result | "
EMBED_PREFIX_QUERY = "task: search query | "
EMBED_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# gemini-embedding-2 supports 128..3072 (default 3072). Recommended: 768 / 1536 / 3072.
#
# MATRYOSHKA (guide M6): the dimensions are ordered by importance, so truncating
# 3072 -> 768 keeps most of the quality at a quarter of the storage. We default
# to 768 because at our corpus size it is free quality-wise and 4x cheaper in
# RAM — and in Session 9 we MEASURE that claim instead of trusting it.
#
# Capacity arithmetic you should be able to do out loud in an interview:
#   768 dims x 4 bytes (float32) = 3 KB/vector
#   x 1.5-2.0 HNSW graph overhead ~= 5 KB/vector
#   x 50M chunks ~= 250 GB RAM. That is why quantization exists.
EMBED_DIM = 768
EMBED_DIM_FULL = 3072  # for the Session 9 Matryoshka comparison

# gemini-embedding-2 input limit is 8192 tokens. We chunk far below that, but
# contextual prefixes + parent documents can push a payload up, so the client
# guards against it rather than letting the API 400.
EMBED_MAX_INPUT_TOKENS = 8192

# Free tier makes re-embedding expensive in wall-clock time (rate limits), not
# just money. An on-disk cache keyed by sha256(model|dim|prefix|text) means
# re-running an experiment costs zero API calls. This is not an optimisation —
# without it, a chunking sweep across 5 configs is unrunnable on a free key.
EMBED_CACHE = CACHE_DIR / "embeddings"

# LLM RESPONSE CACHE for image captions, and it is not optional.
#
# Measured: two loads of the same PNG each cost an API call (8.1s, 6.9s) and
# returned DIFFERENT text. Three consequences, all bad:
#   1. every ingest re-captions every image -- recurring quota on a free key
#   2. the index is not reproducible: chunk counts and chunk_ids move between
#      ingests (image_caption children went 11 -> 12), so an eval run and its
#      baseline can describe different corpora, which defeats the CI gate
#   3. parsed_hash changes every run for images, so the parser-drift detector
#      cannot separate real drift from expected model variance
#
# Keyed on the IMAGE BYTES plus model plus prompt version -- every input to the
# call, the same rule as the embedding cache. Bump CAPTION_PROMPT_VERSION when
# the prompt changes, or old captions will be served for a new question.
CAPTION_CACHE = CACHE_DIR / "captions"
CAPTION_PROMPT_VERSION = 1

# How many texts per embed_content call. Kept small on purpose: a failed batch
# of 100 wastes 100 items' worth of quota, and free-tier 429s are common.
#
# BUT: batch support is a property of the MODEL, and it is not documented
# anywhere I could find. Measured with your key:
#
#   gemini-embedding-001  ->  sent 3, got 3   (batches)
#   gemini-embedding-2    ->  sent 4, got 1   (silently returns only the first)
#
# That second row is why `gemini-embedding-001` is the default. On free tier the
# binding constraint is REQUESTS PER MINUTE, not tokens or money — so 300 chunks
# is ~19 calls on -001 and 300 calls on -2. Same corpus, one runs in seconds and
# the other is a rate-limit endurance test.
#
# The cost of choosing -001: text-only (no direct image embedding) and a
# 2048-token input limit instead of 8192. Neither binds us — images reach the
# index as Gemini-written captions, which is better for citation anyway (you can
# show the user WHY an image matched), and our child chunks are ~300 tokens plus
# a ~100-token contextual prefix.
#
# The client PROBES this per model rather than trusting the table above, because
# a model returning fewer embeddings than you sent is invisible unless you check.
EMBED_BATCH_SIZE = 16
EMBED_MAX_INPUT_TOKENS_BY_MODEL = {
    "gemini-embedding-001": 2048,
    "gemini-embedding-2": 8192,
}


# --------------------------------------------------------------------------
# Chunking (Sessions 2-4)
# --------------------------------------------------------------------------
#
# "What chunk size do you use?" is the guide's named trap (M5). The correct
# answer starts with "it depends on the retrieval unit vs the generation unit"
# — which is why there are TWO sizes here, not one.
#
#   CHILD  = what you EMBED.  Small, topically tight -> precise retrieval.
#   PARENT = what you RETURN. Large, self-contained  -> good generation context.
#
# Decoupling these is the single idea missing from your linear model. Session 3
# implements it; Session 4 measures whether it actually helped.

CHUNK_CHILD_TOKENS = 300
CHUNK_PARENT_TOKENS = 1200
CHUNK_OVERLAP_RATIO = 0.12  # guide's starting point: 10-15%

# Contextual Retrieval (guide M5 — Anthropic 2024; 35% / 49% / 67%).
# A 50-100 token LLM-written blurb prepended before embedding, situating the
# chunk in its document. Fixes "revenue grew 3%" being unretrievable because
# the chunk never names the company or the quarter.
CONTEXTUAL_PREFIX_MAX_TOKENS = 100

# Implicit context caching is automatic on Gemini 2.5+ — nothing to enable.
# But the minimum cacheable prefix on 3.x flash is 4096 tokens, so a document
# shorter than that gets NO cache benefit and contextualising it costs full
# price per chunk. We check this and report it, because "we used prompt caching"
# is worthless if you never verified the cache fired.
IMPLICIT_CACHE_MIN_TOKENS = 4096


# --------------------------------------------------------------------------
# Index provenance — the mixed-index defect
# --------------------------------------------------------------------------
#
# This whole section exists because of your answer to Q2, which was sharper than
# my question. Restating the failure so the settings below have a reason:
#
# Quota does not fail on a random 20% of chunks. It fails at a POINT IN TIME, so
# the un-prefixed chunks are a contiguous tail of ingest order — and ingest order
# correlates with something structural (filename sort, directory walk, upload
# date). The result is not "80% of the corpus is good". It is "documents A-M are
# retrievable and N-Z are systematically harder to find." Whole regions go quiet.
#
# And the bias operates WITHIN a single query: prefixed chunks outrank un-prefixed
# ones by design, for reasons unrelated to relevance. You do not observe degraded
# quality. You observe confident, well-ranked, wrong answers.
#
# A terminal warning is the wrong artifact — it lives for one scroll, the index
# lives for weeks. So the defect gets recorded IN THE INDEX.
#
# GENERALISATION worth making now: `has_contextual_prefix` is one field of a
# per-chunk PROVENANCE record, not a special case. The identical failure shape
# appears whenever any index-time input changes mid-corpus — embedding model,
# output dimensionality, prefix scheme, parser version, chunker version. That is
# the guide's M6 "embedding model versioning" problem, and it is why the answer
# there is a blue/green index swap: a half-migrated index is the same bug at a
# larger blast radius. Recording all of it per chunk is what makes the eval
# harness able to refuse an uninterpretable comparison.
INDEX_PROVENANCE_FIELDS = (
    "has_contextual_prefix",  # the quota-driven one
    "embedding_model",
    "embedding_dim",
    "embedding_prefix_scheme",
    "parser",
    "chunker",
    "ingested_at",
)

# Your fix: resumable backfill. With the flag written per chunk, quota recovery
# re-processes only what is missing instead of re-ingesting 100%. On a free key
# that is the difference between a ten-minute repair and an unrunnable one.
INGEST_RESUMABLE = True

# Your fix: an ingest manifest that outlives the terminal. Counts, the cutoff
# point, and the cause — so "why is recall bad on the Q4 docs" is a thirty-second
# question rather than a re-derivation.
INGEST_MANIFEST_PATH = DATA_INDEX / "ingest_manifest.json"

# Your fix: the eval harness must refuse to blend. If the index holds both kinds,
# a contextual-retrieval A/B either hard-fails or stratifies and reports the two
# populations separately. It never emits one averaged number — a diluted effect
# size makes a real technique look not-worth-the-cost, which is the wrong
# conclusion drawn from correct arithmetic.
EVAL_REFUSE_MIXED_PROVENANCE = True

# Your fix: a mixed index cannot become the CI baseline. A baseline is a promise
# about what was measured, and a mixed index cannot keep it.
CI_BASELINE_REQUIRES_UNIFORM_PROVENANCE = True

# One more entanglement you flagged, worth its own switch: if the golden set was
# generated FROM the corpus, its sampling may itself skew toward the prefixed
# region — so baseline and treatment share a confound and no ablation is
# interpretable. Recording which provenance each golden item's source chunk had
# is what makes that detectable instead of invisible.
GOLDEN_SET_RECORD_SOURCE_PROVENANCE = True


# --------------------------------------------------------------------------
# Retrieval (Sessions 2, 5, 6)
# --------------------------------------------------------------------------
#
# THE TWO-STAGE PRINCIPLE (guide M3), encoded as three numbers:
#   stage 1 optimises RECALL    -> retrieve many (RETRIEVE_K)
#   stage 2 optimises PRECISION -> rerank down to few (RERANK_K)
#   the LLM sees only           -> CONTEXT_K
#
# The ceiling law: a reranker cannot recover a document the retriever never
# returned. So RETRIEVE_K is the hard upper bound on end-to-end quality, and
# tuning it is the first thing you do — not the last.

RETRIEVE_K = 50      # stage 1 candidate pool
RERANK_K = 10        # survivors after the cross-encoder
CONTEXT_K = 5        # what actually reaches the model

# BM25 (guide M2). Written by hand in Session 5 so that "explain BM25" is
# something you have implemented rather than memorised.
#   k1 controls term-frequency SATURATION: the 10th occurrence of a word adds
#      far less than the 2nd. Without saturation, keyword spam wins.
#   b  controls LENGTH NORMALISATION. b=0 disables it (long docs win by having
#      more words); b=1 fully normalises.
BM25_K1 = 1.5   # guide's typical range 1.2-2.0
BM25_B = 0.75   # guide's typical value

# Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum 1/(k + rank_i(d)).
# k=60 is a smoothing constant, not magic — it damps how much the very top rank
# dominates. RRF is RANK-based, which is the entire reason it wins in practice:
# BM25 scores and cosine similarities live on incomparable scales, and rank
# fusion needs no normalisation between them.
# Its weakness: it throws away score MAGNITUDE, so one runaway-confident hit
# gets flattened to "rank 1" like any other rank 1.
RRF_K = 60

# Weights per retrieval leg. Equal by default so Session 5's experiment measures
# the legs honestly before we start tuning.
RRF_WEIGHT_DENSE = 1.0
RRF_WEIGHT_SPARSE = 1.0


# --------------------------------------------------------------------------
# Reranking (Session 6)
# --------------------------------------------------------------------------
#
# Bi-encoder (our embedder): encodes query and doc INDEPENDENTLY. Precomputable,
# fast, lossy — it never sees the pair together.
# Cross-encoder (below): runs joint attention over the concatenated pair. Cannot
# be precomputed, much better, latency LINEAR in candidate count.
# That asymmetry is *why* the two-stage architecture exists at all.

# Default is deliberately the tiny one: 90MB, CPU-fast, runs on your laptop.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# The quality option, so the size/latency/quality tradeoff is something you have
# actually felt rather than read about. ~2.2GB.
RERANK_MODEL_STRONG = "BAAI/bge-reranker-v2-m3"

# MMR diversity (guide M3): lambda=1.0 is pure relevance, 0.0 is pure diversity.
# Without this, a top-5 can be five near-copies of one paragraph and the model
# answers one facet of a multi-facet question.
MMR_LAMBDA = 0.7
# Cosine above this = near-duplicate, collapsed before reranking. Duplicates
# crowd out your top-k, which is failure point #7 (Incomplete) in Barnett et al.
NEAR_DUPLICATE_THRESHOLD = 0.97


# --------------------------------------------------------------------------
# Generation (Session 8)
# --------------------------------------------------------------------------
#
# TOKEN BUDGET AS A DESIGN CONSTRAINT, not an afterthought. You allocate it
# explicitly across four consumers; if retrieved context is unbounded, it
# silently eats the history and the output reserve.

TOKENS_SYSTEM_RESERVE = 800
TOKENS_HISTORY_RESERVE = 2000
TOKENS_CONTEXT_BUDGET = 12000
TOKENS_OUTPUT_RESERVE = 2000

# STRUCTURED OUTPUT COSTS SEVERAL TIMES ITS PROSE, which is why the answer call
# gets its own budget instead of the reserve above.
#
# The reserve was sized by reading an answer and estimating its length. But the
# answer schema does not emit prose -- it emits JSON in which every claim
# carries its citations and each citation carries a VERBATIM QUOTE from the
# source. The quotes are the whole point (they are what makes a citation
# checkable rather than decorative), and they mean the output scales with the
# number of claims times the length of the evidence, not with the length of the
# prose the user reads.
#
# Measured: a 12-source answer over a feedback document ran past 2048 tokens and
# the JSON was cut mid-string. json.loads then failed with "Unterminated string
# at char 6139" and the UI showed the user raw JSON under a banner blaming the
# structured output for not parsing. The model did nothing wrong; the budget was
# wrong, and every layer between reported the symptom instead of the cause.
#
# 8192 is chosen against the observed worst case (~2600 output tokens for 12
# sources) with room for a longer answer, and it is a CEILING, not a spend --
# output is billed on tokens produced, so a short answer still costs a short
# answer. Truncation is now also detected (Usage.truncated) rather than inferred
# from a downstream parse error.
TOKENS_ANSWER_OUTPUT = 8192

# Lost in the Middle (Liu et al., 2023): models attend most strongly to the
# START and END of long context. So after reranking we REORDER — strongest
# evidence to the head and tail, weakest buried in the middle. Cheap, real gain.
# Session 8 measures it against rank-descending and random ordering.
REORDER_LOST_IN_MIDDLE = True

# Grounded abstention. The faithfulness <-> helpfulness tradeoff is a PRODUCT
# decision with a dial, not a bug. A support bot wants high abstention; a
# brainstorming assistant does not.
ABSTAIN_WHEN_UNGROUNDED = True

# Post-hoc citation verification: re-check each generated sentence against the
# chunk it cites, via entailment. This is the answer to "how do you guarantee
# the citations are real?" — because a model can and does fabricate citation IDs.
VERIFY_CITATIONS = True
VERIFY_ENTAILMENT_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Evaluation (Session 10)
# --------------------------------------------------------------------------

GOLDEN_SET_TARGET_SIZE = 120

# Stratification. The guide is specific that a golden set must cover categories
# that FAIL DIFFERENTLY — otherwise your average hides the thing that is broken.
# `out_of_scope` is the one people forget: those questions must be ABSTAINED on,
# and a system that answers them confidently is worse than one that scores lower.
GOLDEN_SET_STRATA = (
    "simple_factual",
    "multi_hop",
    "aggregative",
    "out_of_scope",
    "ambiguous",
    "exact_identifier",
    "table_or_image",
)

# CI gate. Any prompt / chunking / model / index change re-runs the suite and
# fails if quality drops more than this against the stored baseline.
CI_NDCG_REGRESSION_TOLERANCE = 0.03

# --- LLM-as-judge: validation is a GATE, not a warning ---------------------
#
# Revised per your critique, which was correct on both counts.
#
# WHY THE JUDGE MUST NOT BE THE GENERATOR: self-preference bias. A model rates
# its own output higher because "good" partly means "phrased how I would phrase
# it". Grading your own essay. The error is not random — it is loudest exactly
# where the system is most confidently wrong, which is the one place you needed
# the measurement to work.
#
# Note that the three judge biases have DIFFERENT fixes, and people conflate them:
#   position bias    -> randomise presentation order
#   verbosity bias   -> pairwise comparison instead of absolute scoring
#   self-preference  -> a different model. Randomisation does NOT touch this one.
#
# WHY THE FALLBACK IS "WAIT", NOT "SWAP": falling back judge -> workhorse is
# doubly broken. The fallback target IS the generator (self-preference), and it
# is weaker (a judge cannot reliably grade an answer it could not have produced
# — it marks correct-but-sophisticated answers wrong). Two different failure
# modes stacking. So on quota exhaustion the judge WAITS. Judging is an offline
# batch job with generous retries and no latency SLO: a slow grade is fine, a
# wrong grade is not.
JUDGE_ALLOW_MODEL_FALLBACK = False
JUDGE_OFFLINE_MAX_RETRIES = 20
JUDGE_OFFLINE_BACKOFF_SECONDS = 30.0

# Cohen's kappa against YOUR hand labels. This is where the reasoning above
# stops being a plausible story and becomes a measurement.
JUDGE_VALIDATION_SAMPLE = 30
JUDGE_MIN_KAPPA = 0.6

# Your fix, adopted: an unvalidated judge is UNUSABLE, not merely announced.
# The harness refuses to emit judged metrics until kappa passes.
JUDGE_REQUIRE_VALIDATION = True

# Your second fix, adopted: kappa is a property of (judge model, rubric version),
# not of the project. Change either and the old kappa measured a different judge,
# so it is discarded rather than inherited.
JUDGE_KAPPA_RECORD = DATA_EVAL / "judge_kappa.json"  # keyed by model id + rubric hash
JUDGE_RUBRIC_VERSION = 1

# A gotcha worth knowing before you compute it in Session 10: kappa has a
# PREVALENCE PROBLEM. If 95% of your sample is "faithful", judge and human can
# agree on 94 of 100 items and kappa still collapses toward 0, because kappa
# measures agreement ABOVE CHANCE and chance agreement is already ~0.9. The fix
# is not a lower threshold — it is to stratify the validation sample so the
# classes are roughly balanced. Deliberately include known-bad answers.
JUDGE_VALIDATION_BALANCE_CLASSES = True


# --------------------------------------------------------------------------
# Free-tier behaviour
# --------------------------------------------------------------------------
#
# Your requirement: any limitation caused by the free tier must be reported to
# the user with its cause. limits.py implements the mechanism; these are its
# settings.

# Retry policy for 429 RESOURCE_EXHAUSTED. Free-tier limits are per-minute, so
# a short backoff often clears; a per-day limit never will, which is why we cap
# attempts and then degrade loudly instead of hanging.
QUOTA_MAX_RETRIES = 3
QUOTA_BACKOFF_BASE_SECONDS = 2.0

# Set RAGKIT_FAKE_QUOTA_EXHAUSTED=1 to force the quota path and watch the
# degradation notice render, without waiting to be rate-limited for real.
FAKE_QUOTA_EXHAUSTED = os.getenv("RAGKIT_FAKE_QUOTA_EXHAUSTED") == "1"

# Fault injection for a FABRICATED CITATION, following the same precedent as the
# quota flag above.
#
# Why a flag rather than a unit test: the fabricated-citation branch spans the
# generator, the verifier, the API's claim shaping and the UI chip. A unit test
# of any one of them tests a COPY of the path, and "measuring a different
# program" is this project's most expensive recurring mistake. Injecting the
# fault at the generator's output lets the real path run end to end.
#
# It appends a citation whose label was never sent, which is precisely the
# failure the membership check exists to catch: an id that looks valid and
# points at nothing this request delivered.
FAKE_FABRICATED_CITATION = os.getenv("RAGKIT_FAKE_FABRICATED_CITATION") == "1"

# Fault injection for an UNVERIFIABLE QUOTE, so the "found but not quoted" state
# can be seen rather than assumed.
#
# That state is the honest middle ground between asserting and abstaining: the
# passage was retrieved and cited, and the quote did not verify against it, so
# the system declines to present it as a quotation. It cannot be produced on
# demand by a well-behaved model, which is exactly why it needs injecting -- a
# render path never watched execute is unverified, and this one carries a product
# claim ("It's in your documents. I won't quote it.").
FAKE_UNVERIFIABLE_QUOTE = os.getenv("RAGKIT_FAKE_UNVERIFIABLE_QUOTE") == "1"


def api_key() -> str:
    """Fail loudly and usefully, rather than with a stack trace from the SDK."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  1. Get a free key: https://aistudio.google.com/apikey\n"
            "  2. cp .env.example .env\n"
            "  3. Paste the key into .env\n"
        )
    return key


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_INDEX, DATA_EVAL, EMBED_CACHE, EXPERIMENTS_OUT, NOTES):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PUBLIC DEMO MODE
#
# The API has nine write endpoints and no authentication of any kind. That is
# correct for a tool running on localhost against your own corpus, and it is
# indefensible on a public URL. Two of them are actively destructive:
#
#   DELETE /api/documents/{id}   "removes a document and everything derived from
#                                it", with purge_cache=True by DEFAULT -- so an
#                                anonymous request deletes both the corpus and
#                                the cached embeddings that cost API quota to
#                                rebuild. A crawler following links empties the
#                                demo.
#   POST   /api/ingest           triggers a full corpus ingest: hundreds of
#                                Gemini calls. One curl exhausts a free-tier
#                                key for the day.
#
# The fix is NOT authentication. Nobody is going to be issued a credential to
# look at a demo, and a login wall on a portfolio piece is worse than no demo.
# The fix is to make the deployment READ-ONLY and say so out loud: the write
# paths return 403 with the reason, /api/ask is rate-limited per client, and the
# UI states that upload is disabled and why.
#
# Saying it out loud matters. A demo that silently lacks upload looks unfinished;
# one that explains it is protecting a shared free-tier key looks operated.
DEMO_MODE = os.getenv("RAGKIT_DEMO_MODE") == "1"

# Per-client budget for the one endpoint that must stay open. Generation is the
# demo, so /api/ask cannot be disabled -- but a single visitor holding the whole
# free-tier quota means the next visitor sees nothing but 429s. A window keeps
# one client from starving the rest.
#
# 20/hour is roughly "explore it thoroughly, twice" and nowhere near enough to
# drain a day's quota. Both values are env-tunable because the right number
# depends on the key's actual limits, which the free tier does not publish.
DEMO_RATE_LIMIT_REQUESTS = int(os.getenv("RAGKIT_DEMO_RATE_LIMIT", "20"))
DEMO_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RAGKIT_DEMO_RATE_WINDOW", "3600"))

# Deployed origins for CORS. Hardcoded to localhost:5173, which is right in
# development and breaks the moment the app is served from anywhere else -- and
# it breaks as an opaque browser error with a working backend behind it, which is
# a bad hour to spend. Comma-separated; the dev origins stay unconditionally so
# adding a deployment never breaks local work.
CORS_EXTRA_ORIGINS = tuple(
    o.strip() for o in os.getenv("RAGKIT_CORS_ORIGINS", "").split(",") if o.strip()
)

# When set, the built frontend is served by FastAPI from this directory, making
# the whole app ONE service behind ONE URL. Two services would mean two
# deployments, a cross-origin hop and a second thing to be asleep when the
# examiner clicks the link.
WEB_DIST_DIR = os.getenv("RAGKIT_WEB_DIST", "")
