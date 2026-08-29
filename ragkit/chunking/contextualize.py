"""Contextual Retrieval: an LLM-written prefix that situates a chunk. Guide M5.

WHY THIS EXISTS NOW AND NOT BEFORE. D-6 deferred this behind a free baseline --
heading breadcrumbs -- on the rule that the paid version must beat the free one
rather than an empty one. That deferral has since expired on its own predicate
(`child_strict` 86% < 90% with `source_hit` already 1.0), and the failure
histogram gave a second, independent argument: all 23 measured failures are FP2
or FP3, and `source_hit` is 100% at every budget. The right DOCUMENT is always
found; the loss is ranking WITHIN it. A situating prefix is precisely the
intervention that addresses that and nothing else.

---------------------------------------------------------------------------
WE DO NOT SEND THE WHOLE DOCUMENT, AND THE ARITHMETIC IS WHY

Anthropic's version puts the entire document in the prompt for every chunk, and
that is affordable *there* because prompt caching makes the repeated prefix
nearly free. Measured on this corpus, the naive version is:

    15,791,052 input tokens   (sum over children of their document's length)

against ~1.0M for the design below. Implicit caching on Gemini 3.x might have
recovered most of that -- but "might" is not a budget. Implicit caching also
requires a 4,096-token prefix, and SIX of this corpus's fifteen sources are
shorter than that, so those would have paid full price per chunk regardless.
Building a cost estimate on a discount you have not observed is the same
mistake as gating on a metric you have not calibrated.

So the prefix is assembled from two pieces the chunk does not already contain:

  1. a DOCUMENT SYNOPSIS, computed once per document (15 calls, not 814)
  2. the chunk's own PARENT section, which is already in memory and free

Whether that is as good as whole-document context is an empirical question, and
it is the one the eval answers. What it is not is a guess dressed as a saving.

---------------------------------------------------------------------------
THE PREFIX IS NOT QUOTABLE, AND THAT IS STRUCTURAL

This text is written by a model. It carries a real page number, sits directly
against document prose, and reads like the document. That combination is the
image-caption failure exactly: model-written text displayed as a quotation from
a source. So the prefix enters `embed_text` ONLY. `verbatim_text` stays the
body, `display_text` stays the body, and provenance is PREFIXED -- the same
state the breadcrumb already used, which is the whole reason that state exists.

---------------------------------------------------------------------------
CACHED, FOR THE SAME REASON CAPTIONS ARE

The call is non-deterministic. Uncached, every ingest would produce different
prefixes, which produces different `embed_text`, which produces different
`chunk_id`s, which means an eval run and its baseline describe different
corpora -- and the CI gate compares them anyway. Keyed on every input to the
call: document synopsis, parent text, body, model, prompt version.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..gemini import Usage, count_tokens, generate, resolve_models

# Bump when either prompt below changes, or cached prefixes written for the old
# prompt will be served for the new question. Same rule as CAPTION_PROMPT_VERSION.
CONTEXT_PROMPT_VERSION = 1

_CACHE = config.CACHE_DIR / "contexts"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_SYNOPSIS_PROMPT = """\
Write a short factual synopsis of the document below, for use as background
context when indexing its individual passages for search.

State, in at most 120 words:
- what the document is (paper, manual, report, feedback notes) and its title
- the system, product, method or dataset it is about, BY NAME
- the named entities a reader would need in order to disambiguate this document
  from a similar one: method names, model names, benchmark names, version
  strings, identifiers

Use only what the document says. Do not evaluate it, summarise its conclusions,
or add anything not present. Reply with the synopsis alone.

DOCUMENT
--------
{doc}
"""

_SITUATE_PROMPT = """\
Below is a passage from a document, together with background about the document
and the section the passage sits in.

Write one or two sentences that situate this passage so that someone searching
the whole collection could find it from a query. The goal is to supply what the
passage does NOT say about itself.

Rules:
- Resolve references. If the passage says "the method", "this model", "it", or
  "the benchmark", name what it refers to.
- Repeat identifiers verbatim -- model names, method names, version strings,
  table and figure numbers, error codes. Do not paraphrase them.
- State only what the document supports. Never speculate, evaluate, or add a
  fact that is not in the material below.
- Do not begin with "This passage" or "This chunk". Write it as context.
- Reply with the context alone, no preamble, at most {max_words} words.

DOCUMENT BACKGROUND
-------------------
{synopsis}

SECTION THE PASSAGE IS FROM
---------------------------
{parent}

THE PASSAGE
-----------
{body}
"""


# --------------------------------------------------------------------------
# Cache -- every input to the call, the same rule as the embedding cache
# --------------------------------------------------------------------------


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")       # a separator, so ("ab","c") != ("a","bc")
    return h.hexdigest()


def _path(key: str) -> Path:
    return _CACHE / key[:2] / f"{key}.json"


def _read(key: str) -> str | None:
    p = _path(key)
    if not p.exists():
        return None
    try:
        return str(json.loads(p.read_text("utf-8"))["text"])
    except Exception:  # noqa: BLE001 -- a corrupt entry is a miss, not a crash
        return None


def _write(key: str, text: str, *, kind: str, model: str) -> None:
    p = _path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "text": text,
                "kind": kind,
                "model": model,
                "prompt_version": CONTEXT_PROMPT_VERSION,
                # Stored on the entry itself, so a prefix lifted out of this
                # cache by any future reader carries the fact that no human
                # wrote it. Provenance travels with the text or it is not
                # provenance.
                "provenance": "model_generated",
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, p)


# --------------------------------------------------------------------------
# Accounting -- what this cost, and whether it was worth it
# --------------------------------------------------------------------------


@dataclass
class ContextStats:
    """What the contextualiser actually did. Reported, never inferred."""

    synopses_made: int = 0
    synopses_cached: int = 0
    prefixes_made: int = 0
    prefixes_cached: int = 0
    # A refusal is not a failure to be retried -- it is the spend ceiling doing
    # its job, and the chunk falls back to the breadcrumb-only prefix. Counted
    # so a partially-contextualised index can never be mistaken for a complete
    # one.
    prefixes_skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    prefix_tokens: list[int] = field(default_factory=list)

    def note_skip(self, reason: str) -> None:
        self.prefixes_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def to_json(self) -> dict[str, object]:
        n = len(self.prefix_tokens)
        return {
            "synopses_made": self.synopses_made,
            "synopses_cached": self.synopses_cached,
            "prefixes_made": self.prefixes_made,
            "prefixes_cached": self.prefixes_cached,
            "prefixes_skipped": self.prefixes_skipped,
            "skip_reasons": dict(self.skip_reasons),
            "prompt_tokens": self.usage.prompt_tokens,
            "output_tokens": self.usage.output_tokens,
            "cached_tokens": self.usage.cached_tokens,
            "mean_prefix_tokens": round(sum(self.prefix_tokens) / n) if n else 0,
            "max_prefix_tokens": max(self.prefix_tokens) if self.prefix_tokens else 0,
            # THE NUMBER THAT DECIDES WHETHER THIS HELPED OR HURT. A prefix is
            # a fixed cost on every child, and at a FIXED TOKEN BUDGET a longer
            # child means fewer children fit. Recall can fall while ranking
            # improves. Most evaluations measure recall@k and are structurally
            # blind to this; this one is not, so the overhead is recorded next
            # to the score it has to justify.
            "n_prefixed": n,
        }


# --------------------------------------------------------------------------
# The two calls
# --------------------------------------------------------------------------


def synopsis(title: str, doc_text: str, *, stats: ContextStats | None = None) -> str:
    """One background paragraph per DOCUMENT, computed once and cached.

    Truncated at a fixed head of the document rather than sent whole: a synopsis
    is drawn from the front matter, abstract and early sections in practice, and
    a 30k-token paper contributes nothing extra to a 120-word summary except
    cost. The truncation point is part of the cache key by construction, since
    the truncated text IS what gets hashed.
    """
    st = stats or ContextStats()
    model = resolve_models()["workhorse"]
    head = doc_text[: config.CONTEXT_SYNOPSIS_MAX_CHARS]
    key = _key("synopsis", str(CONTEXT_PROMPT_VERSION), model, title, head)

    hit = _read(key)
    if hit is not None:
        st.synopses_cached += 1
        return hit

    text, usage = generate(
        _SYNOPSIS_PROMPT.format(doc=head),
        role="workhorse",
        max_output_tokens=400,
        stage="context_synopsis",
    )
    text = text.strip()
    st.synopses_made += 1
    st.usage.add(usage)
    _write(key, text, kind="synopsis", model=model)
    return text


def situate(
    *,
    doc_synopsis: str,
    parent_text: str,
    body: str,
    stats: ContextStats | None = None,
) -> str:
    """The per-chunk prefix. Empty string means "fall back to the breadcrumb".

    Returning "" rather than raising is deliberate and it is the same shape as
    every other degradation in this codebase: the caller gets a well-formed
    chunk, the manifest gets a count, and the eval refuses a claim it cannot
    support. A contextualiser that raised would turn a spend ceiling into a
    failed ingest.
    """
    st = stats or ContextStats()
    model = resolve_models()["workhorse"]
    # The parent is trimmed for the same reason the document is: a p99 section
    # contributes tail text the sentence being written will never reach.
    parent = parent_text[: config.CONTEXT_PARENT_MAX_CHARS]
    key = _key("situate", str(CONTEXT_PROMPT_VERSION), model,
               doc_synopsis, parent, body)

    hit = _read(key)
    if hit is not None:
        st.prefixes_cached += 1
        st.prefix_tokens.append(count_tokens(hit))
        return hit

    prompt = _SITUATE_PROMPT.format(
        synopsis=doc_synopsis,
        parent=parent,
        body=body,
        max_words=config.CONTEXT_PREFIX_MAX_WORDS,
    )
    try:
        text, usage = generate(
            prompt,
            role="workhorse",
            max_output_tokens=config.CONTEXTUAL_PREFIX_MAX_TOKENS * 2,
            stage="context_prefix",
        )
    except Exception as exc:  # noqa: BLE001
        # Named by type, not swallowed into one bucket: "the spend ceiling
        # refused" and "the model returned nothing" are different findings and
        # a single `skipped: 814` cannot tell them apart.
        st.note_skip(type(exc).__name__)
        return ""

    text = " ".join(text.split()).strip()
    if not text:
        st.note_skip("empty")
        return ""
    st.prefixes_made += 1
    st.usage.add(usage)
    st.prefix_tokens.append(count_tokens(text))
    _write(key, text, kind="prefix", model=model)
    return text


# --------------------------------------------------------------------------
# Cost, before spending it
# --------------------------------------------------------------------------


def estimate(docs: dict[str, tuple[str, int]]) -> dict[str, object]:
    """Token cost of contextualising a corpus, WITHOUT calling anything.

    `docs` maps source_id -> (document text, number of children). Returned so a
    caller can look at the number before authorising the spend, which is the
    difference between a budget and a bill.
    """
    syn_in = syn_out = sit_in = sit_out = 0
    for _sid, (text, n_children) in docs.items():
        head = text[: config.CONTEXT_SYNOPSIS_MAX_CHARS]
        syn_in += count_tokens(_SYNOPSIS_PROMPT.format(doc=head))
        syn_out += 160
        # Per child: synopsis + a trimmed parent + the body + the prompt frame.
        # Approximated from the document mean rather than per chunk, because an
        # estimate that needs the chunks needs the chunker, and this has to be
        # answerable before anything runs.
        per_child = 160 + config.CONTEXT_PARENT_MAX_CHARS // 4 + 300 + 200
        sit_in += per_child * n_children
        sit_out += config.CONTEXTUAL_PREFIX_MAX_TOKENS * n_children

    naive_in = sum(count_tokens(t) * n for t, n in docs.values())
    return {
        "n_documents": len(docs),
        "n_children": sum(n for _t, n in docs.values()),
        "synopsis_input_tokens": syn_in,
        "synopsis_output_tokens": syn_out,
        "prefix_input_tokens": sit_in,
        "prefix_output_tokens": sit_out,
        "total_input_tokens": syn_in + sit_in,
        "total_output_tokens": syn_out + sit_out,
        # What the textbook version would have cost on this corpus. Kept in the
        # output because the design decision only defends itself next to it.
        "whole_document_input_tokens": naive_in,
        "saving_vs_whole_document": naive_in - (syn_in + sit_in),
    }
