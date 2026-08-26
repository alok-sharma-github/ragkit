"""
Cited generation, with the two free checks. Guide module M7.

TWO CITATION FAILURES THAT LOOK IDENTICAL AND ARE NOT

    "HYBRID_RRF achieved 0.831 Recall@10 [2]"

  FABRICATED  -- label [2] was never sent. Nothing was misread; the label points
                 at nothing.
  UNSUPPORTED -- [2] is real and was sent, and it says nothing about 0.831. The
                 label is genuine and attached to the wrong claim.

A footnote to a book that was never published, versus a footnote to a real book
that does not say that. Same-looking output, different fixes -- and the first is
checkable for free, deterministically, with no model in the loop. So it goes
first.

---------------------------------------------------------------------------
WHY THE LABELS ARE [1]..[N] AND NOT CHUNK IDS

Chunk ids are 24-hex-char hashes. A model mistyping one character produces a
string indistinguishable from a fabrication, which invites fuzzy matching -- and
fuzzy matching on identifiers means the check has stopped checking (the same
reason `_squash` must not be loosened until it accepts anything).

Per-request integer labels have a tiny vocabulary. `[6]` when five were sent is
unambiguous fabrication: there is nothing to have typoed toward. The mapping back
to real chunk ids happens in this file; the model never sees an internal id.

MEMBERSHIP IS CHECKED AGAINST WHAT THIS REQUEST SENT, not against the index.
A citation of a chunk that exists but was not retrieved for this query is still
fabrication -- the model never saw it. Checking against the index would turn a
caught bug into an approved one, which is the more dangerous direction.

---------------------------------------------------------------------------
THE QUOTE CHECK, AND THE BUG IT WOULD OTHERWISE REPEAT

Each citation must carry a verbatim quote. String-matching that against the
chunk's `verbatim_text` catches much of the unsupported case, still with no model
call.

The naive version fails on our own repaired tables. A `page_text_clip` chunk
holds:

    HYBRID_RRF
    0.831
    0.604
    58

and a model asked to quote it will write "HYBRID_RRF | 0.831" -- normalising the
newlines, because that is what models do with tabular text. A whitespace-exact
match rejects a verified-correct citation.

That is exactly the earlier cross-extractor comparison bug: two representations
of the same content, compared along a dimension that does not matter. It was
solved once, in loaders._squash (alphanumerics only), so it is REUSED here rather
than re-derived. Rediscovering the same insensitivity requirement a third time
would be the lesson not landing.

THREE OUTCOMES, not two, because the squash still cannot tell a reworded quote
from an invented one:
    verified   -- present after squashing
    paraphrase -- long contiguous overlap; the model restated it
    absent     -- not there
Collapsing paraphrase into absent would blame the model for our repair's
formatting; collapsing it into verified would let real fabrication through.

---------------------------------------------------------------------------
FAIL CLOSED

A fabricated label is evidence the model is generating rather than reading. If it
invented one citation, the surrounding prose is suspect too -- so the response is
not "strip that bracket", it is to mark the whole answer ungrounded and abstain.
A gate that acts, like the build failing on a prefix leak rather than warning.

Every check here errs toward refusal. That is deliberate and it is the property
that makes a WRONG CHECK cheap: a check that fails closed announces its own bugs
immediately on real data, while a check that fails open hides them. Three of the
verification bugs found today were caught within minutes for exactly this reason.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .. import config, gemini, limits
from ..ingest.document import Chunk, TextProvenance
from ..ingest.loaders import _squash  # reused, not re-derived -- see module docstring

QuoteStatus = Literal["verified", "absent", "unquotable", "no_quote"]

# THERE IS NO "PARAPHRASE" VERDICT, and removing it was a measurement result.
#
# I built three outcomes -- verified / paraphrase / absent -- intending the middle
# one to separate "the model reworded our awkward repaired text" from "the model
# invented a quote". Character-coverage cannot do that. Measured, on the fixture:
#
#   "HYBRID_RRF scored 0.831 on recall"   genuine rewording   overlap 0.48
#   "HYBRID_RRF 0.712 0.573 52"           INVENTED numbers    overlap 0.53
#
# The fabrication scores HIGHER than the rewording, because a fake quote that
# reuses the entity name and the table's shape shares as many characters as a
# real restatement. The two classes are ordered the wrong way round, so no
# threshold splits them: any cutoff either admits the fabrication or rejects the
# rewording.
#
# The conclusion is that paraphrase-vs-fabrication is a SEMANTIC judgement and
# belongs to entailment (level 3). I had mislabelled a level-3 problem as a
# free level-1 check. So: non-verbatim is `absent`, and absent blocks.
#
# The overlap NUMBER is still recorded, as diagnostic data rather than a verdict.
# It is what lets the eval ask whether `absent` clusters on repaired chunks with
# high overlap (our formatting is the cause) or low overlap (the model is
# improvising). The number is the instrument; the label was the overreach.
# Quotes shorter than this are not evidence of anything: "0.831" appears in many
# chunks, so matching it proves nothing about which chunk was read.
QUOTE_MIN_CHARS = 12


# --------------------------------------------------------------------------
# Structured output schema
# --------------------------------------------------------------------------
#
# Structured output arrives at level 1 rather than level 2, and the reason is
# mechanical: a per-claim quote check needs (claim, citation) PAIRS. A blob of
# prose with brackets scattered through it does not give you pairs, so there is
# nothing to verify against. Level 2 is infrastructure for the check, not a
# quality upgrade in its own right.

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "abstain_reason": {"type": "string"},
        # THE "ASK FRESH" AFFORDANCE, moved here from the condenser.
        #
        # It was attached to the condenser's `ask_fresh` route, and that route
        # never fires: five deliberately-constructed follow-ups all routed to
        # retrieval instead, which is the model being RIGHT -- retrieval is cheap
        # and the abstention path already handles "nothing supports this" better
        # than refusing to look. So the suggestion lived on a branch that does
        # not execute, making the design's affordance unreachable.
        #
        # An abstention is the moment the suggestion is actually worth something:
        # the system looked, found nothing, and can say what to ask instead.
        "suggested_question": {"type": "string"},
        "answer_markdown": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    # PER-CLAIM ATTRIBUTION. The design's fourth citation state --
                    # grey, "said earlier in this conversation, not from your
                    # documents" -- is only honest if the model declares, per
                    # claim, where the claim came from. Inferring it from whether
                    # citations are present would be a guess, and a claim wrongly
                    # rendered in the "from your documents" colour is exactly the
                    # failure the citation layer exists to prevent.
                    "source": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "integer"},
                                "quote": {"type": "string"},
                            },
                            "required": ["label", "quote"],
                        },
                    },
                },
                "required": ["text", "citations"],
            },
        },
    },
    "required": ["abstained", "answer_markdown", "claims"],
}

SYSTEM = """You answer questions strictly from the numbered sources provided.

Rules:
- Use ONLY the sources. Do not use prior knowledge, even if you are confident.
- Every factual claim must cite at least one source by its number, e.g. [2].
- With each citation, give a VERBATIM quote copied from that source: the exact
  characters, not a summary. The quote must be long enough to identify the
  passage (roughly 12 characters or more).
- For EVERY claim set `source` to one of:
    "documents"    -- the claim comes from the numbered sources. It must cite.
    "conversation" -- the claim is a reading, restatement or judgment over what
                      was already said in this conversation, not something the
                      sources state. It must NOT cite a source, and you must say
                      plainly that it is your reading rather than the documents'.
  Never label a claim "documents" unless a source actually states it.
- If the sources do not contain the answer, set abstained=true and say what is
  missing. When a differently-phrased standalone question would plausibly find it
  in these documents, put that question in suggested_question. Leave
  suggested_question empty if no rephrasing would help -- proposing a question
  that also cannot be answered is worse than proposing none. Abstaining is the correct answer when the sources are insufficient.
  Do not guess and do not hedge: a hedged guess is worse than an abstention.
- Section breadcrumbs shown as "Section:" are metadata added by the retrieval
  system. They are NOT part of the document and must never be quoted."""


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------


def _format_source(label: int, chunk: Chunk) -> str:
    """One numbered source block.

    The breadcrumb is rendered as MARKED METADATA on its own line, never
    concatenated into the body. Concatenating it would put model-visible text
    that is not in the document into the region the citation layer treats as
    quotable -- a fabricated quotation carrying a real page number.
    """
    head = [f"[{label}] {chunk.source_id}"]
    if chunk.page:
        head.append(f"page {chunk.page}" + (f"-{chunk.page_end}" if chunk.page_end else ""))
    if chunk.heading_path:
        head.append("Section: " + " > ".join(chunk.heading_path))

    note = ""
    if chunk.text_source and "page_text_clip" in chunk.text_source:
        # A FORMAT note, not a reliability warning, and the distinction is the
        # whole argument. This tells the model how to PARSE (values run in row
        # order) and has a checkable outcome. A reliability caveat would be
        # unactionable -- the model cannot inspect the PDF -- and would produce a
        # hedge wrapped around the same claim, which reads as caution while
        # laundering a possibly-wrong answer. If this line ever starts producing
        # hedges instead of parses, it is that failure and it comes out.
        note = (
            "\n(This source is tabular data rendered as plain text; values run in "
            "row order, one per line.)"
        )
    elif chunk.text_provenance is TextProvenance.MODEL_GENERATED:
        note = "\n(This source is a system-generated description, not document text.)"

    return f"{' | '.join(head)}{note}\n{chunk.display_text}"


def assemble(chunks: Sequence[Chunk]) -> tuple[str, dict[int, Chunk]]:
    """Numbered sources plus the label -> chunk map the model never sees."""
    mapping = {i + 1: c for i, c in enumerate(chunks)}
    body = "\n\n---\n\n".join(_format_source(i, c) for i, c in mapping.items())
    return body, mapping


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@dataclass
class CitationCheck:
    label: int
    chunk_id: str | None
    quote: str
    fabricated: bool
    quote_status: QuoteStatus
    overlap: float = 0.0
    detail: str = ""


def _check_quote(quote: str, chunk: Chunk) -> tuple[QuoteStatus, float, str]:
    if chunk.text_provenance is TextProvenance.MODEL_GENERATED or not chunk.verbatim_text:
        # Nothing verbatim exists to match against. Structural, from the
        # provenance field -- without it we would be string-matching against
        # breadcrumbs and Gemini captions and reporting nonsense failures.
        return "unquotable", 0.0, "chunk has no verbatim text (model-generated)"
    q = _squash(quote)
    if len(q) < QUOTE_MIN_CHARS:
        return "no_quote", 0.0, f"quote too short after normalisation ({len(q)} chars)"
    body = _squash(chunk.verbatim_text)
    if q in body:
        return "verified", 1.0, ""
    # COVERAGE across matching blocks, not the single longest run. The longest
    # run cannot see a rewording that inserts words: "HYBRID_RRF scored 0.831 on
    # recall" against "HYBRID_RRF 0.831 0.604 58" shares "hybridrrf" (9 chars of
    # 27 = 0.33) and then interleaves. Coverage sums every matching block instead.
    #
    # Blocks below MIN_BLOCK are excluded, because coverage built from many
    # 1-2 character coincidences is not evidence of anything -- any two English
    # strings share letters. That floor is what stops this metric degrading into
    # "everything is a paraphrase".
    MIN_BLOCK = 3
    blocks = difflib.SequenceMatcher(None, q, body).get_matching_blocks()
    covered = sum(b.size for b in blocks if b.size >= MIN_BLOCK)
    overlap = covered / max(len(q), 1)
    return "absent", overlap, f"quote not present verbatim ({overlap:.0%} char overlap)"


def locate_span(quote: str, text: str) -> tuple[int, int] | None:
    """Character offsets of `quote` inside `text`, ignoring whitespace/punctuation.

    The source panel highlights "the exact span this answer cites", which needs
    real offsets in the ORIGINAL text -- but the match has to be squash-insensitive,
    because the model writes "HYBRID_RRF | 0.831" for a chunk that reads
    "HYBRID_RRF
0.831". Two extractors, two renderings, same content.

    So: squash both, find the match in squashed space, then map the squashed
    indices back through a position table built while squashing. Doing it the
    other way round -- fuzzy-matching in the frontend -- would put the same
    normalisation logic in a second language, where it would drift.
    """
    keep: list[int] = []
    buf: list[str] = []
    for i, ch in enumerate(text):
        if ch.isalnum():
            buf.append(ch.lower())
            keep.append(i)
    hay = "".join(buf)
    needle = "".join(c for c in quote.lower() if c.isalnum())
    if not needle or not hay:
        return None
    at = hay.find(needle)
    if at < 0:
        return None
    return keep[at], keep[at + len(needle) - 1] + 1


def locator(chunk: Chunk | None) -> str:
    """Where a citation points, in a form a user can act on.

    A page number is the natural locator for a PDF and does not exist for a
    DOCX -- DOCX has no pages until it is rendered, so `page` is None and the
    naive format printed "manual.docx pNone", which is not a locator at all.
    The section breadcrumb is the DOCX equivalent, and it is the one format where
    that breadcrumb is DECLARED rather than inferred, so it is reliable.
    """
    if chunk is None:
        return "?"
    parts = [chunk.source_id]
    if chunk.page:
        parts.append(f"p{chunk.page}" + (f"-{chunk.page_end}" if chunk.page_end else ""))
    elif chunk.heading_path:
        parts.append(" > ".join(chunk.heading_path[-2:]))
    if chunk.asset_path:
        parts.append(f"image: {chunk.asset_path}")
    return "  ".join(parts)


@dataclass
class Answer:
    text: str
    abstained: bool
    abstain_reason: str = ""
    suggested_question: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    checks: list[CitationCheck] = field(default_factory=list)
    grounded: bool = True
    reasons: list[str] = field(default_factory=list)
    reconciliation: dict[str, int] = field(default_factory=dict)
    usage: gemini.Usage = field(default_factory=gemini.Usage)
    sources: dict[int, Chunk] = field(default_factory=dict)

    def render(self) -> str:
        if self.abstained:
            out = ["ABSTAINED: " + (self.abstain_reason or "sources insufficient")]
        else:
            out = [self.text]
        if not self.grounded:
            out.append("\nUNGROUNDED -- answer withheld. " + "; ".join(self.reasons))
        # One line per SOURCE, not per citation: two claims citing the same
        # chunk printed the same line twice.
        seen: set[int] = set()
        src_lines: list[str] = []
        for c in self.checks:
            if c.fabricated or c.label in seen:
                continue
            seen.add(c.label)
            src_lines.append(
                f"  [{c.label}] {locator(self.sources.get(c.label))}  quote={c.quote_status}"
            )
        if src_lines:
            out.append("\nSources:")
            out += src_lines
        out.append("\n" + " | ".join(f"{k}={v}" for k, v in self.reconciliation.items()))
        return "\n".join(out)


def verify(payload: dict[str, Any], mapping: dict[int, Chunk]) -> tuple[list[CitationCheck], bool, list[str], dict[str, int]]:
    """Both free checks, then the fail-closed decision.

    Reconciliation counts are built here rather than added later: three numbers
    that must agree (sent / emitted / verified) is what surfaced the lost
    text_source label, and building the reconciliation in from the start beats
    promoting it after it bites.
    """
    checks: list[CitationCheck] = []
    for claim in payload.get("claims") or []:
        # A conversation-derived claim carrying document citations is a
        # contradiction: it says "the documents did not state this" and then
        # points at a document. Counted, and the citations are still checked --
        # dropping them would hide the contradiction rather than surface it.
        for cit in claim.get("citations") or []:
            label = cit.get("label")
            quote = cit.get("quote") or ""
            try:
                label = int(label)
            except (TypeError, ValueError):
                checks.append(
                    CitationCheck(-1, None, quote, True, "no_quote", 0.0, f"label {label!r} is not an integer")
                )
                continue
            chunk = mapping.get(label)
            if chunk is None:
                checks.append(
                    CitationCheck(
                        label, None, quote, True, "no_quote", 0.0,
                        f"label [{label}] was never sent (sent 1..{len(mapping)})",
                    )
                )
                continue
            status, overlap, detail = _check_quote(quote, chunk)
            checks.append(CitationCheck(label, chunk.chunk_id, quote, False, status, overlap, detail))

    fabricated = [c for c in checks if c.fabricated]
    absent = [c for c in checks if c.quote_status == "absent"]
    short = [c for c in checks if c.quote_status == "no_quote"]

    reasons: list[str] = []
    if fabricated:
        reasons.append(
            f"{len(fabricated)} fabricated citation(s): "
            + "; ".join(c.detail for c in fabricated[:3])
        )
    if absent:
        reasons.append(f"{len(absent)} citation(s) whose quote is not in the cited source")
    if short:
        reasons.append(f"{len(short)} citation(s) with no usable quote")


    all_claims = payload.get("claims") or []
    n_claims = len(all_claims)
    doc_claims = [c for c in all_claims if (c.get("source") or "documents") == "documents"]
    conv_claims = [c for c in all_claims if (c.get("source") or "") == "conversation"]
    # Only DOCUMENT claims are required to cite. A conversation-derived claim has
    # nothing in the corpus to point at, and demanding a citation for it is what
    # drives a model to attach a plausible-looking one.
    uncited = sum(1 for cl in doc_claims if not (cl.get("citations") or []))
    if uncited and not payload.get("abstained"):
        reasons.append(f"{uncited} document claim(s) carry no citation at all")
    mislabelled = [c for c in conv_claims if (c.get("citations") or [])]
    if mislabelled:
        reasons.append(
            f"{len(mislabelled)} claim(s) labelled 'conversation' but citing a "
            "document -- the label and the evidence disagree"
        )

    recon = {
        "chunks_sent": len(mapping),
        "chunks_cited": len({c.label for c in checks if not c.fabricated}),
        "citations_emitted": len(checks),
        "citations_verified": sum(1 for c in checks if c.quote_status == "verified"),
        "citations_absent": len(absent),
        # Diagnostic, not a verdict: mean character overlap of the absent quotes.
        # High overlap on repaired chunks points at our formatting; low overlap
        # points at the model. The eval slices this by text_source.
        "absent_mean_overlap_pct": (
            round(100 * sum(c.overlap for c in absent) / len(absent)) if absent else 0
        ),
        "citations_fabricated": len(fabricated),
        "citations_unquotable": sum(1 for c in checks if c.quote_status == "unquotable"),
        "claims": n_claims,
        "claims_from_documents": len(doc_claims),
        "claims_from_conversation": len(conv_claims),
        "claims_uncited": uncited,
        "claims_mislabelled": len(mislabelled),
    }
    return checks, (not reasons), reasons, recon


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The conversation-only path
# --------------------------------------------------------------------------

CONVERSATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "abstain_reason": {"type": "string"},
        "suggested_question": {"type": "string"},
        "answer_markdown": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    "required": ["abstained", "answer_markdown", "claims"],
}

CONVERSATION_SYSTEM = """You are answering from THIS CONVERSATION ONLY.

You have no documents for this turn. The user is asking you to read, restate,
compare or judge things already said above.

Rules:
- Use only what is in the conversation. Do not introduce outside facts, even
  ones you are confident about.
- Say plainly that this is your reading of the conversation, not something the
  documents state. The user must not mistake a judgment for a citation.
- If the conversation does not contain enough to answer, set abstained=true, say
  what is missing, and put a standalone question the documents could answer in
  suggested_question.
- Do not invent citations. There are no sources this turn."""


def answer_from_conversation(
    question: str,
    history: str,
    *,
    max_output_tokens: int = 1024,
) -> Answer:
    """Answer a conversation_only turn. Answered, not abstained.

    WHY THIS EXISTS AS A SEPARATE PATH. Calling answer() with zero sources
    produces "retrieval returned nothing for this question", which is a true
    statement about retrieval and the wrong answer to the user: the question was
    routed here precisely BECAUSE retrieval had nothing to add. Abstaining would
    refuse a question the conversation can answer.

    Every claim from this path is labelled source="conversation", so the citation
    layer renders it grey and never in the document colour. The quote checks are
    skipped because there is nothing verbatim to check -- and a citation check
    that cannot run must report that, not pass.
    """
    if not history.strip():
        return Answer(
            text="", abstained=True,
            abstain_reason="there is no conversation yet to read from",
            reconciliation={"chunks_sent": 0},
        )

    prompt = f"Conversation so far:\n{history}\n\nNow answer: {question}"
    try:
        raw, usage = gemini.generate(
            prompt, role="workhorse", system=CONVERSATION_SYSTEM,
            schema=CONVERSATION_SCHEMA, max_output_tokens=max_output_tokens,
            stage="answer_conversation",
        )
    except limits.QuotaExhausted:
        return Answer(text="", abstained=True,
                      abstain_reason=_quota_reason(),
                      reconciliation={"chunks_sent": 0})
    except gemini.EmptyResponse as exc:
        return Answer(text="", abstained=True,
                      abstain_reason=f"model returned no text: {exc}",
                      reconciliation={"chunks_sent": 0})

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Identical defect to the document path -- raw JSON in the answer field
        # and a message naming the parser instead of the output ceiling. Fixed in
        # both places rather than one: the conversation route has no sources, so
        # nobody would have noticed it here until a user hit it.
        return Answer(
            text=_salvage_markdown(raw), grounded=False,
            reasons=[(f"the answer was cut off at the output limit "
                      f"({max_output_tokens} tokens)") if usage.truncated
                     else f"structured output did not parse ({exc})"],
            reconciliation={"chunks_sent": 0, "output_truncated": usage.truncated},
            usage=usage)

    claims = [
        {"text": (c.get("text") or "").strip(), "source": "conversation", "citations": []}
        for c in (payload.get("claims") or [])
        if (c.get("text") or "").strip()
    ]
    return Answer(
        text=(payload.get("answer_markdown") or "").strip(),
        abstained=bool(payload.get("abstained")),
        abstain_reason=payload.get("abstain_reason", ""),
        suggested_question=(payload.get("suggested_question") or "").strip(),
        claims=claims,
        checks=[],
        # GROUNDED IN THE CONVERSATION, which is a different claim from grounded
        # in the documents -- and the route label is what carries that
        # distinction. Marking it ungrounded would imply a fault; marking it
        # grounded without the label would imply document support it does not have.
        grounded=True,
        # EVERY KEY, including the zeroes. The document path emits
        # chunks_cited and citations_unquotable and this path did not, so a
        # consumer reading the reconciliation strip got blanks for two fields on
        # conversation turns. A missing key is ambiguous -- "not applicable" and
        # "forgot to compute" look identical -- while an explicit 0 is a fact.
        # `verification` carries the not-applicable meaning; the counts stay
        # numeric so nothing downstream has to guess a shape.
        reconciliation={
            "chunks_sent": 0,
            "chunks_cited": 0,
            "citations_emitted": 0,
            "citations_verified": 0,
            "citations_absent": 0,
            "citations_fabricated": 0,
            "citations_unquotable": 0,
            "absent_mean_overlap_pct": 0,
            "claims": len(claims),
            "claims_from_documents": 0,
            "claims_from_conversation": len(claims),
            "claims_uncited": 0,
            "claims_mislabelled": 0,
            "verification": "not_applicable_no_sources",
        },
        usage=usage,
    )


def _quota_reason() -> str:
    """The abstain reason for a quota stop, named for the deployment it ran on.

    This was the literal string "free-tier Gemini quota exhausted" in two places.
    On a billed key it is simply untrue, and it is shown to the END USER as the
    reason their question went unanswered -- so it is the worst place to keep a
    stale cause.
    """
    if config.DEPLOYMENT_KIND == "demo":
        return "free-tier Gemini quota exhausted"
    return "Gemini API quota or rate limit reached"


def _salvage_markdown(raw: str) -> str:
    """Pull answer_markdown out of JSON that was cut off mid-document.

    When the output budget runs out the JSON is truncated, usually inside a long
    string, and json.loads gives up on the whole document. But the prose the user
    wants is the FIRST field and is almost entirely present -- throwing it away
    and showing raw JSON instead is a strictly worse outcome than showing the
    prose and saying the citations were lost.

    What is NOT salvaged, deliberately: the claims and their citations. Those are
    the checkable part, and a half-read citation list is exactly the thing that
    must never be presented as verified. So the answer comes back ungrounded --
    readable, and clearly marked as unverifiable.
    """
    i = raw.find('"answer_markdown"')
    if i < 0:
        return ""
    # Both finds must be checked. `find` returns -1 on failure, and -1 + 1 == 0
    # restarts the search from the top of the document -- which found the opening
    # quote of the KEY and cheerfully "salvaged" the string "answer_markdown".
    # A sentinel that arithmetic turns into a valid index is a trap, and this is
    # the same shape as the `not existing` truthiness bug in the judge harness.
    colon = raw.find(":", i + 17)
    if colon < 0:
        return ""
    j = raw.find('"', colon + 1)
    if j < 0:
        return ""
    out: list[str] = []
    k = j + 1
    while k < len(raw):
        c = raw[k]
        if c == "\\":
            out.append(raw[k:k + 2])
            k += 2
            continue
        if c == '"':
            break
        out.append(c)
        k += 1
    body = "".join(out)
    # A cut can land mid-escape ("...\u00e" or a lone trailing backslash), which
    # would make even this one string undecodable. Trim from the right until it
    # decodes rather than giving up on the whole answer.
    for trim in range(0, 8):
        try:
            return json.loads('"' + (body[:-trim] if trim else body) + '"')
        except json.JSONDecodeError:
            continue
    return ""


def answer(
    question: str,
    chunks: Sequence[Chunk],
    *,
    history: str = "",
    # Was a hardcoded 2048, which truncated real answers -- see
    # config.TOKENS_ANSWER_OUTPUT for the measurement and the reason structured
    # output needs its own budget rather than the prose reserve.
    max_output_tokens: int = config.TOKENS_ANSWER_OUTPUT,
    thinking: bool = False,
    # Primitives, not the Retrieved object, so `generate` keeps no import of
    # `index` -- the dependency runs one way and adding a diagnostic is not a
    # reason to reverse it. Callers that have a Retrieved pass
    # `starved=r.starved_by_budget, min_unit_tokens=r.min_unit_tokens`.
    starved: bool = False,
    min_unit_tokens: int | None = None,
    # The budget ACTUALLY used for this call. Without it the remedy reported
    # config.TOKENS_CONTEXT_BUDGET (12000) while the caller had passed 1500 --
    # advice to "raise the budget above 12000" when the real one was 1500, which
    # is the third misattribution in this same code path. A default is not a
    # substitute for the value in use.
    context_budget: int | None = None,
) -> Answer:
    if not chunks:
        # TWO CAUSES, ONE SYMPTOM. This said "retrieval returned nothing for
        # this question" unconditionally, which is false in the case that
        # actually occurs: on one golden-set item retrieval ranked candidates
        # correctly and the strict budget fill admitted none of them, because
        # the top-ranked parent is larger than the whole 1500-token budget. The
        # message sent the reader to the embedding and the index -- the two
        # parts that were working -- while the real fix was one number.
        #
        # Second instance of a warning misattributing its own cause (the first
        # was the INCOMPLETE-source notice naming deferred contextual prefixes).
        # The pattern to watch for: a message written when only one cause was
        # possible, left unchanged after a second one became possible.
        if starved:
            need = min_unit_tokens
            limits.report(
                limits.Degradation(
                    stage="retrieve",
                    cause="context_budget_smaller_than_smallest_unit",
                    impact="no sources could be sent to the model, so this question "
                           "was not answered -- retrieval itself worked",
                    fallback="abstained rather than answer without evidence",
                    remedy=(f"raise the context budget to at least {need} tokens "
                            f"(this call used "
                            f"{context_budget or config.TOKENS_CONTEXT_BUDGET}), or "
                            "retrieve the child unit, whose units are far smaller"
                            if need else
                            "raise the context budget, or retrieve the child unit"),
                    free_tier=False,   # our own budget knob, not a Gemini limit
                )
            )
            return Answer(
                text="", abstained=True,
                abstain_reason=(
                    "context budget too small for the top-ranked unit: the smallest "
                    f"candidate needs {need} tokens" if need else
                    "context budget too small for any ranked unit"),
                reconciliation={"chunks_sent": 0, "starved_by_budget": True,
                                "min_unit_tokens": need},
            )
        return Answer(
            text="", abstained=True,
            abstain_reason="retrieval returned nothing for this question",
            reconciliation={"chunks_sent": 0, "starved_by_budget": False},
        )

    body, mapping = assemble(chunks)
    prompt = (
        (f"Conversation so far:\n{history}\n\n" if history else "")
        + f"Sources:\n\n{body}\n\n---\n\nQuestion: {question}"
    )

    try:
        raw, usage = gemini.generate(
            prompt, role="workhorse", system=SYSTEM, schema=ANSWER_SCHEMA,
            max_output_tokens=max_output_tokens, thinking=thinking, stage="answer",
        )
    except limits.QuotaExhausted:
        limits.report(
            limits.Degradation(
                stage="answer", cause="free_tier_quota_exhausted",
                impact="no answer was generated for this question",
                fallback="none -- the question is unanswered",
                remedy="wait for the per-minute quota to reset and retry",
            )
        )
        return Answer(text="", abstained=True, abstain_reason=_quota_reason(),
                      reconciliation={"chunks_sent": len(mapping)}, sources=mapping)
    except gemini.EmptyResponse as exc:
        return Answer(text="", abstained=True, abstain_reason=f"model returned no text: {exc}",
                      reconciliation={"chunks_sent": len(mapping)}, sources=mapping)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Structured output failing to parse is itself ungrounded: we cannot pair
        # claims with citations, so nothing is checkable. That part was right.
        #
        # What was wrong was the message and the body. It reported "structured
        # output did not parse" -- true, and useless, because it names the last
        # layer to notice rather than the cause -- and it put the RAW JSON in the
        # answer field, so the user read `{"abstained": false, ...}` where the
        # answer should be. Two separate mistakes stacked on one failure.
        salvaged = _salvage_markdown(raw)
        if usage.truncated:
            reason = (
                f"the answer was cut off at the output limit "
                f"({max_output_tokens} tokens), so its citations are incomplete "
                "and none of them could be verified"
            )
            limits.report(
                limits.Degradation(
                    stage="answer", cause="output_token_limit_reached",
                    impact="the answer is incomplete and its citations could not be "
                           "verified, so nothing in it is marked as evidence",
                    fallback=("showed the readable part of the answer without citations"
                              if salvaged else "no answer could be recovered"),
                    remedy=(f"raise config.TOKENS_ANSWER_OUTPUT above {max_output_tokens}, "
                            "or ask a narrower question so fewer claims need quoting"),
                    free_tier=False,   # our own ceiling, not a Gemini one
                )
            )
        else:
            reason = f"structured output did not parse ({exc}); citations unverifiable"
        return Answer(
            # Never the raw JSON. Either the prose, or nothing with a reason.
            text=salvaged, abstained=False, grounded=False,
            reasons=[reason], usage=usage, sources=mapping,
            reconciliation={"chunks_sent": len(mapping),
                            "output_truncated": usage.truncated,
                            "verification": "not_attempted_unparseable_output"},
        )

    if config.FAKE_UNVERIFIABLE_QUOTE:
        # Corrupt the quote, not the label. The citation stays valid -- the chunk
        # was really sent -- so membership passes and only the quote check fails,
        # which is precisely the "found, not quoted" case rather than fabrication.
        for _cl in payload.get("claims") or []:
            for _cit in _cl.get("citations") or []:
                _cit["quote"] = "a paraphrase that appears nowhere in the passage"

    if config.FAKE_FABRICATED_CITATION and (payload.get("claims") or []):
        # Injected BEFORE verification, so the citation travels the real path:
        # membership check -> fabricated flag -> fail-closed -> UI chip. Label
        # len(mapping)+7 cannot have been sent, which is the point -- a plausible
        # integer that was never in this request.
        payload["claims"][0].setdefault("citations", []).append(
            {"label": len(mapping) + 7, "quote": "an invented quotation from nowhere"}
        )

    checks, grounded, reasons, recon = verify(payload, mapping)

    # Dead code removed: I had a recomposition fallback here for an "empty
    # answer_markdown with populated claims" case that never existed. It came
    # from reading a truncated `tail -12` as a missing field. Code written for an
    # imaginary failure reads like protection against a real one, so it goes.
    return Answer(
        text=(payload.get("answer_markdown") or "").strip(),
        abstained=bool(payload.get("abstained")),
        abstain_reason=payload.get("abstain_reason", ""),
        suggested_question=(payload.get("suggested_question") or "").strip(),
        claims=payload.get("claims") or [],
        checks=checks, grounded=grounded, reasons=reasons, reconciliation=recon,
        usage=usage, sources=mapping,
    )
