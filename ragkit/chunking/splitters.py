"""
Chunking. Guide module M5.

THE TWO IDEAS THIS FILE EXISTS FOR

1. DECOUPLING. What you embed and what you return are different objects with
   different optimal sizes. A 300-token child is topically tight, so it embeds
   precisely. A 1200-token parent is self-contained, so it reads well. Retrieve
   the child, return the parent.

2. ORDER OF OPERATIONS. Enrichment must be applied AFTER slicing, per child,
   never before slicing.

Point 2 is the whole design, and it comes from your answer to the breadcrumb
drill. Prefixing a parent and THEN cutting it into children gives the breadcrumb
to child 0 only. Nothing errors. Every chunk is well-formed. But section-opening
chunks become systematically more findable for positional reasons unrelated to
relevance, aggregate recall ticks UP, and the bias reads as a win.

Same shape as the parser bug found an hour earlier: two operations, order
matters, output still looks valid. import-then-measure vs measure-then-import.
prefix-then-slice vs slice-then-prefix. Neither raises. Both produce a
confidently wrong number.

So the ordering is enforced structurally rather than remembered:
  - `_slice_body()` only ever sees raw body text; it cannot see a breadcrumb.
  - `_make_child()` is the only place a breadcrumb is attached, and it attaches
    to exactly one already-sliced body.
  - `assert_no_prefix_leak()` fails the build if a parent's stored text contains
    a breadcrumb, which is the fingerprint of prefix-then-slice.

---------------------------------------------------------------------------
THE BREADCRUMB HAS TWO JOBS AND THEY ARE NOT THE SAME

On the CHILD it is retrieval text: it enters `embed_text` and moves the vector.
On the PARENT it is reading text: it tells the model what section it is looking
at. Recall measures the first and is structurally blind to the second.

Consequence: the parent's breadcrumb is NEVER concatenated into its body. It
stays in `heading_path` and the prompt assembler renders it as marked metadata.
Concatenating it would put model-visible text that is not in the document into a
field the citation layer treats as quotable -- a fabricated quotation carrying a
real page number, which is the image-caption failure arriving through the parent.

`verbatim_text` therefore holds the body only, on both records.

---------------------------------------------------------------------------
FREE BEFORE PAID

heading_path costs nothing: the loaders already extracted it. Contextual
prefixes (Session 4) are the same idea with an LLM call per chunk attached. So
the breadcrumb version gets measured FIRST, and the LLM version has to beat a
real baseline instead of an empty one. If breadcrumbs recover most of the gain,
the entire Session 4 quota budget is saved. If they do not, we learn what the
paid version is actually buying.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

from .. import config
from ..gemini import count_tokens
from ..ingest.document import (
    Block,
    Chunk,
    ChunkKind,
    ChunkRole,
    PipelineVersion,
    Source,
    TextProvenance,
    sha256_text,
)

Strategy = Literal["fixed", "recursive", "header_aware", "header_aware_parent"]

CHUNKER_VERSIONS: dict[str, str] = {
    "fixed": "fixed@1",
    "recursive": "recursive@1",
    "header_aware": "header_aware@1",
    "header_aware_parent": "header_aware_parent@3",
}

# What goes in PipelineVersion.contextualizer. Its own version string, because
# the situating PROMPT can change without the splitter changing a line -- and
# two indexes built by the same splitter from different prompts are not
# comparable either.
CONTEXTUALIZER_BREADCRUMB = "breadcrumb-only"
CONTEXTUALIZER_LLM = "llm-prefix@1"

# Paragraph, then line, then sentence, then word. Recursive splitting tries each
# separator in turn and only falls through when a piece is still too big, so it
# breaks at the most semantically meaningful boundary available rather than at a
# fixed character offset.
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def breadcrumb(heading_path: Sequence[str]) -> str:
    """The free context: 'Section > Subsection'. Empty when there is none."""
    return " > ".join(h for h in heading_path if h)


def _normalize_for_compare(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _prefix_string(src: Source, heading_path: Sequence[str]) -> str:
    """What gets prepended to a CHILD's embed_text.

    Includes the document title, because a breadcrumb alone answers "which
    section" and not "which document" -- and 'Results > Recall rose 8 points' is
    ambiguous across nine papers that all have a Results section.

    DEDUPED against the first heading, because a paper's H1 usually IS its title,
    and 'seven-failure-points > Seven Failure Points When Engineering a
    Retrieval Augmented Generation System' spends the same tokens twice. That
    matters more than it looks: the prefix is a fixed cost on every child, and
    measured overhead is ~9% of a paper chunk and ~39% on manual.docx, where
    bodies are short. Tokens spent on a repeated title are tokens the body does
    not get -- and because breadcrumb length correlates with section depth, that
    cost is unevenly distributed across the document, which is a bias in BM25's
    length normalisation before it is a cost.
    """
    # Keep the MOST INFORMATIVE form of each distinct name. A first pass that
    # merely skipped subsumed candidates kept the filename slug
    # ("seven-failure-points") and discarded the real heading ("Seven Failure
    # Points When Engineering a Retrieval Augmented Generation System") -- the
    # dedupe worked and threw away the better string. So containment REPLACES
    # rather than skips.
    parts: list[str] = []
    for candidate in (src.title, *heading_path):
        cand = (candidate or "").strip()
        if not cand:
            continue
        norm = _normalize_for_compare(cand)
        if not norm:
            continue
        replaced = False
        skip = False
        for idx, existing in enumerate(parts):
            en = _normalize_for_compare(existing)
            if norm == en or norm in en:
                skip = True
                break
            if en in norm:
                parts[idx] = cand
                replaced = True
                break
        if skip or replaced:
            continue
        parts.append(cand)
    sep = "\n\n"
    return (" > ".join(parts) + sep) if parts else ""


# --------------------------------------------------------------------------
# Splitters -- these see BODY TEXT ONLY, never a breadcrumb
# --------------------------------------------------------------------------


def _slice_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Baseline. Breaks semantics at arbitrary offsets, and that is the point:
    it is the control that later strategies must beat."""
    if not text.strip():
        return []
    words = text.split()
    if not words:
        return []
    # Token-per-word ratio measured on this text rather than assumed, so the
    # word window actually lands near `size` tokens.
    ratio = max(count_tokens(text) / len(words), 1e-6)
    win = max(int(size / ratio), 1)
    step = max(win - int(overlap / ratio), 1)
    return [" ".join(words[i : i + win]) for i in range(0, len(words), step) if words[i : i + win]]


def _slice_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Respect paragraph, then sentence, then word boundaries."""
    if not text.strip():
        return []
    if count_tokens(text) <= size:
        return [text.strip()]

    units = [u for u in _PARA.split(text) if u.strip()]
    if len(units) == 1:
        units = [u for u in _SENT.split(text) if u.strip()]
    if len(units) == 1:
        return _slice_fixed(text, size, overlap)

    out: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for u in units:
        t = count_tokens(u)
        if t > size:                      # a single unit too big: recurse into it
            if buf:
                out.append("\n\n".join(buf)); buf, buf_tok = [], 0
            out.extend(_slice_recursive(u, size, overlap))
            continue
        if buf_tok + t > size and buf:
            out.append("\n\n".join(buf))
            # Overlap carried as whole trailing units, not a character tail --
            # a half-sentence of overlap adds tokens without adding meaning.
            keep: list[str] = []
            keep_tok = 0
            for prev in reversed(buf):
                pt = count_tokens(prev)
                if keep_tok + pt > overlap:
                    break
                keep.insert(0, prev); keep_tok += pt
            buf, buf_tok = keep, keep_tok
        buf.append(u); buf_tok += t
    if buf:
        out.append("\n\n".join(buf))
    return [o.strip() for o in out if o.strip()]


def _slice_body(text: str, *, size: int, overlap: int, recursive: bool) -> list[str]:
    return (_slice_recursive if recursive else _slice_fixed)(text, size, overlap)


# --------------------------------------------------------------------------
# Grouping blocks into parents
# --------------------------------------------------------------------------


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    blocks: list[Block]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


def _group_sections(blocks: Iterable[Block], *, max_tokens: int) -> list[_Section]:
    """Group consecutive blocks sharing a heading path, capped at max_tokens.

    A TABLE block is never merged with prose. Two reasons, and the second is the
    one that bites: a table's markdown is dense in a way prose is not, so mixing
    them makes the chunk-size budget meaningless; and merging lets a table lose
    its own heading_path to whatever prose it got attached to, which is how a
    table ends up cited under the wrong section.
    """
    out: list[_Section] = []
    cur: _Section | None = None
    cur_tok = 0
    for b in blocks:
        if not b.text.strip():
            continue
        t = count_tokens(b.text)
        isolate = b.kind in (ChunkKind.TABLE, ChunkKind.TABLE_SUMMARY, ChunkKind.IMAGE_CAPTION)
        if (
            cur is None
            or isolate
            or cur.blocks[-1].kind in (ChunkKind.TABLE, ChunkKind.TABLE_SUMMARY, ChunkKind.IMAGE_CAPTION)
            or b.heading_path != cur.heading_path
            or cur_tok + t > max_tokens
        ):
            cur = _Section(heading_path=tuple(b.heading_path), blocks=[b])
            out.append(cur)
            cur_tok = t
        else:
            cur.blocks.append(b)
            cur_tok += t
    return out


# --------------------------------------------------------------------------
# Chunk construction -- the ONLY place a breadcrumb is attached
# --------------------------------------------------------------------------


# Kinds whose text NOBODY WROTE: a model produced it. Nothing in them is
# quotable from the document, so a quote check against them is meaningless.
_MODEL_GENERATED_KINDS = (ChunkKind.IMAGE_CAPTION, ChunkKind.TABLE_SUMMARY)


def _section_provenance(section: "_Section") -> TextProvenance:
    """Provenance of a whole section, for the PARENT record.

    THE BUG THIS FIXES was live and serious. _make_parent hardcoded
    VERBATIM + verbatim_text=body for every parent, so the parent of an image
    caption asserted that Gemini's description of a chart was literal document
    text. Quote verification then matched against it and reported
    `quote=verified` -- certifying model-written prose as a document quotation
    with a real page number.

    Observed consequence, not hypothetical: an answer cited chart values
    (75%, 61%, 53%) that Gemini had read off an image's PIXELS, and the citation
    checker passed them as verified quotes from the paper.

    The child path had this right from the start (_make_child sets
    MODEL_GENERATED for IMAGE_CAPTION). The parent path is what reaches the
    model, so getting it wrong there defeated the guard entirely -- and the child
    check passing is what made it look fine.
    """
    kinds = {b.kind for b in section.blocks}
    if kinds & set(_MODEL_GENERATED_KINDS):
        # Conservative on a mixed section: if any part is model-written we cannot
        # tell which part a quote came from, so refuse to certify any of it.
        # _group_sections isolates caption/summary blocks, so mixing should not
        # occur -- this is the branch that stays correct if that changes.
        return TextProvenance.MODEL_GENERATED
    return TextProvenance.VERBATIM


def _section_text_source(section: "_Section") -> str:
    """Which extractor produced this section's text.

    A section can mix blocks from different extractors (a repaired table grouped
    with markdown prose), so a single label would be a lie. "mixed:" names both
    rather than silently picking one -- the same reason a mixed-provenance index
    refuses to be a CI baseline instead of averaging.
    """
    srcs = sorted({b.text_source for b in section.blocks})
    if len(srcs) == 1:
        return srcs[0]
    return "mixed:" + "+".join(srcs)


def _make_child(
    *,
    src: Source,
    body: str,
    prefix: str,
    section: _Section,
    # THE LLM-WRITTEN SITUATING SENTENCE, or "". Attached HERE and nowhere else,
    # for exactly the reason the breadcrumb is: this file's whole design is that
    # enrichment happens after slicing, per child. A contextual prefix applied
    # before slicing would land on child 0 alone and make section openers
    # systematically more findable -- the same positional bias, arriving through
    # a new door.
    context: str,
    ordinal: int,
    parent_id: str | None,
    position: int,
    n_siblings: int,
    pipeline: PipelineVersion,
    # NO DEFAULT, DELIBERATELY. Ownership is not something set after the fact --
    # it is a fact about the chunk that must be stated to create one. A default
    # of PUBLIC_OWNER would make "forgot to pass it" and "meant it to be public"
    # the same keystroke, and only one of those is a leak.
    #
    # assert_owned() in NumpyIndex.__init__ is the backstop. This is the version
    # the editor catches, which is cheaper than the version the constructor
    # catches, which is cheaper than the version a visitor catches.
    owner: str,
    origin: str,
) -> Chunk:
    """Build one child from ONE already-sliced body.

    This signature is the enforcement mechanism. `body` is a single slice and
    `prefix` is applied to it here and nowhere else, so prefix-then-slice is not
    expressible: there is no code path where a prefixed string reaches a splitter.
    """
    first = section.blocks[0]
    last = section.blocks[-1]
    kind = first.kind if len(section.blocks) == 1 else ChunkKind.TEXT
    # ORDER: breadcrumb, then situating sentence, then body. The body is last so
    # that BM25's length normalisation sees the same shape it always did, and so
    # a human reading embed_text can see where the document starts.
    ctx = (context.strip() + "\n\n") if context.strip() else ""
    embed_text = prefix + ctx + body

    # PREFIXED, not VERBATIM: the body is a contiguous span of the document and
    # is quotable; the prefix is assembled from headings and is not. So
    # `quote` stays the body and citation() reports highlightable=True for it.
    #
    # A CONTEXTUAL PREFIX DOES NOT WEAKEN THIS, it is the case the state was
    # built for. The situating sentence is model-written, sits against document
    # prose, and carries the chunk's real page number -- the exact combination
    # that makes a fabricated quotation look genuine. It goes into embed_text
    # only; verbatim_text and display_text stay the body, unchanged.
    prov = TextProvenance.PREFIXED if (prefix or ctx) else TextProvenance.VERBATIM
    if kind is ChunkKind.IMAGE_CAPTION:
        prov = TextProvenance.MODEL_GENERATED

    return Chunk(
        owner=owner,
        origin=origin,
        chunk_id=Chunk.make_id(src.source_id, ordinal, embed_text, ChunkRole.CHILD),
        source_id=src.source_id,
        ordinal=ordinal,
        role=ChunkRole.CHILD,
        parent_id=parent_id,
        position_within_parent=position,
        n_siblings=n_siblings,
        embed_text=embed_text,
        has_contextual_prefix=bool(ctx),
        display_text=body,
        verbatim_text=None if prov is TextProvenance.MODEL_GENERATED else body,
        text_provenance=prov,
        kind=kind,
        source_uri=src.uri,
        page=first.page,
        page_end=last.page_end or (last.page if last.page != first.page else None),
        bbox=first.bbox,
        asset_path=first.asset_path,
        heading_path=section.heading_path,
        table_continuation_suspect=any(b.table_continuation_suspect for b in section.blocks),
        table_header_missing=any(b.table_header_missing for b in section.blocks),
        # PROPAGATED, and it was not before. The loader repaired two corrupted
        # tables by swapping in bbox-clipped page text; the repaired TEXT flowed
        # through because it is just Block.text, but the LABEL did not, so all
        # 791 children claimed text_source="markdown". The fix worked and the
        # audit trail did not -- and the audit trail is the half the eval slices
        # on. Caught only because the index report prints the loader's repair
        # count next to the children's page-text count, two numbers that must
        # agree.
        text_source=_section_text_source(section),
        pipeline_fingerprint=pipeline.fingerprint(),
        content_hash=sha256_text(body),
    )


def _make_parent(
    *, src: Source, section: _Section, ordinal: int, pipeline: PipelineVersion,
    owner: str, origin: str,
) -> Chunk:
    """The returned unit. Never embedded, and its body never carries a breadcrumb.

    embed_text is deliberately empty. A parent with embed_text set would be
    embeddable, and something downstream would eventually embed it -- doubling
    the index and putting parent and child in competition for the same top-k.
    """
    first = section.blocks[0]
    last = section.blocks[-1]
    body = section.text
    prov = _section_provenance(section)
    return Chunk(
        owner=owner,
        origin=origin,
        chunk_id=Chunk.make_id(src.source_id, ordinal, body, ChunkRole.PARENT),
        source_id=src.source_id,
        ordinal=ordinal,
        role=ChunkRole.PARENT,
        parent_id=None,
        embed_text="",
        display_text=body,
        # None when nothing here is verbatim: the quote checker treats a missing
        # verbatim_text as "unquotable" rather than failing the citation, which
        # is the correct outcome for text no human wrote.
        verbatim_text=None if prov is TextProvenance.MODEL_GENERATED else body,
        text_provenance=prov,
        kind=ChunkKind.PARENT,
        source_uri=src.uri,
        page=first.page,
        page_end=last.page_end or (last.page if last.page != first.page else None),
        heading_path=section.heading_path,
        table_continuation_suspect=any(b.table_continuation_suspect for b in section.blocks),
        table_header_missing=any(b.table_header_missing for b in section.blocks),
        text_source=_section_text_source(section),
        pipeline_fingerprint=pipeline.fingerprint(),
        content_hash=sha256_text(body),
    )


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def assert_no_prefix_leak(chunks: Sequence[Chunk], src: Source) -> None:
    """Fail the build on the fingerprint of prefix-then-slice.

    Two invariants, both cheap, both catching a defect that produces
    well-formed output:

      1. No PARENT's stored text contains a breadcrumb. If one does, the
         breadcrumb was prepended before slicing.
      2. Every child in a multi-child parent carries the prefix, or none does.
         A parent whose child 0 has a prefix and whose child 3 does not is the
         positional-bias signature exactly.
    """
    for c in chunks:
        if c.role is ChunkRole.PARENT:
            crumb = breadcrumb(c.heading_path)
            body = (c.verbatim_text or "").lstrip()
            # STARTSWITH, not `in`. My first version tested containment anywhere
            # and that predicate is simply wrong: prefix-then-slice puts the
            # breadcrumb at the START of the body, while a legitimate paper can
            # mention its own section title mid-paragraph ("as shown in 2.1 RAG
            # Approaches and Systems"). Containment flags the innocent case.
            #
            # It hard-failed the full-corpus build on a breadcrumb of '{{', which
            # occurs in ordinary text by coincidence -- a guard with the power to
            # stop a build needs a predicate that matches the actual defect
            # signature, not one that merely correlates with it. This is your
            # point about a flag stopping being risk-free once something acts on
            # it, arriving in the most literal form available: the thing acting
            # on it was the build.
            if crumb and len(crumb) >= 8 and body.startswith(crumb):
                raise AssertionError(
                    f"prefix leak: parent {c.chunk_id} body BEGINS with its own "
                    f"breadcrumb ({crumb[:60]!r}). The breadcrumb was applied "
                    "before slicing."
                )
    by_parent: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.role is ChunkRole.CHILD and c.parent_id:
            by_parent.setdefault(c.parent_id, []).append(c)
    for pid, kids in by_parent.items():
        if len(kids) < 2:
            continue
        prefixed = [k.text_provenance is TextProvenance.PREFIXED for k in kids]
        if any(prefixed) and not all(prefixed):
            n = sum(prefixed)
            raise AssertionError(
                f"positional bias: parent {pid} has {n}/{len(kids)} prefixed children. "
                "Enrichment must be per-child and uniform, or position predicts retrieval."
            )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_chunks(
    src: Source,
    blocks: Sequence[Block],
    *,
    strategy: Strategy = "header_aware_parent",
    breadcrumbs: bool = True,
    child_tokens: int | None = None,
    parent_tokens: int | None = None,
    overlap_ratio: float | None = None,
    pipeline: PipelineVersion | None = None,
    # (parent section text, sliced body) -> one situating sentence, or "".
    #
    # A CALLABLE RATHER THAN A FLAG, so this module never imports gemini. The
    # chunker stays free, deterministic and testable, and the paid boundary
    # stays where the audit already knows to look for it. Passing a lambda in a
    # test gives contextual chunking with no API key and no spend.
    contextualizer: Callable[[str, str], str] | None = None,
    # REQUIRED, and it propagates to every chunk this produces. There is exactly
    # one chunking entry point and two Chunk constructors, all in this file, so
    # this parameter is the whole surface: nothing can become searchable without
    # passing through here and stating who owns it.
    owner: str,
    origin: str = "corpus",
) -> list[Chunk]:
    """Blocks -> chunks, under one named strategy.

    Strategies exist as a set because Session 4's experiment compares them on
    the same corpus with the same eval set. Each one is a config, not a rewrite.
    """
    child_tokens = child_tokens or config.CHUNK_CHILD_TOKENS
    parent_tokens = parent_tokens or config.CHUNK_PARENT_TOKENS
    ratio = config.CHUNK_OVERLAP_RATIO if overlap_ratio is None else overlap_ratio
    overlap = int(child_tokens * ratio)
    # THE FINGERPRINT RECORDS WHAT WAS APPLIED, not what the file can do.
    #
    # Contextual prefixes change embed_text, which changes chunk_ids, which
    # means two indexes built from the same bytes are not comparable. The
    # fingerprint has to say so -- that is its entire job. PipelineVersion
    # already carries a `contextualizer` field for exactly this, so the fact
    # goes there rather than into a second, parallel version string.
    #
    # And it is recorded ONLY when a contextualiser was actually passed. Moving
    # the fingerprint for a run that produced a byte-identical index would
    # invalidate a valid baseline and force a re-measurement that could only
    # reproduce it.
    pipeline = pipeline or PipelineVersion(
        chunker=CHUNKER_VERSIONS[strategy],
        contextualizer=CONTEXTUALIZER_LLM if contextualizer else CONTEXTUALIZER_BREADCRUMB,
    )

    recursive = strategy != "fixed"
    header_aware = strategy in ("header_aware", "header_aware_parent")
    use_parents = strategy == "header_aware_parent"

    if header_aware:
        sections = _group_sections(blocks, max_tokens=parent_tokens)
    else:
        # Flat strategies ignore document structure on purpose: one section over
        # the whole document, so the comparison isolates the splitter.
        flat = [b for b in blocks if b.text.strip()]
        sections = [_Section(heading_path=(), blocks=flat)] if flat else []

    chunks: list[Chunk] = []
    ordinal = 0
    for sec in sections:
        body = sec.text
        if not body.strip():
            continue

        parent: Chunk | None = None
        if use_parents:
            parent = _make_parent(src=src, section=sec, ordinal=ordinal,
                              pipeline=pipeline, owner=owner, origin=origin)
            chunks.append(parent)
            ordinal += 1

        # SLICE FIRST. The splitter receives body text and nothing else.
        bodies = _slice_body(body, size=child_tokens, overlap=overlap, recursive=recursive)
        # THEN PREFIX, once per slice.
        prefix = _prefix_string(src, sec.heading_path) if (breadcrumbs and header_aware) else ""
        for pos, piece in enumerate(bodies):
            # THEN CONTEXTUALISE, once per slice, with the slice in hand. Same
            # position in the pipeline as the breadcrumb and for the same
            # reason -- see _make_child.
            ctx = contextualizer(body, piece) if contextualizer else ""
            chunks.append(
                _make_child(
                owner=owner,
                origin=origin,
                    src=src, body=piece, prefix=prefix, context=ctx,
                    section=sec, ordinal=ordinal,
                    parent_id=parent.chunk_id if parent else None,
                    position=pos, n_siblings=len(bodies), pipeline=pipeline,
                )
            )
            ordinal += 1

    assert_no_prefix_leak(chunks, src)
    return chunks


def check_provenance_propagation(
    per_file: Sequence[dict[str, object]]
) -> tuple[bool, list[str]]:
    """Did every repaired block reach the index carrying its label?

    THE PREDICATE MATTERS, and my first version got it wrong in the build itself.
    I asserted `n_repaired_blocks == n_labelled_children` and it fired a false
    alarm on the first run: 2 repaired blocks produced 3 labelled children.
    Correctly. A repaired block of 376 tokens exceeds CHUNK_CHILD_TOKENS=300 and
    SPLITS -- blocks and children are different units, and equality between their
    counts was never the invariant.

    The actual invariants:
      1. per file, repaired_blocks > 0  <=>  labelled_children > 0
         (a repair that produces no labelled child was dropped in the plumbing;
          a labelled child with no repaired block came from nowhere)
      2. labelled_children >= repaired_blocks
         (splitting can only increase the count, never reduce it)

    Same lesson as `crumb in body`: a check with the power to fail a build needs
    a predicate that matches the defect's signature, not one that correlates with
    it. A correlated predicate is fine in a report and disqualifying in a gate.
    """
    problems: list[str] = []
    for row in per_file:
        rep = int(row.get("tables_repaired", 0) or 0)
        lab = int(row.get("children_page_text", 0) or 0)
        name = row.get("file", "?")
        if (rep > 0) != (lab > 0):
            problems.append(
                f"{name}: {rep} repaired block(s) but {lab} labelled child(ren) -- "
                "provenance dropped between loader and chunker"
                if rep
                else f"{name}: {lab} labelled child(ren) with 0 repaired blocks -- label invented"
            )
        elif lab < rep:
            problems.append(
                f"{name}: {lab} labelled children < {rep} repaired blocks; splitting "
                "cannot reduce the count, so at least one repair was lost"
            )
    return (not problems), problems


def stats(chunks: Sequence[Chunk]) -> dict[str, object]:
    kids = [c for c in chunks if c.role is ChunkRole.CHILD]
    parents = [c for c in chunks if c.role is ChunkRole.PARENT]
    tok = [count_tokens(c.embed_text) for c in kids] or [0]
    body_tok = [count_tokens(c.display_text) for c in kids] or [0]
    return {
        "n_children": len(kids),
        "n_parents": len(parents),
        "child_embed_tokens_mean": round(sum(tok) / len(tok), 1),
        "child_embed_tokens_max": max(tok),
        "child_body_tokens_mean": round(sum(body_tok) / len(body_tok), 1),
        "prefix_overhead_tokens_mean": round((sum(tok) - sum(body_tok)) / len(tok), 1),
        "n_prefixed": sum(1 for c in kids if c.text_provenance is TextProvenance.PREFIXED),
        "n_header_missing": sum(1 for c in kids if c.table_header_missing),
        "n_continuation_suspect": sum(1 for c in kids if c.table_continuation_suspect),
        "orphan_children": sum(1 for c in kids if c.parent_id and c.parent_id not in
                               {p.chunk_id for p in parents}),
    }
