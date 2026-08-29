"""
The document model. Written from your field list, plus four things it exposed.

Once 300 chunks are in an index, changing this shape means rebuilding. So it
gets designed before the loaders, not discovered by them.

Your three jobs, and the field that does each:

  (a) retrieval matches the CHILD, generation receives the PARENT
        -> Chunk.parent_id, with parents stored as their own records rather than
           copied into every child. Four children sharing one parent should not
           store the parent's text four times: that is 4x the storage and four
           places to fix a parsing bug.

  (b) a citation must be CLICKABLE, and openable at the right place
        -> source_uri + page + bbox. And, because of the wrinkle you found,
           chunk_kind + asset_path, so the UI can branch.

  (c) deleting one source removes EVERYTHING derived from it
        -> source_id stamped on every derived record, plus a Manifest, because a
           stamp you cannot query is not a delete mechanism.

---------------------------------------------------------------------------
FOUR THINGS YOUR ANSWER EXPOSED

1. THREE TEXTS, NOT ONE. You noticed a caption's text exists nowhere in the
   source, so the UI cannot highlight it. That generalises further than images:
   LLM-written table summaries have the same property, and so do CONTEXTUAL
   PREFIXES -- which means after Session 4 almost nothing we embed is verbatim.
   So a chunk carries three distinct strings:

        embed_text    what went to the embedder (prefix + body)
        display_text  what the user is shown
        verbatim_text what is literally in the document, or None if nothing is

   A single `text` field conflates them and makes honest citation impossible.
   `text_provenance` names which case a chunk is, and the UI branches on it.

2. THE SKIP CHECK NEEDS THE PIPELINE VERSION. "Same source hash -> skip" is
   right and incomplete: the parser and chunker are not in that hash. Fix a
   table-extraction bug, re-run ingest, and every existing document is skipped --
   your improvement applies to nothing, silently. Same lesson as the cache key:
   the skip key must name every input that can change the output. Hence
   PipelineVersion and Source.reingest_reason().

3. chunk_id HAS A TENSION. It must be stable across re-runs (idempotency) and
   must NOT be stable across content changes (or re-ingest leaves old IDs
   pointing at new text). `source_id|ordinal` fails the second. `hash(text)`
   collides on identical boilerplate in different documents. It needs both.

4. THE MANIFEST IS ALSO THE DETECTOR. Hashes reveal CHANGED documents. Nothing
   in the source tells you a file that used to be there is gone -- absence is
   not an event. The manifest is what makes deletion detectable, not merely
   executable, and it doubles as the delete-by-list fallback for stores that
   cannot delete by filter.

AND THE CACHE, which you flagged: the embedding cache is keyed on text, so
deleting a document leaves its vectors behind. Worse, the LLM response cache
holds captions and contextual prefixes -- derived DESCRIPTIONS of the content.
That is the guide's M13 right-to-erasure drill: erasure must reach the vectors
and the derived text, not just the source row. `Manifest.derived_cache_keys`
records what to purge; `purge_source()` uses it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .. import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------
# Enums -- small closed sets, so a typo is an error rather than a silent branch
# --------------------------------------------------------------------------


class DocType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    MARKDOWN = "markdown"
    TEXT = "text"


class ChunkKind(str, Enum):
    """What this chunk IS. The UI branches on it for citation."""

    TEXT = "text"                    # prose from the document
    TABLE = "table"                  # a table serialised to markdown
    TABLE_SUMMARY = "table_summary"  # LLM prose ABOUT a table (M4 strategy b)
    IMAGE_CAPTION = "image_caption"  # LLM prose ABOUT an image
    PARENT = "parent"                # a returned-but-never-embedded parent


class TextProvenance(str, Enum):
    """Where the indexed text came from -- the honest-citation field.

    VERBATIM         : embed_text is in the document. Highlightable.
    PREFIXED         : verbatim body + an LLM-written contextual prefix. The body
                       is highlightable; the prefix is not and must not be shown
                       as if it were quoted from the source.
    MODEL_GENERATED  : nobody wrote this. Captions and table summaries. There is
                       nothing to highlight; the UI shows the ASSET instead.
    """

    VERBATIM = "verbatim"
    PREFIXED = "prefixed"
    MODEL_GENERATED = "model_generated"


class ChunkRole(str, Enum):
    CHILD = "child"    # embedded, retrieved
    PARENT = "parent"  # returned to the generator, never embedded


# --------------------------------------------------------------------------
# Pipeline version -- addition (2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineVersion:
    """Every index-time input that can change the output, in one hashable object.

    Bump a component when you change its behaviour. `fingerprint()` goes into the
    Source record, and reingest_reason() compares it -- so improving the parser
    invalidates the skip and the document is actually reprocessed.

    This is config.INDEX_PROVENANCE_FIELDS given a home. Same idea as the
    embedding cache key, applied to the decision of whether to do the work at all.
    """

    parser: str = "unset"
    chunker: str = "unset"
    embed_model: str = "unset"
    embed_dim: int = 0
    embed_scheme: str = "unset"
    contextualizer: str = "none"  # "none" until Session 4

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256_text(blob)[:16]


# --------------------------------------------------------------------------
# Source -- one document
# --------------------------------------------------------------------------


@dataclass
class Source:
    """A document in the corpus, and the identity questions you raised.

    source_id vs content_hash, which you separated correctly:
      source_id    stable name for THIS DOCUMENT OVER TIME. Normalised, so a
                   rename does not create a second copy of the same document.
      content_hash fingerprint of the current bytes. Answers "did it change?"

    Same id + same hash + same pipeline  -> skip (this is what makes re-running
                                            ingest cheap, and idempotent when
                                            free-tier quota kills a run midway)
    Same id + different hash             -> purge and re-add; the old chunks are
                                            not merely stale, they are WRONG
    Same id + same hash + new pipeline   -> reprocess (addition 2)
    """

    source_id: str
    uri: str
    doc_type: DocType
    content_hash: str
    title: str = ""
    n_pages: int = 0
    bytes_size: int = 0
    ingested_at: str = field(default_factory=_now)
    pipeline_fingerprint: str = ""

    # Hash of the PARSER OUTPUT, not the source bytes. Your determinism check
    # exposed why both are needed: a parser that drifts between runs leaves
    # content_hash and pipeline_fingerprint identical while the indexed text
    # differs -- so every provenance field would certify two different corpora
    # as the same one, and EVAL_REFUSE_MIXED_PROVENANCE would see nothing wrong.
    # Any parser with a model inside it can do this. This field makes it visible.
    parsed_hash: str = ""

    @staticmethod
    def normalize_id(path_or_uri: str | Path) -> str:
        """A stable id that survives a rename within the corpus directory.

        Deliberately NOT the raw absolute path: moving data/raw/ to another
        machine would otherwise re-ingest the whole corpus as new documents.
        Relative-to-corpus, forward slashes, lowercased.
        """
        p = Path(str(path_or_uri))
        try:
            rel = p.resolve().relative_to(config.DATA_RAW.resolve())
        except (ValueError, OSError):
            rel = Path(p.name)
        return rel.as_posix().lower()

    @classmethod
    def from_file(cls, path: Path, doc_type: DocType, *, title: str = "") -> "Source":
        data = path.read_bytes()
        return cls(
            source_id=cls.normalize_id(path),
            uri=str(path),
            doc_type=doc_type,
            content_hash=sha256_bytes(data),
            title=title or path.stem,
            bytes_size=len(data),
        )

    def reingest_reason(self, previous: "Source | None", pipeline: PipelineVersion) -> str | None:
        """None means skip. A string means reprocess, and says why.

        Returning the REASON rather than a bool is not decoration: an ingest run
        that reprocesses 500 documents should be able to tell you whether that
        was because they changed or because you bumped the chunker.
        """
        fp = pipeline.fingerprint()
        if previous is None:
            return "new document"
        if previous.content_hash != self.content_hash:
            return "content changed"
        if previous.pipeline_fingerprint != fp:
            return f"pipeline changed ({previous.pipeline_fingerprint} -> {fp})"
        return None


# --------------------------------------------------------------------------
# Block -- what a loader emits, before chunking
# --------------------------------------------------------------------------


@dataclass
class Block:
    """One structural unit from a parser: a paragraph, a heading, a table, an image.

    Chunking consumes Blocks; it does not consume raw text. That is the whole
    point of header-aware chunking (M5) -- if the loader flattens structure to a
    string, the chunker cannot respect boundaries it can no longer see.
    """

    kind: ChunkKind
    text: str                      # verbatim for TEXT/TABLE; "" for an image
    page: int | None = None
    page_end: int | None = None    # set when this block was stitched across pages
    stitched: bool = False         # True if assembled from >1 page fragment
    # DETECTED, NOT REPAIRED. Session 1 flags suspected table continuations and
    # deliberately does not join them. Detection is ~80% of the work and is
    # needed either way; the join is ~20% and is where the dangerous failure
    # lives -- a wrongly joined table is data that exists in NO source document,
    # with clean headers, uniform columns and a real page citation. Structural
    # validators cannot catch it, because its structure is perfect.
    table_continuation_suspect: bool = False
    table_header_missing: bool = False
    bbox: tuple[float, float, float, float] | None = None
    heading_path: tuple[str, ...] = ()   # ("3 Methods", "3.2 Retrieval")
    asset_path: str | None = None        # extracted image, for IMAGE_CAPTION
    ordinal: int = 0

    # WHICH EXTRACTOR produced `text`. Not decoration: when a table's header is
    # lost, the markdown serialisation is actively wrong (a data row promoted to
    # a column name), and the repair is to fall back to the page's own text layer
    # for that region. The delivered text then came from a different extractor
    # than the default path, and citation, eval and debugging all need to know
    # which -- otherwise "the table looks different from the PDF" is unexplainable.
    text_source: str = "markdown"
    ordinal_on_page: int = 0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["heading_path"] = list(self.heading_path)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Block":
        d = dict(d)
        d["kind"] = ChunkKind(d["kind"])
        d["heading_path"] = tuple(d.get("heading_path") or ())
        if d.get("bbox") is not None:
            d["bbox"] = tuple(d["bbox"])
        return cls(**d)


# --------------------------------------------------------------------------
# Chunk -- what is stored, embedded, retrieved, cited, deleted
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    """One indexed record.

    Parents live here too, with role=PARENT and no embedding. One store means one
    delete path -- job (c) gets harder for every extra place a document's remains
    can hide.
    """

    # identity
    chunk_id: str
    source_id: str                 # job (c): the stamp
    ordinal: int
    role: ChunkRole = ChunkRole.CHILD
    parent_id: str | None = None   # job (a)

    # Index of this child within its parent. Exists for one purpose: the
    # POSITIONAL BIAS PROBE. For any per-chunk enrichment, position within the
    # parent must NOT predict retrieval. If it does, something is positional that
    # should not be -- e.g. a breadcrumb prepended to a parent BEFORE slicing,
    # which lands only on child 0 and makes section-openers systematically more
    # findable for reasons unrelated to relevance. Aggregate recall then rises
    # and the bias reads as a win.
    #
    # Slicing recall by this field turns that into a one-line check, and it
    # generalises to contextual prefixes in Session 4.
    position_within_parent: int = 0
    n_siblings: int = 1

    # the three texts -- addition (1)
    embed_text: str = ""           # what the embedder saw
    display_text: str = ""         # what the user is shown
    verbatim_text: str | None = None  # what is literally in the document
    text_provenance: TextProvenance = TextProvenance.VERBATIM

    # job (b): the address
    kind: ChunkKind = ChunkKind.TEXT
    source_uri: str = ""
    page: int | None = None
    # A table stitched across a page break is ONE logical unit on TWO pages, and
    # a single `page` cannot name it. Citing only page 7 sends the user to a page
    # that does not contain the numbers you quoted -- the trust-destroying
    # outcome. None means single-page; set it when a chunk spans.
    page_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    asset_path: str | None = None
    heading_path: tuple[str, ...] = ()

    # provenance -- config.INDEX_PROVENANCE_FIELDS
    has_contextual_prefix: bool = False
    text_source: str = "markdown"   # which extractor produced the delivered text
    # Third instance of the same convention: a stage that can silently degrade
    # gets a per-chunk boolean, a manifest count, and an eval refusal. Never a
    # stdout warning alone.
    table_continuation_suspect: bool = False
    table_header_missing: bool = False
    embed_model: str = ""
    embed_dim: int = 0
    embed_scheme: str = ""
    pipeline_fingerprint: str = ""
    ingested_at: str = field(default_factory=_now)
    content_hash: str = ""         # of THIS chunk's text; dedup + cache keying

    # WHO CAN RETRIEVE THIS. "" means the public corpus -- visible to everyone.
    #
    # THE EMPTY STRING IS THE DANGEROUS VALUE, and it is dangerous in one
    # direction only. Every other field in this class fails toward a MISS: get it
    # wrong and a chunk is unreachable, which is visible and annoying. Get this
    # one wrong and an uploaded document becomes readable by every visitor, which
    # is a LEAK, and leaks are silent.
    #
    # So `owner` is never merely "set at ingest". `assert_owned()` below refuses to
    # build an index containing an upload-sourced chunk with a public owner, which
    # makes the failure structural rather than a matter of remembering. Same
    # reasoning as _make_parent hardcoding VERBATIM: a value you assert is a value
    # that can drift.
    #
    # This is Phase 2's tenant filter, built early against a lower-stakes threat
    # model. `session_id` becomes `tenant_id` and the mechanism is unchanged.
    owner: str = ""
    # Where the chunk came from. An upload can never be public; the public corpus
    # can never be session-scoped. Both directions are checked.
    origin: str = "corpus"          # "corpus" | "upload"

    @staticmethod
    def make_id(source_id: str, ordinal: int, text: str, role: ChunkRole) -> str:
        """Addition (3): stable across re-runs, unstable across content changes.

        Both halves are required.
          source_id + ordinal alone -> re-chunking shifts ordinals, so an old id
                                       now names different text
          hash(text) alone          -> "Confidential - do not distribute" appears
                                       in 40 documents and collides
        """
        seed = f"{role.value}|{source_id}|{ordinal}|{sha256_text(text)}"
        return sha256_text(seed)[:24]

    def citation(self) -> dict[str, Any]:
        """What the UI needs to render one citation, including how to open it.

        `highlightable` is the field that makes addition (1) actionable: a caption
        or table summary has no source text to highlight, so the UI must show the
        asset instead of pretending to quote the document. Showing model-written
        prose as a quotation from the source is a correctness bug, not a styling
        one -- it is a fabricated citation with a real-looking page number.
        """
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "uri": self.source_uri,
            "page": self.page,
            "page_end": self.page_end,
            "page_label": (
                f"p. {self.page}" if self.page_end in (None, self.page)
                else f"pp. {self.page}-{self.page_end}"
            ),
            "bbox": self.bbox,
            "kind": self.kind.value,
            "section": " > ".join(self.heading_path) if self.heading_path else None,
            "asset_path": self.asset_path,
            "highlightable": self.text_provenance is not TextProvenance.MODEL_GENERATED,
            "quote": self.verbatim_text,
            "provenance": self.text_provenance.value,
        }

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("role", "kind", "text_provenance"):
            d[k] = getattr(self, k).value
        d["heading_path"] = list(self.heading_path)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Chunk":
        d = dict(d)
        d["role"] = ChunkRole(d["role"])
        d["kind"] = ChunkKind(d["kind"])
        d["text_provenance"] = TextProvenance(d["text_provenance"])
        d["heading_path"] = tuple(d.get("heading_path") or ())
        if d.get("bbox") is not None:
            d["bbox"] = tuple(d["bbox"])
        return cls(**d)


# --------------------------------------------------------------------------
# Manifest -- addition (4): the detector, the delete list, the erasure record
# --------------------------------------------------------------------------


@dataclass
class SourceRecord:
    """What one document produced. The answer to "what do I delete?"."""

    source: Source
    chunk_ids: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    asset_paths: list[str] = field(default_factory=list)
    cache_keys: list[str] = field(default_factory=list)  # embedding + LLM cache
    n_uncontextualized: int = 0  # the free-tier partial-ingest count

    # Prevalence, recorded rather than printed. Two numbers, not one, because a
    # detector's own count is circular: it measures what the detector FINDS, not
    # what EXISTS. `n_tables_detected` is bounded below by the detector's blind
    # spots -- strategy="lines" cannot see a whitespace-aligned table at all. So
    # `n_tables_undetected_manual` is filled in from a hand-labelled sample and
    # is what makes the ratio interpretable.
    n_pages: int = 0
    n_tables_detected: int = 0
    n_continuation_suspects: int = 0
    n_tables_undetected_manual: int | None = None


class Manifest:
    """Persisted record of the corpus. Three jobs, all of them job (c).

    1. DETECTOR. Delta detection from hashes finds documents that CHANGED.
       Absence is not an event -- nothing in the source directory announces that
       a file used to be there. `deleted_sources()` diffs what is on disk against
       what the manifest remembers. Without this, a document deleted at the
       source stays retrievable forever, which is the guide's M4 drill verbatim.

    2. DELETE LIST. Storing source_id on every chunk only helps if each store can
       be asked "give me everything with this source_id". Some vector stores
       delete by filter; some do not. The explicit id list is the fallback, so
       the stamp is always usable.

    3. ERASURE RECORD. `cache_keys` is the part that gets forgotten -- the
       embedding cache and the LLM response cache are keyed on TEXT, not on
       source_id, so they survive a deletion that looks complete. For GDPR the
       LLM cache is the more sensitive of the two: it holds Gemini-written
       descriptions of the document's images and passages.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config.DATA_INDEX / "manifest.json")
        self.records: dict[str, SourceRecord] = {}
        self.tombstones: dict[str, str] = {}  # source_id -> deleted_at
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for sid, r in raw.get("records", {}).items():
            s = r["source"]
            s["doc_type"] = DocType(s["doc_type"])
            self.records[sid] = SourceRecord(
                source=Source(**s),
                chunk_ids=r.get("chunk_ids", []),
                parent_ids=r.get("parent_ids", []),
                asset_paths=r.get("asset_paths", []),
                cache_keys=r.get("cache_keys", []),
                n_uncontextualized=r.get("n_uncontextualized", 0),
            )
        self.tombstones = raw.get("tombstones", {})

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": {
                sid: {
                    "source": {**asdict(r.source), "doc_type": r.source.doc_type.value},
                    "chunk_ids": r.chunk_ids,
                    "parent_ids": r.parent_ids,
                    "asset_paths": r.asset_paths,
                    "cache_keys": r.cache_keys,
                    "n_uncontextualized": r.n_uncontextualized,
                }
                for sid, r in self.records.items()
            },
            "tombstones": self.tombstones,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    # -- job (c) --------------------------------------------------------------

    def previous(self, source_id: str) -> Source | None:
        rec = self.records.get(source_id)
        return rec.source if rec else None

    def plan(
        self, present: Iterable[Source], pipeline: PipelineVersion
    ) -> dict[str, Any]:
        """What this ingest run should do, decided BEFORE doing any of it.

        Returning a plan rather than acting immediately means the run can be
        printed and inspected -- 'reprocess 500 documents because the chunker
        changed' is something you want to see before it spends your daily quota.
        """
        present_list = list(present)
        present_ids = {s.source_id for s in present_list}
        todo: list[tuple[Source, str]] = []
        skip: list[str] = []
        for s in present_list:
            reason = s.reingest_reason(self.previous(s.source_id), pipeline)
            (todo.append((s, reason)) if reason else skip.append(s.source_id))
        return {
            "reprocess": todo,
            "skip": skip,
            "delete": sorted(set(self.records) - present_ids),  # detected absence
        }

    def record(self, rec: SourceRecord) -> None:
        rec.source.pipeline_fingerprint = rec.source.pipeline_fingerprint or ""
        self.records[rec.source.source_id] = rec
        self.tombstones.pop(rec.source.source_id, None)

    def tombstone_for_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Which removed document did this chunk belong to?

        A citation in an old conversation outlives the document it points at. The
        design's answer is right: the answer stays as it was given, and the
        citation says its source was removed. That is only expressible if the
        tombstone remembers the chunk ids -- otherwise a dead citation is
        indistinguishable from a bad one, and "this was deleted" gets reported as
        "this never existed".
        """
        for sid, rec in self.tombstones.items():
            if not isinstance(rec, dict):
                continue
            known = set(rec.get("chunk_ids", [])) | set(rec.get("parent_ids", []))
            if chunk_id in known:
                return {"source_id": sid, **rec}
        return None

    def purge_source(self, source_id: str) -> dict[str, Any]:
        """Everything derived from one document, for the caller to remove.

        Returns the work rather than doing it, because the stores it touches
        (dense index, sparse index, parent store, extracted assets, two caches)
        do not exist yet. What matters now is that the LIST is complete -- a
        deletion that misses one store leaves a ghost that looks like a real hit
        until the user clicks it.
        """
        rec = self.records.pop(source_id, None)
        if rec is None:
            self.tombstones[source_id] = {
                "deleted_at": _now(), "chunk_ids": [], "parent_ids": []
            }
            return {"source_id": source_id, "found": False}
        # The tombstone records WHAT was deleted, not merely that something was.
        self.tombstones[source_id] = {
            "deleted_at": _now(),
            "title": rec.source.title,
            "doc_type": rec.source.doc_type.value,
            "chunk_ids": rec.chunk_ids,
            "parent_ids": rec.parent_ids,
            "n_chunks": len(rec.chunk_ids),
        }
        return {
            "source_id": source_id,
            "found": True,
            "chunk_ids": rec.chunk_ids,
            "parent_ids": rec.parent_ids,
            "asset_paths": rec.asset_paths,
            "cache_keys": rec.cache_keys,
        }

    # -- reporting -----------------------------------------------------------

    def mixed_provenance(self) -> bool:
        """True if the index holds both contextualised and un-contextualised chunks.

        config.EVAL_REFUSE_MIXED_PROVENANCE and
        config.CI_BASELINE_REQUIRES_UNIFORM_PROVENANCE are enforced against this.
        """
        partial = sum(r.n_uncontextualized for r in self.records.values())
        total = sum(len(r.chunk_ids) for r in self.records.values())
        return 0 < partial < total

    def summary(self) -> str:
        n_src = len(self.records)
        n_chunks = sum(len(r.chunk_ids) for r in self.records.values())
        n_partial = sum(r.n_uncontextualized for r in self.records.values())
        line = f"manifest: {n_src} sources, {n_chunks} chunks, {len(self.tombstones)} tombstones"
        if n_partial:
            line += (
            # WORDING CORRECTED. This said "lack contextual prefixes", which
            # was accurate when only that could cause incompleteness -- but
            # contextual prefixes are deferred, so the message named a cause
            # that cannot currently occur while the real one (an image whose
            # caption failed, leaving it unretrievable) went unnamed. A
            # warning that misattributes its own cause sends the reader to
            # the wrong place.
                f"\n  WARNING: {n_partial} source(s) indexed INCOMPLETE -- part of "
                "their content is unretrievable (e.g. an image whose caption "
                "failed). A/B results over a mixed index are not interpretable."
            )
        return line


PUBLIC_OWNER = ""
"""Owner value meaning "the shared demo corpus, visible to everyone"."""


class OwnershipViolation(RuntimeError):
    """A chunk's owner and origin disagree in a way that would leak or hide it."""


def assert_owned(chunks: "list[Chunk] | tuple[Chunk, ...]") -> None:
    """Refuse a chunk set whose ownership could leak. Call before indexing.

    TWO DIRECTIONS, because only one of them is obvious:

      upload with a public owner   an uploaded document readable by every
                                   visitor. A LEAK, and silent -- nothing in the
                                   product looks wrong.
      corpus with a session owner  the shared corpus invisible to everyone. A
                                   miss, loud, and mostly harmless.

    The first is why this function exists. It is deliberately a hard failure and
    not a warning: a warning about a leak is a leak with a note attached.
    """
    bad_upload = [c for c in chunks
                  if c.origin == "upload" and c.owner == PUBLIC_OWNER]
    if bad_upload:
        raise OwnershipViolation(
            f"{len(bad_upload)} uploaded chunk(s) carry the PUBLIC owner "
            f"(first: {bad_upload[0].chunk_id}). An upload with a public owner is "
            "readable by every visitor. Refusing to index rather than leaking."
        )
    bad_corpus = [c for c in chunks
                  if c.origin == "corpus" and c.owner != PUBLIC_OWNER]
    if bad_corpus:
        raise OwnershipViolation(
            f"{len(bad_corpus)} corpus chunk(s) carry a session owner "
            f"(first: {bad_corpus[0].chunk_id}). The shared corpus would be "
            "invisible to everyone."
        )
