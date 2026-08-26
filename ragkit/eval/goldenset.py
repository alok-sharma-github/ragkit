"""
Golden set construction. Guide module M9.

THE LABEL IS NOT A CHUNK ID

The guide's shortcut -- generate a question from a chunk, and that chunk IS the
answer -- gives free labels and only works while the chunks stay put. There are
27 chunks under `fixed` and 42 under `header_aware`, with different boundaries,
so chunk #14 means two unrelated things. A golden set labelled against one
strategy's ids scores that strategy and nothing else, which makes it useless for
the only question being asked: which strategy is better. Worse, it is biased
toward whichever strategy produced it.

NOR IS IT A CHARACTER OFFSET. Offsets are anchored to parser output, and this
project's parser output has already moved four times (`+quadcorr-unset`,
`+cleanhdr@2`, `+pagetextrepair@3`, `+bboxflags@1`). A golden set keyed on
offsets silently rots on a PARSER_VERSION bump, which here happens hourly.

THE ANCHOR IS THE ANSWER TEXT ITSELF -- a short verbatim needle plus enough
locating information to disambiguate it:

    source_id : benchmark-report.pdf
    page      : 2
    needle    : "HYBRID_RRF 0.831"

At eval time the needle is FOUND in the current parsed text rather than trusted
at a stored position. A page number breaks when the edition changes; a remembered
sentence finds itself.

Free bonus: if a needle cannot be located at all, either the parser changed or
the corpus did. The golden set doubles as a parser-drift detector, the same role
`parsed_hash` plays. The markdown-`**` bug would have surfaced as needles
suddenly failing to locate.

---------------------------------------------------------------------------
FOUR CONSTRUCTION RULES, each fixing a way the set could lie

1. UNIQUENESS IS VERIFIED, NOT ASSUMED. "0.831" occurs in several papers. An
   ambiguous needle inflates recall, because a chunk from an unrelated document
   containing the same string scores as a hit. So every needle is searched across
   the WHOLE parsed corpus and either extended until unique or recorded with all
   its acceptable locations.

2. GENERATED FROM A STRATEGY-NEUTRAL UNIT. A question written by looking at a
   183-token header_aware chunk is answerable within 183 tokens, phrased in that
   chunk's vocabulary, and scoped to one section -- so feeding it to a system
   chunked the same way is building the exam from the textbook it studied. Here
   questions come from PAGE-level parsed text, which no candidate chunking
   strategy uses.

3. IMAGE QUESTIONS NEED A DIFFERENT ANCHOR, decided before generation rather
   than discovered after. Image chunks are Gemini captions: there is no document
   text to anchor a needle to. They are anchored by `asset_path` instead. This
   is not an edge case to omit -- an image-heavy question was measured at 7 of 9
   citations `unquotable`, so a golden set with no image questions measures a
   system that behaves quite differently from the real one.

4. THE FIXTURE IS QUARANTINED. `benchmark-report.pdf` is a file I authored
   containing the failure I was designing against, with a needle I chose. It is
   the least representative question in the corpus. It belongs in CI as "this
   known failure must not return" and NOT in any average, for the same reason
   the single continuation suspect -- also authored by me -- was not counted as
   prevalence.
"""

from __future__ import annotations

import json
from collections import Counter
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config, gemini, limits
from ..ingest import loaders as L
from ..ingest.document import Block, ChunkKind, DocType, Source
from ..ingest.loaders import _squash

_ASSET_ORIGIN = re.compile(r"([A-Za-z0-9-]+)_p(\d+)\.(?:png|jpg|jpeg|webp)$", re.I)


def asset_origin(source_id: str, known: Iterable[str]) -> tuple[str | None, int | None]:
    """Map a rendered page image back to the document it came from.

    Filename-derived and therefore fragile, so it is CHECKED against the corpus:
    a guessed origin that is not actually an indexed source returns None rather
    than a plausible-looking wrong link.
    """
    m = _ASSET_ORIGIN.search(source_id)
    if not m:
        return None, None
    slug, page = m.group(1).lower(), int(m.group(2))
    for cand in known:
        stem = cand.rsplit(".", 1)[0].lower()
        if stem == slug:
            return cand, page
    return None, None


# Files whose questions are regression tests, never eval items. See rule 4.
QUARANTINE = {"benchmark-report.pdf"}

# A needle must be long enough to identify a passage. "0.831" is 4 characters and
# appears all over a corpus of retrieval papers; matching it proves nothing.
NEEDLE_MIN_CHARS = 14
NEEDLE_MAX_CHARS = 160


@dataclass
class GoldenItem:
    question: str
    answer: str
    stratum: str
    source_id: str
    # Needles are a LIST because multi-hop and aggregative questions cannot be
    # answered from one span by definition. Recall over them is reported strict
    # (all needles covered) and partial (fraction) separately -- the generator
    # needs all of them, so strict is the number that predicts answer quality.
    needles: list[str] = field(default_factory=list)
    page: int | None = None
    asset_path: str | None = None       # image anchor, rule 3
    anchor: str = "needle"              # "needle" | "asset" | "none"
    # The plausible-but-wrong answer, where one can be constructed. Turns "wrong"
    # into "wrong in this specific diagnosable way", so a bad answer names its
    # own cause instead of only lowering a score.
    distractor: str | None = None
    # Which source_ids count as a correct retrieval. Normally just this item's
    # own source, but an asset-anchored item accepts its ORIGIN document too.
    #
    # Reason: the images in this corpus are rendered PAGES of PDFs that are also
    # indexed, so an image question is usually answerable from document text as
    # well. Scoring only the image chunk would mark the better hit wrong -- text
    # is quotable, a caption is not. The filename carries the link
    # (assets/hnsw_p2.png -> hnsw.pdf p2), so accepting both is free.
    accept_sources: list[str] = field(default_factory=list)
    origin_source_id: str | None = None
    origin_page: int | None = None
    needle_occurrences: dict[str, int] = field(default_factory=dict)
    unique: bool = True
    quarantined: bool = False
    generator_model: str = ""
    verified_by_human: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "GoldenItem":
        return cls(**d)


# --------------------------------------------------------------------------
# The corpus index that needles are located against
# --------------------------------------------------------------------------


@dataclass
class Window:
    """A page of parsed text. The strategy-neutral generation unit (rule 2)."""

    source_id: str
    doc_type: str
    page: int | None
    text: str
    asset_path: str | None = None
    has_table: bool = False
    has_identifier: bool = False


def build_windows(*, files: Sequence[Path] | None = None) -> list[Window]:
    """Parsed text grouped by page, plus per-window hints for stratification.

    Goes through ragkit.ingest.loaders, not around it. A generation script that
    reached for pymupdf directly would produce windows from a DIFFERENT PARSER
    than the pipeline indexes -- measured earlier at a 5x difference in table
    counts on identical bytes.
    """
    windows: list[Window] = []
    for path in (files if files is not None else L.corpus_files()):
        kind = L._LOADERS.get(path.suffix.lower())
        if kind is None:
            continue
        src, blocks, _diag = L.load(path, caption_images=True)
        by_page: dict[int | None, list[Block]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)
        for page, blks in by_page.items():
            text = "\n\n".join(b.text for b in blks if b.text.strip()).strip()
            if len(text) < 200:
                continue
            windows.append(
                Window(
                    source_id=src.source_id,
                    doc_type=src.doc_type.value,
                    page=page,
                    text=text,
                    asset_path=next((b.asset_path for b in blks if b.asset_path), None),
                    has_table=any(b.kind is ChunkKind.TABLE for b in blks),
                    has_identifier=any(
                        tok in text for tok in ("ERR_", "PN-", "efSearch", "@10", "k1", "_RRF")
                    ),
                )
            )
    return windows


class NeedleLocator:
    """Finds needles in the CURRENT parsed corpus and counts their occurrences.

    Squash-insensitive on purpose, reusing the same normaliser as the citation
    checker: a needle stored as "HYBRID_RRF 0.831" must still locate when the
    parser renders it "HYBRID_RRF\\n0.831". Two normalisers for the same job is
    how they end up disagreeing -- which already happened once, between
    find_tables() cell text and page text.
    """

    def __init__(self, windows: Sequence[Window]) -> None:
        self.windows = list(windows)
        self._squashed = [(w, _squash(w.text)) for w in self.windows]

    def occurrences(self, needle: str) -> list[Window]:
        n = _squash(needle)
        if not n:
            return []
        return [w for w, sq in self._squashed if n in sq]

    def count(self, needle: str) -> int:
        return len(self.occurrences(needle))

    def locatable(self, needle: str, source_id: str) -> bool:
        return any(w.source_id == source_id for w in self.occurrences(needle))


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "stratum": {"type": "string"},
                    "needles": {"type": "array", "items": {"type": "string"}},
                    "distractor": {"type": "string"},
                },
                "required": ["question", "answer", "stratum", "needles"],
            },
        }
    },
    "required": ["items"],
}

_GEN_SYSTEM = """You write evaluation questions for a document retrieval system.

For each question you MUST supply:
- question: answerable from this page alone, phrased as a user would ask it.
  Do not refer to "this page", "the text above", or "the passage" -- the system
  will not know which page you meant.
- answer: short and factual.
- needles: one or more VERBATIM strings copied character-for-character from the
  page. A needle is the evidence a retrieval system must surface. Each needle
  must be 14-160 characters, and specific enough to identify this page rather
  than any page in a corpus of retrieval papers -- so include a distinguishing
  name, label or identifier, never a bare number like "0.831".
- stratum: one of simple_factual, multi_hop, aggregative, exact_identifier,
  table_or_image, ambiguous.
- distractor (optional): a plausible but WRONG answer that a careless system
  would give, e.g. a neighbouring value from the same table.

Prefer questions whose answers live in hard places: inside tables, in figure
captions, attached to exact identifiers. Do not invent facts."""


def generate_items(
    window: Window, *, n: int = 2, model_role: str = "workhorse"
) -> list[GoldenItem]:
    prompt = (
        f"Document: {window.source_id}"
        + (f", page {window.page}" if window.page else "")
        + f"\n\n{window.text[:6000]}\n\nWrite {n} evaluation questions."
    )
    try:
        raw, _usage = gemini.generate(
            prompt, role=model_role, system=_GEN_SYSTEM, schema=_ITEM_SCHEMA,
            max_output_tokens=2048, stage="goldenset_generate",
        )
    except (limits.QuotaExhausted, gemini.EmptyResponse):
        limits.report(
            limits.Degradation(
                stage="goldenset_generate", cause="free_tier_quota_exhausted",
                impact=f"no questions generated from {window.source_id} p{window.page}",
                fallback="window skipped; the golden set under-covers this document",
                remedy="re-run generation when quota resets; existing items are kept",
            )
        )
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    model = gemini.resolve_models()[model_role]
    out: list[GoldenItem] = []
    for it in payload.get("items") or []:
        needles = [n for n in (it.get("needles") or []) if isinstance(n, str)]
        out.append(
            GoldenItem(
                question=(it.get("question") or "").strip(),
                answer=(it.get("answer") or "").strip(),
                stratum=(it.get("stratum") or "simple_factual").strip(),
                source_id=window.source_id,
                needles=needles,
                page=window.page,
                asset_path=window.asset_path,
                anchor="asset" if window.doc_type == DocType.IMAGE.value else "needle",
                distractor=(it.get("distractor") or None),
                generator_model=model,
                quarantined=window.source_id in QUARANTINE,
            )
        )
    return out


# --------------------------------------------------------------------------
# Verification -- rule 1
# --------------------------------------------------------------------------


@dataclass
class VerifyReport:
    kept: int = 0
    dropped_unlocatable: int = 0
    dropped_short: int = 0
    dropped_no_needle: int = 0
    ambiguous_kept: int = 0
    quarantined: int = 0
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        return (
            f"kept {self.kept} | quarantined {self.quarantined} | "
            f"ambiguous-but-kept {self.ambiguous_kept} | dropped: "
            f"{self.dropped_unlocatable} unlocatable, {self.dropped_short} short-needle, "
            f"{self.dropped_no_needle} no-needle"
        )


def verify_items(
    items: Sequence[GoldenItem], locator: NeedleLocator
) -> tuple[list[GoldenItem], VerifyReport]:
    """Drop items whose anchors do not hold. This is where rule 1 is enforced.

    An item that survives here is one whose evidence provably exists in the
    corpus as currently parsed. An item that fails is not a bad question -- it is
    an unusable LABEL, and keeping it would silently mark correct retrievals wrong.
    """
    rep = VerifyReport()
    kept: list[GoldenItem] = []
    for it in items:
        if it.anchor == "asset":
            # TWO CORRECTIONS, both found by generating a handful and reading them.
            #
            # (a) A NEEDLE FROM A CAPTION IS NOT AN ANCHOR. The generator happily
            #     produced needles for image items, copied from Gemini's caption.
            #     Caption text is regenerated on every re-ingest and is not
            #     deterministic, so a caption needle rots exactly the way a
            #     character offset rots when the parser changes -- the same bug in
            #     a new costume. Caption-derived needles are therefore dropped.
            #
            # (b) OUR IMAGES ARE RENDERED PAGES OF PDFS WE ALREADY INDEX, so an
            #     image's content usually ALSO exists as document text. Anchoring
            #     such an item on asset_path alone would mark a correct retrieval
            #     wrong for surfacing the text chunk -- which is the better hit,
            #     since text is quotable and a caption is not. So: if a needle
            #     locates in real document text SOMEWHERE ELSE in the corpus, the
            #     item becomes needle-anchored and accepts either. Only genuinely
            #     image-only content (a chart's shape, a diagram's layout) stays
            #     asset-anchored.
            elsewhere = [
                n for n in it.needles
                if NEEDLE_MIN_CHARS <= len(n) <= NEEDLE_MAX_CHARS
                and any(w.doc_type != DocType.IMAGE.value for w in locator.occurrences(n))
            ]
            if elsewhere:
                it.anchor = "needle"
                it.needles = elsewhere
                it.needle_occurrences = {n: locator.count(n) for n in elsewhere}
                it.unique = all(v == 1 for v in it.needle_occurrences.values())
                if not it.unique:
                    rep.ambiguous_kept += 1
            else:
                it.needles = []           # see (a)
                it.unique = True
            known = {w.source_id for w in locator.windows}
            origin, opage = asset_origin(it.source_id, known)
            it.origin_source_id, it.origin_page = origin, opage
            it.accept_sources = [it.source_id] + ([origin] if origin else [])
            kept.append(it)
            rep.kept += 1
            rep.quarantined += int(it.quarantined)
            continue

        good: list[str] = []
        for n in it.needles:
            if not (NEEDLE_MIN_CHARS <= len(n) <= NEEDLE_MAX_CHARS):
                rep.dropped_short += 1
                continue
            if not locator.locatable(n, it.source_id):
                rep.dropped_unlocatable += 1
                rep.reasons.append(f"{it.source_id}: needle not found: {n[:48]!r}")
                continue
            good.append(n)
        if not good:
            rep.dropped_no_needle += 1
            continue

        it.needles = good
        it.accept_sources = [it.source_id]
        it.needle_occurrences = {n: locator.count(n) for n in good}
        it.unique = all(v == 1 for v in it.needle_occurrences.values())
        if not it.unique:
            # KEPT, not dropped, and recorded. A needle occurring in two windows
            # of the SAME document is still a usable label; one occurring across
            # documents inflates recall. The count is stored so the eval can
            # decide, rather than this function deciding silently.
            rep.ambiguous_kept += 1
        kept.append(it)
        rep.kept += 1
        rep.quarantined += int(it.quarantined)
    return kept, rep


# --------------------------------------------------------------------------
# Out-of-scope items -- hand written, and they have to be
# --------------------------------------------------------------------------
#
# The stratum everyone forgets, and the one that cannot be generated from a
# window: a question is out of scope precisely because no window contains its
# answer. Generating it from the corpus is a contradiction in terms.
#
# So these are authored, and then VERIFIED ABSENT against the locator -- because
# a hand-written "out of scope" question that the corpus happens to answer is a
# label that marks a correct answer wrong. Topic-adjacent on purpose: "what is
# the capital of France" tests nothing, while "what does RRF cost per query in
# dollars" is exactly the shape of question a user asks and this corpus cannot
# answer.
OUT_OF_SCOPE = [
    ("What is the dollar cost per query of running reciprocal rank fusion in production?",
     "kokolo"),
    ("Which cloud provider does the Meridian Retrieval Appliance ship on?", "kokolo"),
    ("What is the p99 latency of gemini-embedding-001 under load?", "kokolo"),
    ("How many engineers were on the GraphRAG team?", "kokolo"),
    ("What is the licence fee for the BGE reranker?", "kokolo"),
]


def out_of_scope_items(locator: NeedleLocator) -> tuple[list[GoldenItem], list[str]]:
    """Authored questions, verified to be unanswerable from this corpus.

    The verification is crude and deliberate: every content word of the question
    is looked up, and if a single window contains enough of them the item is
    rejected as possibly-answerable. Over-rejecting here is safe; a false
    out-of-scope item silently penalises correct behaviour.
    """
    kept, rejected = [], []
    for q, _ in OUT_OF_SCOPE:
        words = [w for w in _squash(q).split() if w]
        # _squash strips spaces, so probe with the longest tokens of the raw text
        tokens = sorted({t.strip("?,.").lower() for t in q.split() if len(t) > 5},
                        key=len, reverse=True)[:4]
        hits = [w for w in locator.windows
                if sum(1 for t in tokens if t in w.text.lower()) >= max(2, len(tokens) - 1)]
        if hits:
            rejected.append(f"{q!r} may be answerable from {hits[0].source_id}")
            continue
        kept.append(
            GoldenItem(
                question=q,
                answer="ABSTAIN -- not answerable from this corpus",
                stratum="out_of_scope",
                source_id="",
                needles=[],
                anchor="none",
            )
        )
    return kept, rejected


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def select_windows(windows: Sequence[Window], *, target: int) -> list[Window]:
    """Pick windows for coverage, deterministically.

    Priority order matters: the hard strata are scarce, so they are taken first
    and the easy prose windows fill whatever is left. Without this an
    LLM-generated set drifts toward "simple factual from body prose", which is
    the population the system is already best at -- an eval that flatters itself.

    Deterministic (sorted + strided, no randomness) so the same corpus produces
    the same selection, and a re-run is comparable to the previous one.
    """
    ordered = sorted(windows, key=lambda w: (w.source_id, w.page or 0))
    images = [w for w in ordered if w.doc_type == DocType.IMAGE.value]
    tables = [w for w in ordered if w.has_table and w not in images]
    idents = [w for w in ordered if w.has_identifier and w not in images and w not in tables]
    rest = [w for w in ordered if w not in images and w not in tables and w not in idents]

    picked: list[Window] = []
    for bucket, cap in ((images, len(images)), (tables, target // 2), (idents, target // 4)):
        picked += bucket[:cap]
    # Stride the remainder so coverage spreads across documents rather than
    # taking the first N pages of the alphabetically first paper.
    need = max(target - len(picked), 0)
    if need and rest:
        step = max(len(rest) // need, 1)
        picked += rest[::step][:need]
    return picked[:target]


def build(
    *,
    target_windows: int = 60,
    per_window: int = 2,
    files: Sequence[Path] | None = None,
    verbose: bool = True,
    save_path: Path | None = None,
) -> tuple[list[GoldenItem], dict[str, Any]]:
    """Generate, verify and SAVE. Saving happens inside, before returning.

    Because it did not, once: 60 windows and 120 Gemini calls completed, then the
    caller's own print statement raised UnicodeEncodeError on a Turkish dotless i
    and every generated item was lost. Expensive work must be durable before
    anything cheap and fallible runs -- a print is cheap and fallible.
    """
    windows = build_windows(files=files)
    locator = NeedleLocator(windows)
    chosen = select_windows(windows, target=target_windows)
    if verbose:
        print(f"windows: {len(windows)} total, {len(chosen)} selected for generation")

    raw_items: list[GoldenItem] = []
    with limits.collect() as log:
        for i, w in enumerate(chosen, 1):
            got = generate_items(w, n=per_window)
            raw_items += got
            if verbose:
                print(f"  [{i}/{len(chosen)}] {w.source_id} p{w.page}: {len(got)} items", flush=True)

        kept, rep = verify_items(raw_items, locator)
        oos, oos_rejected = out_of_scope_items(locator)
        kept += oos

        report = {
            "n_windows": len(windows),
            "n_windows_selected": len(chosen),
            "generated": len(raw_items),
            "verify": asdict(rep),
            "out_of_scope_kept": len(oos),
            "out_of_scope_rejected": oos_rejected,
            "summary": summarise(kept),
            "degradations": log.to_dicts(),
        }
    # Durable before reportable.
    report["saved_to"] = str(save(kept, save_path))
    return kept, report


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save(items: Sequence[GoldenItem], path: Path | None = None) -> Path:
    path = path or (config.DATA_EVAL / "golden.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")
    return path


def load(path: Path | None = None) -> list[GoldenItem]:
    path = path or (config.DATA_EVAL / "golden.jsonl")
    if not path.exists():
        return []
    return [GoldenItem.from_json(json.loads(ln)) for ln in path.read_text("utf-8").splitlines() if ln.strip()]


def summarise(items: Sequence[GoldenItem]) -> dict[str, Any]:
    evaluable = [i for i in items if not i.quarantined]
    return {
        "total": len(items),
        "evaluable": len(evaluable),
        "quarantined": len(items) - len(evaluable),
        "by_stratum": dict(Counter(i.stratum for i in evaluable)),
        "by_anchor": dict(Counter(i.anchor for i in evaluable)),
        "by_source": dict(Counter(i.source_id for i in evaluable)),
        "multi_needle": sum(1 for i in evaluable if len(i.needles) > 1),
        "ambiguous_needles": sum(1 for i in evaluable if not i.unique),
        "with_distractor": sum(1 for i in evaluable if i.distractor),
        "human_verified": sum(1 for i in evaluable if i.verified_by_human),
    }
