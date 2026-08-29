"""
The pipeline, as one callable thing. Everything before this ran as ad-hoc
scripts I typed by hand, which is a problem in its own right:

  a survey script that reaches around the pipeline measures a DIFFERENT PROGRAM.

That is not a hypothetical -- it cost an hour when a standalone table survey
imported only pymupdf while the loader imported pymupdf4llm, and the two
reported table counts differing by 5x on identical bytes. Neither number was
wrong; they were answers about different parsers.

So `ingest()` and `ask()` live here and everything else calls them. The eval
harness, the CLI and the UI will all go through this module rather than
re-assembling the steps, which makes "the thing I measured" and "the thing that
runs" the same object by construction.

---------------------------------------------------------------------------
WHY INGEST REBUILDS THE WHOLE INDEX

At 791 chunks a full rebuild costs seconds, because the two expensive stages are
both cached on their inputs (parse cache keyed on content_hash + parser_version,
embedding cache keyed on model|dim|scheme|text). So an incremental index would
buy nothing measurable while introducing the class of bug this project keeps
finding -- two representations that must agree and silently do not.

Delta detection still runs, and still matters: it decides what to REPARSE and
what to re-embed, and it detects deletions, which no amount of rebuilding can
infer from the corpus directory alone.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import config, gemini, limits
from .chunking import contextualize
from .chunking import splitters as S
from .generate import answer as A
from .index.numpy_index import Hit, NumpyIndex
from .retrieve.query import CondensedQuery, condense
from .ingest import loaders as L
from .ingest.document import (
    PUBLIC_OWNER,
    Chunk,
    ChunkRole,
    DocType,
    Manifest,
    PipelineVersion,
    Source,
    SourceRecord,
)


def pipeline_version(strategy: str, *, contextual: bool = False) -> PipelineVersion:
    """Every index-time input that can change the output, in one fingerprint."""
    model = gemini.resolve_models()["embedding"]
    return PipelineVersion(
        parser=L.PARSER_VERSION,
        chunker=S.CHUNKER_VERSIONS[strategy],
        embed_model=model,
        embed_dim=config.EMBED_DIM,
        embed_scheme=gemini.task_scheme(model, "document").label,
        # The field was always here waiting for this; "Session 4 replaces this"
        # is now replaced. Two indexes that differ only in whether an LLM wrote
        # a sentence at the top of each child have different fingerprints, so
        # the eval cannot silently compare them.
        contextualizer=(S.CONTEXTUALIZER_LLM if contextual
                        else S.CONTEXTUALIZER_BREADCRUMB),
    )


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


@dataclass
class IngestResult:
    n_sources: int = 0
    n_children: int = 0
    n_parents: int = 0
    seconds: float = 0.0
    per_file: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    index_report: dict[str, Any] = field(default_factory=dict)
    provenance_ok: bool = True
    provenance_problems: list[str] = field(default_factory=list)
    degradations: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"ingested {self.n_sources} sources -> {self.n_children} children, "
            f"{self.n_parents} parents in {self.seconds:.1f}s"
        ]
        by_type: Counter[str] = Counter()
        for r in self.per_file:
            by_type[r["doc_type"]] += r["children"]
        lines.append("  children by type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
        if self.plan.get("skip"):
            lines.append(f"  reparse skipped for {len(self.plan['skip'])} unchanged source(s)")
        if self.plan.get("delete"):
            lines.append(f"  DELETED at source, purged from index: {self.plan['delete']}")
        if not self.provenance_ok:
            lines.append("  PROVENANCE CHECK FAILED:")
            lines += [f"    {p}" for p in self.provenance_problems]
        return "\n".join(lines)


def _contextual_estimate(strategy: str, files, caption_images: bool) -> IngestResult:
    """What contextualising this corpus would cost, without calling anything."""
    paths = list(files) if files is not None else L.corpus_files()
    docs: dict[str, tuple[str, int]] = {}
    for p in paths:
        if L._LOADERS.get(p.suffix.lower()) is None:
            continue
        src, blocks, _diag = L.load(p, caption_images=caption_images)
        chunks = S.build_chunks(src, blocks, strategy=strategy,
                                owner=PUBLIC_OWNER, origin="corpus")
        n_kids = sum(1 for c in chunks if c.role is ChunkRole.CHILD)
        docs[src.source_id] = ('\n\n'.join(b.text for b in blocks if b.text.strip()),
                               n_kids)
    r = IngestResult()
    r.index_report = {"contextualization_estimate": contextualize.estimate(docs)}
    return r


def ingest(
    *,
    strategy: str = "header_aware_parent",
    caption_images: bool = True,
    index_name: str = "numpy_index",
    files: Sequence[Path] | None = None,
    verbose: bool = True,
    on_progress: Callable[[str, int, int, str], None] | None = None,
    # WHO WILL BE ABLE TO RETRIEVE WHAT THIS PRODUCES.
    #
    # Required, no default, all the way down to the Chunk constructor. A default
    # of PUBLIC_OWNER would make "forgot to pass it" and "meant it to be public"
    # the same keystroke, and only one of those is a leak. The CLI states
    # PUBLIC_OWNER explicitly; the upload path states the session id. Both are
    # deliberate acts, visible at the call site.
    owner: str,
    origin: str = "corpus",
    # LLM-WRITTEN CONTEXTUAL PREFIXES (D-6, expired). Off unless asked for, and
    # asked for explicitly at the call site rather than read from the
    # environment here: this is the only paid-per-chunk stage in ingest, and a
    # flag that turns spending on should be visible in the line that spends.
    contextual: bool | None = None,
    # Parse, chunk, and report what contextualising WOULD cost -- then stop
    # before the first paid call. Useless without `contextual`, and ignored
    # rather than an error, because "estimate an ingest that spends nothing" has
    # an obvious answer.
    estimate_only: bool = False,
) -> IngestResult:
    """Corpus -> index, with the manifest recording what each source produced.

    `caption_images` defaults TRUE here and FALSE in the loader, deliberately.
    The loader default protects incidental corpus walks from silently spending a
    free-tier quota. But an image with no caption is UNRETRIEVABLE -- nothing in
    an index matches pixels -- so an ingest that skips captioning produces a
    system that claims to support images and does not. The default belongs to
    the caller's intent, and ingest's intent is "make this searchable".
    """
    t0 = time.time()
    config.ensure_dirs()
    contextual = config.CONTEXTUAL_PREFIXES if contextual is None else contextual
    ctx_stats = contextualize.ContextStats()
    if contextual and estimate_only:
        # THE NUMBER BEFORE THE BILL. Parsing is cached and free; chunking is
        # free; so a full dry run up to the point of spending is cheap enough
        # to be the default way of answering "what will this cost".
        return _contextual_estimate(strategy, files, caption_images)
    pipe = pipeline_version(strategy, contextual=contextual)
    # THE MANIFEST BELONGS TO THE INDEX IT DESCRIBES.
    #
    # It was shared, and building a second index for an A/B silently rewrote the
    # first one's provenance: after `ingest --index numpy_index_ctx` the manifest
    # said fingerprint 4d54ab24 while data/index/numpy_index still held chunks
    # stamped 6fd55e19. Nothing failed. The demo kept serving, the documents
    # sidebar kept rendering, and the record of which pipeline produced the
    # shipped index was simply wrong -- the exact defect A-12 is about, caused by
    # the tool built to investigate a different one.
    #
    # A side-by-side comparison must not be able to damage the thing it is
    # compared against.
    manifest = Manifest(None if index_name == "numpy_index"
                        else config.DATA_INDEX / f"manifest.{index_name}.json")
    result = IngestResult()

    paths = list(files) if files is not None else L.corpus_files()

    # Delta detection: what to reprocess, what to skip, what has been DELETED.
    # Absence is not an event -- nothing in the corpus directory announces that a
    # file used to be there, so this diff is the only way to notice.
    present: list[Source] = []
    for p in paths:
        kind = L._LOADERS.get(p.suffix.lower())
        if kind is None:
            continue
        present.append(Source.from_file(p, kind))
    plan = manifest.plan(present, pipe)
    result.plan = {
        "reprocess": [(s.source_id, why) for s, why in plan["reprocess"]],
        "skip": plan["skip"],
        "delete": plan["delete"],
    }
    for sid in plan["delete"]:
        purged = manifest.purge_source(sid)
        if verbose:
            print(f"  purge {sid}: {len(purged.get('chunk_ids', []))} chunks, "
                  f"{len(purged.get('cache_keys', []))} cache keys")

    all_chunks: list[Chunk] = []
    with limits.collect() as log:
        for p in paths:
            kind = L._LOADERS.get(p.suffix.lower())
            if kind is None:
                continue
            if on_progress:
                # Per DOCUMENT, not per page. pymupdf4llm parses in one opaque
                # call and emits nothing until it returns, so a page counter
                # would be invented. Stage + document name is what we can
                # honestly report, and a fake smooth progress bar over an opaque
                # operation is the same class of lie as filling in a truncated
                # table with plausible numbers.
                on_progress("parsing", len(result.per_file), len(paths), p.name)
            src, blocks, diag = L.load(p, caption_images=caption_images)

            # ONE SYNOPSIS PER DOCUMENT, then a closure over it. The closure is
            # what build_chunks calls, so the chunker never learns that Gemini
            # exists -- the paid boundary stays in this file and in gemini.py,
            # where the guard audit already looks for it.
            situate = None
            if contextual:
                if on_progress:
                    on_progress("contextualising", len(result.per_file), len(paths), p.name)
                doc_text = '\n\n'.join(b.text for b in blocks if b.text.strip())
                syn = contextualize.synopsis(src.title or src.source_id, doc_text,
                                             stats=ctx_stats)

                def situate(parent_text: str, body: str, _syn: str = syn) -> str:
                    return contextualize.situate(doc_synopsis=_syn,
                                                 parent_text=parent_text,
                                                 body=body, stats=ctx_stats)

            chunks = S.build_chunks(src, blocks, strategy=strategy, pipeline=pipe,
                                    owner=owner, origin=origin,
                                    contextualizer=situate)
            all_chunks.extend(chunks)

            kids = [c for c in chunks if c.role is ChunkRole.CHILD]
            parents = [c for c in chunks if c.role is ChunkRole.PARENT]
            src.pipeline_fingerprint = pipe.fingerprint()
            manifest.record(
                SourceRecord(
                    source=src,
                    chunk_ids=[c.chunk_id for c in kids],
                    parent_ids=[c.chunk_id for c in parents],
                    asset_paths=[b.asset_path for b in blocks if b.asset_path],
                    # Closes the erasure hole: the embedding cache is keyed on
                    # text, so without recording these, "delete this document"
                    # cannot reach its vectors.
                    # Embedding keys AND caption keys. The caption cache holds
                    # model-written descriptions of the document's images, which
                    # for right-to-erasure is the more sensitive of the two --
                    # and it is keyed on image bytes, so a source-id delete
                    # cannot see it.
                    cache_keys=(
                        [gemini.cache_key(c.embed_text) for c in kids]
                        + [
                            gemini.caption_cache_key(
                                Path(b).read_bytes(), Path(b).stem.replace("_", " ")
                            )
                            for b in {b.asset_path for b in blocks if b.asset_path}
                            if Path(b).exists()
                        ]
                    ),
                    n_pages=diag.get("n_pages", 0),
                    n_tables_detected=diag.get("n_tables_detected", 0),
                    n_continuation_suspects=diag.get("n_continuation_suspects", 0),
                    # POPULATED AT LAST. This field had FOUR readers -- the
                    # sidebar's SEARCHABLE_INCOMPLETE state,
                    # Manifest.mixed_provenance(), and the manifest summary
                    # warning -- and no writer, so every one of them was
                    # permanently reporting "complete". A UI state whose data
                    # source is never written is worse than a missing state: it
                    # looks live and always says the reassuring thing.
                    #
                    # The ORIGINAL cause can now occur: contextual prefixes
                    # are no longer deferred, and a refused or empty prefix
                    # leaves a child indexed with breadcrumb context only. That
                    # is a weaker chunk, not a broken one -- but a document
                    # where SOME children got the paid treatment and others did
                    # not is a mixed-provenance index, which is exactly what
                    # this field exists to surface. A failed image caption
                    # counts too, and is worse: an uncaptioned image cannot
                    # match any query at all.
                    n_uncontextualized=(
                        (1 if "FAILED" in str(diag.get("detector", "")) else 0)
                        + sum(1 for c in chunks
                              if c.role is ChunkRole.CHILD
                              and contextual and not c.has_contextual_prefix)
                    ),
                )
            )
            row = {
                "file": p.name,
                "doc_type": src.doc_type.value,
                "children": len(kids),
                "parents": len(parents),
                "tables_repaired": diag.get("n_tables_repaired_page_text", 0),
                "tables_headerless": diag.get("n_tables_headerless", 0),
                "children_page_text": sum(1 for c in kids if "page_text_clip" in c.text_source),
                "from_cache": diag.get("from_cache", False),
            }
            result.per_file.append(row)
            if verbose:
                print(f"  {p.name:34s} {src.doc_type.value:8s} kids={len(kids):4d} "
                      f"par={len(parents):4d} repaired={row['tables_repaired']}")

        kids = [c for c in all_chunks if c.role is ChunkRole.CHILD]
        result.n_sources = len(result.per_file)
        result.n_children = len(kids)
        result.n_parents = sum(1 for c in all_chunks if c.role is ChunkRole.PARENT)

        ok, problems = S.check_provenance_propagation(result.per_file)
        result.provenance_ok, result.provenance_problems = ok, problems

        if on_progress:
            on_progress("embedding", len(paths), len(paths), f"{len(kids)} chunks")
        vecs, st = gemini.embed_texts([c.embed_text for c in kids], kind="document")
        # ONE NOTICE, WITH BOTH NUMBERS. `contextualize_skipped` had a writer
        # nowhere -- it was excused by the deferral audit as "waiting, not
        # forgotten", which was true right up until the deferral was acted on.
        # A degradation helper that nothing calls reports "no degradation".
        if contextual and ctx_stats.prefixes_skipped:
            attempted = (ctx_stats.prefixes_made + ctx_stats.prefixes_cached
                         + ctx_stats.prefixes_skipped)
            log.seen("contextualize", n=attempted)
            log.report(limits.contextualize_skipped(
                ctx_stats.prefixes_skipped, attempted),
                n=ctx_stats.prefixes_skipped)
        if verbose:
            print(f"  {st.render()}")
        if on_progress:
            on_progress("indexing", len(paths), len(paths), f"{len(kids)} chunks")
        idx, report = NumpyIndex.build(all_chunks, vecs, strategy=strategy)
        idx.save(index_name)

        report.update(
            {
                "parser_version": L.PARSER_VERSION,
                "chunker_version": S.CHUNKER_VERSIONS[strategy],
                "contextualizer": pipe.contextualizer,
                "pipeline_fingerprint": pipe.fingerprint(),
                # WHAT THE PAID STAGE ACTUALLY DID, next to the index it
                # produced. Including mean prefix length, which is the number
                # that decides whether this helped: a prefix is a fixed cost on
                # every child, and at a fixed token budget a longer child means
                # fewer children fit. Recall can fall while ranking improves,
                # and an evaluation that measures recall@k cannot see it.
                "contextualization": ctx_stats.to_json() if contextual else None,
                "child_kind": dict(Counter(c.kind.value for c in kids)),
                "child_text_source": dict(Counter(c.text_source for c in kids)),
                "child_doc_type": dict(Counter(r["doc_type"] for r in result.per_file)),
                "provenance_cross_check_ok": ok,
                "provenance_problems": problems,
                "per_file": result.per_file,
                "embed_stats": {"hits": st.hits, "misses": st.misses,
                                "api_calls": st.api_calls, "failed": st.failed},
                "degradations": log.to_dicts(),
            }
        )
        # Same rule as the manifest: a non-default index writes its own report
        # rather than overwriting the one that describes the shipped index.
        report_name = ("index_report.json" if index_name == "numpy_index"
                       else f"index_report.{index_name}.json")
        (config.DATA_EVAL / report_name).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        result.index_report = report
        result.degradations = log.to_dicts()
        # THE ARTIFACT, not just the response. I argued at length that a
        # degradation notice must outlive the terminal, built
        # DegradationLog.write_manifest() for exactly that, and then never
        # called it -- only to_dicts(), which reaches the caller and dies with
        # the process. Found by the name-the-caller audit.
        log.write_manifest(
            # Index-scoped for the same reason the manifest and the index report
            # are: three files described the shipped index, and building a second
            # index quietly re-pointed all three at it.
            None if index_name == "numpy_index"
            else config.DATA_INDEX / f"ingest_manifest.{index_name}.json",
            pipeline_fingerprint=pipe.fingerprint(),
            parser_version=L.PARSER_VERSION,
            chunker_version=S.CHUNKER_VERSIONS[strategy],
            n_children=result.n_children,
            n_parents=result.n_parents,
            provenance_ok=ok,
            # uniform() answers "did anything degrade in THIS RUN", which is a
            # different question from the index's uniform_provenance (a property
            # of the corpus). It had no caller; the eval enforced via index
            # metadata instead. Recording it here gives the run-level answer a
            # home rather than leaving a method that looks enforced and is not.
            run_clean=log.uniform(),
        )
        manifest.save()
        if verbose and log:
            print(log.render_cli())

    result.seconds = time.time() - t0
    return result


# --------------------------------------------------------------------------
# Removal
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _index_write_lock(index_name: str, *, timeout: float = 120.0):
    """One writer at a time for an index directory. Cross-process, not a mutex.

    WHY THIS IS NOT A THREADING LOCK. `ingest_upload` is read-modify-write over
    files: load the index, append, save. Two concurrent uploads both read 814
    chunks, each appends its own, and each writes 816 -- so the second save
    silently discards the first visitor's document.

    AND THE FAILURE IS INVISIBLE, WHICH IS WHY IT IS WORTH TWENTY MINUTES. The
    visitor whose upload vanished asks their question and is told "that is not in
    your documents" -- a sentence this system produces correctly, constantly, and
    on purpose. Silent data loss wearing the costume of the abstention behaviour
    the whole product is built to demonstrate. Of every failure mode reachable
    from this path, that is the one that would take longest to notice.

    An O_EXCL lockfile rather than `threading.Lock` because uvicorn may run more
    than one worker, and a lock inside one process protects nothing from the
    next. The lock is advisory and every writer here takes it.

    STALE LOCKS ARE RECLAIMED, not waited on forever. A worker killed mid-ingest
    would otherwise wedge uploads permanently, which converts a rare race into a
    permanent outage -- a worse trade than the race.
    """
    lock = config.DATA_INDEX / f".{index_name}.write.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    fd = None
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            age = time.time() - lock.stat().st_mtime if lock.exists() else 0.0
            if age > timeout:
                # Older than any legitimate ingest: the holder is gone.
                lock.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise TimeoutError(
                    f"another upload has held the {index_name} index for "
                    f"{timeout:.0f}s. Nothing was written."
                ) from None
            time.sleep(0.15)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock.unlink(missing_ok=True)


def ingest_upload(
    paths: Sequence[Path],
    *,
    owner: str,
    index_name: str = "numpy_index",
    caption_images: bool = True,
) -> dict[str, Any]:
    """Add ONE session's files to the live index. Additive, never destructive.

    WHY NOT `ingest(files=[...])`, which looks like it does this. Because it
    would delete the corpus. `Manifest.plan` computes removals as
    `set(records) - present_ids`, so a run whose `files` argument names one
    upload declares every other document ABSENT and purges it -- dense index,
    sparse index, parent store, assets, caches. The convenience argument and the
    delete detector share one notion of "what is present", and for a full corpus
    walk that is correct. For a subset it is a corpus-wiping bug wearing the
    shape of a filter.

    So this path does not go near the planner. It loads, chunks under the
    session's owner, embeds, and APPENDS. Nothing is compared against what was
    there before, because nothing is supposed to be removed.

    THE OWNER IS THE POINT. `build_chunks` requires it with no default, so every
    chunk this produces is session-scoped by construction, and
    `assert_owned()` in NumpyIndex.__init__ refuses the index outright if an
    upload-sourced chunk ever carries the public sentinel.
    """
    if not paths:
        return {"added": 0, "sources": []}
    config.ensure_dirs()
    # Read once outside the lock only to VALIDATE the fingerprint -- cheap, and
    # re-read inside the lock before appending, because anything read out here
    # may be stale by the time the append happens.
    idx = NumpyIndex.load(index_name)
    # The fingerprint of the index being APPENDED TO, not a fresh one. A chunk
    # embedded under different settings than its neighbours is the mixed-index
    # defect, and here it would arrive one upload at a time.
    fp = sorted({c.pipeline_fingerprint for c in idx.children})
    if len(fp) != 1:
        raise RuntimeError(
            f"refusing to append to an index with {len(fp)} fingerprints: {fp}. "
            "Appending to a mixed index makes the mixture permanent."
        )
    pipe = pipeline_version("header_aware_parent",
                            contextual=config.CONTEXTUAL_PREFIXES)
    if pipe.fingerprint() != fp[0]:
        raise RuntimeError(
            f"refusing to append chunks stamped {pipe.fingerprint()} to an index "
            f"of {fp[0]}. Re-ingest the corpus, or the upload is measured under a "
            "pipeline the rest of the index was not built with."
        )

    new_chunks: list[Chunk] = []
    sources: list[str] = []
    manifest = Manifest()
    with limits.collect() as log:
        for p in paths:
            src, blocks, diag = L.load(p, caption_images=caption_images)
            situate = None
            if config.CONTEXTUAL_PREFIXES:
                doc_text = "\n\n".join(b.text for b in blocks if b.text.strip())
                syn = contextualize.synopsis(src.title or src.source_id, doc_text)

                def situate(parent_text: str, body: str, _syn: str = syn) -> str:
                    return contextualize.situate(doc_synopsis=_syn,
                                                 parent_text=parent_text, body=body)
            chunks = S.build_chunks(src, blocks, pipeline=pipe,
                                    owner=owner, origin="upload",
                                    contextualizer=situate)
            new_chunks.extend(chunks)
            src.pipeline_fingerprint = pipe.fingerprint()
            kids = [c for c in chunks if c.role is ChunkRole.CHILD]
            parents = [c for c in chunks if c.role is ChunkRole.PARENT]
            manifest.record(SourceRecord(
                source=src,
                owner=owner,
                chunk_ids=[c.chunk_id for c in kids],
                parent_ids=[c.chunk_id for c in parents],
                asset_paths=[b.asset_path for b in blocks if b.asset_path],
                cache_keys=[gemini.cache_key(c.embed_text) for c in kids],
                n_pages=diag.get("n_pages", 0),
            ))
            sources.append(src.source_id)

        kids = [c for c in new_chunks if c.role is ChunkRole.CHILD]
        vecs, _st = gemini.embed_texts([c.embed_text for c in kids], kind="document")

    # THE ONLY SERIALISED SECTION, and it is deliberately the smallest one.
    # Parsing, captioning, contextualising and embedding are the slow parts and
    # they run concurrently; the lock covers re-read, append, save -- typically
    # milliseconds. Holding it across the whole ingest would make two visitors
    # queue behind each other's API calls for no safety benefit.
    with _index_write_lock(index_name):
        # RE-READ INSIDE THE LOCK. The copy loaded at the top of this function
        # was for validation and may now be missing another visitor's upload.
        # Appending to it would write 816 chunks over somebody else's 816 and
        # silently drop their document.
        idx = NumpyIndex.load(index_name)
        idx.children = list(idx.children) + kids
        idx.vectors = np.vstack([idx.vectors, vecs])
        for parent in (c for c in new_chunks if c.role is ChunkRole.PARENT):
            idx.parents[parent.chunk_id] = parent
        idx._owners = None                  # the ownership mask is memoised
        idx.save(index_name)
        # The manifest is re-read inside the lock for the same reason.
        fresh = Manifest()
        for sid, rec in manifest.records.items():
            if sid in sources:
                fresh.records[sid] = rec
        fresh.save()
    return {"added": len(kids), "sources": sources,
            "degradations": log.to_dicts()}


def remove_source(
    source_id: str,
    *,
    purge_cache: bool = True,
    reindex: bool = True,
    index_name: str = "numpy_index",
    verbose: bool = True,
) -> dict[str, Any]:
    """Delete a document and everything derived from it.

    EVERY STORE A DOCUMENT LEAVES TRACES IN:
      1. the source file in data/raw
      2. its extracted assets
      3. the manifest record (replaced by a tombstone that remembers chunk ids)
      4. the vector index (via full rebuild -- see below)
      5. the embedding cache, keyed on TEXT and therefore invisible to a
         source-id delete. This is the store that gets forgotten, and for
         right-to-erasure it is the one that matters -- the derived cache holds
         model-written descriptions of the document content.

    Missing any one leaves a ghost that looks like a real hit until it is clicked.

    REINDEX BY REBUILD, not surgical removal. A rebuild over 802 chunks costs
    seconds because parse and embed are both cached on their inputs, whereas
    surgical removal would add a second code path whose result must agree with
    the rebuild -- the exact class of bug this project keeps finding.
    """
    manifest = Manifest()
    purged = manifest.purge_source(source_id)
    if not purged.get("found"):
        manifest.save()
        return {
            "source_id": source_id,
            "found": False,
            "note": "no manifest record; nothing was derived from it",
        }

    removed_files: list[str] = []
    root = config.DATA_RAW.resolve()
    for rel in [source_id, *purged.get("asset_paths", [])]:
        cand = Path(rel)
        target = cand if cand.is_absolute() else (config.DATA_RAW / rel)
        try:
            target = target.resolve()
            target.relative_to(root)      # never delete outside the corpus
        except (ValueError, OSError):
            continue
        if target.exists() and target.is_file():
            target.unlink()
            removed_files.append(str(target.relative_to(root)))

    n_cache = 0
    if purge_cache:
        for key in purged.get("cache_keys", []):
            for f in config.EMBED_CACHE.rglob(key):
                f.unlink(missing_ok=True)
                n_cache += 1

    manifest.save()
    result: dict[str, Any] = {
        "source_id": source_id,
        "found": True,
        "chunks_removed": len(purged.get("chunk_ids", [])),
        "parents_removed": len(purged.get("parent_ids", [])),
        "files_removed": removed_files,
        "cache_entries_removed": n_cache,
        "tombstoned": True,
    }
    if reindex:
        # PUBLIC_OWNER, stated. `owner` was made required with no default so that
        # "forgot to pass it" and "meant it to be public" stop being the same
        # keystroke -- and this call site was missed, which made every deletion
        # that reindexes raise TypeError.
        #
        # It was invisible until the session sweep started running, because the
        # only other caller passes through the CLI. The sweep reported it rather
        # than swallowing it -- `failed: [{source_id, error}]` -- and kept the
        # session for the next attempt, which is the whole reason a purge reports
        # what it could not do instead of retrying silently forever.
        #
        # PUBLIC_OWNER is correct here: this rebuilds THE CORPUS from data/raw,
        # and a session's uploads no longer live there.
        res = ingest(index_name=index_name, verbose=verbose,
                     owner=PUBLIC_OWNER, origin="corpus")
        result["reindexed"] = {"children": res.n_children, "parents": res.n_parents}
    return result


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------


@dataclass
class AskResult:
    question: str
    answer: A.Answer
    hits: list[Hit] = field(default_factory=list)
    parents: list[Chunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    degradations: list[dict[str, Any]] = field(default_factory=list)
    understanding: CondensedQuery | None = None

    def render(self, *, show_sources: bool = True) -> str:
        out = []
        if self.understanding:
            u = self.understanding
            out.append(f"[{u.header}]" + (f"  READ AS: {u.read_as}" if u.read_as else ""))
            if u.was_rewritten:
                out.append(f"  searched as: {u.search_query!r}")
            if not u.needs_retrieval:
                out.append("  retrieval skipped: nothing new was read")
        out.append(self.answer.render())
        if show_sources:
            out.append("\nretrieved:")
            for i, p in enumerate(self.parents, 1):
                # Same locator as the citation renderer. Two formatters for the
                # same concept is how "manual.docx pNone" survived in one of them
                # after being fixed in the other.
                out.append(f"  [{i}] {A.locator(p)}  {p.kind.value}/{p.text_source}")
        out.append("  " + " ".join(f"{k}={v:.0f}ms" for k, v in self.timings_ms.items()))
        return "\n".join(out)


def ask(
    question: str,
    *,
    index_name: str = "numpy_index",
    token_budget: int | None = None,
    max_sources: int = 6,
    history: str = "",
    index: NumpyIndex | None = None,
) -> AskResult:
    """One question -> a verified, cited answer.

    Per-stage timings are collected here rather than added later, because
    everything before generation lands inside TTFT (guide M14) -- so a retrieval
    latency number is a user-visible number, not an implementation detail.
    """
    t = {}
    with limits.collect() as log:
        t0 = time.time()
        idx = index or NumpyIndex.load(index_name)
        t["load"] = (time.time() - t0) * 1000

        t0 = time.time()
        cq = condense(question, history)
        t["condense"] = (time.time() - t0) * 1000

        hits: list[Hit] = []
        parents: list[Chunk] = []
        if cq.needs_retrieval:
            t0 = time.time()
            qv = gemini.embed_query(cq.search_query)
            t["embed"] = (time.time() - t0) * 1000

            t0 = time.time()
            rstats: dict = {}
            hits = idx.search_budget(
                qv, token_budget=token_budget or config.TOKENS_CONTEXT_BUDGET,
                unit="parent", stats=rstats,
            )
            parents = idx.parents_for(hits)[:max_sources]
            t["retrieve"] = (time.time() - t0) * 1000

        t0 = time.time()
        if cq.needs_retrieval:
            ans = A.answer(
                cq.original, parents, history=history,
                # An empty context has two causes with opposite fixes, and the
                # generator cannot tell them apart from an empty list alone.
                starved=bool(rstats.get("starved_by_budget")),
                min_unit_tokens=rstats.get("min_unit_tokens"),
                context_budget=token_budget or config.TOKENS_CONTEXT_BUDGET,
            )
        else:
            ans = A.answer_from_conversation(cq.original, history)
        t["generate"] = (time.time() - t0) * 1000

    return AskResult(question, ans, hits, parents, t, log.to_dicts(), understanding=cq)
