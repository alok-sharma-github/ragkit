"""Inspect an untrusted document. Runs as a SUBPROCESS under hard limits.

Everything here touches bytes a stranger uploaded, which is why none of it runs in
the API process. `python -m ragkit.upload_probe <path>` prints one JSON line and
exits; the parent applies a wall-clock timeout and kills it.

WHY THE INSPECTION IS ALSO INSIDE THE SANDBOX. The obvious split is "cheap checks
in-process, expensive parse in a subprocess" -- and it is wrong, because the cheap
checks are themselves parsing. Reading `doc.is_encrypted` means PyMuPDF has already
walked the xref table and trailer of a hostile file. PyMuPDF is a C library behind
a Python wrapper, and C libraries that parse untrusted input have a long history of
memory-safety bugs. So the boundary goes around ALL of it: the parent process never
touches the bytes.

The limits below bound the thing that actually matters. A page cap is a proxy: a
five-page PDF can be a decompression bomb, a deeply nested object graph, or one
page carrying two hundred thousand vector paths. Every one of those passes a page
limit and hangs the parser. Wall clock and address space do not care what shape the
file is.
"""

from __future__ import annotations

import json
import os
import sys


def _apply_limits() -> dict[str, object]:
    """Cap address space and CPU for this process. POSIX only.

    Returns what was actually applied, because a limit you believe is in force
    and is not is worse than no limit: it licenses the caller to relax the checks
    that do work. On Windows there is no RLIMIT_AS, so this reports `false` and
    the parent's wall-clock timeout is the only bound -- which is stated rather
    than glossed, since the container is Linux and the laptop is not.
    """
    applied: dict[str, object] = {"address_space_mb": None, "cpu_seconds": None}
    try:
        import resource  # POSIX only
    except ImportError:
        return applied

    mb = int(os.environ.get("RAGKIT_PARSE_MEM_MB", "512"))
    cpu = int(os.environ.get("RAGKIT_PARSE_CPU_SECONDS", "60"))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mb * 1024 * 1024,) * 2)
        applied["address_space_mb"] = mb
    except Exception:  # noqa: BLE001
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        applied["cpu_seconds"] = cpu
    except Exception:  # noqa: BLE001
        pass
    return applied


# Features that make a document ACTIVE rather than a document. None is needed to
# read text, and each is a known delivery mechanism -- so they are refused rather
# than stripped, because stripping means trusting the stripper.
#
# /OpenAction AND /AA WERE HERE AND HAD TO COME OUT. They are present in most
# ordinary PDFs, where they encode "open at page 1" or "fit to window" -- and
# including them rejected self-rag.pdf, an ordinary academic paper from this
# project's own corpus. A rule that refuses a large share of legitimate documents
# does not make the demo safer, it makes it useless, which is its own failure.
#
# The predicate has to MATCH the condition, not correlate with it: /OpenAction
# correlates with "might do something on open"; only /JavaScript and /Launch mean
# "will run something". An /OpenAction that points AT JavaScript is still caught,
# because the JavaScript keys are scanned independently.
_ACTIVE_KEYS = (b"/JavaScript", b"/JS", b"/Launch", b"/EmbeddedFile", b"/RichMedia")


# Magic bytes, in the order they are cheapest to test. An extension is a claim
# by the uploader; these are a property of the file.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),          # JPEG
    (b"RIFF", "image"),                    # WebP, confirmed below
    (b"PK\x03\x04", "docx"),             # ZIP container, confirmed below
)


def _sniff(raw: bytes) -> str | None:
    for sig, kind in _MAGIC:
        if not raw.startswith(sig):
            continue
        if sig == b"RIFF" and raw[8:12] != b"WEBP":
            return None                    # some other RIFF container
        if sig == b"PK\x03\x04" and b"word/" not in raw[:8192]:
            # A ZIP is not a document. Confirmed by looking for the Word part
            # near the front rather than by trusting the extension.
            return None
        return kind
    return None


def _probe_image(path: str, raw: bytes, out: dict[str, object]) -> dict[str, object]:
    """Images fail differently from PDFs: the threat is PIXELS, not pages.

    A 40 KB PNG can declare 60,000 x 60,000 pixels and expand to gigabytes the
    moment anything decodes it -- the classic decompression bomb. A page cap
    cannot see that and an object count is meaningless here, so the limit that
    matters is the pixel count, checked from the HEADER before any decode.
    """
    import pymupdf

    max_mp = float(os.environ.get("RAGKIT_MAX_IMAGE_MEGAPIXELS", "40"))
    try:
        doc = pymupdf.open(path)
        page = doc[0]
        w, h = int(page.rect.width), int(page.rect.height)
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, code="unreadable", detail=str(exc)[:200])
        return out

    mp = (w * h) / 1_000_000
    out.update(pages=1, width=w, height=h, megapixels=round(mp, 1))
    if mp > max_mp:
        out.update(ok=False, code="image_too_large", limit=max_mp, actual=round(mp, 1))
        return out
    out.update(ok=True, code="accepted")
    return out


def _probe_docx(path: str, raw: bytes, out: dict[str, object]) -> dict[str, object]:
    """A DOCX is a ZIP, and the threat is what it expands to.

    Checked WITHOUT extracting anything: the archive's own directory declares
    each member's uncompressed size, so the total is knowable before a single
    byte is written. A 200 KB file that unpacks to 4 GB is refused on its own
    declaration.
    """
    import zipfile

    max_ratio = float(os.environ.get("RAGKIT_MAX_ZIP_RATIO", "200"))
    max_unpacked_mb = float(os.environ.get("RAGKIT_MAX_UNPACKED_MB", "300"))
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            unpacked = sum(i.file_size for i in infos)
            names = [i.filename for i in infos]
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, code="unreadable", detail=str(exc)[:200])
        return out

    # Zip-slip: a member whose path escapes the archive root. python-docx does
    # not extract to disk, so this is defence in depth rather than the only
    # thing standing between a stranger and the filesystem.
    escaping = [n for n in names if n.startswith("/") or ".." in n.split("/")]
    if escaping:
        out.update(ok=False, code="unsafe_archive", found=escaping[:3],
                   detail="a member path escapes the archive root")
        return out

    ratio = unpacked / max(len(raw), 1)
    out.update(members=len(names), unpacked_mb=round(unpacked / 1048576, 1),
               ratio=round(ratio, 1), pages=0)
    if unpacked / 1048576 > max_unpacked_mb or ratio > max_ratio:
        out.update(ok=False, code="archive_bomb",
                   limit=max_unpacked_mb, actual=round(unpacked / 1048576, 1))
        return out
    out.update(ok=True, code="accepted")
    return out


def probe(path: str) -> dict[str, object]:
    limits = _apply_limits()
    out: dict[str, object] = {"limits_applied": limits}

    raw = open(path, "rb").read()
    out["bytes"] = len(raw)

    # WHICH FORMAT, FROM THE BYTES. Not the extension -- a stranger chooses that.
    #
    # This used to refuse anything not starting with %PDF-, which made the guard
    # PDF-only while `ALLOWED` in the API listed six extensions and the pipeline
    # genuinely reads three families. The corpus itself contains a DOCX and four
    # PNGs, so the capability was demonstrably there and unreachable by upload.
    #
    # Each family gets its OWN checks, because each has its own way of being
    # hostile, and a single "is it safe" answer across three formats would be a
    # guess about two of them.
    kind = _sniff(raw)
    out["kind"] = kind
    if kind is None:
        out.update(ok=False, code="unsupported_type",
                   detail="the file is not a PDF, Word document or image")
        return out
    if kind == "image":
        return _probe_image(path, raw, out)
    if kind == "docx":
        return _probe_docx(path, raw, out)

    # Scanned on the raw bytes rather than via the object model: reaching the
    # object model means the parser has already done the work being guarded
    # against, and a false positive here costs a rejected file, not a crash.
    found = sorted({k.decode().lstrip("/") for k in _ACTIVE_KEYS if k in raw})
    if found:
        out.update(ok=False, code="active_content", found=found,
                   detail="the file contains active content: " + ", ".join(found))
        return out

    import pymupdf  # imported AFTER the limits are in force

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, code="unreadable", detail=str(exc)[:200])
        return out

    if doc.is_encrypted and doc.needs_pass:
        out.update(ok=False, code="encrypted",
                   detail="the file is password protected")
        return out

    pages = doc.page_count
    objects = doc.xref_length()
    out.update(pages=pages, objects=objects)

    max_pages = int(os.environ.get("RAGKIT_MAX_UPLOAD_PAGES", "20"))
    max_objects = int(os.environ.get("RAGKIT_MAX_UPLOAD_OBJECTS", "50000"))

    if pages > max_pages:
        out.update(ok=False, code="too_many_pages", limit=max_pages, actual=pages)
        return out
    if objects > max_objects:
        # Object count catches complexity a page count cannot see: one page can
        # carry a graph large enough to hang extraction.
        out.update(ok=False, code="too_complex", limit=max_objects, actual=objects)
        return out

    out.update(ok=True, code="accepted")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(json.dumps({"ok": False, "code": "usage"}))
        return 2
    try:
        print(json.dumps(probe(argv[1])))
    except MemoryError:
        # The address-space limit firing IS the guard working, so it reports as a
        # refusal rather than as a crash.
        print(json.dumps({"ok": False, "code": "memory_limit"}))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "code": "probe_failed",
                          "detail": type(exc).__name__}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
