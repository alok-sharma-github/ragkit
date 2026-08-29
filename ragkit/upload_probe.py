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


def probe(path: str) -> dict[str, object]:
    limits = _apply_limits()
    out: dict[str, object] = {"limits_applied": limits}

    raw = open(path, "rb").read()
    out["bytes"] = len(raw)

    if not raw.startswith(b"%PDF-"):
        # Checked on the bytes, before any parser is involved.
        out.update(ok=False, code="not_a_pdf",
                   detail="the file does not begin with a PDF header")
        return out

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
