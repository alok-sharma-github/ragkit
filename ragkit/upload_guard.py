"""Decide whether an uploaded document may be parsed, and say why if not.

THE FOURTH AUDIENCE. The refusal messages in this codebase already serve three
readers -- an operator, a customer, a demo visitor asking a question. This is a
different one: a person who just tried to do something and was told no.

That message has to name WHICH LIMIT and WHAT WOULD WORK. "Upload failed" is
useless; so is "exceeds RAGKIT_MAX_UPLOAD_PAGES", which names a knob the reader
cannot turn. The rule is the one already applied to the demo remedy: naming a knob
to someone who cannot turn it is as useless as naming the wrong cause. A visitor
cannot raise the page cap. They can choose a smaller file -- but only if told that
is the fix.

THE CEILING BOUNDS TIME AND MEMORY, NOT PAGES. Pages are a proxy, and a decent one
for wait time (~8s/page measured on this corpus), which is why the cap exists. But
a five-page PDF can be a decompression bomb or a single page with two hundred
thousand vector paths -- passing any page limit and hanging the parser. So the page
cap is the courtesy, and the wall clock plus the address-space limit are the
guarantee. Assert the property that matters, not the one that is easy to count.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class UploadVerdict:
    """May this file be parsed, and if not, what does the visitor need to hear?"""

    ok: bool
    code: str
    # Written for the person who uploaded the file. Names the limit and the fix.
    visitor_message: str = ""
    pages: int | None = None
    bytes: int | None = None
    detail: str = ""
    limits_applied: dict | None = None

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "code": self.code,
            "why": self.visitor_message,
            "pages": self.pages,
            "bytes": self.bytes,
            "detail": self.detail,
        }


def _visitor_message(r: dict) -> str:
    """One sentence: which limit, and what would work instead."""
    code = r.get("code")
    limit, actual = r.get("limit"), r.get("actual")
    if code == "not_a_pdf":
        return ("That file is not a PDF. This demo reads PDFs -- try exporting "
                "your document as one.")
    if code == "encrypted":
        return ("That PDF is password protected, so its text cannot be read. "
                "Remove the password and upload it again.")
    if code == "active_content":
        found = ", ".join(r.get("found") or [])
        return (f"That PDF contains active content ({found}), which this demo will "
                "not open. Printing it to a new PDF usually removes it.")
    if code == "too_many_pages":
        return (f"That document is {actual} pages and this demo accepts up to "
                f"{limit}. Try a shorter document, or split out the section you "
                "want to ask about.")
    if code == "too_complex":
        return ("That PDF is unusually complex internally -- often a scan or a "
                "heavily illustrated export -- and would take too long to read "
                "here. A text-based PDF of the same content will work.")
    if code == "too_large":
        mb = (r.get("limit") or 0) / 1048576
        return (f"That file is larger than {mb:.0f} MB, which is more than this "
                "demo accepts. Try a smaller file.")
    if code == "timeout":
        return ("That document took too long to read and was stopped. Large scans "
                "and heavily illustrated PDFs are the usual cause -- a shorter or "
                "text-based document will work.")
    if code == "memory_limit":
        return ("Reading that document needed more memory than this demo allows. "
                "A shorter or less complex PDF will work.")
    if code in ("unreadable", "probe_failed"):
        return ("That PDF could not be read -- it may be damaged or be a scan with "
                "no text layer. A text-based PDF will work.")
    return "That file could not be accepted."


def check_upload(path: str | Path) -> UploadVerdict:
    """Inspect an untrusted file under hard limits. Never parses in-process."""
    p = Path(path)
    size = p.stat().st_size
    max_bytes = config.MAX_UPLOAD_BYTES
    if size > max_bytes:
        # Checked before the subprocess is even started: no reason to spend a
        # process on a file already over the line.
        r = {"code": "too_large", "limit": max_bytes, "actual": size}
        return UploadVerdict(False, "too_large", _visitor_message(r), bytes=size)

    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "RAGKIT_MAX_UPLOAD_PAGES": str(config.MAX_UPLOAD_PAGES),
        "RAGKIT_MAX_UPLOAD_OBJECTS": str(config.MAX_UPLOAD_OBJECTS),
        "RAGKIT_PARSE_MEM_MB": str(config.PARSE_MEM_MB),
        "RAGKIT_PARSE_CPU_SECONDS": str(config.PARSE_TIMEOUT_SECONDS),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ragkit.upload_probe", str(p)],
            capture_output=True, text=True, env=env,
            timeout=config.PARSE_TIMEOUT_SECONDS,
            cwd=str(config.ROOT),
        )
    except subprocess.TimeoutExpired:
        # The wall clock is the bound that holds regardless of what the file is.
        # A page cap cannot see a bomb; this can.
        r = {"code": "timeout"}
        return UploadVerdict(False, "timeout", _visitor_message(r), bytes=size)

    line = (proc.stdout or "").strip().splitlines()
    if not line:
        # Killed by the CPU/address-space limit, or crashed. Either way the file
        # is refused -- a probe that dies on a document is itself the answer.
        r = {"code": "probe_failed"}
        return UploadVerdict(False, "probe_failed", _visitor_message(r),
                             detail=(proc.stderr or "")[-200:], bytes=size)
    try:
        res = json.loads(line[-1])
    except json.JSONDecodeError:
        r = {"code": "probe_failed"}
        return UploadVerdict(False, "probe_failed", _visitor_message(r), bytes=size)

    return UploadVerdict(
        ok=bool(res.get("ok")),
        code=str(res.get("code", "unknown")),
        visitor_message="" if res.get("ok") else _visitor_message(res),
        pages=res.get("pages"),
        bytes=res.get("bytes", size),
        detail=str(res.get("detail", "")),
        limits_applied=res.get("limits_applied"),
    )
