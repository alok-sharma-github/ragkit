"""ragkit -- a tutorial RAG build over Gemini.

CONSOLE ENCODING IS FIXED HERE, at package import, and that placement is the
whole point.

The default Windows console codepage is cp1252. During this build it has broken
five separate times: it took out a degradation warning mid-print (the error path
destroying the message it existed to deliver), it made correctly-stored accented
author names look like corrupted data three times, and it killed a golden-set
generation run AFTER 120 Gemini calls had succeeded -- on a Turkish dotless i in
one needle -- losing all of the work to a print statement.

A `_fix_console()` existed in cli.py before that last one. It did not help,
because the run was an inline script that never went through the CLI. Same shape
as the citation guard that was correct on the child and absent on the parent: a
fix that is not on the path the traffic takes is not a fix.

So it lives at package import, where nothing that touches ragkit can bypass it.

This is an import-time side effect, which is the exact category of thing that
cost an hour earlier today (pymupdf4llm flipping a global geometry flag on
import). The difference is scope and idempotence: this only sets the encoding of
this process's own output streams, it cannot alter any computation, and running
it twice is the same as running it once. Naming the tension rather than pretending
it is not there.
"""

from __future__ import annotations

import sys as _sys


def _force_utf8() -> None:
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 -- never let this be the thing that fails
            pass


_force_utf8()


def _quiet_sdk_noise() -> None:
    """Silence the google-genai AFC advisory.

    It prints on EVERY generate_content call -- "Direct use of automatic function
    calling ... is not recommended" -- which is advice about a feature this
    project does not use. At ~92 calls in one eval run it buries the actual
    output, and noise that always fires trains you to skim past the line where a
    real warning would appear.
    """
    import logging

    class _DropAFC(logging.Filter):
        def filter(self, record: "logging.LogRecord") -> bool:
            return "automatic function calling" not in record.getMessage().lower()

    for name in ("google_genai", "google_genai.models", "google.genai"):
        logging.getLogger(name).addFilter(_DropAFC())


_quiet_sdk_noise()
