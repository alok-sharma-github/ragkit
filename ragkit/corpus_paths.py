"""Where a corpus path may point. Not an HTTP concern, despite its one caller.

THIS LIVED IN `app/api.py` AND THAT WAS A MISTAKE WITH A COST. The invariant
that guards it has to call the real function -- a check that reimplements what
it checks is testing itself (A-16) -- so `reconcile` imported `app.api`, which
imports FastAPI, which made the EVAL HARNESS depend on the web stack. It crashed
with ModuleNotFoundError in any environment that installs the library without
the server.

The rule itself was never about HTTP. It answers "does this path point inside the
corpus", which is a fact about the corpus. Moving it here lets the endpoint and
the invariant share one implementation without dragging a web framework into a
measurement tool.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def resolve_asset(path: str) -> Path | None:
    """Where an asset path lands, or None if it escapes. THE function, not a copy.

    A SEPARATE FUNCTION SO THE INVARIANT CAN CALL IT. The first version of the
    upload-reachability check re-implemented this logic and passed -- against
    itself. That is this project's oldest failure and it very nearly shipped
    inside the check written to catch a different instance of it. An invariant
    that tests a reimplementation tests the reimplementation.

    TWO BUGS FIXED HERE, and they were the same bug pointing opposite ways.
    `Path(path).resolve()` resolves a RELATIVE path against the process working
    directory, not against the corpus root. Measured consequences:

      `assets/hnsw_p2.png`          the URL the UI itself generates -> 403
      `data/raw/<upload>.pdf`       another session's private document -> 200

    So it refused what it should serve and served what it should refuse. Joining
    to the root first, then checking containment, is correct in both directions.

    Containment is NOT ownership, and this function does not pretend otherwise.
    Uploads are unreachable here because they live outside this root entirely --
    a directory this function cannot address needs no rule about who may read it.
    """
    root = config.DATA_RAW.resolve()
    cand = Path(path)

    # ABSOLUTE PATHS ARE REFUSED, not reinterpreted. The previous version kept
    # their basename -- `/etc/passwd` became `data/raw/passwd` -- which is
    # contained and therefore safe, and which is also PLATFORM-DEPENDENT in a
    # security rule, which is a defect on its own.
    #
    # `Path("/etc/passwd").is_absolute()` is True on Linux and False on Windows,
    # where it is merely rooted. So the same input took different branches on
    # different operating systems: on Windows it escaped the root and was
    # refused; on Linux it was silently rewritten to a corpus filename. The
    # invariant passed on my machine and FAILED IN CI, which is the only reason
    # anybody looked.
    #
    # The UI never sends an absolute path. Refusing them outright makes the
    # behaviour identical everywhere and removes a silent reinterpretation of
    # untrusted input -- and `is_absolute()` is not enough to spot one, because
    # a leading separator is not "absolute" on Windows.
    if cand.is_absolute() or path.startswith(("/", "\\")) or cand.drive:
        return None

    try:
        resolved = (root / cand).resolve()
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


