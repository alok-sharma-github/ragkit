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
    # Join to the root FIRST. An absolute path keeps only its filename, a
    # relative one is interpreted where the caller meant it -- inside the corpus.
    target = (root / cand.name) if cand.is_absolute() else (root / cand)
    try:
        resolved = target.resolve()
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


