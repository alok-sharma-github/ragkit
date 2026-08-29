"""Ephemeral upload sandboxes: who uploaded what, and when it goes away.

A visitor uploading their own document is the point of the public demo -- the
system's differentiator is invisible on our corpus, because a visitor cannot know
the tables are broken, and obvious on theirs, because they know what is in it. The
cost is that a stranger's document must never be readable by the next stranger.

THE SESSION ID IS THE SECRET. 128 bits from `secrets.token_urlsafe(16)`, so a live
session cannot be guessed. That is what makes signing the cookie unnecessary: a
signature would prove the value came from us, and we do not care where it came
from -- only whether it matches a session that exists. A forged id that passes the
shape check opens a fresh empty session and nothing else.

Which means the id must not leak, and there is exactly one way it does: JavaScript
reading the cookie. Hence HttpOnly. Unguessability protects everything except a
value someone can simply read.

WHAT THIS IS NOT. Not accounts, not authentication, not durable storage. A session
is a bucket with a timer. Nothing here survives a container restart, and that is
deliberate for a demo whose uploads are strangers' documents.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from .ingest.document import PUBLIC_OWNER

COOKIE = "ragkit_session"

# 16 random bytes -> 22 urlsafe-base64 characters.
_SHAPE = re.compile(r"^[A-Za-z0-9_-]{22}$")


def new_id() -> str:
    return secrets.token_urlsafe(16)


def resolve(raw: str | None) -> str | None:
    """Turn an untrusted cookie value into an owner, or None.

    The cookie arrives from the browser, so it is attacker-controlled. Two checks,
    and the second is not redundant with the first:

      shape     22 urlsafe characters. Rejects "", rejects a crafted value with
                path separators or quotes, bounds the length.
      identity  it must not BE the public sentinel.

    The shape rule already excludes "" today, and that is exactly why the identity
    check belongs here too: the regex CORRELATES with "not the public sentinel",
    while the identity check IS that property. If PUBLIC_OWNER ever changes from
    "" to something like "public" -- six characters, passes plenty of shape rules
    -- the regex silently stops protecting anything and nothing announces it.

    Same distinction as /OpenAction versus /JavaScript, and `crumb in body` versus
    startswith. Correlation is acceptable in a report and disqualifying in a gate.
    """
    if not raw:
        return None
    if not _SHAPE.match(raw):
        return None
    if secrets.compare_digest(raw, PUBLIC_OWNER):
        # Unreachable while PUBLIC_OWNER is "" and the shape demands 22 chars.
        # Kept because it names the actual property rather than a proxy for it.
        return None
    return raw


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # The documents this session uploaded, by source_id. Purging a session means
    # purging these, through the ordinary deletion path.
    source_ids: list[str] = field(default_factory=list)

    def age(self) -> float:
        return time.time() - self.created_at

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "documents": len(self.source_ids),
            "age_seconds": int(self.age()),
            "expires_in_seconds": max(
                0, int(config.UPLOAD_TTL_SECONDS - self.age())),
        }


class Store:
    """In-memory sessions. Lost on restart, which is the intended lifetime."""

    def __init__(self) -> None:
        self._by_id: dict[str, Session] = {}
        # Uploads mutate the shared index while other requests read it.
        self._lock = threading.RLock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            if session_id and session_id in self._by_id:
                s = self._by_id[session_id]
                s.last_seen = time.time()
                return s
            sid = session_id or new_id()
            s = Session(session_id=sid)
            self._by_id[sid] = s
            return s

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            return self._by_id.get(session_id)

    def note_document(self, session_id: str, source_id: str) -> None:
        with self._lock:
            s = self._by_id.setdefault(session_id, Session(session_id=session_id))
            if source_id not in s.source_ids:
                s.source_ids.append(source_id)

    def expired(self) -> list[Session]:
        ttl = config.UPLOAD_TTL_SECONDS
        with self._lock:
            return [s for s in self._by_id.values() if s.age() > ttl]

    def forget(self, session_id: str) -> Session | None:
        with self._lock:
            return self._by_id.pop(session_id, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sessions": len(self._by_id),
                "documents": sum(len(s.source_ids) for s in self._by_id.values()),
                "ttl_seconds": config.UPLOAD_TTL_SECONDS,
            }


STORE = Store()


def purge_expired(*, verbose: bool = False) -> dict[str, Any]:
    """Delete every expired session's documents, through the NORMAL path.

    THE DELETION IS NOT WRITTEN HERE. `pipeline.remove_source` already knows the
    six stores plus two caches a document leaves traces in, and it exists because
    a deletion that misses one store leaves a ghost that looks like a real hit
    until someone clicks it.

    A second deletion path written for sessions would be that argument all over
    again -- the same case already made against Qdrant, where two datastores make
    "gone from one, still in the other" possible. One deletion path, called twice,
    beats two deletion paths that agree today.

    So the only thing session-specific here is deciding WHICH documents go.
    """
    from . import pipeline

    removed: list[str] = []
    failed: list[dict[str, str]] = []
    for s in STORE.expired():
        for source_id in list(s.source_ids):
            try:
                pipeline.remove_source(source_id, purge_cache=True, verbose=verbose)
                removed.append(source_id)
            except Exception as exc:  # noqa: BLE001
                # A failed purge is recorded, not swallowed: the session record is
                # kept so the next sweep retries. Forgetting the session while its
                # documents survive would orphan them -- owned by a session that
                # no longer exists, unreachable and never collected.
                failed.append({"source_id": source_id, "error": str(exc)[:160]})
        if not failed:
            STORE.forget(s.session_id)
    return {"purged_documents": removed, "failed": failed,
            "sessions_remaining": STORE.stats()["sessions"]}
