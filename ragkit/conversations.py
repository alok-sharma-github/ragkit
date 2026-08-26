"""
Conversation persistence, and the drift accounting it makes possible.

TWO REASONS THIS EXISTS, and the second is the interesting one.

1. The design's sidebar lists saved conversations, and multi-turn retrieval needs
   history to condense against. Without persistence, `chat` is in-memory only and
   a refresh loses the thread.

2. A CONVERSATION IS THE UNIT WHERE GROUNDING DEGRADES. Query condensation can
   route a turn to `conversation_only`, which answers without touching a
   document. That is correct behaviour and it is also a new way to be confidently
   wrong: every citation guard in this project is INERT on that path -- there is
   nothing verbatim to verify against, so `verification` reports
   `not_applicable_no_sources`.

   One such turn is fine. A conversation that starts grounded and drifts into
   answering from itself is the exact degradation the whole provenance design was
   built to make visible -- and it is invisible per-turn, because each individual
   turn looks like a reasonable judgment call. Only the trajectory shows it.

   So the route of every turn is stored, and `drift()` reports the trend rather
   than the level. A conversation that has been self-referential from turn one has
   the highest possible share and is not drifting at all; it is a different kind
   of conversation. Level says nothing, trend says everything -- the same lesson
   as the recall curve, where a single point at a 12,000-token budget reported
   100% and meant nothing.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .retrieve.query import route_drift

DIR = config.DATA_INDEX / "conversations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Turn:
    question: str
    answer_markdown: str = ""
    route: str = "documents"
    header: str = ""
    read_as: str = ""
    search_query: str = ""
    was_rewritten: bool = False
    retrieval_ran: bool = True
    abstained: bool = False
    grounded: bool = True
    # Kept per turn rather than recomputed: the answer was given under a specific
    # index and a specific set of sources, and those can be removed later. A
    # citation whose document is gone must still render as it was given, which is
    # only possible if the turn remembers what it cited.
    claims: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    evidence_mix: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    pipeline_fingerprint: str = ""
    created_at: str = field(default_factory=_now)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    id: str
    title: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    turns: list[Turn] = field(default_factory=list)

    # -- history for condensation -------------------------------------------

    def history(self, *, max_turns: int = 6) -> str:
        """Recent turns as plain text, for the condenser and the generator.

        A WINDOW, not the whole thing. Two reasons: the token budget is allocated
        deliberately (config.TOKENS_HISTORY_RESERVE), and a condenser given fifty
        turns is more likely to drag stale subject matter into a rewritten query --
        which is the failure mode condensation itself introduces.
        """
        lines: list[str] = []
        for t in self.turns[-max_turns:]:
            lines.append(f"user: {t.question}")
            body = t.answer_markdown or (
                "(abstained) " + (t.reconciliation.get("abstain_reason") or "")
            )
            lines.append(f"assistant: {body}")
        return "\n".join(lines)

    def drift(self) -> dict[str, Any]:
        return route_drift([t.route for t in self.turns])

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "n_turns": len(self.turns),
            "drift": self.drift(),
            "turns": [t.to_json() for t in self.turns],
        }

    def summary(self) -> dict[str, Any]:
        """Sidebar row: enough to choose a conversation, not enough to render it."""
        d = self.drift()
        return {
            "id": self.id,
            "title": self.title or (self.turns[0].question[:60] if self.turns else "(empty)"),
            "n_turns": len(self.turns),
            "updated_at": self.updated_at,
            "conversation_only_share": d["conversation_only_share"],
            "drifting": d["drifting"],
        }


class Store:
    """One JSON file per conversation.

    A file per conversation rather than one index file: concurrent turns on two
    conversations then cannot clobber each other, and a corrupt write loses one
    thread instead of all of them.
    """

    def __init__(self, dirpath: Path | None = None) -> None:
        self.dir = dirpath or DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cid: str) -> Path:
        return self.dir / f"{cid}.json"

    def create(self, title: str = "") -> Conversation:
        c = Conversation(id=uuid.uuid4().hex[:12], title=title)
        self.save(c)
        return c

    def get(self, cid: str) -> Conversation | None:
        p = self._path(cid)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        turns = [Turn(**t) for t in d.get("turns", [])]
        return Conversation(
            id=d["id"], title=d.get("title", ""),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            turns=turns,
        )

    def save(self, c: Conversation) -> Path:
        c.updated_at = _now()
        p = self._path(c.id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(c.to_json(), indent=1), encoding="utf-8")
        tmp.replace(p)
        return p

    def append(self, cid: str, turn: Turn) -> Conversation:
        c = self.get(cid) or self.create()
        c.turns.append(turn)
        if not c.title and c.turns:
            c.title = c.turns[0].question[:60]
        self.save(c)
        return c

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        out: list[Conversation] = []
        for p in self.dir.glob("*.json"):
            c = self.get(p.stem)
            if c:
                out.append(c)
        out.sort(key=lambda c: c.updated_at, reverse=True)
        return [c.summary() for c in out[:limit]]

    def delete(self, cid: str) -> bool:
        p = self._path(cid)
        if p.exists():
            p.unlink()
            return True
        return False


STORE = Store()
