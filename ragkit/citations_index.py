"""Which stored turns cited a document — asked BEFORE it is deleted.

Tombstones answer the question afterwards: a citation whose chunk is gone renders
as a tombstone rather than a broken reference. This is the other half, and it is a
different structure: the removal screen has to name what will be affected *while
the confirm dialog is still open*.

TURN GRANULARITY, NOT CONVERSATION. The UI copy only needs conversation names, so
per-conversation would have been enough for the screen and wrong for the record.
`Turn` stores `sources` **and** `pipeline_fingerprint`, and a conversation can span
fingerprint changes — so "three turns cited this, two of them under a pipeline you
no longer run" is a true statement that a per-conversation index cannot make.

The cost of choosing right is zero today and unrecoverable later: the granularity
is fixed at backfill time, and coarsening is free while refining means re-reading
every conversation file. Same argument as capturing `owner_id` at ingest rather
than backfilling it once there are customers.

WHY AN INDEX RATHER THAN A SCAN. Answering from the files directly means reading
every conversation to service one delete. Fine at nine conversations, a full scan
at nine thousand, and it becomes a migration exactly when the system is busiest.

DERIVED, NOT AUTHORITATIVE. Conversations are the source of truth; this is a
cache. `rebuild()` regenerates it from them at any time, so a corrupt or missing
index costs a rebuild rather than data. It is therefore safe to delete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

PATH = config.DATA_INDEX / "citation_index.json"
FORMAT_VERSION = 1


@dataclass
class Citing:
    """One turn that cited a document."""

    conversation_id: str
    conversation_title: str
    turn_index: int
    question: str
    # The pipeline that produced the citation. A turn cited under a fingerprint
    # you no longer run is still a real citation, and still worth naming before a
    # delete -- but it is a weaker claim about the current index, and collapsing
    # the two would overstate the blast radius.
    pipeline_fingerprint: str = ""
    created_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "turn_index": self.turn_index,
            "question": self.question,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "created_at": self.created_at,
        }


@dataclass
class Impact:
    """What deleting one document would touch in stored conversations."""

    source_id: str
    turns: list[Citing] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def conversations(self) -> list[dict[str, Any]]:
        """Deduplicated to conversations, for the UI, keeping the turn counts."""
        by: dict[str, dict[str, Any]] = {}
        for t in self.turns:
            row = by.setdefault(
                t.conversation_id,
                {"conversation_id": t.conversation_id,
                 "title": t.conversation_title, "n_turns": 0},
            )
            row["n_turns"] += 1
        return sorted(by.values(), key=lambda r: -r["n_turns"])

    def stale_turns(self, current_fingerprint: str) -> int:
        """Turns cited under a pipeline that is no longer the current one."""
        if not current_fingerprint:
            return 0
        return sum(
            1 for t in self.turns
            if t.pipeline_fingerprint and t.pipeline_fingerprint != current_fingerprint
        )

    def to_json(self, *, current_fingerprint: str = "") -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "n_turns": self.n_turns,
            "n_conversations": len(self.conversations),
            "conversations": self.conversations,
            "turns": [t.to_json() for t in self.turns],
            "n_turns_under_older_pipeline": self.stale_turns(current_fingerprint),
        }


def _source_ids_of(turn: dict[str, Any]) -> set[str]:
    """Every source_id a turn cited.

    Reads `sources` (what was SENT) rather than `claims` (what was cited), because
    the removal question is "what did this turn read", and a source that was sent
    and not cited still shaped the answer -- and still renders in the turn's source
    panel, which is what breaks when the document goes.
    """
    out: set[str] = set()
    for s in turn.get("sources") or []:
        sid = (s or {}).get("source_id")
        if sid:
            out.add(str(sid))
    return out


def rebuild(store: Any | None = None) -> dict[str, Any]:
    """Regenerate the index from the conversations. Safe to run at any time."""
    from . import conversations as C

    store = store or C.STORE
    index: dict[str, list[dict[str, Any]]] = {}
    n_conv = n_turns = 0

    for p in sorted(Path(store.dir).glob("*.json")):
        try:
            doc = json.loads(p.read_text("utf-8"))
        except Exception:  # noqa: BLE001 -- one unreadable file is not a failure
            continue
        n_conv += 1
        title = doc.get("title", "")
        for i, turn in enumerate(doc.get("turns") or []):
            n_turns += 1
            for sid in _source_ids_of(turn):
                index.setdefault(sid, []).append(
                    Citing(
                        conversation_id=doc.get("id", p.stem),
                        conversation_title=title,
                        turn_index=i,
                        question=(turn.get("question") or "")[:160],
                        pipeline_fingerprint=turn.get("pipeline_fingerprint", ""),
                        created_at=turn.get("created_at", ""),
                    ).to_json()
                )

    payload = {
        "format_version": FORMAT_VERSION,
        "n_conversations_scanned": n_conv,
        "n_turns_scanned": n_turns,
        "n_sources_cited": len(index),
        "by_source": index,
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(PATH)
    return payload


def _read() -> dict[str, Any]:
    if not PATH.exists():
        return rebuild()
    try:
        doc = json.loads(PATH.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return rebuild()
    # A format bump invalidates rather than mis-reads. The index is derived, so
    # rebuilding is always available and always cheaper than guessing.
    if doc.get("format_version") != FORMAT_VERSION:
        return rebuild()
    return doc


def record(conversation: Any) -> None:
    """Fold one conversation's turns into the index. Called on save.

    Rewrites that conversation's entries wholesale rather than appending, so
    saving the same conversation twice cannot double-count -- the failure the
    histogram just paid for at a different layer.
    """
    doc = _read()
    by = doc.setdefault("by_source", {})
    cid = conversation.id

    for sid in list(by):
        by[sid] = [e for e in by[sid] if e.get("conversation_id") != cid]
        if not by[sid]:
            del by[sid]

    for i, t in enumerate(conversation.turns):
        # THE SAME EXTRACTOR AS rebuild(), not a second copy of it.
        #
        # This iterated `t.sources` directly and appended once per source DICT,
        # while rebuild() used _source_ids_of() which returns a SET. A turn that
        # delivered two parents from the same document therefore produced two
        # entries here and one there -- so re-saving a conversation grew the count
        # (measured: 7 -> 8) and the incremental index drifted from the rebuilt
        # one.
        #
        # Two code paths computing the same quantity, disagreeing. Exactly the
        # failure the pymupdf4llm survey script caused at the parser layer and the
        # histogram caused at the counting layer, here at the indexing layer. One
        # implementation, called twice.
        for sid in sorted(_source_ids_of({"sources": t.sources})):
            by.setdefault(str(sid), []).append(
                Citing(
                    conversation_id=cid,
                    conversation_title=conversation.title,
                    turn_index=i,
                    question=(t.question or "")[:160],
                    pipeline_fingerprint=getattr(t, "pipeline_fingerprint", "") or "",
                    created_at=getattr(t, "created_at", "") or "",
                ).to_json()
            )

    doc["n_sources_cited"] = len(by)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(PATH)


def impact(source_id: str) -> Impact:
    """Which stored turns would be affected by removing this document."""
    doc = _read()
    rows = (doc.get("by_source") or {}).get(str(source_id)) or []
    return Impact(
        source_id=str(source_id),
        turns=[
            Citing(
                conversation_id=r.get("conversation_id", ""),
                conversation_title=r.get("conversation_title", ""),
                turn_index=int(r.get("turn_index", 0)),
                question=r.get("question", ""),
                pipeline_fingerprint=r.get("pipeline_fingerprint", ""),
                created_at=r.get("created_at", ""),
            )
            for r in rows
        ],
    )
