"""
The query understanding layer. Guide module M1.

WHY THIS IS THE #1 GAP in a linear load->store->retrieve->generate model: the raw
user query is rarely the thing you should embed. "What about the other one?" and
"would that hold for older patients?" embed to noise -- they contain almost no
retrievable content, and the information needed to make them retrievable is in
the conversation, not the question.

Without this layer multi-turn chat is broken BY DESIGN, not by bug. That is why
`ragkit.cli chat` prints a warning saying so.

---------------------------------------------------------------------------
THE ROUTE IS THE PRODUCT, and the design already worked this out

The design's per-answer headers are not decoration, they are a routing taxonomy:

    FROM YOUR DOCUMENTS · 6 passages read
    YOUR DOCUMENTS + THIS CONVERSATION · 2 new passages
    THIS CONVERSATION · nothing new was read
    THIS CONVERSATION · two steps from your documents   (+ "Ask fresh: ...")

Each states WHERE the answer came from before the user reads a word of it. That
is the same discipline as the citation colours, moved up a level: the grounding
of an answer is declared, not inferred by the reader.

So condensation returns a ROUTE, and the route decides whether retrieval runs at
all:

    documents               -> retrieve with the condensed query
    documents+conversation  -> retrieve, and history is also evidence
    conversation_only       -> DO NOT RETRIEVE; the answer is a reading of what
                               was already said, and must be labelled as such
    ask_fresh               -> the conversation cannot answer it and neither can
                               a follow-up; propose a standalone question

MEASURED, AND WORTH RECORDING: `ask_fresh` does not fire in practice. Five
deliberately-constructed follow-ups ("would it hold at 10x corpus size?", "what
about the third site?", "how does that compare to the industry average?") all
routed to `documents` or `documents_and_conversation` instead.

The model is right to do that. Retrieval is cheap, and if nothing supports an
answer the generation layer abstains WITH a reason -- which is strictly more
useful than refusing to look. So the "Ask fresh: ..." suggestion was moved to the
abstention path in generate/answer.py, where it fires on a branch that actually
executes. The route is left in the enum because the model may still choose it,
but nothing depends on it.

`conversation_only` is the one that earns its keep twice: it saves a retrieval
(guide M1's retrieval-necessity classification, which also removes the
irrelevant-context degradation that comes with retrieving for "summarise that")
AND it is what makes the design's grey citation state honest. A claim derived
from the conversation must not be rendered in the "from your documents" colour.

---------------------------------------------------------------------------
THE FAILURE MODE OF CONDENSATION ITSELF

Rewriting drags stale context across topic shifts. Ask about dosing for four
turns, then ask "what is the retention policy?", and a naive rewriter produces
"what is the retention policy for renagliptin dosing" -- worse than the raw
question. So the rewriter is instructed to detect topic shift explicitly and
return the question UNCHANGED when the new question already stands alone, and
`was_rewritten=False` records that it declined. A rewriter that always rewrites
cannot be audited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .. import gemini, limits

Route = Literal["documents", "documents_and_conversation", "conversation_only", "ask_fresh"]

ROUTE_HEADER: dict[str, str] = {
    "documents": "FROM YOUR DOCUMENTS",
    "documents_and_conversation": "YOUR DOCUMENTS + THIS CONVERSATION",
    "conversation_only": "THIS CONVERSATION",
    "ask_fresh": "THIS CONVERSATION",
}

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": [
                "documents",
                "documents_and_conversation",
                "conversation_only",
                "ask_fresh",
            ],
        },
        "search_query": {"type": "string"},
        "was_rewritten": {"type": "boolean"},
        "topic_shift": {"type": "boolean"},
        "read_as": {"type": "string"},
        "suggested_fresh_question": {"type": "string"},
    },
    "required": ["route", "search_query", "was_rewritten", "topic_shift", "read_as"],
}

_SYSTEM = """You prepare a user's question for a document-retrieval system.

Decide ONE route:
- documents: the question asks for facts from the documents. Rewrite it to stand
  alone if it depends on the conversation (resolve pronouns, ellipsis, "that",
  "the second one").
- documents_and_conversation: it needs new facts AND something already said.
- conversation_only: it can be answered from what has already been said, or it is
  a judgment/synthesis over it ("so which would you pick?", "summarise that"), or
  it needs no documents at all ("hi", "say that in French"). Retrieval would add
  nothing.
- ask_fresh: the conversation cannot answer it and rewriting will not help --
  it needs a new standalone question. Supply suggested_fresh_question.

Rules:
- If the new question ALREADY stands alone, return it UNCHANGED and set
  was_rewritten=false. Do not append context it does not need.
- Set topic_shift=true when the question moves to a different subject than the
  recent turns. On a topic shift you must NOT carry earlier subject matter into
  search_query -- that produces a worse query than the raw one.
- read_as: one short clause describing how you interpreted the request, in the
  user's terms. This is shown to the user, so it must be honest about judgment
  vs lookup.
- Never answer the question. Only route and rewrite it."""


@dataclass
class CondensedQuery:
    original: str
    search_query: str
    route: Route = "documents"
    was_rewritten: bool = False
    topic_shift: bool = False
    read_as: str = ""
    suggested_fresh_question: str = ""
    # True when condensation did not run or failed. The raw question is used, and
    # saying so beats letting a degraded path look like a chosen one.
    fallback: bool = False
    fallback_reason: str = ""

    @property
    def needs_retrieval(self) -> bool:
        return self.route in ("documents", "documents_and_conversation")

    @property
    def header(self) -> str:
        return ROUTE_HEADER[self.route]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["needs_retrieval"] = self.needs_retrieval
        d["header"] = self.header
        return d


def route_drift(routes: "list[str]") -> dict[str, Any]:
    """Is the conversation increasingly answering from itself?

    `conversation_only` is correct behaviour AND a new way to be confidently
    wrong: a turn that never touches a document cannot be checked against one.
    Every citation guard in this project is inert on that path.

    So the rate is tracked, and -- more importantly -- its TREND. A conversation
    that starts grounded and drifts toward self-reference is the degradation the
    whole provenance design exists to make visible, and it is invisible in a
    per-turn view: each individual turn looks like a reasonable judgment call.

    `drifting` compares the second half of the conversation to the first. It is a
    signal to surface, not a verdict -- on a short conversation the halves are
    tiny, so `sufficient` says whether the comparison means anything, the same
    discipline as the eval's rate floor.
    """
    n = len(routes)
    conv = sum(1 for r in routes if r == "conversation_only")
    half = n // 2
    first = routes[:half]
    second = routes[half:]

    def share(rs: list[str]) -> float | None:
        return (sum(1 for r in rs if r == "conversation_only") / len(rs)) if rs else None

    a, b = share(first), share(second)
    sufficient = n >= 6 and half >= 3
    return {
        "turns": n,
        "conversation_only": conv,
        "conversation_only_share": round(conv / n, 3) if n else None,
        "first_half_share": None if a is None else round(a, 3),
        "second_half_share": None if b is None else round(b, 3),
        "sufficient": sufficient,
        "drifting": bool(sufficient and a is not None and b is not None and b > a + 0.25),
        "note": "a turn routed conversation_only is unverifiable against documents: "
                "every citation check is inert on that path",
    }


def condense(
    question: str,
    history: str = "",
    *,
    role: str = "cheap",
) -> CondensedQuery:
    """Route and, if needed, rewrite the question.

    Runs on the CHEAP model. The task is classification plus a short rewrite, and
    spending the workhorse on it would put an avoidable LLM call in front of every
    query -- condensation sits inside TTFT (guide M14), so its latency is
    user-visible in a way retrieval's is not.

    FIRST TURN SHORT-CIRCUITS. With no history there is nothing to resolve, so a
    single-turn question skips the call entirely. That keeps the common case free
    and means condensation's cost only appears where it buys something.
    """
    if not history.strip():
        return CondensedQuery(
            original=question, search_query=question, route="documents",
            read_as="a direct question about your documents",
        )

    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"New question: {question}\n\n"
        "Route and rewrite it."
    )
    try:
        raw, _usage = gemini.generate(
            prompt, role=role, system=_SYSTEM, schema=_SCHEMA,
            max_output_tokens=512, stage="condense",
        )
        payload = json.loads(raw)
    except (limits.QuotaExhausted, gemini.EmptyResponse, json.JSONDecodeError) as exc:
        # DEGRADE TO THE RAW QUESTION, loudly. A failed rewrite must not become a
        # silent quality drop attributed to retrieval: the raw query is used, the
        # route defaults to `documents`, and the fallback is recorded so a bad
        # answer can be traced to condensation rather than blamed on the index.
        limits.report(
            limits.Degradation(
                stage="condense",
                cause="free_tier_quota_exhausted" if isinstance(exc, limits.QuotaExhausted)
                      else "condense_failed",
                impact="the question was searched as typed, so pronouns and "
                       "follow-up references were not resolved",
                fallback="raw question used, route defaulted to documents",
                remedy="retry, or ask the question in a standalone form",
                free_tier=isinstance(exc, limits.QuotaExhausted),
            )
        )
        return CondensedQuery(
            original=question, search_query=question, route="documents",
            fallback=True, fallback_reason=f"{type(exc).__name__}: {exc}"[:200],
            read_as="searched as typed (rewriting was unavailable)",
        )

    route: Route = payload.get("route", "documents")  # type: ignore[assignment]
    sq = (payload.get("search_query") or "").strip() or question

    # GUARD: on a declared topic shift, a rewrite that grew a lot has almost
    # certainly imported the previous subject. Prefer the raw question -- it is
    # the failure this layer is most likely to cause, so it gets a cheap check
    # rather than trust.
    if payload.get("topic_shift") and len(sq) > len(question) * 1.6:
        sq = question
        payload["was_rewritten"] = False
        payload["read_as"] = (payload.get("read_as") or "") + \
                             " (topic changed, so earlier context was not carried over)"

    return CondensedQuery(
        original=question,
        search_query=sq,
        route=route,
        was_rewritten=bool(payload.get("was_rewritten")),
        topic_shift=bool(payload.get("topic_shift")),
        read_as=(payload.get("read_as") or "").strip(),
        suggested_fresh_question=(payload.get("suggested_fresh_question") or "").strip(),
    )
