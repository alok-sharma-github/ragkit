"""The spend ceiling.

Until now there was none. `gemini.Usage` recorded what each call cost and nothing
aggregated or persisted it -- survivable only while the key was free-tier,
because the quota wall failed CLOSED: you hit a 429 and it cost nothing. Attaching
a card removes the wall, so the identical code now fails OPEN, into a bill.

TWO LAYERS, AND THEY FAIL IN OPPOSITE DIRECTIONS. That is the whole design, so it
is worth being explicit about which one is load-bearing:

  per-operation   Checked BEFORE the first paid call, from the size of the work in
                  front of it. The dangerous case -- someone uploading 500 PDFs --
                  is ONE operation, and its cost is knowable in advance: the
                  chunks exist, their token counts exist. So this refuses the
                  whole operation up front instead of discovering the problem
                  halfway through a paid run. Immune to restarts, because it
                  remembers nothing.

  cumulative day  A file counter, and on ephemeral disk it RESETS ON RESTART, so
                  it fails OPEN across restarts. Kept anyway -- it catches slow
                  drift within a process lifetime -- but it is explicitly NOT the
                  thing standing between a stranger and a large bill.

A ceiling that is never seen to refuse is not known to work, so `check_operation`
is exercised by the tests from the exceeding side, not by reading it.

WHY REFUSAL IS AN EXCEPTION AND NOT A RETURN VALUE. Every caller of a paid route
would have to remember to check a boolean, and the one that forgets spends the
money anyway. Raising means the default behaviour of forgetting is "the operation
stops", which is the direction this has to fail in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from . import config, limits


class BudgetExceeded(RuntimeError):
    """Refused BEFORE spending anything. Carries the numbers, not just a flag."""

    def __init__(
        self,
        *,
        stage: str,
        wanted: int,
        cap: int,
        layer: str,
        spent_today: int = 0,
    ) -> None:
        self.stage = stage
        self.wanted = wanted
        self.cap = cap
        self.layer = layer
        self.spent_today = spent_today
        super().__init__(
            f"{stage}: refused before spending -- {wanted:,} tokens exceeds the "
            f"{layer} ceiling of {cap:,}"
            + (f" (already spent {spent_today:,} today)" if spent_today else "")
        )


# ---------------------------------------------------------------------------
# Wording. Three audiences, and the honest cause differs for each.
# ---------------------------------------------------------------------------


def _whose_limit() -> str:
    # "spending limit", not "ingest limit". This cap guards THREE routes -- embed,
    # generate and caption -- so calling it an ingest limit was accurate only on
    # one of them, and a refusal on an answer told the reader their ingest was too
    # large. A small slip, and the same species as everything else in this file:
    # a message describing a neighbouring thing rather than the actual one.
    kind = config.DEPLOYMENT_KIND
    if kind == "customer":
        label = config.DEPLOYMENT_LABEL
        return f"the spending limit configured for your account{f' ({label})' if label else ''}"
    if kind == "demo":
        return "the spending limit on this shared public demo"
    return "the spending limit configured for this deployment"


def _remedy(layer: str) -> str:
    kind = config.DEPLOYMENT_KIND
    # OPERATOR ADVICE ONLY REACHES OPERATORS. Naming an env var to someone who
    # cannot set one is the same defect as naming the wrong cause: the sentence is
    # true and useless. Only `internal` is read by whoever owns the config.
    # ROUTE-NEUTRAL WORDING, enforced by scripts/audit_guards.py.
    #
    # These said "split the upload" and "...to ingest documents". Both were true
    # when the ceiling guarded only embedding, and both became false the moment it
    # was widened to generation: a refused ANSWER told the reader to split an
    # upload they never made. Widening a guard invalidates every string written
    # while it was narrow, which is why the audit greps these for words that name
    # a single route rather than trusting them to be reviewed.
    if kind == "customer":
        if layer == "daily":
            return (
                "wait for the daily limit to reset, or contact us to raise the "
                "limit on your account"
            )
        return (
            "send fewer documents at a time, or contact us to raise the limit on "
            "your account"
        )
    if kind == "demo":
        return (
            "this public demo has a shared daily budget -- try again later, or "
            "clone the repo and run it against your own key"
        )
    knob = (
        "RAGKIT_MAX_OPERATION_TOKENS" if layer == "per-operation"
        else "RAGKIT_DAILY_TOKEN_CAP"
    )
    return f"raise {knob}, or split the work into smaller batches"


# ---------------------------------------------------------------------------
# The daily ledger. Durability is a known hole -- see the module docstring.
# ---------------------------------------------------------------------------


def _today() -> str:
    return date.today().isoformat()


def _load() -> dict[str, Any]:
    p = config.SPEND_LEDGER_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:  # noqa: BLE001 -- a corrupt ledger must not stop work
        # Fails OPEN deliberately, and loudly. The alternative -- refusing all
        # paid work because a counter file is malformed -- turns a bookkeeping
        # problem into an outage.
        return {}


def _save(d: dict[str, Any]) -> None:
    p = config.SPEND_LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def spent_today() -> int:
    return int(_load().get(_today(), {}).get("tokens", 0))


def remaining_today() -> int:
    return max(0, config.GEMINI_DAILY_TOKEN_CAP - spent_today())


def record(*, prompt_tokens: int = 0, output_tokens: int = 0, stage: str = "") -> None:
    """Add a completed call's cost to today's total.

    Called AFTER the fact, so it can only inform the next decision -- which is
    why the per-operation check does not depend on it.
    """
    n = int(prompt_tokens) + int(output_tokens)
    if n <= 0:
        return
    d = _load()
    day = d.setdefault(_today(), {"tokens": 0, "calls": 0, "by_stage": {}})
    day["tokens"] = int(day.get("tokens", 0)) + n
    day["calls"] = int(day.get("calls", 0)) + 1
    if stage:
        by = day.setdefault("by_stage", {})
        by[stage] = int(by.get(stage, 0)) + n
    # Keep the file small: only the last 30 days are of any use.
    for k in sorted(d)[:-30]:
        d.pop(k, None)
    _save(d)


# ---------------------------------------------------------------------------
# The check that matters
# ---------------------------------------------------------------------------


def check_operation(
    estimated_tokens: int,
    *,
    stage: str,
    n_items: int | None = None,
) -> None:
    """Refuse an over-budget operation BEFORE any paid call is made.

    `estimated_tokens` must count only work that will actually be BILLED -- cache
    hits cost nothing, and a ceiling that refused a fully-cached re-ingest would
    be measuring the wrong thing.

    Raises BudgetExceeded. Also emits a Degradation, so the refusal reaches the
    CLI footer and the UI banner rather than only an exception trace.
    """
    est = int(estimated_tokens)
    if est <= 0:
        return

    cap = config.GEMINI_MAX_OPERATION_TOKENS
    if est > cap:
        _report(stage, est, cap, "per-operation", n_items)
        raise BudgetExceeded(
            stage=stage, wanted=est, cap=cap, layer="per-operation"
        )

    already = spent_today()
    daily = config.GEMINI_DAILY_TOKEN_CAP
    if already + est > daily:
        _report(stage, est, daily, "daily", n_items, already)
        raise BudgetExceeded(
            stage=stage, wanted=est, cap=daily, layer="daily", spent_today=already
        )


def _report(
    stage: str,
    est: int,
    cap: int,
    layer: str,
    n_items: int | None,
    already: int = 0,
) -> None:
    scale = f"{n_items} item(s), " if n_items else ""
    limits.report(
        limits.Degradation(
            stage=stage,
            cause=f"{layer.replace('-', '_')}_budget_exceeded",
            impact=(
                f"this work was NOT started: {scale}about {est:,} tokens would be "
                f"billed, which exceeds {_whose_limit()} of {cap:,} tokens"
                + (f" ({already:,} already spent today)" if already else "")
            ),
            fallback="refused before any paid request was sent, so it cost nothing",
            remedy=_remedy(layer),
            # Our own ceiling, not a Gemini one. Mislabelling it would send the
            # reader to Google's quota page for a limit we set ourselves.
            free_tier=False,
        )
    )


# ---------------------------------------------------------------------------
# Reporting surface, so the ceiling is visible before it bites
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    deployment_kind: str
    per_operation_cap: int
    daily_cap: int
    spent_today: int
    remaining_today: int
    durable: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "deployment_kind": self.deployment_kind,
            "per_operation_cap": self.per_operation_cap,
            "daily_cap": self.daily_cap,
            "spent_today": self.spent_today,
            "remaining_today": self.remaining_today,
            # Stated, not implied. A caller reading `spent_today` deserves to
            # know the number resets when the container does.
            "daily_counter_durable": self.durable,
            "note": (
                "the daily counter is a file on ephemeral disk and resets on "
                "restart; the per-operation cap does not depend on it"
            ),
        }


def snapshot() -> Snapshot:
    return Snapshot(
        deployment_kind=config.DEPLOYMENT_KIND,
        per_operation_cap=config.GEMINI_MAX_OPERATION_TOKENS,
        daily_cap=config.GEMINI_DAILY_TOKEN_CAP,
        spent_today=spent_today(),
        remaining_today=remaining_today(),
        durable=False,
    )
