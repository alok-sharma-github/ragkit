"""
Free-tier quota handling and the degradation-notice channel.

YOUR REQUIREMENT: "any restriction or limitation should give user feedback that
this limitation is due to free tier gemini."

YOUR CRITIQUE OF MY FIRST ATTEMPT, which reshaped this file: a terminal warning
is the wrong artifact. It lives for one scroll; the index lives for weeks. Two
days later you are tuning RETRIEVE_K against a corpus you have forgotten is
mixed, and nothing in the index can tell you which chunks are which.

So a Degradation here has THREE destinations, and the code makes all three
mandatory rather than optional:

  1. the user, now        -> rendered in the CLI footer / Streamlit banner
  2. the artifact         -> written into per-chunk provenance and the ingest manifest
  3. the eval harness     -> which refuses comparisons it cannot interpret

Only (1) is what most projects build. (2) is what makes the defect queryable and
the repair resumable. (3) is what stops a diluted effect size from being read as
"the technique isn't worth it".

---------------------------------------------------------------------------
THE DISTINCTION THAT DRIVES THE RETRY POLICY

Free-tier limits come in two shapes and they need opposite handling:

  per-MINUTE (RPM/TPM)  -> transient. Sleeping clears it. Retry.
  per-DAY (RPD)         -> terminal for this session. No amount of sleeping
                           clears it. Retrying is just a slower failure.

Both arrive as the same `429 RESOURCE_EXHAUSTED`. The API does not reliably tell
you which one you hit. So the policy is: retry a bounded number of times with
backoff (clears the per-minute case), and if it still fails, treat it as terminal
and DEGRADE LOUDLY rather than hanging forever on a limit that resets tomorrow.

That is why QUOTA_MAX_RETRIES is 3 and not 50. A long retry loop against a daily
cap looks identical to a hung process, which is the worst failure mode for
something you are running interactively while learning.

The judge path is the deliberate exception - see `patient_retry`.
"""

from __future__ import annotations

import contextvars
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from . import config

T = TypeVar("T")


# --------------------------------------------------------------------------
# The notice
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Degradation:
    """One capability lost, with its cause and consequence named.

    Every field is required on purpose. A notice that says "quota exceeded" and
    stops has told the user nothing actionable: they cannot tell whether their
    answer is wrong, whether to re-run, or whether to wait a minute or a day.
    """

    stage: str          # "contextualize" | "caption_image" | "embed" | "judge" | ...
    cause: str          # short machine-ish reason: "free_tier_quota_exhausted"
    impact: str         # what is now WORSE, in the user's terms
    fallback: str       # what happened instead ("skipped", "un-prefixed", "waited")
    remedy: str         # what the user can actually do about it
    free_tier: bool = True  # was this caused by the free tier specifically?

    def render(self) -> str:
        tier = " (free-tier Gemini limit)" if self.free_tier else ""
        return (
            f"[{self.stage}]{tier} {self.impact}\n"
            f"    what happened instead: {self.fallback}\n"
            f"    what you can do:       {self.remedy}"
        )


@dataclass
class DegradationLog:
    """Collects notices for one operation, aggregating repeats.

    Aggregation matters: contextualising 300 chunks and hitting quota on 40 of
    them must produce ONE notice saying "40 of 300", not 40 identical lines. A
    flood of identical warnings is functionally the same as no warning - the
    reader scrolls past it.
    """

    notices: list[Degradation] = field(default_factory=list)
    counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    totals: Counter[str] = field(default_factory=Counter)

    def report(self, d: Degradation, *, n: int = 1) -> None:
        key = (d.stage, d.cause)
        if key not in self.counts:
            self.notices.append(d)
        self.counts[key] += n

    def seen(self, stage: str, *, n: int = 1) -> None:
        """Record an ATTEMPT, so a notice can say '40 of 300' rather than '40'."""
        self.totals[stage] += n

    def __bool__(self) -> bool:
        return bool(self.notices)

    def render_cli(self) -> str:
        # ASCII only, deliberately. The default Windows console codepage is
        # cp1252 and cannot encode box-drawing characters or bullets - a
        # UnicodeEncodeError while PRINTING A WARNING would swallow the very
        # message the user needs. Found this the honest way, by hitting it.
        if not self.notices:
            return ""
        lines = ["", "-" * 72, "DEGRADED - this run did not do everything it was asked to:"]
        for d in self.notices:
            n = self.counts[(d.stage, d.cause)]
            total = self.totals.get(d.stage)
            scope = f"{n} of {total}" if total else f"{n}"
            lines.append(f"  * {scope} x {d.render()}")
        lines.append("-" * 72)
        return "\n".join(lines)

    def to_dicts(self) -> list[dict[str, Any]]:
        """For the Streamlit banner and the ingest manifest."""
        out = []
        for d in self.notices:
            row = asdict(d)
            row["count"] = self.counts[(d.stage, d.cause)]
            row["attempted"] = self.totals.get(d.stage)
            out.append(row)
        return out

    def write_manifest(self, path: Path | None = None, **extra: Any) -> Path:
        """Destination (2): the artifact.

        This is the part that outlives the terminal. Without it, 'why is recall
        bad on the Q4 documents' is a re-derivation instead of a lookup.
        """
        path = path or config.INGEST_MANIFEST_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"degradations": self.to_dicts(), **extra}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def uniform(self) -> bool:
        """True if nothing degraded - i.e. this index/run is interpretable.

        `EVAL_REFUSE_MIXED_PROVENANCE` and
        `CI_BASELINE_REQUIRES_UNIFORM_PROVENANCE` are enforced against this.
        """
        return not self.notices


# Ambient log, so a function five frames deep can report a degradation without
# every intermediate signature growing a `log` parameter. `collect()` scopes it.
_current: contextvars.ContextVar[DegradationLog | None] = contextvars.ContextVar(
    "ragkit_degradation_log", default=None
)


def current_log() -> DegradationLog:
    log = _current.get()
    if log is None:  # unscoped call - give it somewhere to go rather than crash
        log = DegradationLog()
        _current.set(log)
    return log


class collect:
    """Scope a fresh log:  `with collect() as log: ...`  then read `log`."""

    def __init__(self) -> None:
        self.log = DegradationLog()
        self._token: contextvars.Token[DegradationLog | None] | None = None

    def __enter__(self) -> DegradationLog:
        self._token = _current.set(self.log)
        return self.log

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _current.reset(self._token)


def report(d: Degradation, *, n: int = 1) -> None:
    current_log().report(d, n=n)


def seen(stage: str, *, n: int = 1) -> None:
    current_log().seen(stage, n=n)


# --------------------------------------------------------------------------
# Detecting a quota error
# --------------------------------------------------------------------------


class QuotaExhausted(RuntimeError):
    """Raised when retries are spent. Callers decide whether to degrade or abort."""

    def __init__(self, stage: str, attempts: int, original: BaseException | None = None):
        super().__init__(
            f"{stage}: free-tier Gemini quota still exhausted after {attempts} attempts"
        )
        self.stage = stage
        self.attempts = attempts
        self.original = original


class _FakeQuotaError(RuntimeError):
    """Injected by RAGKIT_FAKE_QUOTA_EXHAUSTED so the degraded path is testable."""


def is_quota_error(exc: BaseException) -> bool:
    """Detect 429 / RESOURCE_EXHAUSTED defensively.

    Deliberately duck-typed rather than catching a specific SDK exception class.
    Reason: the class names and module paths in google-genai have moved before,
    and a rename would silently turn 'degrade with a clear message' into an
    unhandled crash. Matching on the status code and the documented status
    string survives that.
    """
    if isinstance(exc, _FakeQuotaError):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    blob = f"{type(exc).__name__} {exc}".upper()
    return "RESOURCE_EXHAUSTED" in blob or "429" in blob and "QUOTA" in blob


def retry_after_seconds(exc: BaseException, attempt: int) -> float:
    """Honour a server-provided retry delay when there is one, else back off.

    Gemini may return RetryInfo in the error details. Using it is strictly better
    than guessing - guessing too short wastes an attempt against a limit that has
    not reset, and guessing too long wastes wall-clock you are sitting through.
    """
    for attr in ("details", "response_json", "response"):
        blob = getattr(exc, attr, None)
        if not blob:
            continue
        text = json.dumps(blob, default=str) if not isinstance(blob, str) else blob
        # e.g. "retryDelay": "17s"
        import re

        m = re.search(r'"retry[_-]?[Dd]elay"\s*:\s*"?(\d+(?:\.\d+)?)s?"?', text)
        if m:
            return float(m.group(1))
    # Exponential backoff with jitter. Jitter is not decoration: without it,
    # a batch of 16 parallel embed calls all retry in lockstep and re-trip the
    # same per-minute limit together.
    base = config.QUOTA_BACKOFF_BASE_SECONDS * (2 ** attempt)
    return base * (0.5 + random.random())


# --------------------------------------------------------------------------
# The two retry policies
# --------------------------------------------------------------------------


def guard(
    fn: Callable[[], T],
    *,
    stage: str,
    max_retries: int | None = None,
) -> T:
    """Request-path policy: bounded retries, then raise QuotaExhausted.

    Bounded on purpose. See the module docstring: an unbounded loop against a
    per-DAY limit is indistinguishable from a hang.
    """
    attempts = config.QUOTA_MAX_RETRIES if max_retries is None else max_retries
    last: BaseException | None = None
    for attempt in range(attempts):
        if config.FAKE_QUOTA_EXHAUSTED:
            last = _FakeQuotaError("RAGKIT_FAKE_QUOTA_EXHAUSTED=1")
            break
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not quota
            if not is_quota_error(exc):
                raise
            last = exc
            if attempt < attempts - 1:
                time.sleep(retry_after_seconds(exc, attempt))
    raise QuotaExhausted(stage, attempts, last)


def patient_retry(
    fn: Callable[[], T],
    *,
    stage: str = "judge",
    on_wait: Callable[[int, float], None] | None = None,
) -> T:
    """Offline-batch policy: many retries, long backoff, NEVER a model swap.

    This exists because of your correction. The eval judge must not fall back to
    the workhorse model on quota exhaustion:

      - the fallback target IS the generator  -> self-preference bias
      - and it is weaker                      -> it cannot grade an answer it
                                                 could not have produced

    Two distinct failure modes stacking, and both push the score in the
    flattering direction. Meanwhile judging has no latency SLO at all. So the
    judge waits. A slow grade is fine; a wrong grade is not.
    """
    last: BaseException | None = None
    for attempt in range(config.JUDGE_OFFLINE_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not is_quota_error(exc):
                raise
            last = exc
            delay = max(
                retry_after_seconds(exc, min(attempt, 4)),
                config.JUDGE_OFFLINE_BACKOFF_SECONDS,
            )
            if on_wait:
                on_wait(attempt + 1, delay)
            time.sleep(delay)
    raise QuotaExhausted(stage, config.JUDGE_OFFLINE_MAX_RETRIES, last)


# --------------------------------------------------------------------------
# Prebuilt notices
# --------------------------------------------------------------------------
#
# Centralised so the wording is consistent and each one actually names an impact
# and a remedy. Writing these as literals at call sites is how you end up with
# "warning: quota" in six places and no way to explain any of them.


def contextualize_skipped(count: int, total: int) -> Degradation:
    return Degradation(
        stage="contextualize",
        cause="free_tier_quota_exhausted",
        impact=(
            f"{count} of {total} chunks were indexed WITHOUT a contextual prefix. "
            "Those chunks are systematically harder to retrieve - a chunk that "
            "says 'revenue grew 12%' without naming the company or quarter will "
            "lose to a prefixed chunk regardless of which one is actually relevant"
        ),
        fallback="indexed un-prefixed, and flagged has_contextual_prefix=False per chunk",
        remedy=(
            "the index is MIXED and A/B results over it are not interpretable. "
            "Re-run ingest when quota resets to backfill only the missing chunks "
            "(resumable via the per-chunk flag), or enable billing"
        ),
    )


def caption_skipped(count: int, total: int) -> Degradation:
    return Degradation(
        stage="caption_image",
        cause="free_tier_quota_exhausted",
        impact=(
            f"{count} of {total} images have no caption, so they are UNRETRIEVABLE. "
            "An image with no text representation cannot match any query"
        ),
        fallback="image stored and displayable, but absent from the retrieval index",
        remedy="re-run ingest when quota resets; only uncaptioned images are reprocessed",
    )


def embed_failed(count: int, total: int) -> Degradation:
    return Degradation(
        stage="embed",
        cause="free_tier_quota_exhausted",
        impact=(
            f"{count} of {total} chunks have no embedding and are absent from the "
            "dense index entirely. Recall is bounded below by this, and the gap is "
            "a contiguous tail of ingest order, not a random sample"
        ),
        fallback="chunk kept for BM25/sparse retrieval only",
        remedy="re-run ingest to embed the remainder; the on-disk cache means finished work is not repeated",
    )


def judge_waiting(attempt: int, delay: float) -> Degradation:
    return Degradation(
        stage="judge",
        cause="free_tier_quota_exhausted",
        impact=f"evaluation is slower (waiting {delay:.0f}s, attempt {attempt})",
        fallback="WAITING - deliberately not falling back to the generator model",
        remedy=(
            "nothing; this is the correct behaviour. Swapping the judge to the "
            "workhorse would make the score self-graded and unusable"
        ),
    )


def judge_unvalidated(kappa: float | None, model: str) -> Degradation:
    got = "never measured" if kappa is None else f"kappa={kappa:.2f}"
    return Degradation(
        stage="judge",
        cause="judge_not_validated",
        impact=(
            f"judged metrics are WITHHELD: judge '{model}' has not shown adequate "
            f"agreement with your hand labels ({got}, need "
            f">= {config.JUDGE_MIN_KAPPA})"
        ),
        fallback="retrieval-tier metrics reported; generation-tier metrics suppressed",
        remedy=(
            f"hand-label {config.JUDGE_VALIDATION_SAMPLE} stratified items and run "
            "the judge validation. An unvalidated judge is not a weak signal, it is "
            "an unknown one"
        ),
        free_tier=False,
    )
