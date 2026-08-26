"""
The only file in this project that talks to the Gemini API.

WHY THE BOUNDARY: `embed_content`'s task type moved from a `task_type` parameter
to a text prompt prefix between gemini-embedding-001 and -2, and
`gemini-3.1-flash-lite-preview` was retired in May 2026. Both are the same class
of event, and both should cost one file's worth of editing. Nothing else in
ragkit imports google.genai.

---------------------------------------------------------------------------
THE EMBEDDING CACHE - built from your answer

You worked out that a cache keyed on sha256(chunk_text) silently returns
768-dim vectors to a 3072-dim experiment, and that the result "768 and 3072
perform identically" is worse than a crash because it is the answer you were
half expecting. Everything below follows from that.

  1. THE KEY NAMES EVERY INPUT.  embed(text, model, dim, prefix) takes four
     arguments, so the key has four parts plus a format version you bump by
     hand. A key naming fewer inputs than the function takes is not a cache; it
     is a machine for confidently answering questions you did not ask.

  2. THE ENTRY RE-STATES ITS OWN KEY, and read verifies it. Your proposed guard
     -- assert len(vec) == EMBED_DIM -- catches the dimension case but NOT the
     model or prefix case: gemini-embedding-001@768 and gemini-embedding-2@768
     are the same length and live in incompatible spaces. So the stored metadata
     is checked field by field, and a mismatch is a MISS, never a silent hit.

  3. THE STATS ARE REPORTED. Your tell -- "notice when expensive things are
     suddenly free" -- becomes an artifact instead of an instinct. Every call
     returns hits/misses/api_calls, and the experiment scripts print them. A run
     claiming to compare two embedding configs while reporting 100% cache hits
     is visibly broken on its own output.

  4. CACHE AT FULL DIMENSION, TRUNCATE LOCALLY. Embedding is billed per INPUT
     token; output_dimensionality is free. Matryoshka orders dimensions by
     importance, so 768 is a prefix of 3072. Cache 3072 once, slice and
     re-normalise for anything smaller -- the whole dimension sweep costs one
     embedding pass instead of one per dimension. On a free key that is the
     difference between runnable and not.

     That is a CLAIM, so `verify_truncation()` checks it against the API instead
     of assuming it. If local truncation does not match API truncation, I want
     to know before the experiment, not after.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from . import config, limits

# Bump when the cache PAYLOAD format changes (not when a key input changes --
# those are already in the key). Old entries then miss instead of deserialising
# into something subtly wrong.
CACHE_VERSION = 1

Role = Literal["workhorse", "cheap", "judge", "embedding", "embedding_multimodal"]
EmbedKind = Literal["document", "query"]

_ROLES: tuple[str, ...] = ("workhorse", "cheap", "judge", "embedding", "embedding_multimodal")


class EmptyResponse(RuntimeError):
    """The model returned no text. Raised rather than returning "".

    An empty string is indistinguishable from a legitimately empty answer, so a
    caller three layers up writes it into an index or a citation. See generate().
    """


class BatchContract(RuntimeError):
    """The API returned a different number of embeddings than we sent.

    Its own section because this bug bit me in this very file: I zipped request
    items against response embeddings, and gemini-embedding-2 returns ONE
    embedding for a multi-item batch. zip() truncates silently, so the unmatched
    rows stayed zero vectors -- present in the index, scoring 0.0 against every
    query, invisible. Reported success. Same class of defect as the cache bug you
    diagnosed, one file later, in my code.
    """


# --------------------------------------------------------------------------
# Client + model resolution
# --------------------------------------------------------------------------

_client: Any = None


def client() -> Any:
    """Lazily construct the SDK client so importing ragkit needs no API key."""
    global _client
    if _client is None:
        from google import genai  # imported here, not at module top, on purpose

        _client = genai.Client(api_key=config.api_key())
    return _client


def available_models() -> list[str]:
    """Bare model IDs your key can actually reach ('models/x' -> 'x')."""
    out: list[str] = []
    for m in client().models.list():
        name = getattr(m, "name", "") or ""
        out.append(name.split("/", 1)[-1] if name.startswith("models/") else name)
    return sorted(n for n in out if n)


def resolve_models(*, force: bool = False, verbose: bool = False) -> dict[str, str]:
    """Pick the first reachable candidate per role and persist the choice.

    This exists because I asserted four model IDs came from live docs and only
    two of them did. The fix is not better memory, it is asking your key what it
    can reach -- and then RECORDING the answer, because an eval baseline that
    does not say which model produced it cannot be compared to anything.
    """
    if not force and config.RESOLVED_MODELS_PATH.exists():
        return json.loads(config.RESOLVED_MODELS_PATH.read_text(encoding="utf-8"))["resolved"]

    have = set(available_models())
    resolved: dict[str, str] = {}
    notes: dict[str, Any] = {}

    for role in _ROLES:
        candidates: tuple[str, ...] = getattr(config.MODELS, role)
        pick = next((c for c in candidates if c in have), None)
        if pick is None:
            # Prefix match rescue: providers often expose 'gemini-3.7-flash-001'
            # style variants. Better than failing outright, but recorded as an
            # inexact resolution so it is not mistaken for a clean hit.
            for c in candidates:
                near = sorted(n for n in have if n.startswith(c))
                if near:
                    pick = near[0]
                    notes[role] = f"inexact: wanted {c}, using {pick}"
                    break
        if pick is None:
            raise RuntimeError(
                f"No candidate reachable for role '{role}'.\n"
                f"  wanted (in order): {list(candidates)}\n"
                f"  your key exposes:  {sorted(have)[:40]}\n"
                f"Update config.Models.{role}."
            )
        if pick != candidates[0] and role not in notes:
            notes[role] = f"fell back: first choice {candidates[0]} unavailable"
        resolved[role] = pick

    # The judge must not be the generator -- self-preference bias. Resolution can
    # collide the two if the judge candidates are all unavailable, so it is
    # checked here rather than assumed from config.
    if resolved["judge"] == resolved["workhorse"]:
        notes["judge_collision"] = (
            f"judge resolved to the same model as workhorse ({resolved['judge']}). "
            "Judged metrics will be self-graded and are not trustworthy."
        )

    config.RESOLVED_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESOLVED_MODELS_PATH.write_text(
        json.dumps({"resolved": resolved, "notes": notes, "available": sorted(have)}, indent=2),
        encoding="utf-8",
    )
    if verbose:
        for role, mid in resolved.items():
            flag = f"   <- {notes[role]}" if role in notes else ""
            print(f"  {role:10s} {mid}{flag}")
        if "judge_collision" in notes:
            print(f"  WARNING: {notes['judge_collision']}")
    return resolved


def model_for(role: Role) -> str:
    return resolve_models()[role]


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Approximate token count, computed LOCALLY.

    Deliberately not the API's count_tokens: the chunker calls this per candidate
    boundary, which would be thousands of network round trips against a free-tier
    rate limit. tiktoken's cl100k is not Gemini's tokenizer, so this is an
    ESTIMATE -- fine for budgeting a context window, wrong for billing. Where the
    exact number matters (verifying we are under EMBED_MAX_INPUT_TOKENS before a
    call that would 400), we leave headroom rather than pretend to precision.
    """
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 -- estimation must never be the thing that fails
        return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class Usage:
    """What a call cost. Collected so Session 11's cost waterfall is real data."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # implicit-cache hits; the proof that caching fired
    # WHY GENERATION STOPPED. This was read only on the empty-text path and
    # thrown away otherwise -- so the lesson ("an empty response must carry its
    # finish_reason, never look like a legitimate empty answer") was learned for
    # text == "" and not for the case one step along: text that is non-empty and
    # CUT OFF. A truncated JSON answer returned normally, failed json.loads with
    # "Unterminated string at char 6139", and reported itself to the user as
    # "structured output did not parse" -- blaming the schema for our own output
    # budget. Same defect, and the earlier fix stopped just short of it.
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        return "MAX_TOKENS" in self.finish_reason.upper()

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        # A truncation anywhere in an aggregate makes the aggregate truncated.
        # Summing costs but dropping the one flag that says "this output is
        # incomplete" is how the fact gets lost on the way up.
        if other.truncated and not self.truncated:
            self.finish_reason = other.finish_reason


def _usage_of(response: Any) -> Usage:
    """Read usage defensively -- field names on this object have moved before."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return Usage()

    def g(*names: str) -> int:
        for n in names:
            v = getattr(um, n, None)
            if isinstance(v, int):
                return v
        return 0

    return Usage(
        prompt_tokens=g("prompt_token_count", "input_token_count"),
        output_tokens=g("candidates_token_count", "output_token_count"),
        cached_tokens=g("cached_content_token_count", "total_cached_tokens"),
    )


def generate(
    prompt: str,
    *,
    role: Role = "workhorse",
    system: str | None = None,
    schema: Any = None,
    images: Sequence[tuple[bytes, str]] = (),
    max_output_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    stage: str = "generate",
    patient: bool = False,
) -> tuple[str, Usage]:
    """One text completion. Returns (text, usage).

    `temperature=0.0` by default and that is a measurement decision, not a style
    preference: an eval suite whose generator samples differently each run cannot
    attribute a metric change to the change you made.

    THINKING IS OFF BY DEFAULT, and finding out why cost a debugging round.
    Gemini 3.x flash thinks by default, and thinking tokens are charged against
    `max_output_tokens`. A call with max_output_tokens=16 asking for the literal
    text "OK" returned an EMPTY STRING: finish_reason=MAX_TOKENS,
    thoughts_token_count=11, candidates_token_count=1. Eleven of sixteen tokens
    went to reasoning about a two-character answer, and the visible text was
    truncated to nothing.

    That is the failure mode you named earlier -- quietly wrong beats loudly
    broken only for whoever is not debugging it. So: thinking off unless asked
    for, and an empty response RAISES with its finish_reason rather than
    returning "" for a caller to mistake for a legitimately empty answer.

    Turn it on for genuinely hard work (multi-hop synthesis, judge verdicts);
    leave it off for extraction, classification, captioning and routing, where
    it buys nothing and eats the output budget.

    `patient=True` switches to the never-swap-the-model retry policy. The judge
    uses it. See limits.patient_retry.
    """
    from google.genai import types

    parts: list[Any] = []
    for data, mime in images:
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(types.Part.from_text(text=prompt))

    model = model_for(role)
    cfg_kwargs: dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if thinking:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    elif supports_thinking_disabled(model):
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    # else: omit the field. The model cannot be told not to think, so asking it
    # to is a 400. Omitting is the only working way to say "do not think much",
    # and it means max_output_tokens must leave room for thinking tokens -- which
    # is exactly the trap that made generate() return "" on a 16-token budget.
    if system:
        cfg_kwargs["system_instruction"] = system
    if schema is not None:
        # Structured output. Used for metadata-filter extraction (M1) and the
        # judge's verdicts (M9) -- anywhere a parsed field beats parsing prose.
        cfg_kwargs["response_mime_type"] = "application/json"
        cfg_kwargs["response_schema"] = schema

    def call() -> Any:
        return client().models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )

    if patient:
        response = limits.patient_retry(
            call,
            stage=stage,
            on_wait=lambda a, d: limits.report(limits.judge_waiting(a, d)),
        )
    else:
        response = limits.guard(call, stage=stage)

    usage = _usage_of(response)
    text = getattr(response, "text", None) or ""
    # Read ALWAYS, not only when the text is empty -- see Usage.finish_reason.
    cands = getattr(response, "candidates", None) or []
    reason = str(getattr(cands[0], "finish_reason", "unknown")) if cands else "unknown"
    usage.finish_reason = reason
    if not text.strip():
        raise EmptyResponse(
            f"{stage}: {model} returned no text (finish_reason={reason}, "
            f"thinking={'on' if thinking else 'off'}, "
            f"max_output_tokens={max_output_tokens}, "
            f"prompt_tokens={usage.prompt_tokens}, output_tokens={usage.output_tokens}). "
            "If finish_reason is MAX_TOKENS, the output budget was consumed -- raise "
            "max_output_tokens, or check that thinking is not eating it."
        )
    return text, usage


_thinking_disable_support: dict[str, bool] = {}


def supports_thinking_disabled(model: str) -> bool:
    """Can this model be told NOT to think? Measured, once, per model.

    MEASURED BECAUSE I ASSUMED IT AND WAS WRONG. `gemini-3.5-flash-lite` returns
    400 INVALID_ARGUMENT for thinking_budget=0, while accepting -1 (dynamic) or
    the field omitted entirely. Since generate() sent 0 for every call with
    thinking=False, the ENTIRE `cheap` model tier was unusable from the moment it
    was written -- and nothing noticed, because nothing called it until the
    condensation layer did. A tier that exists to save latency and quota, dead on
    arrival, with no test exercising it.

    Same pattern as supports_batching(): one cheap probe, cached for the process,
    rather than a hardcoded table that goes stale as the model line churns.

    CONSEQUENCE WORTH KNOWING: on a model that cannot disable thinking, the
    "cheap" tier still spends thinking tokens. It is cheaper per token, not
    thinking-free -- so the cascade's saving is smaller than the price list
    suggests.
    """
    if model in _thinking_disable_support:
        return _thinking_disable_support[model]
    from google.genai import types

    try:
        client().models.generate_content(
            model=model,
            contents=[types.Part.from_text(text="ok")],
            config=types.GenerateContentConfig(
                max_output_tokens=16,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        _thinking_disable_support[model] = True
    except Exception:  # noqa: BLE001 -- any rejection means "cannot disable"
        _thinking_disable_support[model] = False
    return _thinking_disable_support[model]


def caption_cache_key(image_bytes: bytes, context: str = "") -> str:
    """Cache key for one caption: image bytes + model + prompt version + context."""
    model = resolve_models()["workhorse"]
    blob = json.dumps(
        {
            "v": config.CAPTION_PROMPT_VERSION,
            "model": model,
            "context": context,
            "image_sha": hashlib.sha256(image_bytes).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _caption_path(key: str) -> Path:
    return config.CAPTION_CACHE / key[:2] / f"{key}.json"


def caption_image(
    image_bytes: bytes,
    mime_type: str,
    *,
    context: str = "",
) -> tuple[str, Usage]:
    """Describe an image so it becomes RETRIEVABLE.

    The caption is what goes in the index; the image is what you show the user.
    An image with no text representation cannot match any query -- which is why
    a skipped caption is a retrieval failure, not a cosmetic one, and why
    limits.caption_skipped says exactly that.

    The prompt asks for identifiers verbatim on purpose: model numbers, error
    codes and axis labels are precisely what BM25 will later match on (M2), and
    a caption that paraphrases them ("an error code") destroys the only signal
    that made the image findable.
    """
    prompt = (
        "Describe this image so that someone searching a document collection could "
        "find it from a text query.\n"
        "- State what kind of image it is (chart, diagram, screenshot, table, photo).\n"
        "- Transcribe ALL text, labels, axis names, legends and numbers VERBATIM. "
        "Do not paraphrase identifiers, error codes, part numbers or version strings.\n"
        "- If it is a chart, state what is plotted against what, and the trend.\n"
        "- Do not speculate about anything not visible.\n"
    )
    if context:
        prompt += f"\nSurrounding document context:\n{context}\n"

    # CACHED, because the call is non-deterministic and the index must be
    # reproducible. The stored entry carries its own provenance (model, prompt
    # version) so a caption can never be mistaken for document text later.
    key = caption_cache_key(image_bytes, context)
    path = _caption_path(key)
    if path.exists():
        try:
            payload = json.loads(path.read_text("utf-8"))
            return payload["caption"], Usage()      # no tokens spent on a hit
        except Exception:  # noqa: BLE001 -- a corrupt entry is a miss, not a crash
            pass

    text, usage = generate(
        prompt,
        role="workhorse",
        images=[(image_bytes, mime_type)],
        max_output_tokens=600,
        stage="caption_image",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "caption": text,
                "model": resolve_models()["workhorse"],
                "prompt_version": config.CAPTION_PROMPT_VERSION,
                "provenance": "model_generated",
                "image_sha": hashlib.sha256(image_bytes).hexdigest(),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return text, usage


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


@dataclass
class EmbedStats:
    """Destination (3) of your 'expensive things suddenly free' tell.

    Printed by every experiment. A dimension sweep that reports api_calls=0 on
    its second config did not run that config.
    """

    hits: int = 0
    misses: int = 0
    api_calls: int = 0
    rejected: int = 0  # cache entries found but REFUSED on metadata mismatch
    failed: int = 0    # never embedded: quota died
    prompt_tokens: int = 0

    def render(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        extra = f", rejected={self.rejected}" if self.rejected else ""
        extra += f", FAILED={self.failed}" if self.failed else ""
        return (
            f"embed: {total} texts | cache {self.hits} hit / {self.misses} miss "
            f"({rate:.0f}% hit) | {self.api_calls} API calls{extra}"
        )


@dataclass(frozen=True)
class TaskScheme:
    """How this model is told 'document' vs 'query'.

    Asymmetric embedding: a question and the passage answering it do not look
    alike. Embedding both identically asks the model to cluster questions near
    questions, which is the opposite of what retrieval needs.

    The two models express that through different APIs:
      gemini-embedding-2   -> a text prefix on the content
      gemini-embedding-001 -> a task_type parameter
    `label` goes into the cache key, because a vector produced under one scheme
    is not interchangeable with one produced under the other even at identical
    model and dimension -- and both are the same length, so a length check
    cannot catch the difference.
    """

    prefix: str
    task_type: str | None
    label: str

    def payload(self, text: str) -> str:
        return self.prefix + text


def task_scheme(model: str, kind: EmbedKind) -> TaskScheme:
    doc = kind == "document"
    if model.startswith("gemini-embedding-001"):
        tt = config.EMBED_TASK_TYPE_DOCUMENT if doc else config.EMBED_TASK_TYPE_QUERY
        return TaskScheme(prefix="", task_type=tt, label=f"task_type={tt}")
    prefix = config.EMBED_PREFIX_DOCUMENT if doc else config.EMBED_PREFIX_QUERY
    return TaskScheme(prefix=prefix, task_type=None, label=f"prefix={prefix.strip()}")


def max_input_tokens(model: str) -> int:
    for known, limit in config.EMBED_MAX_INPUT_TOKENS_BY_MODEL.items():
        if model.startswith(known):
            return limit
    return config.EMBED_MAX_INPUT_TOKENS


_batch_support: dict[str, bool] = {}


def supports_batching(model: str) -> bool:
    """Probe whether this model honours a multi-item batch. Measured, not assumed.

    Costs one two-item call, once per process. Worth it: whether 300 chunks is 19
    requests or 300 requests is the single biggest factor in whether an ingest run
    completes on a free key, and it is not documented anywhere I could find.
    """
    if model in _batch_support:
        return _batch_support[model]
    from google.genai import types

    probe = ["batch support probe alpha", "batch support probe beta"]
    try:
        resp = limits.guard(
            lambda: client().models.embed_content(
                model=model,
                contents=probe,
                config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
            ),
            stage="probe_batching",
        )
        _batch_support[model] = len(resp.embeddings) == len(probe)
    except Exception:  # noqa: BLE001 -- an unprobeable model is treated as unbatched
        _batch_support[model] = False
    return _batch_support[model]


def _key_meta(text: str, model: str, dim: int, prefix: str) -> dict[str, Any]:
    """Every input to the function being cached. Four arguments, four parts."""
    return {
        "v": CACHE_VERSION,
        "model": model,
        "dim": dim,
        "prefix": prefix,
        "text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _cache_path(meta: dict[str, Any]) -> Path:
    blob = json.dumps(meta, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    # Two-level fan-out: 300 files in one directory is fine, 300k is not.
    d = config.EMBED_CACHE / h[:2]
    return d / f"{h}.json"


def _cache_read(meta: dict[str, Any], stats: EmbedStats) -> np.ndarray | None:
    """Read, then VERIFY. A mismatch is a miss, never a silent hit.

    The key already encodes model/dim/prefix, so a mismatch here should be
    impossible -- which is exactly why it is checked. This catches a hash
    collision, a hand-edited cache file, and (the realistic one) a future change
    to _cache_path that stops including a field the reader still relies on.
    Cheap assertions on invariants you believe are how quietly-wrong becomes
    immediately-broken.
    """
    p = _cache_path(meta)
    if not p.exists():
        stats.misses += 1
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        stored = payload["meta"]
        if any(stored.get(k) != v for k, v in meta.items()):
            stats.rejected += 1
            stats.misses += 1
            return None
        vec = np.asarray(payload["vector"], dtype=np.float32)
        if vec.shape[0] != meta["dim"]:  # your guard, kept as a second net
            stats.rejected += 1
            stats.misses += 1
            return None
    except Exception:  # noqa: BLE001 -- a corrupt entry is a miss, not a crash
        stats.rejected += 1
        stats.misses += 1
        return None
    stats.hits += 1
    return vec


def _cache_write(meta: dict[str, Any], vec: np.ndarray) -> None:
    """Atomic write: temp file + rename.

    Free-tier quota can kill a batch mid-write. A half-written JSON file is a
    corrupt cache entry that survives on disk, and the class of bug we are
    guarding against is precisely 'stale data returned confidently'.
    """
    p = _cache_path(meta)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "vector": [float(x) for x in vec]}
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: slice the prefix, then re-normalise.

    Re-normalising is not optional. Cosine similarity assumes unit vectors; a
    truncated vector has lost some of its norm, and comparing vectors of
    differing norms silently biases toward whichever ones kept more of theirs.
    """
    if vectors.shape[-1] == dim:
        out = vectors
    elif vectors.shape[-1] < dim:
        raise ValueError(f"cannot expand {vectors.shape[-1]} -> {dim}; Matryoshka truncates only")
    else:
        out = vectors[..., :dim]
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


def embed_texts(
    texts: Sequence[str],
    *,
    kind: EmbedKind,
    dim: int | None = None,
    stats: EmbedStats | None = None,
    stage: str = "embed",
    degrade_on_quota: bool = True,
) -> tuple[np.ndarray, EmbedStats]:
    """Embed texts, cached at full dimension and truncated locally.

    Returns (matrix, stats). Rows that could not be embedded because quota died
    are ZERO VECTORS, and `stats.failed` counts them -- a zero row scores 0
    against every query, so such a chunk is absent from dense retrieval. That is
    reported via limits.embed_failed rather than left to be discovered.
    """
    stats = stats or EmbedStats()
    want = dim or config.EMBED_DIM
    model = resolve_models()["embedding"]
    scheme = task_scheme(model, kind)
    cache_dim = config.EMBED_DIM_FULL  # see (4) in the module docstring

    payloads = [scheme.payload(t) for t in texts]
    metas = [_key_meta(p, model, cache_dim, scheme.label) for p in payloads]

    vectors: list[np.ndarray | None] = [_cache_read(m, stats) for m in metas]
    todo = [i for i, v in enumerate(vectors) if v is None]

    token_cap = max_input_tokens(model)
    over = [(i, count_tokens(payloads[i])) for i in todo if count_tokens(payloads[i]) > token_cap]
    if over:
        raise ValueError(
            f"{len(over)} text(s) exceed {model}'s {token_cap}-token input limit "
            f"(largest ~{max(t for _, t in over)} tokens). The chunker should have "
            "prevented this. Deliberately NOT truncating: silently dropping the tail "
            "of a chunk indexes content the user believes is searchable and is not."
        )

    # Batch width is a property of the model, probed once. gemini-embedding-2
    # returns one embedding for a multi-item batch, so batching it would be a
    # silent data-loss bug (see BatchContract).
    width = config.EMBED_BATCH_SIZE if (todo and supports_batching(model)) else 1

    limits.seen(stage, n=len(todo))
    for start in range(0, len(todo), width):
        batch_idx = todo[start : start + width]
        batch = [payloads[i] for i in batch_idx]

        from google.genai import types

        cfg: dict[str, Any] = {"output_dimensionality": cache_dim}
        if scheme.task_type:
            cfg["task_type"] = scheme.task_type

        def call() -> Any:
            return client().models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(**cfg),
            )

        try:
            resp = limits.guard(call, stage=stage)
        except limits.QuotaExhausted:
            if not degrade_on_quota:
                raise
            stats.failed += len(todo) - start
            limits.report(
                limits.embed_failed(stats.failed, len(texts)), n=stats.failed
            )
            break

        # The assertion that would have caught my bug. zip() truncates silently;
        # this does not. Never relax it into a warning: the failure it prevents is
        # zero vectors sitting in the index scoring 0.0 against every query.
        got = list(resp.embeddings or [])
        if len(got) != len(batch):
            raise BatchContract(
                f"{model}: sent {len(batch)} contents, received {len(got)} embeddings. "
                f"Refusing to zip mismatched lists -- the unmatched rows would become "
                f"zero vectors that are present in the index and unreachable by any "
                f"query. Set EMBED_BATCH_SIZE=1 for this model, or use a model that "
                f"batches (measured: gemini-embedding-001 does, gemini-embedding-2 "
                f"does not)."
            )

        stats.api_calls += 1
        for i, emb in zip(batch_idx, got):
            vec = np.asarray(emb.values, dtype=np.float32)
            _cache_write(metas[i], vec)
            vectors[i] = vec

    full = np.zeros((len(texts), cache_dim), dtype=np.float32)
    for i, v in enumerate(vectors):
        if v is not None:
            full[i] = v
    return truncate(full, want), stats


def capabilities() -> dict[str, Any]:
    """What the API can actually do, MEASURED at startup rather than configured.

    The rule this encodes, learned three times: config describes intent, only a
    real call describes reality. A config file documenting a capability nobody
    has exercised is a plan, and it reads identically to a fact.

    Three probes, each from a bug:
      models      -- I asserted four model IDs from "live docs"; two were not
                     verifiable and one whole family had churned
      batching    -- gemini-embedding-2 returns ONE embedding for N inputs, which
                     zip() then truncated silently into zero vectors
      thinking    -- gemini-3.5-flash-lite rejects thinking_budget=0, which made
                     the entire cheap tier fail on its first ever call
    """
    resolved = resolve_models()
    out: dict[str, Any] = {"resolved": resolved, "probes": {}}
    for role, model in resolved.items():
        entry: dict[str, Any] = {"model": model}
        if role.startswith("embedding"):
            entry["batches"] = supports_batching(model)
            entry["max_input_tokens"] = max_input_tokens(model)
            entry["task_scheme"] = task_scheme(model, "document").label
        else:
            entry["can_disable_thinking"] = supports_thinking_disabled(model)
            if not entry["can_disable_thinking"]:
                # Role-specific, because the same fact means opposite things. On
                # the CHEAP tier it erodes the saving the cascade was built for.
                # On the JUDGE it is desirable -- a grader should reason. Emitting
                # one generic note said "cheaper per token" about the pro model,
                # which is simply false.
                entry["note"] = (
                    "thinking cannot be disabled: this tier is cheaper per token "
                    "but not thinking-free, so the cascade saves less than the "
                    "price list implies"
                    if role == "cheap"
                    else "thinking cannot be disabled, which is what we want here"
                )
        out["probes"][role] = entry
    return out


def cache_key(text: str, *, kind: EmbedKind = "document") -> str:
    """The embedding cache key for one text, as a stable string.

    Exists so the manifest can record what a document's ingest DERIVED, closing
    the erasure hole you identified: the embedding cache is keyed on text, not on
    source_id, so deleting a document leaves its vectors behind. Without this,
    "delete everything derived from this document" cannot reach the cache -- and
    the LLM response cache is the more sensitive of the two, since it holds
    model-written descriptions of the document's images and passages.
    """
    model = resolve_models()["embedding"]
    scheme = task_scheme(model, kind)
    meta = _key_meta(scheme.payload(text), model, config.EMBED_DIM_FULL, scheme.label)
    return _cache_path(meta).name


def embed_query(text: str, *, dim: int | None = None) -> np.ndarray:
    vecs, _ = embed_texts([text], kind="query", dim=dim, stage="embed_query")
    return vecs[0]


# --------------------------------------------------------------------------
# The honesty check for design decision (4)
# --------------------------------------------------------------------------


def verify_truncation(sample: str = "Reciprocal rank fusion combines ranked lists.") -> dict[str, Any]:
    """Does locally truncating 3072 -> 768 match asking the API for 768?

    The whole cache-at-full-dimension design rests on this being true. It is a
    documented property of Matryoshka embeddings, which is a reason to believe it
    and not a reason to skip checking it -- 'the docs say so' is how you end up
    with a dimension sweep comparing 768 to itself.

    Cosine ~1.0 => the design holds and the sweep costs one embedding pass.
    Anything lower => cache per requested dimension instead, and pay for it.
    """
    model = resolve_models()["embedding"]
    scheme = task_scheme(model, "document")
    from google.genai import types

    def one(d: int) -> np.ndarray:
        cfg: dict[str, Any] = {"output_dimensionality": d}
        if scheme.task_type:
            cfg["task_type"] = scheme.task_type
        resp = limits.guard(
            lambda: client().models.embed_content(
                model=model,
                contents=[scheme.payload(sample)],
                config=types.EmbedContentConfig(**cfg),
            ),
            stage="verify_truncation",
        )
        return np.asarray(resp.embeddings[0].values, dtype=np.float32)

    api_small = truncate(one(config.EMBED_DIM), config.EMBED_DIM)
    local_small = truncate(one(config.EMBED_DIM_FULL), config.EMBED_DIM)
    cos = float(np.dot(api_small, local_small))
    return {
        "model": model,
        "full_dim": config.EMBED_DIM_FULL,
        "small_dim": config.EMBED_DIM,
        "cosine_api_vs_local_truncation": cos,
        "holds": cos > 0.999,
        "verdict": (
            "local truncation matches the API; cache at full dim and slice"
            if cos > 0.999
            else "MISMATCH -- cache per requested dimension, do not slice locally"
        ),
    }
