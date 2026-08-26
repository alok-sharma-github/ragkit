"""
LLM-as-judge, and the gate it has to pass first. Guide module M9, tier two.

WHAT IS MISSING WITHOUT IT: the retrieval tier answers "was the evidence
retrievable" and is silent on "was the answer correct". Those are the two tiers
the guide calls essential, and having one is a credibility gap rather than a
missing feature -- every claim this project makes rests on measurement, so an
unmeasured half is load-bearing.

---------------------------------------------------------------------------
THREE CONSTRAINTS WRITTEN DOWN EARLY. THIS IS WHERE THEY GET TESTED.

1. KAPPA BEFORE VERDICTS COUNT. An unvalidated judge is not a weak signal, it is
   an unknown one. `validate()` compares the judge against hand labels and
   `score()` REFUSES to emit judged metrics until agreement clears
   config.JUDGE_MIN_KAPPA. Withheld, not warned about.

2. JUDGE != GENERATOR. Self-preference bias: a model rates its own output higher
   because "good" partly means "phrased how I would phrase it". Checked at
   runtime, not assumed from config, because model resolution can collide the two
   when a candidate is unavailable.

3. KAPPA IS CACHED PER (MODEL, RUBRIC), so a quota fallback invalidates it. A
   kappa measured on a different judge describes a different judge. The judge is
   also forbidden from falling back at all (limits.patient_retry): the fallback
   target IS the generator, and it is weaker -- two failures stacking, both
   pushing the score in the flattering direction.

---------------------------------------------------------------------------
THE PREVALENCE TRAP, which is why validation samples are stratified

Cohen's kappa measures agreement ABOVE CHANCE. If 95% of a sample is "faithful",
chance agreement is already ~0.9, so judge and human can agree on 94 of 100 items
and kappa still collapses toward zero. The fix is not a lower threshold -- it is a
validation sample with balanced classes, which means deliberately including
answers known to be bad. `validation_sample()` builds those by CORRUPTING good
answers in ways with known ground truth, so the negative class does not depend on
finding organic failures.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from .. import config, gemini, limits
from .metrics import MIN_N_FOR_RATE

Faithful = Literal["supported", "partly_supported", "unsupported"]

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "string"},
        "faithfulness_reason": {"type": "string"},
        "answers_question": {"type": "boolean"},
        "answers_question_reason": {"type": "string"},
    },
    "required": ["faithfulness", "faithfulness_reason", "answers_question"],
}

_SYSTEM = """You grade an answer against the sources it was given. You are not
answering the question and not improving the answer.

faithfulness -- is every factual claim in the answer supported by the sources?
  "supported"        every claim is stated in or directly follows from the sources
  "partly_supported" some claims are supported and at least one is not
  "unsupported"      the central claim is not in the sources

Judge ONLY against the provided sources. If the answer is true in the world but
absent from the sources, it is unsupported -- that is the property being measured.

answers_question -- does the answer address what was asked? An answer can be
perfectly faithful and still not answer the question.

A correct abstention ("the sources do not contain this") is "supported" and
answers_question=true when the sources genuinely lack the answer.

Be specific in the reasons and quote the deciding span. Do not reward length,
confidence or hedging."""


@dataclass
class Verdict:
    faithfulness: Faithful = "unsupported"
    faithfulness_reason: str = ""
    answers_question: bool = False
    answers_question_reason: str = ""
    judge_model: str = ""
    order_swapped: bool = False       # position-bias control, recorded per verdict
    failed: bool = False
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _render_sources(sources: Sequence[dict[str, Any]]) -> str:
    out = []
    for i, s in enumerate(sources, 1):
        head = f"[{i}] {s.get('source_id', '?')}"
        if s.get("page"):
            head += f" p{s['page']}"
        note = ""
        if "page_text_clip" in (s.get("text_source") or ""):
            note = "\n(tabular data as plain text; row structure inferred)"
        elif s.get("provenance") == "model_generated":
            note = "\n(a system-generated image description, not document text)"
        out.append(f"{head}{note}\n{s.get('text', '')}")
    return "\n\n---\n\n".join(out)


def judge_answer(
    question: str,
    answer: str,
    sources: Sequence[dict[str, Any]],
    *,
    swap_order: bool | None = None,
) -> Verdict:
    """One judged verdict, with position bias controlled.

    `swap_order` reverses the source order. Position bias is real and cheap to
    mitigate: the caller randomises it, and the choice is RECORDED on the verdict
    so a suspicious result can be re-examined rather than merely distrusted.
    """
    resolved = gemini.resolve_models()
    jm, wm = resolved["judge"], resolved["workhorse"]
    if jm == wm:
        # Constraint 2, checked rather than assumed.
        return Verdict(
            failed=True, judge_model=jm,
            error=("judge and generator resolved to the same model "
                   f"({jm}); every verdict would be self-graded"),
        )

    srcs = list(sources)
    if swap_order is None:
        swap_order = random.random() < 0.5
    if swap_order:
        srcs = list(reversed(srcs))

    prompt = (
        f"Question:\n{question}\n\n"
        f"Sources:\n\n{_render_sources(srcs)}\n\n"
        f"Answer to grade:\n{answer or '(the system abstained)'}\n\n"
        "Grade it."
    )

    def call() -> tuple[str, gemini.Usage]:
        return gemini.generate(
            prompt, role="judge", system=_SYSTEM, schema=_VERDICT_SCHEMA,
            max_output_tokens=1024, thinking=True, stage="judge", patient=True,
        )

    try:
        raw, _usage = call()
        payload = json.loads(raw)
    except limits.QuotaExhausted as exc:
        # Constraint 3: the judge WAITS (patient_retry) and, if it still cannot
        # run, the verdict FAILS. It never falls back to the workhorse.
        return Verdict(failed=True, judge_model=jm, error=f"quota exhausted: {exc}")
    except (gemini.EmptyResponse, json.JSONDecodeError) as exc:
        return Verdict(failed=True, judge_model=jm, error=f"{type(exc).__name__}: {exc}")

    f = (payload.get("faithfulness") or "").strip()
    if f not in ("supported", "partly_supported", "unsupported"):
        return Verdict(failed=True, judge_model=jm,
                       error=f"judge returned an unknown faithfulness value: {f!r}")
    return Verdict(
        faithfulness=f,  # type: ignore[arg-type]
        faithfulness_reason=(payload.get("faithfulness_reason") or "").strip(),
        answers_question=bool(payload.get("answers_question")),
        answers_question_reason=(payload.get("answers_question_reason") or "").strip(),
        judge_model=jm,
        order_swapped=swap_order,
    )


# --------------------------------------------------------------------------
# Validation -- the gate
# --------------------------------------------------------------------------


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> dict[str, Any]:
    """Agreement above chance, with the prevalence caveat reported alongside.

    Returns raw agreement TOO, because kappa alone is misleading on an imbalanced
    sample and a reader needs both to interpret either. A kappa of 0.2 with 94%
    raw agreement means "the classes are imbalanced", not "the judge is bad".
    """
    if not a or len(a) != len(b):
        return {"kappa": None, "n": 0, "sufficient": False,
                "note": "no paired labels"}
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(
        (sum(1 for x in a if x == lab) / n) * (sum(1 for y in b if y == lab) / n)
        for lab in labels
    )
    kappa = 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)
    # Class balance, because it is what makes kappa interpretable.
    counts = {lab: sum(1 for x in a if x == lab) for lab in labels}
    dominant = max(counts.values()) / n if counts else 1.0
    return {
        "kappa": round(kappa, 3),
        "raw_agreement": round(po, 3),
        "chance_agreement": round(pe, 3),
        "n": n,
        "class_counts": counts,
        "dominant_class_share": round(dominant, 3),
        "prevalence_warning": dominant > 0.8,
        "sufficient": n >= config.JUDGE_VALIDATION_SAMPLE,
    }


def _kappa_record_path() -> Path:
    return config.JUDGE_KAPPA_RECORD


def kappa_key() -> str:
    """Kappa is a property of (judge model, rubric version), not of the project."""
    return f"{gemini.resolve_models()['judge']}::rubric-v{config.JUDGE_RUBRIC_VERSION}"


def load_kappa() -> dict[str, Any] | None:
    p = _kappa_record_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get(kappa_key())
    except Exception:  # noqa: BLE001
        return None


def save_kappa(result: dict[str, Any]) -> Path:
    p = _kappa_record_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    # Keyed, not overwritten: a kappa measured on a different judge or a changed
    # rubric describes a different judge, so both records coexist and neither
    # silently stands in for the other.
    existing[kappa_key()] = result
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return p


def is_validated() -> tuple[bool, dict[str, Any]]:
    rec = load_kappa()
    if not rec:
        return False, {"reason": "no kappa recorded for this judge and rubric",
                       "key": kappa_key()}
    k = rec.get("kappa")
    ok = (
        k is not None
        and k >= config.JUDGE_MIN_KAPPA
        and rec.get("n", 0) >= config.JUDGE_VALIDATION_SAMPLE
    )
    return ok, {"key": kappa_key(), **rec, "threshold": config.JUDGE_MIN_KAPPA}


def gate() -> dict[str, Any]:
    """Whether judged metrics may be emitted at all."""
    ok, detail = is_validated()
    resolved = gemini.resolve_models()
    collision = resolved["judge"] == resolved["workhorse"]
    return {
        "may_emit_judged_metrics": bool(ok and not collision),
        "validated": ok,
        "judge_model": resolved["judge"],
        "generator_model": resolved["workhorse"],
        "self_grading_collision": collision,
        "detail": detail,
        "withheld_because": (
            "judge and generator are the same model" if collision
            else None if ok else
            "judge has not passed its kappa gate against hand labels"
        ),
        "note": limits.judge_unvalidated(
            detail.get("kappa"), resolved["judge"]
        ).render() if not ok else "",
    }

# --------------------------------------------------------------------------
# Validation sample, blind labelling, and the kappa that gates everything
# --------------------------------------------------------------------------

LABELS_PATH = config.DATA_EVAL / "judge_labels.json"
VERDICTS_PATH = config.DATA_EVAL / "judge_verdicts.json"
JUDGED_PATH = config.DATA_EVAL / "judged_results.json"


@dataclass
class SampleItem:
    """One thing to be graded, with ground truth where it was constructed."""

    item_id: str
    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    # "supported" / "unsupported" when we MADE the item, None when it came from a
    # real run. Constructed negatives exist so the sample can be balanced without
    # waiting for organic failures.
    constructed_truth: str | None = None
    how_constructed: str = ""
    text_source: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _corrupt(
    answer: str, sources: Sequence[dict[str, Any]], want: str
) -> tuple[str, str, str] | None:
    """Make a KNOWN-BAD answer from a good one. Returns (answer, how, truth).

    THE TRUTH LABEL DEPENDS ON THE CORRUPTION, and getting this wrong invalidated
    the first sample I built. Two corruptions, two different correct labels:

      swap_number -> "unsupported"
          The central value is replaced with one absent from the sources, so the
          claim the answer exists to make is not supported.

      add_claim   -> "partly_supported"
          An unsupported sentence is APPENDED to an otherwise-good answer. The
          original claims remain supported. Labelling this "unsupported" is simply
          wrong, and I labelled all 15 negatives that way -- 11 of them
          incorrectly. A judge answering "partly_supported" would have been RIGHT
          and scored as a disagreement, depressing kappa and leading me to
          conclude the judge was unreliable when the ground truth was.

    Constructed negatives are still the right approach: organic failures in a
    ~90%-faithful population give ~3 negatives at n=30, and kappa then swings on
    a single relabel. But a constructed negative is only as good as the label
    attached to it, and the label has to follow the construction.
    """
    import re

    # THE TARGET CLASS DRIVES THE CORRUPTION, not the other way round. The first
    # version picked whichever corruption the answer happened to allow -- and
    # since only 4 of 15 answers contained a decimal, the sample came out
    # 15/11/4 across three classes. A kappa over a class with four members is
    # exactly what this project's own prevalence warning exists to flag, so the
    # sampler now asks each candidate for a SPECIFIC class and moves on if the
    # answer cannot supply it.
    blob = " ".join(str(x.get("text", "")) for x in sources)

    def _claim_count(text: str) -> int:
        """Roughly how many separate factual assertions the answer makes.

        Bullets, sentences and clause separators. Crude on purpose -- it only has
        to distinguish "one fact" from "several".
        """
        import re as _re
        bullets = len(_re.findall(r"^\s*[-*]\s+", text, _re.M))
        sentences = len([x for x in _re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 15])
        conj = len(_re.findall(r",\s+and\s+the\s+|;\s+", text))
        return max(bullets, sentences + conj, 1)

    if want == "unsupported":
        # DECIMALS FIRST, and never a number glued to an identifier. The first
        # version matched integers too, so it found the 10 in 'Recall@10'
        # before the value '0.831' and produced 'Recall@9.999 of 0.831'. That
        # is a MALFORMED claim rather than a falsified one: the judge rejects
        # it for the wrong reason, which makes the negative class artificially
        # easy and inflates the discrimination estimate this sample exists to
        # measure. Integers remain a fallback, but only ones not attached to a
        # word or an @.
        nums = [m.group(0) for m in re.finditer(r"(?<![\w@.\-])\d+\.\d+(?![\w.\-])", answer)]
        if not nums:
            nums = [m.group(0) for m in re.finditer(r"(?<![\w@.\-])\d{2,}(?![\w.\-])", answer)]
        if nums:
            target = nums[0]
            fake = "9.999" if "9.999" not in blob else "8.888"
            # THE LABEL FOLLOWS WHAT THE ANSWER CLAIMS, not the corruption
            # applied. Assigning "unsupported" to every swap_number was wrong,
            # and a human labeller caught it on four items: in
            # "the compatible model is MRA-8800 only, and the lead time is
            # 9.999 days" the model compatibility IS in the source -- one claim
            # supported, one falsified, which is partly_supported by the very
            # definition in the rubric.
            #
            # Third time this reference standard has been wrong, and the same
            # mistake each time: labelling by the process that produced the item
            # rather than by the item. A bad ground truth corrupts everything
            # measured against it and has nothing downstream to catch it.
            multi = _claim_count(answer) > 1
            return (answer.replace(target, fake, 1),
                    f"swap_number {target}->{fake}"
                    + (" (multi-claim answer)" if multi else " (single-claim answer)"),
                    "partly_supported" if multi else "unsupported")
        # No number to break: replace the answer's substance, so the central
        # claim is genuinely absent from the sources.
        #
        # ROTATED, not fixed. A single boilerplate sentence appeared in 4 of 10
        # unsupported items, which lets a judge pattern-match the sentence instead
        # of reasoning about support -- inflating kappa without measuring
        # anything. The variants are seeded by the answer so the choice is
        # deterministic and reproducible.
        if len(answer) > 60:
            variants = [
                ("This approach was deprecated in the following release and is no "
                 "longer recommended for new deployments."),
                ("The authors report that this was validated on a second corpus of "
                 "roughly twice the size, with consistent results."),
                ("This behaviour is configurable, and the default was changed after "
                 "the initial evaluation was completed."),
                ("The same result was reproduced independently by a separate team "
                 "using a different embedding model."),
                ("Performance degrades sharply beyond a context length of thirty-two "
                 "thousand tokens for this configuration."),
            ]
            pick = variants[sum(ord(c) for c in answer[:40]) % len(variants)]
            return (pick, "replace_substance (central claim absent from sources)",
                    "unsupported")
        return None

    if want == "partly_supported" and len(answer) > 40:
        return (
            answer.rstrip(". ") + ". This was measured over a 12-month follow-up period.",
            "add_claim (unsupported sentence appended to a supported answer)",
            "partly_supported",
        )
    return None


def validation_sample(
    *, target: int | None = None, seed: int = 11
) -> list[SampleItem]:
    """A class-BALANCED sample, deliberately oversampling hard cases.

    Two decisions, both from the prevalence problem:

    1. BALANCED, not random. A random draw from a 90%-faithful population gives
       ~3 negatives at n=30, and kappa then swings wildly on a single relabel.
       Half the sample is constructed-unsupported.

    2. OVERSAMPLED TOWARD DISAGREEMENT -- items whose sources are
       `page_text_clip` or model-generated captions are included preferentially,
       because those are where the judge and a human are most likely to differ
       (a repaired table has values without labels).

    CONSEQUENCE, and it must be stated wherever the kappa is quoted: this is a
    DISCRIMINATION estimate, not a population statistic. It answers "can this
    judge tell good from bad on hard cases", not "how faithful is the corpus".
    Reporting it as the latter would be a different claim than the one measured.
    """
    from . import goldenset as G
    from ..index.hybrid import HybridIndex
    from ..generate import answer as A
    from .. import gemini as GM

    n_target = target or config.JUDGE_VALIDATION_SAMPLE
    rng = random.Random(seed)

    # THREE CLASSES, EVEN QUOTAS. Kappa is computed over the label set actually
    # present, so a class with four members makes the statistic swing on a single
    # relabel. Equal thirds is the only distribution where every class carries
    # comparable weight.
    per_class = {
        "supported": n_target - 2 * (n_target // 3),
        "partly_supported": n_target // 3,
        "unsupported": n_target // 3,
    }
    filled = {k: 0 for k in per_class}

    items = [i for i in G.load()
             if i.stratum != "out_of_scope" and i.anchor != "none" and not i.quarantined]
    rng.shuffle(items)

    hyb = HybridIndex.load()
    out: list[SampleItem] = []

    for it in items:
        if len(out) >= n_target:
            break
        qv = GM.embed_query(it.question)
        r = hyb.retrieve(it.question, qv, mode="dense", token_budget=JUDGE_TOKEN_BUDGET, unit="parent")
        parents = r.parents[:5]
        if not parents:
            continue
        ans = A.answer(it.question, parents)
        if not ans.text.strip():
            continue

        srcs = [
            {"source_id": p.source_id, "page": p.page, "text": p.display_text,
             "text_source": p.text_source, "provenance": p.text_provenance.value}
            for p in parents
        ]
        ts = ",".join(sorted({p.text_source for p in parents}))
        if filled["supported"] < per_class["supported"]:
            out.append(SampleItem(
                item_id=f"pos-{len(out):03d}", question=it.question, answer=ans.text,
                sources=srcs, constructed_truth="supported",
                how_constructed="real answer over real sources", text_source=ts,
            ))
            filled["supported"] += 1
            continue

        # Ask for whichever negative class is furthest from its quota, rather
        # than accepting whichever corruption this answer happens to allow.
        # Content-driven selection produced 15/11/4 across three classes, and
        # a kappa over a four-member class swings on one relabel.
        want = min(
            ("partly_supported", "unsupported"),
            key=lambda c: filled[c] / max(per_class[c], 1),
        )
        if filled[want] >= per_class[want]:
            continue
        c = _corrupt(ans.text, srcs, want)
        if c:
            bad, how, truth = c
            out.append(SampleItem(
                item_id=f"neg-{len(out):03d}", question=it.question, answer=bad,
                sources=srcs, constructed_truth=truth,
                how_constructed=how, text_source=ts,
            ))
            filled[truth] += 1
    return out[:n_target]


def sample_limitations(items: Sequence[SampleItem]) -> dict[str, Any]:
    """What this sample cannot measure, stated with the sample.

    The "oversamples hard cases" intent is only partly achievable: retrieval can
    only surface `page_text_clip` sources if the corpus contains them, and it
    contains THREE children. So the repaired-table cluster -- the one where a
    strict judge provably calls a correct answer unsupported -- cannot reach a
    sufficient sub-sample at this corpus size. That is a corpus limitation, not a
    sampling one, and no cleverness in the sampler fixes it.

    Recorded next to the sample so the kappa is never quoted as covering it.
    """
    from collections import Counter
    from .metrics import MIN_N_FOR_RATE

    by_ts = Counter(i.text_source for i in items)
    hard = sum(v for k, v in by_ts.items() if "page_text_clip" in k or "caption" in k)
    return {
        "n": len(items),
        "class_counts": dict(Counter(i.constructed_truth or "organic" for i in items)),
        "by_text_source": dict(by_ts),
        "hard_case_items": hard,
        "hard_case_sufficient": hard >= MIN_N_FOR_RATE,
        "cannot_measure": (
            [] if hard >= MIN_N_FOR_RATE else
            ["judge/human agreement on page_text_clip (repaired-table) sources: "
             f"only {hard} item(s); the corpus contains 3 such children, so this "
             "cluster is unmeasurable at this corpus size"]
        ),
        "estimate_type": "discrimination on a class-balanced sample, NOT a "
                         "population faithfulness statistic",
    }


def save_sample(items: Sequence[SampleItem]) -> Path:
    p = config.DATA_EVAL / "judge_sample.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "limitations": sample_limitations(items),
        "items": [i.to_json() for i in items],
    }, indent=1), encoding="utf-8")
    return p


def load_sample() -> list[SampleItem]:
    p = config.DATA_EVAL / "judge_sample.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text("utf-8"))
    raw = doc["items"] if isinstance(doc, dict) else doc     # tolerate the old shape
    return [SampleItem(**d) for d in raw]


def load_sample_limitations() -> dict[str, Any]:
    p = config.DATA_EVAL / "judge_sample.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text("utf-8"))
    return doc.get("limitations", {}) if isinstance(doc, dict) else {}


def label_blind(items: Sequence[SampleItem] | None = None) -> Path:
    """Hand-label WITHOUT seeing the judge's verdicts.

    This function never reads VERDICTS_PATH. That is the point: labelling after
    reading the judge's output measures your agreement with something you already
    saw, kappa comes out high, and it means nothing. Blindness is enforced by
    what the code can access, not by remembering to look away.

    It also records whether verdicts already existed on disk when labelling
    happened. If they did, the kappa is marked possibly-anchored -- an integrity
    flag rather than a claim of innocence.
    """
    items = list(items or load_sample())
    if not items:
        raise RuntimeError("no sample -- build one first (judge sample)")

    existing: dict[str, Any] = {}
    if LABELS_PATH.exists():
        existing = json.loads(LABELS_PATH.read_text("utf-8"))
    labels: dict[str, Any] = dict(existing.get("labels", {}))

    verdicts_existed = VERDICTS_PATH.exists()
    print(f"{len(items)} items. Labels are BLIND: judge verdicts are not shown.")
    if verdicts_existed:
        print("NOTE: verdicts already exist on disk. This kappa will be marked")
        print("possibly-anchored, because you may have seen them.")
    print("For each: [s] supported  [p] partly  [u] unsupported  [?] skip  [q] quit")
    print()

    for i, it in enumerate(items, 1):
        if it.item_id in labels:
            continue
        print("=" * 70)
        print(f"({i}/{len(items)})  Q: {it.question}")
        # FULL TEXT, EVERY SOURCE. This truncated to 3 sources at 300 chars
        # each, so a labeller saw ~900 characters of a 6,154-character
        # evidence set -- 15% -- while the grader receives all of it. Any
        # disagreement would then partly measure MY TRUNCATION rather than a
        # difference in judgment, which makes the kappa meaningless: the
        # human and the model would be answering about different evidence.
        #
        # This is the "measuring two different programs" error, inside the
        # harness built to measure agreement. It costs reading time, and the
        # earlier "~10 minutes" estimate was derived from the broken view --
        # the real task is longer. `q` saves and quits, so it need not be one
        # sitting.
        print(f"\nSOURCES ({it.text_source}) -- the same text the grader sees:")
        for n, s in enumerate(it.sources, 1):
            print(f"\n  --- source [{n}] {s.get('source_id')} "
                  f"p{s.get('page')} ({len(str(s.get('text', '')))} chars) ---")
            print("  " + str(s.get("text", "")).replace("\n", "\n  "))
        print(f"\nANSWER TO JUDGE:\n{it.answer}")
        print()
        try:
            key = input("  your label > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key == "q":
            break
        mapping = {"s": "supported", "p": "partly_supported", "u": "unsupported"}
        if key in mapping:
            labels[it.item_id] = mapping[key]

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps({
        "labels": labels,
        # PREDICATE FIXED. This tested `not existing`, where `existing` is the
        # whole loaded JSON document -- truthy even when it holds ZERO labels.
        # An aborted first run writes {"labels": {}}, so the very next session
        # was stamped possibly-anchored despite nothing having been seen. An
        # integrity flag that cries wolf gets ignored, which is worse than not
        # having one. What matters is whether any LABEL already existed.
        "labelled_before_verdicts": not verdicts_existed and not existing.get("labels"),
        "n": len(labels),
    }, indent=2), encoding="utf-8")
    print(f"\n{len(labels)} labels -> {LABELS_PATH}")
    return LABELS_PATH


def run_judge_on_sample(items: Sequence[SampleItem] | None = None) -> Path:
    """Judge every sample item and store the verdicts separately from the labels."""
    items = list(items or load_sample())
    verdicts: dict[str, Any] = {}
    for i, it in enumerate(items, 1):
        v = judge_answer(it.question, it.answer, it.sources)
        verdicts[it.item_id] = v.to_json()
        print(f"  [{i}/{len(items)}] {it.item_id}: "
              f"{v.faithfulness if not v.failed else 'FAILED'}", flush=True)
    VERDICTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICTS_PATH.write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    return VERDICTS_PATH


def validate() -> dict[str, Any]:
    """Join blind human labels with judge verdicts and compute the gate."""
    if not LABELS_PATH.exists() or not VERDICTS_PATH.exists():
        return {"error": "need both judge_labels.json and judge_verdicts.json",
                "have_labels": LABELS_PATH.exists(),
                "have_verdicts": VERDICTS_PATH.exists()}

    lab_doc = json.loads(LABELS_PATH.read_text("utf-8"))
    labels: dict[str, str] = lab_doc.get("labels", {})
    verdicts: dict[str, Any] = json.loads(VERDICTS_PATH.read_text("utf-8"))

    pairs = [
        (labels[k], verdicts[k]["faithfulness"])
        for k in labels
        if k in verdicts and not verdicts[k].get("failed")
    ]
    human = [a for a, _ in pairs]
    judged = [b for _, b in pairs]
    k = cohens_kappa(human, judged)

    by_source: dict[str, dict[str, int]] = {}
    sample = {i.item_id: i for i in load_sample()}
    for key in labels:
        if key not in verdicts or verdicts[key].get("failed"):
            continue
        ts = (sample.get(key).text_source if sample.get(key) else "") or "unknown"
        b = by_source.setdefault(ts, {"n": 0, "agree": 0})
        b["n"] += 1
        b["agree"] += int(labels[key] == verdicts[key]["faithfulness"])

    result = {
        **k,
        "judge_model": gemini.resolve_models()["judge"],
        "rubric_version": config.JUDGE_RUBRIC_VERSION,
        "labelled_before_verdicts": lab_doc.get("labelled_before_verdicts", False),
        "estimate_type": "discrimination (sample is class-balanced and oversamples "
                         "hard cases), NOT a population faithfulness statistic",
        "agreement_by_text_source": by_source,
        "sample_limitations": load_sample_limitations(),
        "passes_gate": bool(
            k.get("kappa") is not None
            and k["kappa"] >= config.JUDGE_MIN_KAPPA
            and k.get("n", 0) >= config.JUDGE_VALIDATION_SAMPLE
        ),
    }
    save_kappa(result)
    return result


JUDGE_TOKEN_BUDGET = 1500
"""The budget the generation tier is scored at.

Named because it is now RECORDED in the artifact, and a literal repeated at
the call site could drift from the stamp -- which would produce an artifact
confidently labelled with a budget it was not produced at. Worse than no
label.
"""


def score(*, limit: int | None = None) -> dict[str, Any]:
    """Judged metrics over the golden set -- WITHHELD unless the gate passes."""
    g = gate()
    if not g["may_emit_judged_metrics"]:
        return {"withheld": True, "gate": g,
                "note": "no judged numbers are produced until the judge is validated; "
                        "an unvalidated judge is an unknown signal, not a weak one"}

    from . import goldenset as G
    from . import metrics as M
    from ..index.hybrid import HybridIndex
    from ..generate import answer as A
    from .. import gemini as GM

    items = [i for i in G.load()
             if i.stratum != "out_of_scope" and i.anchor != "none" and not i.quarantined]
    if limit:
        items = items[:limit]
    hyb = HybridIndex.load()

    # PROGRESS AND INCREMENTAL SAVE.
    #
    # This loop is ~12s per item over ~92 items -- about 18 minutes -- and the
    # first version printed nothing and saved nothing until the end. Two separate
    # faults:
    #
    #   silence      an opaque long operation with no output is indistinguishable
    #                from a hung one, which is the worst failure mode for
    #                something a person is waiting on
    #   no durability an interrupt at minute 17 discarded all 92 answer+judge
    #                calls. Exactly the bug fixed in the golden-set builder
    #                ("expensive work becomes durable before anything cheap
    #                runs") and not fixed here -- same lesson, different file.
    #
    # Partial rows are written as they complete, so Ctrl-C keeps the work and a
    # re-run resumes from where it stopped rather than starting over.
    partial = config.DATA_EVAL / "judged_partial.json"
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if partial.exists():
        try:
            rows = json.loads(partial.read_text("utf-8"))
            done = {r["question"] for r in rows}
            print(f"  resuming: {len(rows)} item(s) already judged")
        except Exception:  # noqa: BLE001
            rows, done = [], set()

    todo = [i for i in items if i.question not in done]
    for n, it in enumerate(todo, 1):
        qv = GM.embed_query(it.question)
        r = hyb.retrieve(it.question, qv, mode="dense", token_budget=JUDGE_TOKEN_BUDGET, unit="parent")
        parents = r.parents[:5]
        ans = A.answer(it.question, parents,
                       starved=r.starved_by_budget,
                       min_unit_tokens=r.min_unit_tokens,
                       context_budget=JUDGE_TOKEN_BUDGET)
        srcs = [{"source_id": p.source_id, "page": p.page, "text": p.display_text,
                 "text_source": p.text_source, "provenance": p.text_provenance.value}
                for p in parents]
        v = judge_answer(it.question, ans.text, srcs)
        rows.append({
            "question": it.question, "stratum": it.stratum,
            "text_source": ",".join(sorted({p.text_source for p in parents})),
            "faithfulness": v.faithfulness, "answers_question": v.answers_question,
            "failed": v.failed, "abstained": ans.abstained,
            "citations_verified": ans.reconciliation.get("citations_verified", 0),
            # Recorded per row, not recomputed later: a starved retrieval is a
            # property of THIS call and cannot be reconstructed from the summary.
            "starved_by_budget": bool(ans.reconciliation.get("starved_by_budget")),
        })
        partial.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        verdict = "FAILED" if v.failed else v.faithfulness
        print(f"  [{len(rows)}/{len(items)}] {verdict:17s} {it.stratum:16s} "
              f"{it.question[:44]}", flush=True)

    def share(rs: list[dict[str, Any]], key: str, val: Any) -> dict[str, Any]:
        n = len(rs)
        hits = sum(1 for r in rs if r[key] == val)
        if n < MIN_N_FOR_RATE:
            return {"n": n, "hits": hits, "rate": None, "label": f"{hits} of {n}"}
        return {"n": n, "hits": hits, "rate": round(hits / n, 3),
                "label": f"{hits}/{n} = {hits / n:.0%}"}

    ok = [r for r in rows if not r["failed"]]

    # ABSTENTIONS ARE NOT ANSWERS, AND MUST NOT BE POOLED WITH THEM.
    #
    # The first run of this reported "supported 90/91 = 99%" and it was an
    # artifact of the denominator. 17 of those 91 rows were ABSTENTIONS -- the
    # pipeline declining to answer -- and the judge marked each one `supported`,
    # correctly by its own rubric: an answer that asserts nothing asserts nothing
    # false. So a fifth of the headline was made of vacuous passes, and the
    # metric moved UP whenever the system answered FEWER questions. A faithfulness
    # score that rewards silence is measuring the wrong direction.
    #
    # Worse, it hid the finding. 10 of the 17 abstentions are table_or_image, and
    # a 27% decline rate on that stratum is the most actionable number in the
    # whole run -- invisible while it was being counted as success.
    #
    # Split three ways instead:
    #   answered    faithfulness over answers that actually claim something.
    #               This is the number that means what "faithfulness" says.
    #   abstained    a COVERAGE fact, not a faithfulness fact. Reported as its
    #               own rate because declining to answer is a real cost even
    #               when it is the honest response.
    #   starved      abstained because the context budget admitted no sources at
    #               all. Not the generator's behaviour in any sense, and it must
    #               never be filed under either of the above.
    starved = [r for r in ok if r.get("starved_by_budget")]
    abstained = [r for r in ok if r["abstained"] and not r.get("starved_by_budget")]
    answered = [r for r in ok if not r["abstained"]]

    by_ts: dict[str, Any] = {}
    for ts in sorted({r["text_source"] for r in answered}):
        sub = [r for r in answered if r["text_source"] == ts]
        by_ts[ts] = {"supported": share(sub, "faithfulness", "supported"),
                     "answers_question": share(sub, "answers_question", True)}

    by_stratum_abst: dict[str, Any] = {}
    for st in sorted({r["stratum"] for r in ok}):
        sub = [r for r in ok if r["stratum"] == st]
        by_stratum_abst[st] = share(
            [{"a": bool(x["abstained"])} for x in sub], "a", True)

    out = {
        "withheld": False,
        "gate": g,
        "n": len(rows),
        "n_failed_verdicts": sum(1 for r in rows if r["failed"]),
        # Denominator = answers, not rows. See the note above.
        "n_answered": len(answered),
        "n_abstained": len(abstained),
        "n_starved": len(starved),
        "supported": share(answered, "faithfulness", "supported"),
        "partly": share(answered, "faithfulness", "partly_supported"),
        "unsupported": share(answered, "faithfulness", "unsupported"),
        "answers_question": share(answered, "answers_question", True),
        # Coverage, reported alongside so faithfulness can never be read without
        # the rate of questions it declined to attempt.
        "abstention_rate": share([{"a": bool(r["abstained"])} for r in ok], "a", True),
        "abstention_by_stratum": by_stratum_abst,
        # THE SLICE THAT MATTERS. A repaired table has values without column
        # labels, so a strict judge calls a CORRECT answer unsupported. Sliced,
        # that is "our repair produces unverifiable text"; averaged, it is
        # indistinguishable from "the model is improvising".
        "by_text_source": by_ts,
        # WHAT THIS WAS PRODUCED UNDER. Absent until now, which made these rows
        # unpoolable with any other artifact -- see metrics.artifact_stamp.
        "stamp": M.artifact_stamp(token_budget=JUDGE_TOKEN_BUDGET),
        "rows": rows,
    }
    JUDGED_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # The complete result supersedes the partial, so the next run starts fresh
    # rather than resuming a finished job.
    (config.DATA_EVAL / "judged_partial.json").unlink(missing_ok=True)
    return out
