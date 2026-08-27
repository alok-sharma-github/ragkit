"""
Guard coverage. For a guard, COVERAGE is the invariant -- not existence.

WHY THIS EXISTS, and why audit_reachability.py could not catch it.

That audit asks "is this reachable from an entry point?" It cannot ask "is this
reached from EVERY entry point that needs it?" -- and for a guard those are
different questions with the same-looking answer.

The bug it missed, found instead by running the CLI:

    check_operation() was wired into embed_texts() only. `ragkit ask` under a
    5-token ceiling sailed through and spent ~2,500 tokens. The one check in the
    codebase correctly saw zero billable EMBEDDING work -- the query embedding was
    cached -- and nothing examined the generate call at all.

That is a new variant of this project's dominant bug family. Not "built, correct,
and off the request path" -- the earlier seven. This one is:

    BUILT, CORRECT, AND ON *SOME* OF THE TRAFFIC.

Which is worse, because it reports a pass. `ragkit spend` printed real numbers,
the notice rendered correctly, the unit test went green -- while the most-called
paid route was unguarded. Nothing in any output said "one of three". A partial
pass is the kind you stop looking at.

THE ENUMERATION IS DERIVED, NOT LISTED. A hand-maintained roster of paid routes
is the same class of object as a hardcoded count sitting beside a derived one:
correct on the day it is written and silently wrong afterwards. So "paid" is
computed -- any function in gemini.py that reaches `client()` spends money -- and
a fourth paid route added next year fails this audit the day it is written rather
than being remembered into a list, or not.

EXEMPTIONS ARE BY EXPLICIT NAME, and a stale one is an error. Fuzzy matching
already burned this project once: `Chunk.citation` was waved through the
reachability audit because the word "citation" appeared in an unrelated
deferral's prose. An exemption naming a function that no longer exists is worse
than no exemption -- it is a hole waiting for something to be renamed into it.

LIMITS, stated because a tool overstating its certainty is the mistake it hunts:
module-level functions only, AST names only. It cannot follow getattr, registries,
decorators that rewrap, or dynamic dispatch. It would not catch a paid call made
through a variable holding the client.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Exemptions: functions that reach client() but legitimately spend nothing the
# ceiling should meter. Each needs a reason, and each must still exist.
# ---------------------------------------------------------------------------

EXEMPT: dict[str, str] = {
    "client": "constructs the SDK client; makes no request itself",
    "available_models": "models.list() is metadata, not inference -- not billed per token",
    "resolve_models": "only calls available_models(); no inference",
    "capabilities": "startup probes, reported via the exemptions below",
    "supports_batching": (
        "one-off capability probe at startup. Deliberately unmetered: the ceiling "
        "exists to stop bulk spend, and refusing a probe would make the system "
        "unable to discover what it can do -- which is how the zip()/zero-vector "
        "bug shipped in the first place"
    ),
    "supports_thinking_disabled": "one-off capability probe, same reasoning as supports_batching",
    "verify_truncation": (
        "diagnostic that checks Matryoshka truncation against the API. Run by hand, "
        "not on any request path"
    ),
}


def _call_names(node: ast.AST) -> set[str]:
    """Every callable name invoked inside this node, bare and dotted."""
    out: set[str] = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name):
            out.add(f.id)
        elif isinstance(f, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST = f
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            out.add(".".join(reversed(parts)))
            out.add(f.attr)
    return out


def _module_funcs(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _reaches(
    name: str,
    funcs: dict[str, ast.FunctionDef],
    hit: set[str],
    seen: set[str] | None = None,
    stop: frozenset[str] = frozenset(),
) -> bool:
    """Does `name` invoke anything in `hit`, directly or through this module?

    `stop` names functions the traversal must NOT descend into. This matters for
    the paid-route computation and the audit was wrong without it: `cache_key()`
    calls `resolve_models()` to name the model in a cache key, and
    `resolve_models()` reaches `client()` -- so a pure string function was
    reported as an unguarded paid route.

    An exemption therefore has to be a CUT POINT in the graph, not merely a name
    filtered from the result. "Spends nothing metered" is a claim about
    everything downstream of it too, otherwise every caller of a metadata helper
    inherits a debt it does not owe. Filtering the output instead would have
    demanded an exemption for each of those callers -- growing the hand-written
    list this audit exists to avoid.
    """
    seen = seen if seen is not None else set()
    if name in seen or name not in funcs:
        return False
    seen.add(name)
    calls = _call_names(funcs[name])
    if calls & hit:
        return True
    return any(
        _reaches(c, funcs, hit, seen, stop)
        for c in calls
        if c in funcs and c not in stop
    )


# ---------------------------------------------------------------------------
# Check 1 -- paid_routes subset-of guarded_routes
# ---------------------------------------------------------------------------


def check_paid_routes_guarded() -> tuple[bool, list[str]]:
    funcs = _module_funcs(ROOT / "ragkit" / "gemini.py")
    lines: list[str] = []

    # Exempt functions are cut points: reaching client() only THROUGH one of
    # them is not paying. See _reaches.
    cut = frozenset(EXEMPT)
    pays = {n for n in funcs if _reaches(n, funcs, {"client"}, stop=cut)}
    guarded = {
        n for n in funcs
        if _reaches(n, funcs, {"check_operation", "budget.check_operation"})
    }

    unguarded = sorted(pays - guarded - set(EXEMPT))
    # A stale exemption is a hole waiting for a rename to fall into it.
    stale = sorted(set(EXEMPT) - set(funcs))

    lines.append(f"  paid routes (reach client()):  {len(pays)}")
    lines.append(f"  guarded (reach check_operation): {len(guarded)}")
    lines.append(f"  exempt by name:                {len(EXEMPT)}")
    for n in sorted(pays & guarded):
        lines.append(f"    [guarded ] {n}")
    for n in sorted(pays & set(EXEMPT)):
        lines.append(f"    [exempt  ] {n}")

    ok = True
    if unguarded:
        ok = False
        lines.append("")
        lines.append("  FAIL: paid routes with no ceiling on them:")
        for n in unguarded:
            lines.append(f"    [UNGUARDED] {n}")
        lines.append(
            "  A ceiling on some paid routes reports itself as protection while "
            "the rest spend freely."
        )
    if stale:
        ok = False
        lines.append("")
        lines.append("  FAIL: exemptions naming functions that no longer exist:")
        for n in stale:
            lines.append(f"    [STALE] {n}")
    return ok, lines


# ---------------------------------------------------------------------------
# Check 2 -- the demo write-guard's allowlist, same shape
# ---------------------------------------------------------------------------


def check_demo_allowlist() -> tuple[bool, list[str]]:
    """Every allowlist pattern must match a route that actually exists.

    The demo guard denies by METHOD and opens by pattern, so a newly added write
    endpoint is safe by default -- that direction is already fine. The failure
    mode here is the reverse: a pattern that matches nothing. Dead config that
    reads as protection, and the open/denied split printed below makes any
    future widening visible in a diff rather than buried in a regex.
    """
    src = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
    routes = set(
        re.findall(
            r'@app\.(?:post|delete|put|patch)\(\s*[\'"]([^\'"]+)[\'"]', src
        )
    )
    # FastAPI path params -> a regex the allowlist patterns can be tried against
    concrete = {r: re.sub(r"\{[^}]+\}", "x", r) for r in routes}

    from app.api import _DEMO_ALLOWED_WRITES

    lines = [f"  write endpoints registered: {len(routes)}"]
    ok = True

    matched_by: dict[str, list[str]] = {p.pattern: [] for p in _DEMO_ALLOWED_WRITES}
    for raw, path in sorted(concrete.items()):
        hits = [p.pattern for p in _DEMO_ALLOWED_WRITES if p.match(path)]
        for h in hits:
            matched_by[h].append(raw)
        lines.append(f"    [{'OPEN   ' if hits else 'denied '}] {raw}")

    dead = [pat for pat, hits in matched_by.items() if not hits]
    if dead:
        ok = False
        lines.append("")
        lines.append("  FAIL: allowlist patterns matching no registered route:")
        for pat in dead:
            lines.append(f"    [DEAD] {pat}")
        lines.append(
            "  A pattern that matches nothing is dead config that reads as intent."
        )
    return ok, lines


# ---------------------------------------------------------------------------
# Check 3 -- every message the guard emits is true on every route it covers
# ---------------------------------------------------------------------------

# Words that name ONE paid route. The ceiling now covers embedding, generation
# and captioning, so a message using any of these describes the neighbouring
# thing rather than the actual one -- which is exactly what "the ingest limit"
# did on an `answer` refusal.
ROUTE_SPECIFIC = ("ingest", "upload", "embed", "caption", "index")


def check_guard_messages_route_neutral() -> tuple[bool, list[str]]:
    import importlib

    from ragkit import budget, config

    lines: list[str] = []
    ok = True
    for kind in ("demo", "internal", "customer"):
        config.DEPLOYMENT_KIND = kind
        importlib.reload(budget)
        config.DEPLOYMENT_KIND = kind  # reload re-reads config; pin it again
        for label, text in (
            ("limit", budget._whose_limit()),
            ("remedy(per-operation)", budget._remedy("per-operation")),
            ("remedy(daily)", budget._remedy("daily")),
        ):
            bad = [w for w in ROUTE_SPECIFIC if w in text.lower()]
            flag = "FAIL" if bad else "ok  "
            lines.append(f"    [{flag}] {kind:9s} {label:22s} {text[:64]}")
            if bad:
                ok = False
                lines.append(f"           names one route: {bad}")
    if not ok:
        lines.append("")
        lines.append(
            "  FAIL: a message naming one route is false on the others the guard "
            "now covers. Widening a guard invalidates strings written when it was "
            "narrow."
        )
    return ok, lines


# ---------------------------------------------------------------------------
# Check 4 -- runtime state under data/index/ is not tracked by git
# ---------------------------------------------------------------------------

# Paths under data/index/ that are BUILD OUTPUT and belong in the repo, each with
# the reason it ships. Everything else found there must be gitignored.
#
# The direction of failure is what makes this safe: a path that is neither declared
# here nor ignored FAILS. So a new runtime store added next year is caught the day
# it appears, rather than shipping someone's data because nobody remembered to
# extend a list in .gitignore.
# Matches `config.DATA_INDEX / "name"` with either quote style. Defined at module
# scope rather than inline because a regex containing both quote characters is
# exactly the string that gets mangled when a patch script rewrites this file --
# which it did, producing a SyntaxError.
_DATA_INDEX_LITERAL = r"""DATA_INDEX\s*/\s*["']([^"']+)["']"""

BUILD_OUTPUT: dict[str, str] = {
    "numpy_index": "the prebuilt index -- the reason data/index/ is tracked at all",
    "ingest_manifest.json": "what was ingested, and under which parser version",
    "manifest.json": "the corpus record; delta detection reads it",
    "resolved_models.json": "which model IDs this key resolved to, recorded not guessed",
}


def check_runtime_state_untracked() -> tuple[bool, list[str]]:
    """Every path under data/index/ is either declared build output or ignored.

    WHY THIS EXISTS. data/index/ was deliberately un-ignored so the prebuilt index
    ships and a fresh clone can answer a question immediately. That was right, and
    it captured more than it named: `conversations/` and `jobs/` live under the
    same directory and are RUNTIME USER STATE. Eight ad-hoc debugging conversations
    and nine job records were staged for a public repo.

    Fixing it by listing those two paths in .gitignore would have restated the
    distinction as a literal, and a literal drifts the moment a third runtime
    directory appears -- which is the hand-maintained-list problem this file exists
    to avoid, one layer down.

    DERIVED FROM TWO SOURCES, UNIONED, because either alone under-covers:

      code literals   `config.DATA_INDEX / "x"` -- catches a store that is named
                      but not yet created
      the filesystem  catches paths composed dynamically. `numpy_index` comes from
                      `DATA_INDEX / name` with a parameter default, so no literal
                      grep can see it -- and a check that silently missed the most
                      important directory under audit would be the "correct on some
                      of the traffic" failure again.
    """
    import subprocess

    idx_dir = ROOT / "data" / "index"
    lines: list[str] = []

    named: set[str] = set()
    for path in (ROOT / "ragkit").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(_DATA_INDEX_LITERAL, src):
            named.add(m.group(1))
    on_disk = {e.name for e in idx_dir.iterdir()} if idx_dir.exists() else set()
    candidates = sorted((named | on_disk) - {".gitkeep"})

    lines.append(f"  paths named in code: {len(named)} | on disk: {len(on_disk)}")

    ok = True
    for name in candidates:
        target = idx_dir / name
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(target)],
            cwd=ROOT, capture_output=True,
        ).returncode == 0
        tracked = bool(subprocess.run(
            ["git", "ls-files", str(target)],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.strip())
        declared = name in BUILD_OUTPUT

        if declared and not tracked and target.exists():
            ok = False
            lines.append(f"    [MISSING ] {name} -- declared build output, not tracked")
        elif declared:
            lines.append(f"    [ships   ] {name}")
        elif ignored and not tracked:
            lines.append(f"    [ignored ] {name}")
        elif tracked:
            ok = False
            lines.append(
                f"    [TRACKED!] {name} -- runtime state staged for the repo. "
                "Declare it in BUILD_OUTPUT with a reason, or gitignore it."
            )
        else:
            lines.append(f"    [untracked] {name} (not ignored, but nothing staged)")

    stale = sorted(set(BUILD_OUTPUT) - on_disk)
    if stale:
        ok = False
        lines.append("")
        lines.append("  FAIL: BUILD_OUTPUT names paths that do not exist:")
        for n in stale:
            lines.append(f"    [STALE] {n}")
    return ok, lines


def main() -> int:
    print("guard coverage -- for a guard, coverage is the invariant\n")
    results = [
        ("paid routes are all guarded", check_paid_routes_guarded()),
        ("demo allowlist has no dead patterns", check_demo_allowlist()),
        ("guard messages are route-neutral", check_guard_messages_route_neutral()),
        ("runtime state is not tracked by git", check_runtime_state_untracked()),
    ]
    failed = 0
    for title, (ok, lines) in results:
        print(f"[{'ok  ' if ok else 'FAIL'}] {title}")
        print("\n".join(lines))
        print()
        failed += 0 if ok else 1
    print(f"{len(results) - failed} passing, {failed} failing")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
