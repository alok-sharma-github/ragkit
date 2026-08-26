"""
"Name the caller" -- as REACHABILITY FROM ENTRY POINTS, not reference counting.

WHY THIS REPLACES audit_callers.py. v1 asked "is this name referenced anywhere?"
That is the wrong question, and it leaves open exactly the loophole this bug
class likes: a test file, an experiment, or a scratch script that touches a
function is enough to hide it forever. `capabilities()` could be exercised by a
test and pass the audit while being absent from every request.

So the question is "is this reachable from something a user can trigger?" Roots
are the real entry points -- HTTP handlers, CLI subcommands, main() -- and a
reference from experiments/ or scripts/ does not count.

THE BUG CLASS, seven confirmed instances:

    child provenance guard   on the child        the model reads the parent
    _fix_console()           in cli.py           ad-hoc scripts bypassed the CLI
    text_source label        set on the Block    never copied to the Chunk
    condense()               in its module       /api/ask never called it
    capabilities()           written, described  wired nowhere
    write_manifest()         argued for, built   only to_dicts() was called
    route_drift()            written             no caller until conversations exist

Every one produced a WORKING system. Three were described to the user as
complete -- accurately, as code. "It works" and "it runs" are different claims
and read almost identically in prose.

    A module is not done when it works. It is done when something on the
    request path calls it.

LIMITS, stated because a tool that overstates its certainty is the same mistake
it hunts: AST name/attribute references traversed from the roots. It cannot
follow getattr, registries, or names assembled from strings.

FIXED BLIND SPOT -- ALIASED IMPORTS. This used to follow bare NAMES only, so
`from x import explain as explain_fusion` broke the trail and `explain` reported
unreachable while being live on /api/ask. Per-file alias maps now resolve it. Kept
in the record because the shape is worth recognising: the audit was accurate about
its own limitation for weeks, and being accurate about a hole is not the same as
not having one -- it also kept this script permanently non-zero, which is how a
gate stops being read. "Unreachable" means
"name the caller or delete it", never "provably dead". It errs toward FALSE
ALARMS on purpose -- v1 briefly had the opposite bias (it skipped same-file
calls) and the resulting noise is what got it fixed. A check that errs toward
rejection announces its own bugs; one that errs toward acceptance hides them.

CROSS-REFERENCED WITH THE DEFERRAL LIST, because there are two kinds of dead
code and this tool cannot tell them apart. Dead-because-forgotten is a bug;
dead-because-deferred is a decision. Six of nine were the second kind, and
eyeballing that weekly is how a forgotten one eventually gets waved through on
the strength of the prior -- so the comparison happens in code.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCAN_DIRS = ("ragkit", "app")

EXEMPT_NAMES = {
    "to_json", "from_json", "render", "to_dicts", "summary", "main", "app",
    "__enter__", "__exit__", "__bool__", "__init__", "__post_init__",
}


class Defn:
    __slots__ = ("qual", "short", "file", "line", "kind", "refs", "is_root")

    def __init__(self, qual: str, short: str, file: str, line: int, kind: str) -> None:
        self.qual, self.short, self.file, self.line, self.kind = qual, short, file, line, kind
        self.refs: set[str] = set()
        self.is_root = False


def _names_used(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
    return out


def _is_route(node: ast.AST) -> bool:
    """A FastAPI entry point: @app.get / @app.post / ... or @app.middleware.

    MIDDLEWARE IS A ROOT, and omitting it was a real hole rather than a nicety.
    `demo_guard` runs on EVERY request -- it is the most reachable function in the
    web layer -- and it reported as unreachable because this predicate only knew
    about verb decorators. So the audit was calling the demo's entire write
    protection dead code.

    That is this tool's own bug class turned on itself: a check whose predicate
    covers most of the thing it names. Verb handlers were the only entry shape
    when it was written, and middleware became one later without the predicate
    being widened -- the same "correct when written, narrow afterwards" failure as
    the ceiling wired to one of three paid routes.
    """
    for dec in getattr(node, "decorator_list", []) or []:
        f = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(f, ast.Attribute) and f.attr in {
            "get", "post", "put", "delete", "patch",
            "middleware",          # @app.middleware("http") -- runs on every request
            "exception_handler",   # error paths are reachable too
            "on_event",            # startup/shutdown hooks
        }:
            return True
    return False


def collect() -> tuple[dict[str, Defn], dict[str, set[str]]]:
    defs: dict[str, Defn] = {}
    module_refs: dict[str, set[str]] = {}
    files = [p for d in SCAN_DIRS for p in (ROOT / d).rglob("*.py")
             if "__pycache__" not in str(p)]

    for path in sorted(files):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        # ALIASED IMPORTS, per file. This audit resolves references by SHORT NAME,
        # so `from ragkit.index.fusion import explain as explain_fusion` broke the
        # trail: /api/ask calls `explain_fusion`, no definition is named that, and
        # `explain` reported unreachable while being live on the request path.
        #
        # The docstring called this a known blind spot and warned that "the next
        # person to alias an import will silently drop a function out of coverage".
        # A recorded blind spot is better than an unrecorded one and still lets
        # real dead code hide behind a false positive nobody re-reads -- and it
        # kept this audit permanently non-zero, which is how a gate becomes noise.
        #
        # Built per FILE rather than as one global map. A global map would mark a
        # genuinely dead `foo` reachable because some unrelated module aliased
        # something else to `foo` -- weakening the check to close a blind spot,
        # which is the wrong trade.
        aliases: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.asname and a.asname != a.name:
                        aliases[a.asname] = a.name.split(".")[-1]

        def _expand(refs: set[str]) -> set[str]:
            return refs | {aliases[r] for r in refs if r in aliases}

        # Module-level statements execute on import, so their references are live.
        mod_level: set[str] = set()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mod_level |= _names_used(node)
        module_refs[rel] = _expand(mod_level)

        def add(qual: str, short: str, line: int, kind: str, body: ast.AST, root: bool) -> None:
            d = Defn(qual, short, rel, line, kind)
            d.refs = _expand(_names_used(body))
            d.is_root = root
            defs[qual] = d

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                root = _is_route(node) or node.name == "main" or node.name.startswith("cmd_")
                add(f"{rel}::{node.name}", node.name, node.lineno, "def", node, root)
            elif isinstance(node, ast.ClassDef):
                add(f"{rel}::{node.name}", node.name, node.lineno, "class", node, False)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(f"{rel}::{node.name}.{sub.name}", sub.name, sub.lineno,
                            "method", sub, False)
    return defs, module_refs


def reachable(defs: dict[str, Defn], module_refs: dict[str, set[str]]) -> set[str]:
    by_short: dict[str, list[str]] = defaultdict(list)
    for q, d in defs.items():
        by_short[d.short].append(q)

    seen: set[str] = set()
    queue: deque[str] = deque()
    for q, d in defs.items():
        if d.is_root:
            seen.add(q)
            queue.append(q)

    seed: set[str] = set()
    for names in module_refs.values():
        seed |= names
    for name in seed:
        for q in by_short.get(name, []):
            if q not in seen:
                seen.add(q)
                queue.append(q)

    while queue:
        cur = defs[queue.popleft()]
        for name in cur.refs:
            for q in by_short.get(name, []):
                if q not in seen:
                    seen.add(q)
                    queue.append(q)
    return seen


def main() -> int:
    defs, module_refs = collect()
    live = reachable(defs, module_refs)

    # EXPLICIT symbol names, not fuzzy text. The fuzzy version excused
    # `Chunk.citation` on the strength of the word "citation" appearing in an
    # unrelated deferral -- a forgotten orphan waved through by a keyword.
    excused: dict[str, str] = {}
    try:
        from ragkit import deferred
        for d in deferred.review()["deferrals"]:
            for sym in d.get("orphans", ()):
                excused[sym] = d["name"]
    except Exception:  # noqa: BLE001
        pass

    orphans = [d for q, d in sorted(defs.items())
               if q not in live and d.short not in EXEMPT_NAMES]

    explained, unexplained = [], []
    for d in orphans:
        tail = d.qual.split("::")[-1]
        why = excused.get(tail) or excused.get(d.short)
        if why:
            explained.append((d, why))
        else:
            unexplained.append(d)

    print(f"scanned {len(defs)} definitions across {len(module_refs)} files")
    print(f"reachable from entry points : {len(live)}")
    print(f"NOT reachable               : {len(orphans)}"
          f"  ({len(explained)} explained by a deferral, {len(unexplained)} unexplained)")
    print()
    print("Roots = HTTP handlers, CLI subcommands, main(). A reference from")
    print("experiments/ or scripts/ does NOT count -- that was v1's loophole.")
    print()
    print("UNEXPLAINED -- name the caller or delete it:" if unexplained
          else "UNEXPLAINED: none")
    for d in unexplained:
        print(f"    {d.kind:6s} {d.file}:{d.line}  {d.qual.split('::')[-1]}")
    if explained:
        print()
        print("explained by a deferred decision (waiting, not forgotten):")
        for d, why in explained:
            print(f"    {d.kind:6s} {d.file}:{d.line}  {d.qual.split('::')[-1]}"
                  f"   <- deferral: {why}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
