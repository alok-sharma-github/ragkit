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

KNOWN BLIND SPOT -- ALIASED IMPORTS. This audit follows NAMES, so
`from x import explain as explain_fusion` breaks the trail: `explain` is wired
into /api/ask and still reports unreachable. That is how a live function hides
here, and it is the one false-positive shape to expect. Written down because the
next person to alias an import will silently drop a function out of coverage,
and a blind spot nobody has recorded is indistinguishable from a clean result. "Unreachable" means
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
    """A FastAPI handler: @app.get / @app.post / ..."""
    for dec in getattr(node, "decorator_list", []) or []:
        f = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(f, ast.Attribute) and f.attr in {"get", "post", "put", "delete", "patch"}:
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

        # Module-level statements execute on import, so their references are live.
        mod_level: set[str] = set()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mod_level |= _names_used(node)
        module_refs[rel] = mod_level

        def add(qual: str, short: str, line: int, kind: str, body: ast.AST, root: bool) -> None:
            d = Defn(qual, short, rel, line, kind)
            d.refs = _names_used(body)
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
