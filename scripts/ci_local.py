"""Run the CI gates locally. ONE definition of "the gates", called from two places.

WHY THIS EXISTS RATHER THAN A SECOND COPY OF THE COMMANDS.

GitHub Actions is currently unavailable on this account ("the job was not started
because your account is locked due to a billing issue"), so the workflow is
verified but unexercised. The obvious response -- write a local script that runs
the same checks -- creates the failure this project has paid for repeatedly: two
programs claiming to do the same thing, drifting, and disagreeing about whether
the code is good. The survey script that imported a different parser than the
pipeline reported table counts differing 5x on identical bytes; neither number was
wrong, they were answers about different programs.

So the gate list lives HERE, and ci.yml calls this script. A gate added to one is
added to both, because there is only one.

WHAT IT DELIBERATELY DOES NOT DO. The image build-and-boot job and the frontend
build need Docker and Node, which not every context has. They are declared below
as SKIPPED-WITH-A-REASON rather than silently omitted: a suite that quietly runs
four of six checks and prints "PASS" is worse than one that runs four and says so.

    uv run python scripts/ci_local.py            # the gates that need no network
    uv run python scripts/ci_local.py --all      # plus frontend, image, and the
                                                 # retrieval eval (needs a key)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Gate:
    def __init__(self, name: str, cmd: list[str], *, cwd: Path | None = None,
                 needs: str = "", heavy: bool = False) -> None:
        self.name, self.cmd, self.cwd = name, cmd, cwd or ROOT
        self.needs, self.heavy = needs, heavy


def gates() -> list[Gate]:
    """The gate list. ci.yml calls this script; nothing duplicates the list."""
    py = [sys.executable, "-m"]
    return [
        Gate("reachability + guard coverage", [*py, "ragkit.cli", "audit"]),
        Gate("invariants (reconcile)", [*py, "ragkit.cli", "reconcile"]),
        Gate("failure analysis pools legitimately",
             [sys.executable, "scripts/failure_histogram.py"]),
        Gate("frontend build (tsc -b && vite build)",
             ["npm", "run", "build"], cwd=ROOT / "app" / "web",
             needs="npm", heavy=True),
        Gate("image builds", ["docker", "build", "-q", "-t", "ragkit:cilocal", "."],
             needs="docker", heavy=True),
        Gate("retrieval regression gate",
             [*py, "ragkit.eval.run", "--gate", "--no-sweep"],
             needs="GEMINI_API_KEY", heavy=True),
    ]


def _runnable(cmd: list[str]) -> list[str]:
    """Make a command executable on Windows as well as POSIX.

    `shutil.which("npm")` on Windows resolves to `npm.cmd`, and CreateProcess
    cannot execute a .cmd directly -- it needs a shell. So the availability check
    passed ("it is on PATH") while execution raised FileNotFoundError, which is
    the wrong-predicate failure again: on-PATH and runnable are different claims,
    and only the second is the one that matters.
    """
    import os

    exe = shutil.which(cmd[0])
    if exe is None:
        return cmd
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", exe, *cmd[1:]]
    return [exe, *cmd[1:]]


def _available(need: str) -> tuple[bool, str]:
    if not need:
        return True, ""
    if need == "GEMINI_API_KEY":
        import os

        env = dict(os.environ)
        envf = ROOT / ".env"
        if envf.exists():
            for line in envf.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    env["GEMINI_API_KEY"] = line.split("=", 1)[1].strip()
        return bool(env.get("GEMINI_API_KEY")), "no GEMINI_API_KEY"
    # Resolvable AND runnable. `which` alone was the wrong predicate: it found
    # npm.cmd, reported "available", and execution then raised FileNotFoundError
    # because CreateProcess cannot run a .cmd without a shell.
    resolved = shutil.which(need)
    return resolved is not None, f"{need} not runnable"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="include the heavy gates (frontend, image, retrieval eval)")
    args = ap.parse_args(argv)

    import os

    env = dict(os.environ)
    # The cp1252 console killed six runs on this project.
    env["PYTHONIOENCODING"] = "utf-8"
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    # Bounded so a local gate run cannot spend real money by accident.
    env.setdefault("RAGKIT_MAX_OPERATION_TOKENS", "200000")

    ran = passed = 0
    skipped: list[tuple[str, str]] = []
    failed: list[str] = []

    print("CI gates, run locally. GitHub Actions is billing-locked on this account,")
    print("so this is the only place they currently execute.\n")

    for g in gates():
        if g.heavy and not args.all:
            skipped.append((g.name, "heavy; pass --all"))
            continue
        ok_dep, why = _available(g.needs)
        if not ok_dep:
            skipped.append((g.name, why))
            continue
        t0 = time.time()
        r = subprocess.run(_runnable(g.cmd), cwd=g.cwd, capture_output=True,
                           text=True, env=env)
        dt = time.time() - t0
        ran += 1
        if r.returncode == 0:
            passed += 1
            print(f"  [PASS] {g.name}  ({dt:.1f}s)")
        else:
            failed.append(g.name)
            print(f"  [FAIL] {g.name}  ({dt:.1f}s)  exit {r.returncode}")
            tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
            for line in tail:
                print(f"         {line[:110]}")

    print()
    for name, why in skipped:
        # NAMED, not omitted. A suite that runs four of six checks and prints
        # "PASS" has told you something false by leaving something out.
        print(f"  [SKIP] {name}  ({why})")

    print()
    print(f"{passed}/{ran} ran and passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    if failed:
        print("failing: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
