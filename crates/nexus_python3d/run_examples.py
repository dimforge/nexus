#!/usr/bin/env python3
"""Run every ``nexus3d`` example, one after another.

Each example opens a viewer window and runs until you close it; the next
example then launches automatically. Examples run as **separate subprocesses**
(each ends with ``os._exit`` and the GPU/windowing state must not be reused
in-process), so a crash in one example never stops the rest.

Build the module first (see the README), e.g.::

    maturin develop --release -m crates/nexus_python3d/Cargo.toml --features metal

Usage::

    python crates/nexus_python3d/run_examples.py             # every example
    python crates/nexus_python3d/run_examples.py boxes3 sand3   # only these, in this order
    python crates/nexus_python3d/run_examples.py --list      # list what would run, then exit

Note: `urdf3` and `mujoco_menagerie3` need external assets supplied via
environment variables (see the README); without them they exit with an error,
which the runner reports before moving on.

Press Ctrl+C in this terminal to stop the whole run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# Curated run order, grouped by category. Any example file on disk that is not
# listed here is appended afterwards, sorted, so newly added ports are still
# picked up automatically.
ORDER = [
    # Rigid bodies
    "boxes3",
    "balls3",
    "boxes_and_balls3",
    "primitives3",
    "pyramid3",
    "many_pyramids3",
    "keva3",
    "compound3",
    "trimesh3",
    "dynamic_rbd3",
    # Joints & articulations
    "joint_ball3",
    "joint_fixed3",
    "joint_prismatic3",
    "joint_revolute3",
    "joints3",
    "multibody_pendulum3",
    "joint_revolute_batch3",
    "many_pyramids_batch3",
    # Robots (need external assets via env vars; see README)
    "urdf3",
    "mujoco_menagerie3",
]


def discover():
    """All example stems present on disk (excluding this runner)."""
    return sorted(
        p.stem
        for p in EXAMPLES_DIR.glob("*.py")
        if not p.stem.startswith("_")
    )


def build_run_list(names):
    on_disk = set(discover())

    if names:
        missing = [n for n in names if n not in on_disk]
        if missing:
            sys.exit(f"unknown example(s): {', '.join(missing)}")
        return names  # explicit list runs verbatim.

    ordered = [n for n in ORDER if n in on_disk]
    # Append any examples on disk not covered by ORDER.
    extras = sorted(on_disk - set(ORDER))
    return ordered + extras


def main():
    parser = argparse.ArgumentParser(
        description="Run the nexus3d examples one after another.",
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="specific example(s) to run (default: all self-contained ones)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the examples that would run, then exit",
    )
    args = parser.parse_args()

    run_list = build_run_list(args.names)
    if not run_list:
        sys.exit("no examples found — did you build the module and run from the repo?")

    if args.list:
        print("\n".join(run_list))
        return

    total = len(run_list)
    failures = []
    for i, name in enumerate(run_list, 1):
        path = EXAMPLES_DIR / f"{name}.py"
        print(f"\n[{i}/{total}] ▶ {name} — close the window to continue…", flush=True)
        try:
            result = subprocess.run([sys.executable, str(path)])
        except KeyboardInterrupt:
            print(f"\nInterrupted during {name}; stopping.")
            break
        if result.returncode == 0:
            print(f"[{i}/{total}] ✓ {name}")
        else:
            print(f"[{i}/{total}] ✗ {name} (exit {result.returncode})")
            failures.append(name)

    print(f"\nDone. {total - len(failures)}/{total} finished cleanly.")
    if failures:
        print("Non-zero exit:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
