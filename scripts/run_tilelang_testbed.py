#!/usr/bin/env python3
"""Launch the visual TileLang testbed with the verified khal CPU compiler profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ports/tilelang"))
import cpu_oracle  # noqa: E402


def interpreter_path(value):
    # A venv's python is commonly a symlink. Dereferencing it changes sys.prefix
    # and loses the environment's installed compiler packages.
    path = Path(os.path.abspath(Path(value).expanduser()))
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c", "metal"), default="c")
    parser.add_argument("--python", default=os.environ.get("NEXUS_TILELANG_PYTHON"))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("testbed_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.python:
        parser.error("--python or NEXUS_TILELANG_PYTHON must name the pinned TileLang interpreter")
    env = cpu_oracle.environment()
    env["NEXUS_TILELANG_PYTHON"] = interpreter_path(args.python)
    env["NEXUS_TILELANG_PORT"] = str(ROOT / "ports/tilelang")
    evidence = ROOT / "target/tilelang-testbed"
    profile = cpu_oracle.verify(evidence, env)
    (evidence / "cpu-oracle-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    features = "tilelang,metal" if args.backend == "metal" else "tilelang"
    command = ["cargo", "run", "--locked", "-p", "nexus_examples_3d", "--bin", "all_examples3", "--features", features]
    if args.release:
        command.append("--release")
    command += ["--", "--example", "tileLangComparison", "--tilelang", args.backend, "--run"]
    if args.compare:
        command.append("--compare")
    extra = args.testbed_args
    command += extra[1:] if extra[:1] == ["--"] else extra
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
