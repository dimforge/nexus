#!/usr/bin/env python3
"""Build a pinned Ascend-enabled TileLang wheel without CANN or NPU access.

Build in a dedicated directory. Existing checkouts must match the pin and have
no tracked edits; this tool never resets a checkout or changes the current venv.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_source(source, lock):
    if git(source, "rev-parse", "HEAD") != lock["revision"]:
        raise ValueError(
            "Compiler revision differs from compiler.lock.json; refusing to reset it"
        )
    if git(source, "diff", "HEAD", "--exit-code"):
        raise ValueError("Compiler checkout has tracked edits")
    for path, revision in lock["submodules"].items():
        if git(source / path, "rev-parse", "HEAD") != revision:
            raise ValueError(f"Unexpected submodule revision: {path}")
    submodules = git(source, "submodule", "status", "--recursive")
    if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
        raise ValueError("Uninitialized or changed compiler submodule")
    return submodules


def receipt(source, wheels, lock, cuda_codegen_only=False):
    return {
        "schema": 1,
        "level": "compiler_built",
        "revision": lock["revision"],
        "submodules": verify_source(source, lock),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            **lock["environment"],
            "USE_METAL": "ON" if sys.platform == "darwin" else "OFF",
        },
        "wheels": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(wheels.glob("tilelang-*.whl"))
        ],
        "local_patches": (
            {
                "compiler_cuda_codegen.patch": hashlib.sha256(
                    (HERE / "compiler_cuda_codegen.patch").read_bytes()
                ).hexdigest()
            }
            if cuda_codegen_only
            else {}
        ),
        "cuda_source_lowering_only": cuda_codegen_only,
        "device_execution": "unvalidated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Dedicated build directory")
    parser.add_argument(
        "--source", type=Path, help="Existing checkout at the exact pinned revision"
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--cuda-codegen-only",
        action="store_true",
        help="Apply the recorded compiler-only CUDA patch; no toolkit, driver or device execution",
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Record an already-built wheel in DIRECTORY/wheels",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if sys.platform == "win32":
        parser.error("Use Linux/WSL or macOS for Ascend compiler builds")
    lock = json.loads((HERE / "compiler.lock.json").read_text())
    root = args.directory.resolve()
    source = args.source.resolve() if args.source else root / "source"
    wheels = root / "wheels"
    root.mkdir(parents=True, exist_ok=True)
    if not args.record_only:
        if not source.exists():
            run(["git", "clone", "--no-checkout", lock["repository"], source])
            run(["git", "-C", source, "checkout", "--detach", lock["revision"]])
        if git(source, "rev-parse", "HEAD") != lock["revision"]:
            raise ValueError(
                "Existing checkout differs from pin; use another directory"
            )
        run(
            [
                "git",
                "-C",
                source,
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--depth",
                "1",
            ]
        )
        verify_source(source, lock)
        build_env = root / "build-env"
        venv.EnvBuilder(with_pip=True).create(build_env)
        python = build_env / "bin/python"
        # Normal venv: native library rpaths resolve local z3/tvm-ffi dependencies.
        run([python, "-m", "pip", "install", *lock["build_packages"]])
        if sys.platform.startswith("linux"):
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "torch==2.9.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )
        run([python, "-m", "pip", "install", "-r", HERE / "requirements.txt"])
        wheels.mkdir(exist_ok=True)
        environment = {
            **os.environ,
            **lock["environment"],
            "USE_METAL": "ON" if sys.platform == "darwin" else "OFF",
            "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
            "TILELANG_CUDA_CODEGEN_ONLY": "ON" if args.cuda_codegen_only else "OFF",
        }
        # CMake arguments override cached backend settings in an existing build.
        environment["CMAKE_ARGS"] = " ".join(
            f"-D{k}={v}"
            for k, v in environment.items()
            if k in {*lock["environment"], "USE_METAL", "TILELANG_CUDA_CODEGEN_ONLY"}
        )
        patch = HERE / "compiler_cuda_codegen.patch"
        if args.cuda_codegen_only:
            run(["git", "-C", source, "apply", "--check", patch])
            run(["git", "-C", source, "apply", patch])
        try:
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    wheels,
                    source,
                ],
                env=environment,
            )
        finally:
            if args.cuda_codegen_only:
                run(["git", "-C", source, "apply", "--reverse", patch])
    result = receipt(source, wheels, lock, args.cuda_codegen_only)
    if not result["wheels"]:
        raise ValueError("No compiler wheel found")
    (root / "compiler-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(root / "compiler-receipt.json")


if __name__ == "__main__":
    main()
