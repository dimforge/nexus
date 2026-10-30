#!/usr/bin/env python3
"""Pinned khal CPU oracle, with mutable-reference noalias disabled and checked."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
FLAGS = ("-Zmutable-noalias=no",)
PROFILE = "khal-cpu-noalias-disabled-v1"


def toolchain():
    return tomllib.loads((REFERENCE / "rust-toolchain.toml").read_text())["toolchain"][
        "channel"
    ]


def environment(inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    # Respect Cargo's precedence, then reject unrelated flags rather than silently
    # dropping user settings or claiming to have tested an unrecorded profile.
    flags = (
        env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        if env.get("CARGO_ENCODED_RUSTFLAGS")
        else shlex.split(env.get("RUSTFLAGS", ""))
    )
    if flags and flags != list(FLAGS):
        raise ValueError(
            "Unset custom RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS for the pinned CPU oracle"
        )
    for name in ("RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        if env.get(name):
            raise ValueError(f"Unset {name} for the pinned CPU oracle")
    env.pop("RUSTFLAGS", None)
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(FLAGS)
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTUP_TOOLCHAIN"] = toolchain()
    return env


def run(command, env):
    result = subprocess.run(
        [str(x) for x in command],
        check=False,
        cwd=REFERENCE,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"CPU oracle command failed: {command}\n{result.stderr}\n{result.stdout}"
        )
    return result.stdout.strip()


def check_llvm(ir):
    signatures = {}
    for kind in ("mutable", "shared"):
        match = re.search(
            rf"^define [^\n]*@{kind}_reference_probe\(([^\n]*)\)", ir, re.MULTILINE
        )
        if not match:
            raise ValueError(f"Missing {kind} reference LLVM probe")
        signatures[kind] = bool(re.search(r"\bnoalias\b", match[1]))
    if signatures != {"mutable": False, "shared": True}:
        raise ValueError(f"Unexpected reference noalias attributes: {signatures}")
    return signatures


def verify(directory, env):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = REFERENCE / "noalias_probe.rs"
    ir_path = directory / "cpu-oracle-noalias.ll"
    # Keep frontend attributes visible before LLVM infers attributes of its own.
    run(
        [
            "rustc",
            f"+{toolchain()}",
            *FLAGS,
            "-Copt-level=3",
            "-Cno-prepopulate-passes",
            "--emit=llvm-ir",
            source,
            "-o",
            ir_path,
        ],
        env,
    )
    attributes = check_llvm(ir_path.read_text())
    return {
        "profile": PROFILE,
        "rustc": run(["rustc", f"+{toolchain()}", "--version"], env),
        "rustflags": list(FLAGS),
        "rustc_bootstrap": env["RUSTC_BOOTSTRAP"],
        "mutable_reference_noalias": attributes["mutable"],
        "shared_reference_noalias": attributes["shared"],
        "probe_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "llvm_ir_sha256": hashlib.sha256(ir_path.read_bytes()).hexdigest(),
    }


def verify_probe_file(path, profile):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile.get("llvm_ir_sha256"):
        raise ValueError("Modified CPU oracle LLVM probe")
    check_llvm(path.read_text())


def cargo(arguments, env=None):
    if not arguments:
        raise ValueError("Supply a Cargo subcommand")
    return run(
        [
            "cargo",
            f"+{toolchain()}",
            arguments[0],
            "--manifest-path",
            REFERENCE / "Cargo.toml",
            *arguments[1:],
        ],
        environment() if env is None else env,
    )


def export_fixtures(binary, directory, extra=()):
    env = environment()
    profile = verify(Path(directory).parent, env)
    cargo(["run", "--locked", "--bin", binary, "--", directory, *extra], env)
    return profile


def export_native_scenes(directory):
    """Export the real Nexus CPU conversion without a GPU or Metal dependency."""
    env = environment()
    directory = Path(directory).resolve()
    profile = verify(directory.parent, env)
    run(
        [
            "cargo",
            f"+{toolchain()}",
            "run",
            "--locked",
            "--manifest-path",
            HERE / "native" / "Cargo.toml",
            "--no-default-features",
            "--features",
            "scene-upload",
            "--bin",
            "runtime-upload",
            "--target-dir",
            HERE.parent.parent / "target",
            "--",
            directory,
        ],
        env,
    )
    return profile


if __name__ == "__main__":
    print(cargo(sys.argv[1:]))
