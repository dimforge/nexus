#!/usr/bin/env python3
"""Replay body stages and no-contact composition against live Rust entry points."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from parity import evidence_inputs
from replay import (
    HERE,
    ROOT,
    build_native,
    compiler_libraries,
    execute_c,
    run,
    write_json,
)

from body_validation import BUFFERS, OPERATIONS, validate_case, check_outputs


def main():
    import numpy as np
    import tilelang
    from body_kernels import make_kernel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=OPERATIONS,
        default="free_step",
    )
    parser.add_argument(
        "--target", choices=["c", "metal", "ascend", "cuda"], default="c"
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.target in ("ascend", "cuda") and not args.compile_only:
        parser.error("CUDA/Ascend execution is unavailable; use --compile-only")
    output = args.output.resolve() / args.operation / args.target
    output.mkdir(parents=True, exist_ok=True)
    inputs = evidence_inputs(ROOT)
    report = {"schema": 1, "status": "preparing", "target": args.target}
    write_json(output / "report.json", report)
    run(
        [
            "cargo",
            "run",
            "--locked",
            "--manifest-path",
            HERE / "reference/Cargo.toml",
            "--bin",
            "bodies",
            "--",
            output / "fixtures",
        ],
        cwd=HERE / "reference",
    )
    layout = json.loads((output / "fixtures/layout.json").read_text())
    fixtures = output / f"fixtures/{args.operation}"
    suite = json.loads((fixtures / "suite.json").read_text())
    contract = json.loads((HERE / f"contracts/{args.operation}.json").read_text())
    report.update(
        feature=contract["feature"],
        inputs=inputs,
        suite="portable",
        level="source_lowered" if args.compile_only else "executed",
        reference_commit=run(["git", "-C", ROOT, "rev-parse", "HEAD"]),
        layout_sha256=hashlib.sha256(
            (output / "fixtures/layout.json").read_bytes()
        ).hexdigest(),
        compiler_libraries=compiler_libraries(),
        cases=[],
        environment={
            "python": platform.python_version(),
            "os": platform.platform(),
            "rustflags": os.environ.get("RUSTFLAGS", ""),
            "rustc": run(["rustc", "--version"], cwd=HERE / "reference"),
            **{
                key: importlib.metadata.version(package)
                for key, package in [
                    ("tilelang", "tilelang"),
                    ("torch", "torch"),
                    ("numpy", "numpy"),
                    ("tvm_ffi", "apache-tvm-ffi"),
                    ("z3", "z3-solver"),
                ]
            },
        },
    )
    active_case = None
    try:
        runner = (
            build_native() if args.target == "metal" and not args.compile_only else None
        )
        for case in suite["cases"]:
            active_case = case["name"]
            directory = fixtures / case["name"]
            before = {
                n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32) for n in BUFFERS
            }
            validate_case(layout, case, before)
            requested = (
                {"kind": "cuda", "arch": "sm_80"}
                if args.target == "cuda"
                else args.target
            )
            artifact = tilelang.lower(
                make_kernel(layout, case, args.target),
                target=requested,
                enable_host_codegen=False,
                enable_device_compile=False,
            )
            functions = list(artifact.device_mod.functions.values())
            if len(functions) != 1:
                raise ValueError("Expected one body entry point")
            function = functions[0]
            order = [str(param.name) for param in function.params]
            if len(set(order)) != len(order) or not set(order).issubset(BUFFERS):
                raise ValueError("Unexpected body bindings")
            source = artifact.kernel_source
            if args.target == "metal":
                source = "#pragma clang fp contract(off)\n" + source
            bindings = {
                "schema": 1,
                "entry": str(function.attrs["global_symbol"]),
                "buffers": order,
                "storage_buffers": list(BUFFERS),
                "buffer_bytes": {n: a.nbytes for n, a in before.items()},
                "blocks": max((case["slots"] + 63) // 64, 1),
                "threads": 64,
                "iterations": case["iterations"],
            }
            write_json(directory / "bindings.json", bindings)
            (directory / f"kernel.{args.target}").write_text(source)
            outcome = {
                "name": case["name"],
                "count": case["count"],
                "status": "passed",
                "fixture": case,
                "bindings": bindings,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
            if not args.compile_only:
                actual = {n: a.copy() for n, a in before.items()}
                if args.target == "c":
                    report["environment"].update(
                        execute_c(source, bindings, directory, actual)
                    )
                else:
                    report["environment"].update(json.loads(run([runner, directory])))
                    actual = {
                        n: np.fromfile(directory / f"{n}.actual.bin", dtype=np.uint32)
                        for n in BUFFERS
                    }
                expected = {
                    n: np.fromfile(directory / f"{n}.expected.bin", dtype=np.uint32)
                    if n in contract["outputs"]
                    else before[n]
                    for n in BUFFERS
                }
                outcome["max_absolute_errors"] = check_outputs(
                    layout, case, before, actual, expected, contract
                )
                for n, a in actual.items():
                    a.tofile(directory / f"{n}.actual.bin")
            names = (
                {
                    "bindings.json",
                    f"kernel.{args.target}",
                }
                | {f"{n}.bin" for n in BUFFERS}
                | {f"{n}.expected.bin" for n in contract["outputs"]}
                | {"RbdParams.bin", "BatchIndices.bin", "Gravity.bin"}
            )
            if not args.compile_only:
                names.update(f"{n}.actual.bin" for n in BUFFERS)
            outcome["files"] = {
                n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
                for n in sorted(names)
            }
            report["cases"].append(outcome)
            print(f"{args.target}: {args.operation}/{case['name']}: passed", flush=True)
        if evidence_inputs(ROOT) != inputs:
            raise ValueError("Sources changed during body replay")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error), failed_case=active_case)
        raise
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
