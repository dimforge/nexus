#!/usr/bin/env python3
"""Replay irregular integer operations against the actual Rust GPU entry function."""

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

BUFFERS = ("Counts", "Ids", "Sorted")


def validate_case(case, arrays):
    import numpy as np

    rows, storage = case["rows"], case["storage"]
    if not 0 <= rows <= 65535 or not 1 <= storage <= 2**31 - 64:
        raise ValueError("Invalid primitive dimensions")
    if arrays["Counts"].size != max(rows, 1) or any(
        arrays[n].size != storage for n in ("Ids", "Sorted")
    ):
        raise ValueError("Invalid primitive buffer sizes")
    ends = arrays["Counts"][:rows].astype(np.uint64)
    if (
        np.any(ends[1:] < ends[:-1])
        or (ends.size and ends[-1] != case["count"])
        or not 0 <= case["count"] <= storage
        or (not rows and case["count"])
    ):
        raise ValueError("Invalid cumulative row ends")


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(case, before)
    errors = {}
    for name in BUFFERS:
        reference = expected[name] if name in contract["outputs"] else before[name]
        np.testing.assert_array_equal(
            actual[name], reference, err_msg=f"{case['name']}: {name}"
        )
        errors[name] = 0.0
    # Sentinel output words past the active prefix must remain exactly unchanged.
    np.testing.assert_array_equal(
        actual["Sorted"][case["count"] :], before["Sorted"][case["count"] :]
    )
    return errors


def main():
    import numpy as np
    import tilelang
    from primitive_kernels import make_kernel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["stabilize"], default="stabilize")
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
            "primitives",
            "--",
            output / "fixtures",
        ],
        cwd=HERE / "reference",
    )
    layout = json.loads((output / "fixtures/layout.json").read_text())
    fixtures = output / "fixtures/stabilize"
    suite = json.loads((fixtures / "suite.json").read_text())
    contract = json.loads((HERE / "contracts/stabilize.json").read_text())
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
            validate_case(case, before)
            requested = (
                {"kind": "cuda", "arch": "sm_80"}
                if args.target == "cuda"
                else args.target
            )
            artifact = tilelang.lower(
                make_kernel(case, args.target),
                target=requested,
                enable_host_codegen=False,
                enable_device_compile=False,
            )
            functions = list(artifact.device_mod.functions.values())
            if len(functions) != 1:
                raise ValueError("Expected one primitive entry point")
            function = functions[0]
            order = [str(param.name) for param in function.params]
            if len(set(order)) != len(order) or not set(order).issubset(BUFFERS):
                raise ValueError("Unexpected primitive bindings")
            source = artifact.kernel_source
            bindings = {
                "schema": 1,
                "entry": str(function.attrs["global_symbol"]),
                "buffers": order,
                "storage_buffers": list(BUFFERS),
                "buffer_bytes": {n: a.nbytes for n, a in before.items()},
                "blocks": max(case["rows"], 1),
                "threads": 64,
                "iterations": 1,
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
            names = {
                "bindings.json",
                f"kernel.{args.target}",
                "Sorted.expected.bin",
            } | {f"{n}.bin" for n in BUFFERS}
            if not args.compile_only:
                names.update(f"{n}.actual.bin" for n in BUFFERS)
            outcome["files"] = {
                n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
                for n in sorted(names)
            }
            report["cases"].append(outcome)
            print(f"{args.target}: stabilize/{case['name']}: passed", flush=True)
        if evidence_inputs(ROOT) != inputs:
            raise ValueError("Sources changed during primitive replay")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error), failed_case=active_case)
        raise
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
