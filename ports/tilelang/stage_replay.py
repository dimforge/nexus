#!/usr/bin/env python3
"""Shared live-Rust fixture replay, native execution and hash-bound evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from cpu_oracle import export_fixtures, verify_probe_file
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


def parse_args(operations, default):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=operations,
        default=default,
    )
    parser.add_argument(
        "--target", choices=["c", "metal", "ascend", "cuda"], default="c"
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.target in ("ascend", "cuda") and not args.compile_only:
        parser.error("CUDA/Ascend execution is unavailable; use --compile-only")
    return args


def run_stage(
    args,
    *,
    oracle,
    buffers,
    make_kernel,
    validate_case,
    check_outputs,
    launch,
    extra_files=(),
    make_program=None,
    diagnostics=None,
    oracle_args=(),
    exporter=None,
):
    """Shared evidence/runtime path; operation modules own their ABI and comparisons."""
    import numpy as np
    import tilelang

    output = args.output.resolve() / args.operation / args.target
    output.mkdir(parents=True, exist_ok=True)
    inputs = evidence_inputs(ROOT)
    report = {"schema": 1, "status": "preparing", "target": args.target}
    write_json(output / "report.json", report)
    oracle_profile = (
        exporter(output / "fixtures")
        if exporter is not None
        else export_fixtures(oracle, output / "fixtures", extra=oracle_args)
    )
    # Fail at export, before accepting a report that cannot be handed off.
    verify_probe_file(output / "cpu-oracle-noalias.ll", oracle_profile)
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
            "rustflags": oracle_profile["rustflags"],
            "rustc": oracle_profile["rustc"],
            "cpu_oracle": oracle_profile,
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
    from ordered import CompilationSession

    compiler_session = CompilationSession()
    active_case = None
    try:
        runner = (
            build_native() if args.target == "metal" and not args.compile_only else None
        )
        for case in suite["cases"]:
            active_case = case["name"]
            directory = fixtures / case["name"]
            before = {
                n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32) for n in buffers
            }
            validate_case(layout, case, before)
            if make_program is not None:
                from ordered import lower_program

                sources, bindings = lower_program(
                    make_program(layout, case, args.target),
                    before,
                    args.target,
                    iterations=case["iterations"],
                    session=compiler_session,
                )
                source = sources[f"kernel.{args.target}"]
            else:
                requested = (
                    {"kind": "cuda", "arch": "sm_80"}
                    if args.target == "cuda"
                    else args.target
                )
                from ordered import validate_launch

                kernel = make_kernel(layout, case, args.target)
                blocks, threads, iterations = launch(case)
                validate_launch(kernel, args.target, blocks, threads)
                artifact = tilelang.lower(
                    kernel,
                    target=requested,
                    enable_host_codegen=False,
                    enable_device_compile=False,
                )
                functions = list(artifact.device_mod.functions.values())
                if len(functions) != 1:
                    raise ValueError("Expected one stage entry point")
                function = functions[0]
                order = [str(param.name) for param in function.params]
                if len(set(order)) != len(order) or not set(order).issubset(buffers):
                    raise ValueError("Unexpected stage bindings")
                source = artifact.kernel_source
                if args.target == "metal":
                    source = "#pragma clang fp contract(off)\n" + source
                bindings = {
                    "schema": 1,
                    "entry": str(function.attrs["global_symbol"]),
                    "buffers": order,
                    "storage_buffers": list(buffers),
                    "buffer_bytes": {n: a.nbytes for n, a in before.items()},
                    "blocks": blocks,
                    "threads": threads,
                    "iterations": iterations,
                }
                sources = {f"kernel.{args.target}": source}
            write_json(directory / "bindings.json", bindings)
            for filename, content in sources.items():
                (directory / filename).write_text(content)
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
                    if make_program is not None:
                        from ordered import execute_c_program

                        execution = execute_c_program(
                            sources,
                            bindings,
                            directory,
                            actual,
                            session=compiler_session,
                        )
                    else:
                        execution = execute_c(source, bindings, directory, actual)
                    report["environment"].update(execution)
                else:
                    report["environment"].update(json.loads(run([runner, directory])))
                    actual = {
                        n: np.fromfile(directory / f"{n}.actual.bin", dtype=np.uint32)
                        for n in buffers
                    }
                expected = {
                    n: np.fromfile(directory / f"{n}.expected.bin", dtype=np.uint32)
                    if n in contract["outputs"]
                    else before[n]
                    for n in buffers
                }
                outcome["max_absolute_errors"] = check_outputs(
                    layout, case, before, actual, expected, contract
                )
                if diagnostics is not None:
                    outcome["diagnostics"] = diagnostics(
                        layout, case, before, actual, expected, contract
                    )
                for n, a in actual.items():
                    a.tofile(directory / f"{n}.actual.bin")
            names = (
                {
                    "bindings.json",
                }
                | set(sources)
                | {f"{n}.bin" for n in buffers}
                | {f"{n}.expected.bin" for n in contract["outputs"]}
                | set(extra_files)
            )
            if not args.compile_only:
                names.update(f"{n}.actual.bin" for n in buffers)
            outcome["files"] = {
                n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
                for n in sorted(names)
            }
            report["cases"].append(outcome)
            print(f"{args.target}: {args.operation}/{case['name']}: passed", flush=True)
        if evidence_inputs(ROOT) != inputs:
            raise ValueError("Sources changed during stage replay")
        report["compilation_reuse"] = compiler_session.hits
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error), failed_case=active_case)
        raise
    finally:
        write_json(output / "report.json", report)
