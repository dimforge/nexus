#!/usr/bin/env python3
"""Triangulate Rust CPU, original Rust Metal, and TileLang Metal on identical inputs.

This diagnostic never promotes execution evidence or relaxes parity tolerances.
A failed comparison returns a nonzero exit status after saving every result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from parity import evidence_inputs
from replay import (
    BUFFERS,
    HERE,
    ROOT,
    build_native,
    check_outputs,
    lower,
    run,
    validate_case,
    write_json,
)


def compare(layout, case, before, actual, expected, contract):
    import numpy as np

    masks = validate_case(layout, case, before)
    result = {"status": "passed", "buffers": {}}
    for name in BUFFERS:
        mask = masks[name]
        a = actual[name].view(np.float32)[mask]
        e = expected[name].view(np.float32)[mask]
        result["buffers"][name] = {
            "max_absolute_error": float(np.max(np.abs(a - e))) if a.size else 0.0,
            "outside_tolerance": int(
                np.count_nonzero(
                    ~np.isclose(a, e, atol=contract["atol"], rtol=contract["rtol"])
                )
            ),
        }
    try:
        check_outputs(layout, case, before, actual, expected, contract)
    except AssertionError as error:
        result.update(status="failed", error=str(error))
    return result


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results-numerics")
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Use portable fixtures instead of retained cancellation stress",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "level": "diagnostic_only",
        "status": "preparing",
        "inputs": evidence_inputs(ROOT),
        "suite": "portable" if args.portable else "boundary-stress",
        "cases": [],
    }
    write_json(output / "report.json", report)
    try:
        fixtures = output / "fixtures"
        run(
            [
                "cargo",
                "run",
                "--locked",
                "--manifest-path",
                HERE / "reference/Cargo.toml",
                "--",
                fixtures,
            ]
            + ([] if args.portable else ["--boundary-stress"]),
            cwd=HERE / "reference",
        )
        original = build_native(reference=True)
        generated = build_native()
        layout = json.loads((fixtures / "layout.json").read_text())
        suite = json.loads((fixtures / "suite.json").read_text())
        contract = json.loads((HERE / "contract.json").read_text())
        # Never leave old reference results available after a failed new run.
        for path in fixtures.glob("*/*.rust-metal.bin"):
            path.unlink()
        run([original, fixtures])
        for case in suite["cases"]:
            item = {"name": case["name"]}
            report["cases"].append(item)
            if not case["use_bias"] and not case["solve_friction"]:
                item.update(
                    status="unavailable",
                    reason="Original production kernel has no unbiased/no-friction mode",
                )
                continue
            directory = fixtures / case["name"]
            before = {
                name: np.fromfile(directory / f"{name}.bin", dtype=np.uint32)
                for name in BUFFERS
            }
            validate_case(layout, case, before)
            source, bindings = lower(layout, case, "metal")
            write_json(directory / "bindings.json", bindings)
            (directory / "kernel.metal").write_text(source)
            item["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            report["environment"] = json.loads(run([generated, directory]))
            actual = {
                name: np.fromfile(directory / f"{name}.actual.bin", dtype=np.uint32)
                for name in BUFFERS
            }

            def expected(suffix):
                return {
                    name: np.fromfile(
                        directory / f"{name}.{suffix}.bin", dtype=np.uint32
                    )
                    if name in contract["outputs"]
                    else before[name]
                    for name in BUFFERS
                }

            cpu, gpu = expected("expected"), expected("rust-metal")
            item["comparisons"] = {
                "tilelang_vs_rust_cpu": compare(
                    layout, case, before, actual, cpu, contract
                ),
                "tilelang_vs_rust_metal": compare(
                    layout, case, before, actual, gpu, contract
                ),
                "rust_metal_vs_rust_cpu": compare(
                    layout, case, before, gpu, cpu, contract
                ),
            }
            item["status"] = (
                "failed"
                if any(c["status"] == "failed" for c in item["comparisons"].values())
                else "passed"
            )
            print(f"{case['name']}: {item['status']}", flush=True)
        if evidence_inputs(ROOT) != report["inputs"]:
            raise RuntimeError("Sources changed during diagnostic")
        report["status"] = (
            "failed"
            if any(c["status"] == "failed" for c in report["cases"])
            else "passed_with_unavailable_modes"
        )
    except Exception as error:
        report.update(status="error", error=str(error))
        raise
    finally:
        write_json(output / "report.json", report)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
