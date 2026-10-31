#!/usr/bin/env python3
"""Bounded representative Rust/TileLang execution; never full parity qualification."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matrix
from parity import ROOT, evidence_inputs, smoke_evidence_state, write_json

HERE = Path(__file__).resolve().parent
# A fixed subset of existing contracts; never change their comparison policies.
CASES = {
    "free_step": (
        "n0-b1-c0-s1-i3-g1",
        "n65-b1-c70-s4-i32-g1",
        "n33-b3-c35-s3-i16-g1",
    ),
    "colored_program": ("w33-b3-m1-s21", "w33-b1-m2-s42-shared-static"),
    "joint_program": ("n12-b3-z0",),
    "analytic_collision": ("n3-b2-mixed", "n33-b2-rotated", "n33-b2-filters"),
    "twist_solve": ("n5-b1-f1-i3", "n65-b0-f1-i3"),
}
OPERATIONS = tuple(CASES)
UNIT_MODULES = (
    "test_kernel_cache",
    "test_preparation",
    "test_execution_optimizations",
    "test_frame_cpu",
    "test_runtime_service",
    "test_matrix",
    "test_parity",
    "test_smoke",
)
MAX_SECONDS = 290  # Leave headroom for child cleanup and writing the receipt.


def selected_features(rows=None):
    by_operation = {
        r["operation"]: r for r in (matrix.features() if rows is None else rows)
    }
    missing = set(OPERATIONS) - by_operation.keys()
    if missing:
        raise ValueError(f"Smoke operations missing from registry: {sorted(missing)}")
    rows = [by_operation[op] for op in OPERATIONS]
    for row in rows:
        path = HERE / row.get("contract", "contract.json").removeprefix(
            "ports/tilelang/"
        )
        required = json.loads(path.read_text())["required_cases"]
        if not set(CASES[row["operation"]]).issubset(required):
            raise ValueError(f"Smoke case missing from contract: {row['operation']}")
    return rows


def run_command(command, cwd, log, deadline, env=None):
    """One shared wall-clock deadline, including compilation and fixture export."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Smoke wall-clock budget exhausted")
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("Smoke wall-clock budget exhausted") from error
            if code:
                raise RuntimeError(f"Command exited {code}; see {log}")
        finally:
            # Cancel the entire owned group (Cargo/compiler/oracle grandchildren),
            # including descendants left behind by an interrupted parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)


def validate_report(path, row, target, inputs):
    report = json.loads(path.read_text())
    if report.get("feature") != row["id"] or report.get("target") != target:
        raise ValueError(f"Wrong feature or target in {path}")
    state = smoke_evidence_state(ROOT, report, inputs, CASES[row["operation"]])
    if state != "executed":
        raise ValueError(
            f"Smoke requires current selected-case execution: {path}: {state}"
        )
    return len(report["cases"])


def run_smoke(target, output, budget):
    started = time.monotonic()
    deadline = started + budget
    output.mkdir(parents=True, exist_ok=True)
    # No successful old run can survive a failing new invocation.
    directory = Path(tempfile.mkdtemp(prefix=f"{target}-", dir=output))
    receipt = {
        "schema": 1,
        "suite": "representative_smoke",
        "status": "running",
        "target": target,
        "budget_seconds": budget,
        "unit_backend": "c" if target in ("cuda", "ascend") else target,
        "full_matrix": "not_run; explicit user request required",
        "steps": [],
        "omissions": [
            "long scene trajectories and production-runtime acceptance",
            "exhaustive geometry, LBVH, hubs and scheduling coverage",
            "execution on targets other than the selected backend",
        ],
    }
    path = directory / "smoke.json"
    write_json(path, receipt)
    print(f"Smoke output: {directory}", flush=True)
    try:
        inputs = evidence_inputs(ROOT)
        receipt["inputs"] = inputs
        rows = selected_features()
        steps = [
            (
                "unit-regressions",
                [sys.executable, "-m", "unittest", "-v", *UNIT_MODULES],
                HERE / "tests",
                None,
            ),
            *[
                (
                    row["operation"],
                    matrix.command(row, target, directory)
                    + [
                        arg
                        for case in CASES[row["operation"]]
                        for arg in ("--case", case)
                    ],
                    ROOT,
                    row,
                )
                for row in rows
            ],
        ]
        if target in ("cuda", "ascend"):
            steps.insert(
                0,
                (
                    "device-preflight",
                    [
                        sys.executable,
                        str(HERE / "device_replay.py"),
                        "--target",
                        target,
                        "--probe",
                    ],
                    ROOT,
                    None,
                ),
            )
        for name, command, cwd, row in steps:
            step = {
                "name": name,
                "command": command,
                "status": "running",
                "log": f"{name}.log",
            }
            receipt["steps"].append(step)
            write_json(path, receipt)
            before = time.monotonic()
            print(f"Smoke: {name} ({before - started:.1f}s elapsed)", flush=True)
            try:
                run_command(
                    command,
                    cwd,
                    directory / step["log"],
                    deadline,
                    env=dict(
                        os.environ, NEXUS_TILELANG_TEST_TARGET=receipt["unit_backend"]
                    ),
                )
                if row is not None:
                    step["cases"] = validate_report(
                        directory / name / target / "report.json",
                        row,
                        target,
                        inputs,
                    )
                step["status"] = "passed"
            except BaseException:
                step["status"] = "failed"
                raise
            finally:
                step["elapsed_seconds"] = round(time.monotonic() - before, 3)
                write_json(path, receipt)
        if evidence_inputs(ROOT) != inputs:
            raise ValueError("Sources changed during smoke run")
        if time.monotonic() >= deadline:
            raise TimeoutError("Smoke wall-clock budget exhausted")
        receipt["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - record failure, return nonzero
        receipt["status"] = "timeout" if isinstance(error, TimeoutError) else "failed"
        receipt["error"] = str(error) or type(error).__name__
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(path, receipt)
    print(
        f"Smoke {receipt['status']}: {receipt['elapsed_seconds']:.1f}s; {path}",
        flush=True,
    )
    if receipt.get("error"):
        print(receipt["error"], file=sys.stderr)
    return 0 if receipt["status"] == "passed" else 1


def budget_seconds(value):
    seconds = int(value)
    if not 1 <= seconds <= MAX_SECONDS:
        raise argparse.ArgumentTypeError(f"Budget must be 1..{MAX_SECONDS} seconds")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("c", "metal", "cuda", "ascend"), default="c"
    )
    parser.add_argument("--output", type=Path, default=HERE / "results-smoke")
    parser.add_argument("--budget-seconds", type=budget_seconds, default=MAX_SECONDS)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                {
                    "operations": [r["operation"] for r in selected_features()],
                    "cases": CASES,
                    "unit_modules": UNIT_MODULES,
                    "budget_seconds": args.budget_seconds,
                },
                indent=2,
            )
        )
        return 0
    if os.name != "posix":
        parser.error("The fixture runners and process cleanup require POSIX")
    return run_smoke(args.target, args.output.resolve(), args.budget_seconds)


if __name__ == "__main__":
    sys.exit(main())
