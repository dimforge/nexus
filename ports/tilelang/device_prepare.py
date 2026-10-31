#!/usr/bin/env python3
"""Prepare one complete 3D scene for device bring-up, without full parity or hardware.

Exports a cold-start native mesh/contact/joint scene and specializes the full
resident pipeline. Checks every launch ABI and Ascend wrapper. This is a bounded
engineering diagnostic, not parity qualification or vendor compilation.
"""

import argparse
import json
import time
from pathlib import Path

from cpu_oracle import export_native_scenes
from device_compiler import ascend_wrapper
from parity import ROOT, evidence_inputs, write_json
from runtime_compile import load_variant, specialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("cuda", "ascend"), action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--upload", type=Path, help="Optional existing trusted native cold-start scene"
    )
    parser.add_argument("--pair-capacity", type=int, default=64)
    parser.add_argument("--cache-directory", type=Path)
    args = parser.parse_args()
    if len(set(args.target)) != len(args.target):
        parser.error("Duplicate target")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = evidence_inputs(ROOT)
    receipt = {
        "schema": 1,
        "suite": "device_source_preparation",
        "status": "running",
        "inputs": inputs,
        "vendor_compilation": "not_run",
        "device_execution": "not_run",
        "full_matrix": "not_run",
        "targets": [],
    }
    path = args.output / "preparation.json"
    write_json(path, receipt)
    try:
        archive = args.upload
        if archive is None:
            export_native_scenes(args.output / "uploads")
            archive = args.output / "uploads/mesh-contact-joint"
        for target in args.target:
            directory = args.output / target
            metadata = specialize(
                archive,
                directory,
                target,
                args.pair_capacity,
                solver_colors=8,
                coloring_rounds=7,
                cache_directory=args.cache_directory,
            )
            variant = load_variant(directory, expected_target=target)
            stages = variant["manifest"]["program"]["stages"]
            if any("indirect" in s for s in stages):
                raise ValueError("Device target retained native indirect dispatch")
            for stage in stages:
                bindings = stage["bindings"]
                if type(bindings.get("shared_memory_bytes")) is not int:
                    raise ValueError("Missing dynamic shared-memory ABI")
                if target == "ascend":
                    ascend_wrapper(variant["sources"][stage["source"]], bindings)
            receipt["targets"].append(
                {
                    "target": target,
                    "level": "source_lowered",
                    "stage_count": len(stages),
                    "virtual_grid_stages": sum("virtual_grid" in s for s in stages),
                    "cache_swaps": len(variant["manifest"].get("swaps", [])),
                    "specialization": str(directory.resolve()),
                    "files": metadata["files"],
                    "timings_seconds": metadata["timings_seconds"],
                }
            )
            write_json(path, receipt)
        if evidence_inputs(ROOT) != inputs:
            raise ValueError("Sources changed during preparation")
        receipt["status"] = "passed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
