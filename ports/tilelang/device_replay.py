#!/usr/bin/env python3
"""Compile/replay exported CUDA or Ascend buffers; no expected input or CPU fallback."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from device_frame import DeviceFrameSession
from ordered import validate_program


def create_device(target, ordinal=None):
    ordinal = (
        int(os.environ.get("NEXUS_TILELANG_DEVICE", "0"))
        if ordinal is None
        else ordinal
    )
    if target == "cuda":
        from device_cuda import CudaDevice

        return CudaDevice(ordinal)
    if target == "ascend":
        from device_ascend import AscendDevice

        return AscendDevice(ordinal)
    raise ValueError("Expected cuda or ascend")


def as_program(sources, bindings):
    if bindings.get("schema") == 2:
        return validate_program(bindings)
    if bindings.get("schema") != 1 or len(sources) != 1:
        raise ValueError("Unsupported standalone device bindings")
    return validate_program(
        {
            "schema": 2,
            "storage_buffers": bindings["storage_buffers"],
            "buffer_bytes": bindings["buffer_bytes"],
            "iterations": 1,
            "stages": [
                {
                    "name": "kernel",
                    "source": next(iter(sources)),
                    "bindings": bindings,
                    "buffer_map": {n: n for n in bindings["storage_buffers"]},
                }
            ],
        }
    )


def execute_program(sources, bindings, directory, arrays, target, *, device=None):
    program = as_program(sources, bindings)
    iterations = program["iterations"]
    frame = {
        "schema": 1,
        "max_steps": iterations,
        "program": dict(program, iterations=1),
        "gates": [],
        "persistent": {
            n: {"buffer": n, "word_start": 0, "word_len": size // 4}
            for n, size in program["buffer_bytes"].items()
        },
    }
    owned = device is None
    device = create_device(target) if owned else device
    runtime = DeviceFrameSession(
        [{"manifest": frame, "sources": sources, "initial": arrays}],
        directory,
        device,
        owns_device=owned,
    )
    try:
        for _ in range(iterations):
            runtime.step()
        for name in arrays:
            arrays[name][:] = runtime.read(name)
        return dict(device.environment)
    finally:
        runtime.close()


def load_frame(directory):
    from frame_manifest import validate_frame

    directory = Path(directory)
    manifest = validate_frame(json.loads((directory / "frame.json").read_text()))
    program = manifest["program"]
    return {
        "manifest": manifest,
        "sources": {
            s["source"]: (directory / s["source"]).read_text()
            for s in program["stages"]
        },
        "initial": {
            n: np.fromfile(directory / f"{n}.bin", np.uint32)
            for n in program["storage_buffers"]
        },
    }


def load_bundle(directory, target, steps=None):
    directory = Path(directory)
    if (directory / "specialization.json").is_file():
        from runtime_compile import load_variant

        variants = [load_variant(directory, expected_target=target)]
        count = 1 if steps is None else steps
    else:
        if steps is not None:
            raise ValueError(
                "--steps is only for a prepared runtime scene; fixtures retain their declared iteration count"
            )
        bindings = json.loads((directory / "bindings.json").read_text())
        if bindings.get("schema") == 3:
            paths = bindings["variants"]
            if not paths or any(
                Path(p).is_absolute() or ".." in Path(p).parts for p in paths
            ):
                raise ValueError("Frame variant path escapes the bundle")
            variants = [load_frame(directory / p) for p in paths]
            count = bindings["max_steps"]
        else:
            names = (
                {s["source"] for s in bindings["stages"]}
                if bindings.get("schema") == 2
                else {f"kernel.{target}"}
            )
            if any(Path(n).name != n for n in names):
                raise ValueError("Kernel source path escapes the bundle")
            sources = {n: (directory / n).read_text() for n in names}
            program = as_program(sources, bindings)
            count = program["iterations"]
            manifest = {
                "schema": 1,
                "max_steps": count,
                "program": dict(program, iterations=1),
                "gates": [],
                "persistent": {
                    n: {"buffer": n, "word_start": 0, "word_len": size // 4}
                    for n, size in program["buffer_bytes"].items()
                },
            }
            variants = [
                {
                    "manifest": manifest,
                    "sources": sources,
                    "initial": {
                        n: np.fromfile(directory / f"{n}.bin", np.uint32)
                        for n in program["storage_buffers"]
                    },
                }
            ]
    if type(count) is not int or not 1 <= count <= 0xFFFFFFFF:
        raise ValueError("Invalid frame count")
    for variant in variants:
        DeviceFrameSession._validate_variant(variant)
        if any("indirect" in s for s in variant["manifest"]["program"]["stages"]):
            raise ValueError("Device bundle requires virtual-grid lowering")
    return variants, count


def compile_directory(directory, target):
    """Run vendor compilation without opening a device, retaining warm-cache artifacts."""
    from device_compiler import DeviceCompiler

    variants, _ = load_bundle(directory, target)
    compiler = DeviceCompiler(target)
    artifacts = {}
    for variant in variants:
        for stage in variant["manifest"]["program"]["stages"]:
            binary, receipt = compiler.compile(
                variant["sources"][stage["source"]],
                stage["bindings"],
                Path(directory) / "device-build",
            )
            artifacts[str(binary)] = receipt["binary_sha256"]
    return {
        "device_compiler": compiler.identity,
        "level": "device_compiled",
        "artifacts": artifacts,
        "device_execution": "not_run",
    }


def replay_directory(directory, target, steps=None):
    directory = Path(directory)
    variants, count = load_bundle(directory, target, steps)
    device = create_device(target)
    runtime = None
    try:
        runtime = DeviceFrameSession(
            variants, directory / "device-build", device, owns_device=False
        )
        stats = [runtime.step() for _ in range(count)]
        for name in runtime.variants[runtime.current]["manifest"]["program"][
            "storage_buffers"
        ]:
            runtime.read(name).tofile(directory / f"{name}.actual.bin")
        (directory / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        environment = dict(device.environment)
        (directory / "device-environment.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
        return {
            **environment,
            "level": "executed",
            "steps": count,
            "comparison": "not_run; use the contract replay runner",
        }
    finally:
        if runtime is not None:
            runtime.close()
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--target", choices=("cuda", "ascend"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument(
        "--compile-only",
        action="store_true",
        help="Vendor binary compilation, no device required",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Frame count for a prepared runtime scene (default one)",
    )
    args = parser.parse_args()
    if args.probe:
        if args.steps is not None:
            parser.error("--probe does not execute frames")
        device = create_device(args.target)
        try:
            from device_compiler import DeviceCompiler

            compiler = DeviceCompiler(args.target)
            result = {
                **device.environment,
                "device_compiler": compiler.identity,
                "kernel_execution": "not_run",
            }
        finally:
            device.close()
    else:
        if args.directory is None:
            parser.error("Provide a fixture/prepared scene directory or --probe")
        report = {
            "target": args.target,
            "status": "running",
            "device_execution": "not_run",
        }
        path = args.directory / "device-run.json"
        path.write_text(json.dumps(report) + "\n")
        try:
            result = (
                compile_directory(args.directory, args.target)
                if args.compile_only
                else replay_directory(args.directory, args.target, args.steps)
            )
            report.update(
                result,
                status="passed",
                device_execution="not_run" if args.compile_only else "executed",
            )
        except Exception as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
