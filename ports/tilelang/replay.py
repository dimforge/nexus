#!/usr/bin/env python3
"""Compile and replay actual Rust buffers; produce revision-bound parity evidence.

`--target c` runs generated TileLang code on the CPU with no accelerator.
`--target metal` runs generated code through a small native Metal executable.
Other targets currently support source lowering only, explicitly labelled so.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUFFERS = ("Tiles", "Velocities", "Poses", "Params")


def run(command, **kwargs):
    completed = subprocess.run(
        [str(x) for x in command], text=True, capture_output=True, **kwargs
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command}\n{completed.stderr}\n{completed.stdout}"
        )
    return completed.stdout.strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_case(layout, case, arrays):
    """Reject races and out-of-range references before any device execution."""
    import numpy as np

    n = case["count"]
    if n < 0 or case["tiles"] < max((n + 31) // 32, 1) or case["iterations"] < 1:
        raise ValueError("Invalid replay dimensions")
    expected = {
        "Tiles": case["tiles"] * layout["tile_words"],
        "Velocities": case["bodies"] * layout["velocity_words"],
        "Poses": case["bodies"] * layout["pose_words"],
        "Params": 6,
    }
    for name in BUFFERS:
        if arrays[name].size != expected[name]:
            raise ValueError(f"Invalid {name} buffer length")
    masks = {name: np.zeros(arrays[name].size, dtype=bool) for name in BUFFERS}
    used = set()
    tiles = arrays["Tiles"]
    h, p = layout["header"], layout["point"]
    for i in range(n):
        base, lane = (i // 32) * layout["tile_words"], i % 32
        lane4 = lane * 4
        count = int(tiles[base + h["normal"] + lane4 + 3])
        if count > 4:
            raise ValueError("Invalid contact count")
        for key in ("body_a", "body_b", "diag_a", "off_a"):
            index = int(tiles[base + h[key] + lane4 + 3])
            if index >= case["bodies"]:
                raise ValueError("Out-of-range body/velocity index")
            if key in ("diag_a", "off_a"):
                if index in used:
                    raise ValueError(
                        "Velocity slots overlap: replay must be one conflict-free color"
                    )
                used.add(index)
                for field in ("velocity_linear", "velocity_angular"):
                    start = index * layout["velocity_words"] + layout[field]
                    masks["Velocities"][start : start + 3] = True
        for k in range(count):
            pb = base + layout["points"] + k * layout["point_words"] + lane4
            masks["Tiles"][pb + p["arm_a"] + 3] = True
            if case["solve_friction"]:
                masks["Tiles"][pb + p["friction"] : pb + p["friction"] + 2] = True
    return masks


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks = validate_case(layout, case, before)
    errors = {}
    atol, rtol = contract["atol"], contract["rtol"]
    for name in BUFFERS:
        mask = masks[name]
        np.testing.assert_array_equal(
            actual[name][~mask],
            before[name][~mask],
            err_msg=f"Unexpected write to {name}",
        )
        if not mask.any():
            errors[name] = 0.0
            continue
        a, e = (
            actual[name].view(np.float32)[mask],
            expected[name].view(np.float32)[mask],
        )
        if not np.isfinite(a).all() or not np.isfinite(e).all():
            raise AssertionError(f"Non-finite {name} output")
        np.testing.assert_allclose(
            a, e, rtol=rtol, atol=atol, err_msg=f"{case['name']}: {name}"
        )
        errors[name] = float(np.max(np.abs(a - e)))
    tiles = actual["Tiles"].view(np.float32)
    h, p = layout["header"], layout["point"]
    for i in range(case["count"]):
        base, lane = (i // 32) * layout["tile_words"], i % 32
        count = int(actual["Tiles"][base + h["normal"] + lane * 4 + 3])
        friction = float(tiles[base + h["tangent"] + lane * 4 + 3])
        for k in range(count):
            pb = base + layout["points"] + k * layout["point_words"] + lane * 4
            normal = float(tiles[pb + p["arm_a"] + 3])
            if normal < 0.0:
                raise AssertionError("Negative normal impulse")
            if case["solve_friction"]:
                tangent = tiles[pb + p["friction"] : pb + p["friction"] + 2]
                limit = friction * normal
                if float(np.linalg.norm(tangent)) > limit + atol + rtol * abs(limit):
                    raise AssertionError("Friction impulse outside Coulomb disk")
    return errors


def lower(layout, case, target):
    import tilelang
    from coulomb import make_kernel

    requested = {"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target
    artifact = tilelang.lower(
        make_kernel(layout, case, target),
        target=requested,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    functions = list(artifact.device_mod.functions.values())
    if len(functions) != 1:
        raise ValueError("Expected a single generated entry point")
    function = functions[0]
    # Code generation may reorder arguments. Never assume Python declaration order.
    order = [str(param.name) for param in function.params]
    if len(set(order)) != len(order) or not set(order).issubset(BUFFERS):
        raise ValueError(f"Unexpected generated ABI: {order}")
    source = artifact.kernel_source
    if target == "metal":
        # Metal fastMathEnabled=false alone still permits FP contraction.
        # Preserve the scalar expression order used by the portable contract.
        source = "#pragma clang fp contract(off)\n" + source
    return source, {
        "entry": str(function.attrs["global_symbol"]),
        "buffers": order,
        "storage_buffers": list(BUFFERS),
        "blocks": max((case["count"] + 63) // 64, 1),
        "threads": 64,
        "iterations": case["iterations"],
    }


def execute_c(source, bindings, directory, arrays):
    import tilelang

    source_path = directory / "kernel.cpp"
    source_path.write_text(source)
    library = directory / ("kernel.dylib" if sys.platform == "darwin" else "kernel.so")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    run(
        compiler
        + [
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-strict-aliasing",
            "-shared",
            "-fPIC",
            "-I",
            str(Path(tilelang.__file__).parent / "src"),
            str(source_path),
            "-o",
            str(library),
        ]
    )
    native = ctypes.CDLL(str(library))
    function = getattr(native, bindings["entry"])
    function.argtypes = [ctypes.c_void_p] * len(bindings["buffers"])
    function.restype = ctypes.c_int32
    for _ in range(bindings["iterations"]):
        result = function(*(arrays[name].ctypes.data for name in bindings["buffers"]))
        if result != 0:
            raise RuntimeError(f"Generated CPU kernel failed: {result}")
    return {
        "device": platform.machine(),
        "runtime": "native TileLang C++",
        "compiler": run(compiler + ["--version"]).splitlines()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["c", "metal", "cuda", "ascend"], default="c"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Lower source only; does not certify toolkit compilation or execution",
    )
    parser.add_argument(
        "--boundary-stress",
        action="store_true",
        help="Reproduce the known rounding-sensitive zero-distance branch; not release evidence",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.target in ("cuda", "ascend") and not args.compile_only:
        parser.error(
            "CUDA/Ascend execution runners are not implemented; use --compile-only"
        )
    import numpy as np
    from parity import evidence_inputs

    output = args.output.resolve() / args.target
    output.mkdir(parents=True, exist_ok=True)
    inputs = evidence_inputs(ROOT)
    # A setup/build failure must not leave a previous successful report in place.
    write_json(
        output / "report.json",
        {"schema": 1, "target": args.target, "status": "preparing"},
    )
    # Always regenerate fixtures from the current Rust source. Do not accept stale golden files.
    run(
        [
            "cargo",
            "run",
            "--locked",
            "--manifest-path",
            HERE / "reference/Cargo.toml",
            "--",
            output / "fixtures",
        ]
        + (["--boundary-stress"] if args.boundary_stress else []),
        cwd=HERE / "reference",
    )
    fixture_dir = output / "fixtures"
    layout = json.loads((fixture_dir / "layout.json").read_text())
    suite = json.loads((fixture_dir / "suite.json").read_text())
    contract = json.loads((HERE / "contract.json").read_text())
    if evidence_inputs(ROOT) != inputs:
        raise RuntimeError("Sources changed during Rust fixture export")
    runner = output / "native_metal"
    if args.target == "metal" and not args.compile_only:
        run(["swiftc", HERE / "native_metal.swift", "-o", runner])
    result = {
        "schema": 1,
        "feature": suite["feature"],
        "target": args.target,
        "suite": "boundary-stress" if args.boundary_stress else "portable",
        "status": "running",
        "level": "source_lowered" if args.compile_only else "executed",
        "inputs": inputs,
        "reference_commit": run(["git", "-C", ROOT, "rev-parse", "HEAD"]),
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "rustc": run(["rustc", "--version"], cwd=HERE / "reference"),
            "rustflags": os.environ.get("RUSTFLAGS", ""),
            "torch": importlib.metadata.version("torch"),
            "tilelang": importlib.metadata.version("tilelang"),
            "numpy": np.__version__,
            "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "z3": importlib.metadata.version("z3-solver"),
        },
        "layout_sha256": hashlib.sha256(
            (fixture_dir / "layout.json").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    try:
        for case in suite["cases"]:
            directory = fixture_dir / case["name"]
            before = {
                name: np.fromfile(directory / f"{name}.bin", dtype=np.uint32)
                for name in BUFFERS
            }
            validate_case(layout, case, before)
            arrays = {name: value.copy() for name, value in before.items()}
            source, bindings = lower(layout, case, args.target)
            write_json(directory / "bindings.json", bindings)
            (directory / f"kernel.{args.target}").write_text(source)
            outcome = {
                "name": case["name"],
                "count": case["count"],
                "status": "passed",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bindings": bindings,
            }
            if not args.compile_only:
                if args.target == "c":
                    result["environment"].update(
                        execute_c(source, bindings, directory, arrays)
                    )
                else:
                    result["environment"].update(json.loads(run([runner, directory])))
                    arrays = {
                        name: np.fromfile(
                            directory / f"{name}.actual.bin", dtype=np.uint32
                        )
                        for name in BUFFERS
                    }
                for name, array in arrays.items():
                    array.tofile(directory / f"{name}.actual.bin")
                expected = {
                    name: (
                        np.fromfile(directory / f"{name}.expected.bin", dtype=np.uint32)
                        if name in contract["outputs"]
                        else before[name]
                    )
                    for name in BUFFERS
                }
                outcome["max_absolute_errors"] = check_outputs(
                    layout, case, before, arrays, expected, contract
                )
            result["cases"].append(outcome)
            print(f"{args.target}: {case['name']}: passed", flush=True)
        if evidence_inputs(ROOT) != inputs:
            raise RuntimeError("Sources changed during replay; discard this evidence")
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["failed_case"] = case["name"]
        raise
    finally:
        write_json(output / "report.json", result)
    print(f"Evidence: {output / 'report.json'}")


if __name__ == "__main__":
    main()
