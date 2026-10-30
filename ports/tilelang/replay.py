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
import platform
import shlex
import subprocess
import sys
from pathlib import Path

from cpu_oracle import export_fixtures

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
    kind = case.get("kind")
    if kind == "integrate":
        count = case["active_bodies"]
        if not 0 <= count <= case["bodies"]:
            raise ValueError("Invalid active body count")
        params = arrays["Params"].view(np.float32)
        if not np.isfinite(params).all() or params[0] <= 0 or (params[1:3] < 0).any():
            raise ValueError("Invalid integration timestep or speed caps")
        for i in range(count):
            q = i * layout["pose_words"] + layout["pose_rotation"]
            rotation = arrays["Poses"].view(np.float32)[q : q + 4]
            if (
                not np.isfinite(rotation).all()
                or abs(float(np.linalg.norm(rotation)) - 1.0) > 2e-6
            ):
                raise ValueError("Invalid input rotation")
            for field, length in (("pose_translation", 3), ("pose_rotation", 4)):
                start = i * layout["pose_words"] + layout[field]
                masks["Poses"][start : start + length] = True
            for field in ("velocity_linear", "velocity_angular"):
                start = i * layout["velocity_words"] + layout[field]
                masks["Velocities"][start : start + 3] = True
        return masks
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
    if kind == "warmstart":
        masks["Tiles"][:] = False
    if kind == "effective_mass":
        masks["Velocities"][:] = False
        masks["Tiles"][:] = False
        for i in range(n):
            base, lane = (i // 32) * layout["tile_words"], i % 32
            count = int(tiles[base + h["normal"] + lane * 4 + 3])
            for k in range(count):
                pb = base + layout["points"] + k * layout["point_words"] + lane * 4
                masks["Tiles"][pb + p["arm_b"] + 3] = True
                masks["Tiles"][pb + p["friction"] + 2 : pb + p["friction"] + 4] = True
                masks["Tiles"][base + layout["coupling"] + lane * 4 + k] = True
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
    if case.get("kind"):
        if case["kind"] == "integrate":
            poses = actual["Poses"].view(np.float32)
            velocities = actual["Velocities"].view(np.float32)
            params = before["Params"].view(np.float32)
            for i in range(case["active_bodies"]):
                q = i * layout["pose_words"] + layout["pose_rotation"]
                if abs(float(np.linalg.norm(poses[q : q + 4])) - 1.0) > 2e-6:
                    raise AssertionError("Quaternion normalization invariant")
                for field, limit in (
                    ("velocity_linear", params[1]),
                    ("velocity_angular", params[2]),
                ):
                    v = i * layout["velocity_words"] + layout[field]
                    if np.linalg.norm(
                        velocities[v : v + 3]
                    ) > limit + atol + rtol * abs(limit):
                        raise AssertionError("Velocity cap invariant")
        return errors
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
    from kernels import make_kernel

    requested = {"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target
    from ordered import validate_launch

    kernel = make_kernel(layout, case, target)
    blocks = max((case.get("active_bodies", case["count"]) + 63) // 64, 1)
    validate_launch(kernel, target, blocks, 64)
    artifact = tilelang.lower(
        kernel,
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
        "schema": 1,
        "entry": str(function.attrs["global_symbol"]),
        "buffer_bytes": {
            "Tiles": case["tiles"] * layout["tile_words"] * 4,
            "Velocities": case["bodies"] * layout["velocity_words"] * 4,
            "Poses": case["bodies"] * layout["pose_words"] * 4,
            "Params": 24,
        },
        "buffers": order,
        "storage_buffers": list(BUFFERS),
        "blocks": blocks,
        "threads": 64,
        "iterations": case["iterations"],
    }


def compile_c(source, bindings, directory):
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
    return (
        native,
        function,
        {
            "device": platform.machine(),
            "runtime": "native TileLang C++",
            "compiler": run(compiler + ["--version"]).splitlines()[0],
        },
    )


def execute_c(source, bindings, directory, arrays, *, compiled=None):
    # Keep the CDLL handle alive while its entry point is in use.
    native, function, environment = compiled or compile_c(source, bindings, directory)
    for _ in range(bindings["iterations"]):
        result = function(*(arrays[name].ctypes.data for name in bindings["buffers"]))
        if result != 0:
            raise RuntimeError(f"Generated CPU kernel failed: {result}")
    return environment


def compiler_libraries():
    import tilelang

    directory = Path(tilelang.__file__).parent / "lib"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in (".so", ".dylib", ".dll")
    }


def build_native(reference=False, *, frame=False):
    """Build pinned Rust execution adapters; the returned process needs no Python."""
    if reference and frame:
        raise ValueError("Select either the reference or frame adapter")
    target = ROOT / "target"
    binary = (
        "rust-reference" if reference else "native-frame" if frame else "native-replay"
    )
    command = [
        "cargo",
        "build",
        "--locked",
        "--manifest-path",
        HERE / "native/Cargo.toml",
        "--target-dir",
        target,
    ]
    command += ["--features", "rust-reference"] if reference else ["--bin", binary]
    run(command, cwd=HERE / "native")
    return target / "debug" / binary


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
    parser.add_argument(
        "--operation",
        choices=["all", "solve", "effective_mass", "warmstart", "integrate"],
        default="solve",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.target in ("cuda", "ascend") and not args.compile_only:
        parser.error(
            "CUDA/Ascend execution runners are not implemented; use --compile-only"
        )
    if args.boundary_stress and args.operation != "solve":
        parser.error("Boundary stress applies only to solve")
    if args.operation == "all":
        from matrix import command, features

        for feature in features():
            subprocess.run(
                command(feature, args.target, args.output, args.compile_only),
                check=True,
            )
        return
    import numpy as np
    from parity import evidence_inputs

    output = (
        args.output.resolve() / args.target
        if args.operation == "solve"
        else args.output.resolve() / args.operation / args.target
    )
    output.mkdir(parents=True, exist_ok=True)
    inputs = evidence_inputs(ROOT)
    # A setup/build failure must not leave a previous successful report in place.
    write_json(
        output / "report.json",
        {"schema": 1, "target": args.target, "status": "preparing"},
    )
    # Always regenerate fixtures from the current Rust source. Do not accept stale golden files.
    oracle_profile = export_fixtures(
        "nexus-tilelang-reference" if args.operation == "solve" else "operations",
        output / "fixtures",
        ["--boundary-stress"] if args.boundary_stress else [],
    )
    fixture_dir = output / "fixtures"
    layout = json.loads((fixture_dir / "layout.json").read_text())
    if args.operation != "solve":
        fixture_dir = fixture_dir / args.operation
    suite = json.loads((fixture_dir / "suite.json").read_text())
    contract_path = (
        HERE / "contract.json"
        if args.operation == "solve"
        else HERE / "contracts" / f"{args.operation}.json"
    )
    contract = json.loads(contract_path.read_text())
    suite["feature"] = contract["feature"]
    if evidence_inputs(ROOT) != inputs:
        raise RuntimeError("Sources changed during Rust fixture export")
    runner = None
    if args.target == "metal" and not args.compile_only:
        runner = build_native()
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
            "rustc": oracle_profile["rustc"],
            "rustflags": oracle_profile["rustflags"],
            "cpu_oracle": oracle_profile,
            "torch": importlib.metadata.version("torch"),
            "tilelang": importlib.metadata.version("tilelang"),
            "numpy": np.__version__,
            "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "z3": importlib.metadata.version("z3-solver"),
        },
        "compiler_libraries": compiler_libraries(),
        "layout_sha256": hashlib.sha256(
            (output / "fixtures/layout.json").read_bytes()
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
            outcome["fixture"] = case
            artifact_names = {"bindings.json", f"kernel.{args.target}", "RbdParams.bin"}
            artifact_names.update(f"{name}.bin" for name in BUFFERS)
            artifact_names.update(
                f"{name}.expected.bin" for name in contract["outputs"]
            )
            if not args.compile_only:
                artifact_names.update(f"{name}.actual.bin" for name in BUFFERS)
            outcome["files"] = {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in sorted(artifact_names)
            }
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
