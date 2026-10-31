"""Specialize a native cold-start archive without executing reference physics."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from pathlib import Path

import numpy as np
from cpu_oracle import export_fixtures
from frame_manifest import validate_frame
from program_cache import prepare_program
from parity import digest, evidence_inputs
from replay import ROOT, write_json


def archive_hashes(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "scene.json").read_text())
    names = ["scene.json", *[n + ".bin" for n in metadata["buffer_bytes"]]]
    if any(Path(n).name != n for n in names):
        raise ValueError("Invalid native archive filename")
    return {n: digest(directory / n) for n in sorted(names)}


def specialize(
    archive,
    output,
    target,
    capacity,
    trace_steps=None,
    contact_reduction=False,
    solver_colors=None,
    coloring_rounds=None,
    cache_directory=None,
):
    from runtime_scene_validation import validate_case

    started = perf_counter()
    timings = {}
    workers = int(
        os.environ.get(
            "NEXUS_TILELANG_COMPILE_JOBS", min(4, max(1, (os.cpu_count() or 1) // 2))
        )
    )
    if not 1 <= workers <= 16:
        raise ValueError("NEXUS_TILELANG_COMPILE_JOBS must be 1..16")
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if target not in ("c", "metal", "ascend", "cuda"):
        raise ValueError("Unsupported compiler target")
    if type(contact_reduction) is not bool:
        raise ValueError("Invalid contact reduction mode")
    if type(capacity) is not int or not 1 <= capacity <= 0x1FFFFFFF:
        raise ValueError("Invalid runtime pair capacity")
    if trace_steps is not None and (
        type(trace_steps) is not int or not 1 <= trace_steps <= 0xFFFFFFFF
    ):
        raise ValueError("Invalid runtime trace length")
    if (solver_colors is None) != (coloring_rounds is None):
        raise ValueError("Specify both coloring limits")
    if solver_colors is not None and (
        type(solver_colors) is not int
        or not 1 <= solver_colors <= 63
        or type(coloring_rounds) is not int
        or not 2 <= coloring_rounds < 0xFFFFFFFF
    ):
        raise ValueError("Invalid coloring limits")
    output.mkdir(parents=True, exist_ok=False)
    inputs = evidence_inputs(ROOT)
    archive_before = archive_hashes(archive)
    # This mode calls native typed constructors and allocation only. It never
    # advances the Rust CPU solver or uploads expected outputs to the target.
    before = perf_counter()
    profile = export_fixtures(
        "runtime-scenes",
        output / "native",
        (
            archive,
            str(trace_steps or 1),
            str(capacity),
            "upload",
            "reduced" if contact_reduction else "unreduced",
            *(
                (str(solver_colors), str(coloring_rounds))
                if solver_colors is not None
                else ()
            ),
        ),
    )
    timings["native_export"] = perf_counter() - before
    layout = json.loads((output / "native/layout.json").read_text())
    operation = (
        "runtime_twist_scene"
        if layout.get("friction_model") == "twist"
        else "runtime_scene"
    )
    directory = output / "native" / operation / "native-upload"
    case = json.loads((directory / "case.json").read_text())
    initial = {
        n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in case["buffer_words"]
    }
    if trace_steps is None:
        for name in ("Trajectory", "FrameIndex"):
            del initial[name]
            del case["buffer_words"][name]
    case["color_checks"] = True
    initial["ColorBudget"] = np.array([case["rounds"], 0xDEADBEEF], np.uint32)
    initial["ColorRequirements"] = np.full(5, 0xDEADBEEF, np.uint32)
    case["buffer_words"].update(ColorBudget=2, ColorRequirements=5)
    if target == "metal":
        from scene_execution import grid_words

        case["native_schedule"] = True
        initial["Dispatches"] = np.zeros(grid_words(case), np.uint32)
        case["buffer_words"]["Dispatches"] = len(initial["Dispatches"])
    shapes = initial["Shapes"].reshape(-1, layout["shape_words"])[
        : case["collider_count"]
    ]
    shape_kinds = shapes[:, layout["shape"]["a"] + 3]
    case["has_composite_shapes"] = bool(
        np.isin(
            shape_kinds,
            [layout["shape_types"]["trimesh"], layout["shape_types"]["polyline"]],
        ).any()
    )
    initial["BatchIds"][layout["batch"]["contact_sort_collider_shift"]] = max(
        1, case["collider_count"].bit_length()
    )
    validate_case(layout, case, initial)
    before = perf_counter()
    cache = None
    if cache_directory is not None:
        from kernel_cache import KernelCache

        cache = KernelCache(cache_directory)
    timings["compiler_identity"] = perf_counter() - before
    before = perf_counter()
    sources, program, frame = prepare_program(
        layout,
        case,
        initial,
        target,
        cache,
        workers=workers,
        timings=timings,
    )
    timings["program_preparation"] = perf_counter() - before
    before = perf_counter()
    for name, source in sources.items():
        (output / name).write_text(source)
    libraries = {}
    if cache is not None and target in ("c", "metal"):
        jobs = {}
        for stage in program["stages"]:
            name = stage["source"]
            abi = {k: stage["bindings"][k] for k in ("schema", "entry", "buffers")}
            key = (sources[name], json.dumps(abi, sort_keys=True))
            jobs.setdefault(key, (sources[name], abi, []))[2].append(name)

        def build_library(job):
            source, abi, names = job
            payload = (
                cache.c_library(source, abi)
                if target == "c"
                else cache.metal_library(source)
            )
            return names, payload

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for names, payload in pool.map(build_library, jobs.values()):
                if payload is not None:
                    filename = names[0] + (
                        ".metallib" if target == "metal" else ".native"
                    )
                    (output / filename).write_bytes(payload)
                    for name in names:
                        libraries[name] = filename
    timings["native_libraries"] = perf_counter() - before
    before = perf_counter()
    for name, words in initial.items():
        words.tofile(output / f"{name}.bin")
    write_json(output / "frame.json", frame)
    write_json(output / "layout.json", layout)
    write_json(output / "case.json", case)
    if inputs != evidence_inputs(ROOT) or archive_before != archive_hashes(archive):
        raise ValueError(
            "Compiler sources or native archive changed during specialization"
        )
    timings["write_inputs"] = perf_counter() - before
    timings["total"] = perf_counter() - started
    lowering_tasks = timings.pop("lowering_tasks", [])
    metadata = {
        "lowering_tasks": lowering_tasks,
        "timings_seconds": timings,
        "compile_workers": workers,
        "schema": 1,
        "libraries": libraries,
        "cache": cache.stats if cache is not None else None,
        "target": target,
        "level": "source_lowered",
        "inputs": inputs,
        "archive_sha256": archive_before,
        "cpu_oracle": profile,
        "pair_capacity": capacity,
        "solver_colors": case["color_stride"] - 2,
        "coloring_rounds": case["rounds"],
        "contact_reduction": contact_reduction,
        "max_steps": trace_steps,
        "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "specialization.json", metadata)
    return metadata


def load_variant(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "specialization.json").read_text())
    if metadata["schema"] != 1:
        raise ValueError("Invalid specialization metadata")
    for name, expected in metadata["files"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError("Specialization artifact changed")
    libraries = metadata.get("libraries", {})
    if any(Path(n).name != n or n not in metadata["files"] for n in libraries.values()):
        raise ValueError("Invalid compiled library inventory")
    frame = validate_frame(json.loads((directory / "frame.json").read_text()))
    program = frame["program"]
    if not libraries.keys() <= {s["source"] for s in program["stages"]}:
        raise ValueError("Library has no corresponding source")
    required = {"frame.json", "case.json", "layout.json"}
    required |= {s["source"] for s in program["stages"]}
    required |= {n + ".bin" for n in program["storage_buffers"]}
    if not required.issubset(metadata["files"]):
        raise ValueError("Specialization omits required artifacts")
    return {
        "libraries": {n: directory / p for n, p in libraries.items()},
        "manifest": frame,
        "sources": {
            s["source"]: (directory / s["source"]).read_text()
            for s in program["stages"]
        },
        "initial": {
            n: np.fromfile(directory / f"{n}.bin", np.uint32)
            for n in program["storage_buffers"]
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", choices=("c", "metal", "ascend", "cuda"), required=True)
    p.add_argument("--pair-capacity", type=int, required=True)
    p.add_argument("--trace-steps", type=int)
    p.add_argument("--contact-reduction", action="store_true")
    p.add_argument("--solver-colors", type=int)
    p.add_argument("--coloring-rounds", type=int)
    p.add_argument("--cache-directory", type=Path)
    a = p.parse_args()
    specialize(
        a.upload,
        a.output,
        a.target,
        a.pair_capacity,
        a.trace_steps,
        a.contact_reduction,
        a.solver_colors,
        a.coloring_rounds,
        a.cache_directory,
    )


if __name__ == "__main__":
    main()
