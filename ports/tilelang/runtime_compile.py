"""Specialize a native cold-start archive without executing reference physics."""

import argparse
import json
from pathlib import Path

import numpy as np
from cpu_oracle import export_fixtures
from frame_manifest import from_program, validate_frame
from ordered import CompilationSession, lower_program
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
    archive, output, target, capacity, trace_steps=None, contact_reduction=False
):
    from runtime_scene_validation import validate_case
    from scene_kernels import make_program

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
    output.mkdir(parents=True, exist_ok=False)
    inputs = evidence_inputs(ROOT)
    archive_before = archive_hashes(archive)
    # This mode calls native typed constructors and allocation only. It never
    # advances the Rust CPU solver or uploads expected outputs to the target.
    profile = export_fixtures(
        "runtime-scenes",
        output / "native",
        (
            archive,
            str(trace_steps or 1),
            str(capacity),
            "upload",
            "reduced" if contact_reduction else "unreduced",
        ),
    )
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
    validate_case(layout, case, initial)
    plans = make_program(layout, case, target)
    sources, program = lower_program(
        plans, initial, target, session=CompilationSession()
    )
    frame = from_program(layout, case, plans, program)
    for name, source in sources.items():
        (output / name).write_text(source)
    for name, words in initial.items():
        words.tofile(output / f"{name}.bin")
    write_json(output / "frame.json", frame)
    write_json(output / "layout.json", layout)
    write_json(output / "case.json", case)
    if inputs != evidence_inputs(ROOT) or archive_before != archive_hashes(archive):
        raise ValueError(
            "Compiler sources or native archive changed during specialization"
        )
    metadata = {
        "schema": 1,
        "target": target,
        "level": "source_lowered",
        "inputs": inputs,
        "archive_sha256": archive_before,
        "cpu_oracle": profile,
        "pair_capacity": capacity,
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
    frame = validate_frame(json.loads((directory / "frame.json").read_text()))
    program = frame["program"]
    required = {"frame.json", "case.json", "layout.json"}
    required |= {s["source"] for s in program["stages"]}
    required |= {n + ".bin" for n in program["storage_buffers"]}
    if not required.issubset(metadata["files"]):
        raise ValueError("Specialization omits required artifacts")
    return {
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
    a = p.parse_args()
    specialize(
        a.upload,
        a.output,
        a.target,
        a.pair_capacity,
        a.trace_steps,
        a.contact_reduction,
    )


if __name__ == "__main__":
    main()
