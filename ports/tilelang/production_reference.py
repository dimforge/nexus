"""Independently qualify the composed Rust oracle against RbdPipeline::step."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scene_trajectory import check_trajectory


def hashes(directory):
    directory = Path(directory)
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def check_upload(layout, composed, production):
    """Only native allocator capacities and deterministic enablement may differ."""
    if json.loads((composed / "scene.json").read_text()) != json.loads(
        (production / "scene.json").read_text()
    ):
        raise ValueError("Production/composed scene metadata differs")
    a, b = hashes(composed), hashes(production)
    if set(a) != set(b):
        raise ValueError("Production/composed upload file set differs")
    for name in a:
        if name != "BatchIds.bin" and a[name] != b[name]:
            raise ValueError(f"Production/composed scene input differs: {name}")
    a, b = (np.fromfile(d / "BatchIds.bin", np.uint32) for d in (composed, production))
    if a.shape != b.shape:
        raise ValueError("Production/composed native batch layout differs")
    keep = np.ones(a.size, bool)
    fields = layout["batch"]
    for name in ("collision_pairs_capacity", "contacts_capacity", "deterministic"):
        keep[fields[name]] = False
    if b[fields["deterministic"]] != 1:
        raise ValueError("Production reference must use deterministic ordering")
    np.testing.assert_array_equal(
        a[keep], b[keep], err_msg="Production native batch inputs"
    )


def check_case(layout, case, before, expected, trace, metadata, contract):
    n, pw, vw = case["body_count"], layout["pose_words"], layout["velocity_words"]
    stride = n * (pw + vw)
    frames = metadata.get("frames")
    if (
        metadata.get("kind") != "production_rbd_pipeline"
        or type(frames) is not int
        or frames < case["frames"]
        or metadata.get("bodies") != n
        or metadata.get("batches") != case["batches"]
        or metadata.get("broadphase") != "lbvh"
        or metadata.get("deterministic") is not True
        or metadata.get("contact_reduction") is not case.get("contact_reduction")
        or case["trace_stride"] != stride
        or trace.dtype != np.uint32
        or trace.shape != (frames * stride,)
    ):
        raise ValueError("Invalid production pipeline trace metadata or storage")
    if not np.isfinite(trace.view(np.float32)).all():
        raise ValueError("Nonfinite production pipeline trajectory")
    # Production owns no fixture guard slots. Preserve those supplied by the
    # composed oracle; all actual body padding is still compared exactly.
    actual = {
        name: expected[name].copy()
        for name in ("Trajectory", "FrameIndex", "Poses", "Velocities")
    }
    length = case["frames"] * stride
    actual["Trajectory"][:length] = trace[:length]
    actual["Poses"][: n * pw] = trace[length - stride : length - n * vw]
    actual["Velocities"][: n * vw] = trace[length - n * vw : length]
    return check_trajectory(layout, case, before, actual, expected, contract)


def check_suite(directory, production, operation="runtime_scene"):
    directory, production = Path(directory), Path(production)
    layout = json.loads((directory / "layout.json").read_text())
    contract = json.loads(
        (Path(__file__).parent / f"contracts/{operation}.json").read_text()
    )
    cases = json.loads((directory / operation / "suite.json").read_text())["cases"]
    results = {}
    for case in cases:
        name = case["name"].rsplit("-f", 1)[0]
        reference = production / name
        check_upload(layout, directory / "native-inputs" / name, reference / "upload")
        fixture = directory / operation / case["name"]
        before = {
            n: np.fromfile(fixture / f"{n}.bin", np.uint32)
            for n in ("FrameIndex", "Trajectory")
        }
        expected = {
            n: np.fromfile(fixture / f"{n}.expected.bin", np.uint32)
            for n in ("FrameIndex", "Trajectory", "Poses", "Velocities")
        }
        results[case["name"]] = check_case(
            layout,
            case,
            before,
            expected,
            np.fromfile(reference / "Trajectory.bin", np.uint32),
            json.loads((reference / "scene.json").read_text()),
            contract,
        )
    return {
        "scope": "all-frame body poses and velocities from production RbdPipeline::step",
        "max_absolute_errors": results,
        "files": hashes(production),
    }
