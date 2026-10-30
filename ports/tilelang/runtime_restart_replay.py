"""Live native uploads and identical-state restarts for compact-friction numerics."""

import json
import shutil
from pathlib import Path

from cpu_oracle import export_fixtures, export_native_scenes
from production_reference import hashes
from runtime_replay import BUFFERS
from runtime_restart_validation import check_outputs, validate_case
from scene_kernels import make_program
from stage_replay import parse_args, run_stage


def export(directory):
    directory = Path(directory)
    upload = directory / "native-inputs"
    profile = export_native_scenes(upload, "twist")
    trajectory = directory / "native-trajectory"
    reference_profile = export_fixtures(
        "runtime-scenes",
        trajectory,
        (upload / "mesh-overlap-unreduced", "8", "64", "snapshots", "unreduced"),
    )
    if profile != reference_profile:
        raise ValueError("Restart/native upload CPU profiles differ")
    # Nested exporters place their probe beside their own fixture directories.
    # The replay/handoff ABI requires it beside this operation's report instead.
    shutil.copyfile(
        directory / "cpu-oracle-noalias.ll",
        directory.parent / "cpu-oracle-noalias.ll",
    )
    shutil.copyfile(trajectory / "layout.json", directory / "layout.json")
    source = trajectory / "runtime_twist_scene/native-upload"
    original = json.loads((source / "case.json").read_text())
    cases = []
    operation = directory / "runtime_twist_restart"
    for frame in range(8):
        case = dict(
            original,
            name=f"mesh-overlap-unreduced-restart-{frame}",
            start_frame=frame,
            iterations=1,
        )
        dest = operation / case["name"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in original["buffer_words"]:
            for suffix in ("", ".expected"):
                filename = f"{name}{suffix}.bin"
                candidates = (
                    source / f"restart-{frame}" / filename,
                    source / filename,
                    source / f"{name}.bin",
                )
                shutil.copyfile(
                    next(p for p in candidates if p.exists()), dest / filename
                )
        (dest / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        cases.append(case)
    (operation / "suite.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    profile["native_scene_uploads"] = hashes(upload)
    profile["native_restart_trajectory"] = hashes(trajectory)
    return profile


def main():
    args = parse_args(["runtime_twist_restart"], "runtime_twist_restart")
    run_stage(
        args,
        oracle="runtime-scenes",
        exporter=export,
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
