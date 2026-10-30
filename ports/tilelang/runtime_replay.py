"""Replay production Nexus cold-start uploads through complete generated frames."""

import shutil
import tempfile
from pathlib import Path

from cpu_oracle import export_fixtures, export_native_scenes, export_production_scenes
from general_scene_validation import EXTRA as COLLISION
from joint_scene_validation import EXTRA as JOINTS
from production_reference import check_suite, hashes
from runtime_scene_validation import check_outputs, validate_case
from scene_broadphase import EXTRA as LBVH
from scene_hubs import BUFFERS as HUBS
from scene_kernels import make_program
from scene_validation import BUFFERS as SCENE
from stage_replay import parse_args, run_stage

BUFFERS = (
    SCENE + COLLISION + JOINTS + LBVH + HUBS + tuple(f"Color{i}" for i in range(8, 11))
)


def export(directory, model="coulomb"):
    operation = "runtime_twist_scene" if model == "twist" else "runtime_scene"
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="native-upload-", dir=directory.parent
    ) as tmp:
        upload = Path(tmp) / "inputs"
        native_profile = export_native_scenes(upload, model)
        profile = export_fixtures(
            "runtime-scenes",
            directory,
            (upload, "suite-twist" if model == "twist" else "suite"),
        )
        if native_profile != profile:
            raise ValueError("Native scene exporter and shader oracle profiles differ")
        archive = directory / "native-inputs"
        shutil.copytree(upload, archive, dirs_exist_ok=True)
        profile["native_scene_uploads"] = hashes(archive)
        production = directory / "production-reference"
        if production.exists():
            shutil.rmtree(production)
        production_profile = export_production_scenes(production, model)
        if native_profile != production_profile:
            raise ValueError("Production pipeline CPU profile differs")
        profile["production_pipeline"] = check_suite(directory, production, operation)
        return profile


def main():
    args = parse_args(["runtime_scene", "runtime_twist_scene"], "runtime_scene")
    run_stage(
        args,
        oracle="runtime-scenes",
        exporter=lambda d: export(
            d, "twist" if args.operation == "runtime_twist_scene" else "coulomb"
        ),
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
