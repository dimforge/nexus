"""Python port of `crates/examples3d/mujoco_menagerie3.rs`.

Loads a MuJoCo Menagerie MJCF model and simulates it on the GPU rigid-body
pipeline. The Rust example has a runtime model picker (egui); this port keeps
it simple and loads a single scene chosen up front.

Clone `google-deepmind/mujoco_menagerie` and point `MUJOCO_MENAGERIE_DIR` at it
(defaults to `../mujoco_menagerie` next to the nexus workspace). Pick a model
with `MUJOCO_MENAGERIE_SCENE` (substring match, default `unitree_a1`).
"""

import glob
import os

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    GpuTimestamps,
    Vec3,
)

MENAGERIE_DIR = os.environ.get(
    "MUJOCO_MENAGERIE_DIR",
    os.path.join(os.path.dirname(__file__), "../../../../mujoco_menagerie"),
)
WANTED = os.environ.get("MUJOCO_MENAGERIE_SCENE", "unitree_a1")


def discover_scenes(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "*", "scene*.xml")))


def run(viewer: NexusViewer, pipeline: NexusPipeline, scene: str) -> NexusState:
    # MJCF models are Z-up: orient the camera accordingly (preserved across the
    # per-model set_camera in insert_mjcf).
    viewer.set_up_axis(Vec3.Z)

    state = NexusState()
    info = state.insert_mjcf(viewer, scene, render_colliders=False)
    if not info.loaded:
        print(f"failed to frame scene {scene}")

    timestamps = GpuTimestamps(viewer, 2048)
    state.finalize(viewer)
    # MJCF is Z-up: gravity along -Z (set after finalize).
    if info.z_up:
        state.set_rbd_gravity(viewer, Vec3(0.0, 0.0, -9.81))

    while viewer.render_frame():
        if viewer.simulating():
            pipeline.simulate(viewer, state, timestamps)
        viewer.sync(state, timestamps)

    return state


def main() -> None:
    scenes = discover_scenes(MENAGERIE_DIR)
    if not scenes:
        print(
            f"No MuJoCo Menagerie scenes found under '{MENAGERIE_DIR}'.\n"
            "Clone google-deepmind/mujoco_menagerie there, or set "
            "MUJOCO_MENAGERIE_DIR."
        )
        return
    scene = next((s for s in scenes if WANTED in s), scenes[0])
    print(f"Loading MJCF scene `{scene}`.")

    viewer = NexusViewer()
    viewer.init_backend()
    pipeline = NexusPipeline()
    pipeline.preload_pipelines(viewer)
    run(viewer, pipeline, scene)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
