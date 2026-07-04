"""Python port of `crates/examples3d/urdf3.rs`.

Loads a robot from a URDF file as a multibody and keeps it in slow continuous
motion by re-randomizing each joint's AngX motor target velocity every 5 seconds.

Set the URDF path via the `NEXUS_URDF` environment variable (defaults to the
path used by the Rust example).
"""

import math
import os
import random

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    GpuTimestamps,
    UrdfLoaderOptions,
    JointAxis,
    Vec3,
    Pose,
    Quat,
)

URDF_PATH = os.environ.get(
    "NEXUS_URDF",
    "/Users/sebcrozet/work/nexus-demos/XoQ/js/examples/assets/openarm_v10.urdf",
)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    scale = 40.0
    options = UrdfLoaderOptions(
        create_colliders_from_collision_shapes=True,
        # openarm_v10 ships only visual meshes, so fall back to those.
        create_colliders_from_visual_shapes=True,
        apply_imported_mass_props=True,
        make_roots_fixed=True,
        scale=scale,
        # URDF is Z-up but the viewer is Y-up: lift the robot and rotate -90° X.
        shift=Pose.from_parts(Vec3(0.0, scale, 0.0), Quat.from_rotation_x(-math.pi / 2)),
    )

    robot = state.insert_urdf(URDF_PATH, options, actuate_angx_motors=True)
    num_links = robot.num_links
    for body, shape, local_pose in robot.render_shapes:
        viewer.insert_visual_shape(0, body, shape, local_pose)

    timestamps = GpuTimestamps(viewer, 2048)
    state.finalize(viewer)

    dt = 1.0 / 60.0
    sim_time = 0.0
    next_change_at = 0.0
    interval = 5.0

    while viewer.render_frame():
        if sim_time >= next_change_at:
            next_change_at = sim_time + interval
            num_batches = state.rbd_num_batches()
            for batch in range(num_batches):
                for link_id in range(num_links):
                    target_vel = random.uniform(-0.6, 0.6)
                    state.set_multibody_motor_velocity(
                        viewer, batch, link_id, JointAxis.AngX, target_vel
                    )

        if viewer.simulating():
            pipeline.simulate(viewer, state, timestamps)
            sim_time += dt
        viewer.sync(state, timestamps)

    return state


def main() -> None:
    viewer = NexusViewer()
    viewer.init_backend()
    pipeline = NexusPipeline()
    pipeline.preload_pipelines(viewer)
    run(viewer, pipeline)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
