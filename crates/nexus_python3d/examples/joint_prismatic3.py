"""Python port of `crates/examples3d/joint_prismatic3.rs`.

Many chains of prismatic (sliding) joints with alternating axes and limits.
"""

import math

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    PrismaticJointBuilder,
    GpuTimestamps,
    Vec3,
    Pose,
)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    rad = 0.4
    num = 10
    shift = 1.0

    for m in range(20):
        z = m * shift * (num + 2.0)

        for l in range(20):
            y = l * shift * num * 2.0

            for j in range(30):
                x = j * shift * 4.0

                ground = RigidBodyBuilder.fixed().translation(Vec3(x, y, z)).build()
                collider = ColliderBuilder.cuboid(rad, rad, rad).build()
                shape = collider.shared_shape()
                curr_parent = state.insert_rigid_body(ground, collider)
                viewer.insert_shape(curr_parent, shape, Pose.IDENTITY)

                for i in range(num):
                    zi = z + (i + 1) * shift
                    density = 1.0
                    rigid_body = (
                        RigidBodyBuilder.dynamic()
                        .translation(Vec3(x, y, zi))
                        .build()
                    )
                    collider = (
                        ColliderBuilder.cuboid(rad, rad, rad).density(density).build()
                    )
                    shape = collider.shared_shape()
                    curr_child = state.insert_rigid_body(
                        rigid_body, collider
                    )
                    viewer.insert_shape(curr_child, shape, Pose.IDENTITY)

                    if i % 2 == 0:
                        axis = Vec3(1.0, 1.0, 0.0)
                    else:
                        axis = Vec3(-1.0, 1.0, 0.0)
                    inv_len = 1.0 / math.sqrt(
                        axis.x * axis.x + axis.y * axis.y + axis.z * axis.z
                    )
                    axis = axis * inv_len

                    prism = (
                        PrismaticJointBuilder.new(axis)
                        .local_anchor2(Vec3(0.0, 0.0, -shift))
                        .limits(-2.0, 0.0)
                    )
                    state.insert_impulse_joint(curr_parent, curr_child, prism)

                    curr_parent = curr_child

    timestamps = GpuTimestamps(viewer, 2048)
    viewer.add_directional_light(Vec3(1.0, -2.0, 3.0))
    state.finalize(viewer)

    while viewer.render_frame():
        if viewer.simulating():
            pipeline.simulate(viewer, state, timestamps)
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
