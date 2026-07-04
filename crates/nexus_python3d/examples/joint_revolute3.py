"""Python port of `crates/examples3d/joint_revolute3.rs`.

Many square loops of revolute joints chained together.
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    RevoluteJointBuilder,
    GpuTimestamps,
    Vec3,
    Pose,
)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    rad = 0.4
    num = 10
    shift = 2.0
    nk = 10
    nj = 50

    for k in range(nk):
        for l in range(4):
            y = l * shift * num * 3.0

            for j in range(nj):
                x = (j - nj / 2.0) * shift * 4.0
                z0 = (k - nk / 2.0) * num * shift * 2.1

                ground = RigidBodyBuilder.fixed().translation(Vec3(x, y, z0)).build()
                collider = ColliderBuilder.cuboid(rad, rad, rad).build()
                shape = collider.shared_shape()
                curr_parent = state.insert_rigid_body(ground, collider)
                viewer.insert_shape(curr_parent, shape, Pose.IDENTITY)

                for i in range(num):
                    z = z0 + i * shift * 2.0 + shift
                    positions = [
                        Pose.from_translation(Vec3(x, y, z)),
                        Pose.from_translation(Vec3(x + shift, y, z)),
                        Pose.from_translation(Vec3(x + shift, y, z + shift)),
                        Pose.from_translation(Vec3(x, y, z + shift)),
                    ]

                    handles = [curr_parent] * 4
                    for m in range(4):
                        density = 1.0
                        rigid_body = (
                            RigidBodyBuilder.dynamic().pose(positions[m]).build()
                        )
                        collider = (
                            ColliderBuilder.cuboid(rad, rad, rad)
                            .density(density)
                            .build()
                        )
                        shape = collider.shared_shape()
                        handles[m] = state.insert_rigid_body(
                            rigid_body, collider
                        )
                        viewer.insert_shape(handles[m], shape, Pose.IDENTITY)

                    ax = Vec3.X
                    az = Vec3.Z

                    revs = [
                        RevoluteJointBuilder.new(az).local_anchor2(
                            Vec3(0.0, 0.0, -shift)
                        ),
                        RevoluteJointBuilder.new(ax).local_anchor2(
                            Vec3(-shift, 0.0, 0.0)
                        ),
                        RevoluteJointBuilder.new(az).local_anchor2(
                            Vec3(0.0, 0.0, -shift)
                        ),
                        RevoluteJointBuilder.new(ax).local_anchor2(
                            Vec3(shift, 0.0, 0.0)
                        ),
                    ]

                    state.insert_impulse_joint(curr_parent, handles[0], revs[0])
                    state.insert_impulse_joint(handles[0], handles[1], revs[1])
                    state.insert_impulse_joint(handles[1], handles[2], revs[2])
                    state.insert_impulse_joint(handles[2], handles[3], revs[3])

                    curr_parent = handles[3]

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
