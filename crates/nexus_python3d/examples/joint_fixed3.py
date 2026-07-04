"""Python port of `crates/examples3d/joint_fixed3.rs`.

Many grids of balls rigidly linked together with fixed joints.
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    FixedJointBuilder,
    GpuTimestamps,
    Vec3,
    Pose,
)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    rad = 0.4
    num = 10
    shift = 1.0

    body_handles = []

    for m in range(10):
        z = m * shift * (num + 2.0)

        for l in range(10):
            y = l * shift * 3.0

            for j in range(10):
                x = j * shift * num * 2.0

                for k in range(num):
                    for i in range(num):
                        fk = float(k)
                        fi = float(i)

                        # NOTE: the num - 2 test is to avoid two consecutive
                        # fixed bodies. Because physx will crash if we add
                        # a joint between these.
                        fixed = i == 0 and (
                            k % 4 == 0 and k != num - 2 or k == num - 1
                        )

                        if fixed:
                            rigid_body = (
                                RigidBodyBuilder.fixed()
                                .translation(Vec3(x + fk * shift, y, z + fi * shift))
                                .build()
                            )
                        else:
                            rigid_body = (
                                RigidBodyBuilder.dynamic()
                                .translation(Vec3(x + fk * shift, y, z + fi * shift))
                                .build()
                            )
                        collider = ColliderBuilder.ball(rad).build()
                        shape = collider.shared_shape()
                        child_handle = state.insert_rigid_body(
                            rigid_body, collider
                        )
                        viewer.insert_shape(child_handle, shape, Pose.IDENTITY)

                        # Vertical joint.
                        if i > 0:
                            parent_handle = body_handles[-1]
                            joint = FixedJointBuilder.new().local_anchor2(
                                Vec3(0.0, 0.0, -shift)
                            )
                            state.insert_impulse_joint(
                                parent_handle, child_handle, joint
                            )

                        # Horizontal joint.
                        if k > 0:
                            parent_index = len(body_handles) - num
                            parent_handle = body_handles[parent_index]
                            joint = FixedJointBuilder.new().local_anchor2(
                                Vec3(-shift, 0.0, 0.0)
                            )
                            state.insert_impulse_joint(
                                parent_handle, child_handle, joint
                            )

                        body_handles.append(child_handle)

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
