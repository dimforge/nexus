"""Python port of `crates/examples3d/joint_ball3.rs`.

A large grid of spherical (ball) joints, like a piece of cloth, with a
pile of rigid bodies falling on top.
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    SphericalJointBuilder,
    GpuTimestamps,
    Vec3,
    Pose,
)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    rad = 0.4
    ni = 200
    nk = 301
    shift = 1.0
    center = Vec3(nk * shift / 2.0, 0.0, ni * shift / 2.0)

    body_handles = []

    # A lot of joints. Kind of like a piece of cloth.
    for k in range(nk):
        for i in range(ni):
            fk = float(k)
            fi = float(i)

            fixed = ((i == 0 or i == ni - 1) and (k % 4 == 0 or k == ni - 1)) or (
                (k == 0 or k == nk - 1) and (i % 4 == 0 or i == nk - 1)
            )

            if fixed:
                rigid_body = (
                    RigidBodyBuilder.fixed()
                    .translation(Vec3(fk * shift, 0.0, fi * shift) - center)
                    .build()
                )
                collider = ColliderBuilder.cuboid(rad, rad, rad).build()
            else:
                rigid_body = (
                    RigidBodyBuilder.dynamic()
                    .translation(Vec3(fk * shift, 0.0, fi * shift) - center)
                    .build()
                )
                collider = ColliderBuilder.ball(rad).density(10.0).build()

            shape = collider.shared_shape()
            child_handle = state.insert_rigid_body(rigid_body, collider)
            viewer.insert_shape(child_handle, shape, Pose.IDENTITY)

            # Vertical joint.
            if i > 0:
                parent_handle = body_handles[-1]
                joint = SphericalJointBuilder.new().local_anchor2(
                    Vec3(0.0, 0.0, -shift)
                )
                state.insert_impulse_joint(parent_handle, child_handle, joint)

            # Horizontal joint.
            if k > 0:
                parent_index = len(body_handles) - ni
                parent_handle = body_handles[parent_index]
                joint = SphericalJointBuilder.new().local_anchor2(
                    Vec3(-shift, 0.0, 0.0)
                )
                state.insert_impulse_joint(parent_handle, child_handle, joint)

            body_handles.append(child_handle)

    # Some rigid-bodies to fall on top.
    nj = 10
    nk = nk // 3
    ni = ni // 6
    rad = rad * 2.5

    for k in range(nk):
        for i in range(ni):
            for j in range(nj):
                body = (
                    RigidBodyBuilder.dynamic()
                    .translation(
                        Vec3(
                            (k - nk / 2.0) * rad * 2.1,
                            j * rad * 2.1 + 2.0,
                            (i - ni / 2.0) * rad * 2.1,
                        )
                    )
                    .build()
                )
                collider = ColliderBuilder.cuboid(rad, rad, rad).build()
                shape = collider.shared_shape()
                handle = state.insert_rigid_body(body, collider)
                viewer.insert_shape(handle, shape, Pose.IDENTITY)

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
