"""Python port of `crates/examples3d/multibody_pendulum3.rs`.

A multi-link pendulum modeled with multibody revolute joints. A fixed root
body is anchored at the origin and dynamic links hang from revolute joints
about the Z axis, swinging under gravity.
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    InteractionGroups,
    RevoluteJointBuilder,
    GpuTimestamps,
    Vec3,
    Pose,
)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    rad = 0.4
    link_len = 2.0
    num_links = 20

    # Fixed root at origin.
    root_body = RigidBodyBuilder.fixed().build()
    root_collider = ColliderBuilder.cuboid(rad, rad, rad).build()
    root_shape = root_collider.shared_shape()
    parent_handle = state.insert_rigid_body(root_body, root_collider)
    viewer.insert_shape(parent_handle, root_shape, Pose.IDENTITY)

    for i in range(num_links):
        # Each link hangs `link_len` below its parent.
        x = (i + 1.0) * link_len
        rigid_body = (
            RigidBodyBuilder.dynamic().translation(Vec3(x, 0.0, 0.0)).build()
        )
        # Disable link-vs-link collisions so the chain can fold on itself.
        collider = (
            ColliderBuilder.cuboid(link_len * 0.5, rad, rad)
            .collision_groups(InteractionGroups.none())
            .build()
        )
        shape = collider.shared_shape()
        handle = state.insert_rigid_body(rigid_body, collider)
        viewer.insert_shape(handle, shape, Pose.IDENTITY)

        # Revolute joint about Z: anchor on parent is at its bottom
        # (or at origin for the root), anchor on child is at its top.
        if i == 0:
            parent_anchor = Vec3.ZERO
        else:
            parent_anchor = Vec3(link_len * 0.8, 0.0, 0.0)
        joint = (
            RevoluteJointBuilder.new(Vec3.Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3(-link_len * 0.8, 0.0, 0.0))
        )
        state.insert_multibody_joint(parent_handle, handle, joint)

        parent_handle = handle

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
