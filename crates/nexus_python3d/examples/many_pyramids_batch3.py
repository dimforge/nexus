"""Python port of `crates/examples3d/many_pyramids_batch3.rs`.

Many independent pyramids, each in its own batched simulation environment.
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    GpuTimestamps,
    Vec3,
    Pose,
)

PYRAMID_COUNT = 40


def create_pyramid(state, viewer, env, offset, stack_height, rad):
    shift = rad * 2.0
    for i in range(stack_height):
        for j in range(i, stack_height):
            x = (i * shift / 2.0) + (j - i) * shift
            y = i * shift
            body = RigidBodyBuilder.dynamic().translation(Vec3(x, y, 0.0) + offset).build()
            collider = ColliderBuilder.cuboid(rad, rad, rad).build()
            shape = collider.shared_shape()
            handle = state.insert_rigid_body_in(env, body, collider)
            viewer.insert_shape_in(env, handle, shape, Pose.IDENTITY)


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()
    spacing = 4.0

    for pyramid_index in range(PYRAMID_COUNT):
        # Environment 0 already exists; allocate a fresh one for the rest.
        env = 0 if pyramid_index == 0 else state.add_environment()

        rad = 0.5

        # Ground.
        ground_size = 100.0
        ground_height = 0.1
        body = RigidBodyBuilder.fixed().translation(Vec3(0.0, -ground_height, 0.0)).build()
        collider = ColliderBuilder.cuboid(
            ground_size, ground_height, PYRAMID_COUNT * spacing / 2.0 + ground_size
        ).build()
        shape = collider.shared_shape()
        ground_handle = state.insert_rigid_body_in(env, body, collider)
        viewer.insert_shape_in(env, ground_handle, shape, Pose.IDENTITY)

        # Cubes.
        bottomy = rad
        z = (pyramid_index - PYRAMID_COUNT / 2.0) * spacing
        create_pyramid(state, viewer, env, Vec3(0.0, bottomy, z), 60, rad)
        create_pyramid(state, viewer, env, Vec3(-75.0, bottomy, z), 60, rad)

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
