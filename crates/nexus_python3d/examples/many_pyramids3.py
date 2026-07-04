"""Python port of `crates/examples3d/many_pyramids3.rs`. Two rows of many cuboid pyramids on a long ground plane.

Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/many_pyramids3.py
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


def add_body(state, viewer, body, collider):
    """Inserts a body + collider into the state and registers its render shape."""
    shape = collider.shared_shape()
    handle = state.insert_rigid_body(body, collider)
    viewer.insert_shape(handle, shape, Pose.IDENTITY)
    return handle


def create_pyramid(state, viewer, offset, stack_height, rad):
    shift = rad * 2.0

    for i in range(stack_height):
        for j in range(i, stack_height):
            fj = float(j)
            fi = float(i)
            x = (fi * shift / 2.0) + (fj - fi) * shift
            y = fi * shift

            add_body(
                state,
                viewer,
                RigidBodyBuilder.dynamic()
                .translation(Vec3(x, y, 0.0) + offset)
                .build(),
                ColliderBuilder.cuboid(rad, rad, rad).build(),
            )


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    rad = 0.5
    pyramid_count = 40
    spacing = 4.0

    # Ground
    ground_size = 100.0
    ground_height = 0.1

    add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(Vec3(0.0, -ground_height, 0.0)).build(),
        ColliderBuilder.cuboid(
            ground_size,
            ground_height,
            pyramid_count * spacing / 2.0 + ground_size,
        ).build(),
    )

    # Create the cubes
    for pyramid_index in range(pyramid_count):
        bottomy = rad
        create_pyramid(
            state,
            viewer,
            Vec3(
                0.0,
                bottomy,
                (pyramid_index - pyramid_count / 2.0) * spacing,
            ),
            60,
            rad,
        )

        create_pyramid(
            state,
            viewer,
            Vec3(
                -75.0,
                bottomy,
                (pyramid_index - pyramid_count / 2.0) * spacing,
            ),
            60,
            rad,
        )

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
    # The work is done once the window closes. Exit without running interpreter
    # finalization: some GPU/windowing thread-locals abort if destroyed during
    # Python teardown (an upstream ordering quirk, harmless to skip).
    import os

    os._exit(0)
