"""Python port of `crates/examples3d/pyramid3.rs`. A tall 3D pyramid stack of cuboids resting on the ground.

Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/pyramid3.py
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


def create_pyramid(state, viewer, offset, stack_height, half_extents):
    shift = half_extents * 2.5
    for i in range(stack_height):
        for j in range(i, stack_height):
            for k in range(i, stack_height):
                fi = float(i)
                fj = float(j)
                fk = float(k)
                x = (
                    (fi * shift.x / 2.0)
                    + (fk - fi) * shift.x
                    + offset.x
                    - stack_height * half_extents.x
                )
                y = fi * shift.y + offset.y
                z = (
                    (fi * shift.z / 2.0)
                    + (fj - fi) * shift.z
                    + offset.z
                    - stack_height * half_extents.z
                )

                add_body(
                    state,
                    viewer,
                    RigidBodyBuilder.dynamic().translation(Vec3(x, y, z)).build(),
                    ColliderBuilder.cuboid(
                        half_extents.x, half_extents.y, half_extents.z
                    ).build(),
                )


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    # Ground
    ground_size = 200.0
    ground_height = 0.1

    add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(Vec3(0.0, -ground_height, 0.0)).build(),
        ColliderBuilder.cuboid(ground_size, ground_height, ground_size).build(),
    )

    # Create the cubes
    cube_size = 1.0
    hext = Vec3(cube_size, cube_size, cube_size)
    bottomy = cube_size
    create_pyramid(state, viewer, Vec3(0.0, bottomy, 0.0), 50, hext)

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
