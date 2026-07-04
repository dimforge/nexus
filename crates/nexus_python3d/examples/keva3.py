"""Python port of `crates/examples3d/keva3.rs`. A tapering tower of stacked Keva planks built from cuboid blocks.

Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/keva3.py
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


def build_block(state, viewer, half_extents, shift, dims):
    numx, numy, numz = dims
    dimensions = [
        half_extents,
        Vec3(half_extents.z, half_extents.y, half_extents.x),
    ]
    block_width = 2.0 * half_extents.z * numx
    block_height = 2.0 * half_extents.y * numy
    spacing = (half_extents.z * numx - half_extents.x) / (numz - 1.0)

    for i in range(numy):
        numx, numz = numz, numx
        dim = dimensions[i % 2]
        y = dim.y * i * 2.0

        for j in range(numx):
            if i % 2 == 0:
                x = spacing * j * 2.0
            else:
                x = dim.x * j * 2.0

            for k in range(numz):
                if i % 2 == 0:
                    z = dim.z * k * 2.0
                else:
                    z = spacing * k * 2.0

                add_body(
                    state,
                    viewer,
                    RigidBodyBuilder.dynamic()
                    .translation(
                        Vec3(
                            x + dim.x + shift.x,
                            y + dim.y + shift.y,
                            z + dim.z + shift.z,
                        )
                    )
                    .build(),
                    ColliderBuilder.cuboid(dim.x, dim.y, dim.z).build(),
                )

    # Close the top.
    dim = Vec3(half_extents.z, half_extents.x, half_extents.y)

    for i in range(int(block_width / (dim.x * 2.0))):
        for j in range(int(block_width / (dim.z * 2.0))):
            add_body(
                state,
                viewer,
                RigidBodyBuilder.dynamic()
                .translation(
                    Vec3(
                        i * dim.x * 2.0 + dim.x + shift.x,
                        dim.y + shift.y + block_height,
                        j * dim.z * 2.0 + dim.z + shift.z,
                    )
                )
                .build(),
                ColliderBuilder.cuboid(dim.x, dim.y, dim.z).build(),
            )


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    # Ground
    ground_size = 70.0
    ground_height = 2.0

    add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(Vec3(0.0, -ground_height, 0.0)).build(),
        ColliderBuilder.cuboid(ground_size, ground_height, ground_size).build(),
    )

    # Create the cubes
    half_extents = Vec3(0.02, 0.1, 0.4) * (1.0 / 2.0 * 10.0)
    block_height = 0.0
    # These should only be set to odd values otherwise
    # the blocks won't align in the nicest way.
    numy = [0, 9, 13, 17, 21, 41]

    for i in range(5, 0, -1):
        numx = int(-(-(i * 2.5) // 1))  # (i * 2.5).ceil()
        numy_i = numy[i]
        numz = numx * 3 + 1
        block_width = numx * half_extents.z * 2.0
        build_block(
            state,
            viewer,
            half_extents,
            Vec3(-block_width / 2.0, block_height, -block_width / 2.0),
            (numx, numy_i, numz),
        )
        block_height += numy_i * half_extents.y * 2.0 + half_extents.x * 2.0

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
