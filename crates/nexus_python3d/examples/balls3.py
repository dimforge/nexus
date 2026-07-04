"""Python port of `crates/examples3d/balls3.rs`. A grid of dynamic balls falls into a walled box.

Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/balls3.py
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    GpuTimestamps,
    Vec3,
    Vec4,
    Pose,
)

NXZ = 30
NY = 70


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    # Floor made of large cuboids.
    thick = NXZ * 1.3
    walls_color = Vec4(0.6, 0.8, 1.0, 0.3)
    height = 7.0
    walls = [
        (Vec3(0.0, -0.5, 0.0), Vec3(thick, 0.5, thick)),
        (Vec3(thick, height, 0.0), Vec3(0.5, height, thick)),
        (Vec3(-thick, height, 0.0), Vec3(0.5, height, thick)),
        (Vec3(0.0, height, thick), Vec3(thick, height, 0.5)),
        (Vec3(0.0, height, -thick), Vec3(thick, height, 0.5)),
    ]
    for wall_pos, wall_sz in walls:
        body = RigidBodyBuilder.fixed().build()
        collider = (
            ColliderBuilder.cuboid(wall_sz.x, wall_sz.y, wall_sz.z)
            .translation(wall_pos)
            .build()
        )
        shape = collider.shared_shape()
        handle = state.insert_rigid_body(body, collider)
        viewer.insert_shape_with_color(
            handle, shape, Pose.from_translation(wall_pos), walls_color
        )

    # Falling dynamic objects.
    for j in range(NY):
        max_ik = NXZ // 2
        for i in range(-max_ik, max_ik):
            for k in range(-max_ik, max_ik):
                x = i * 1.1 + j * 0.01
                y = j * 1.1
                z = k * 1.1
                pos = Vec3(x, y, z)

                body = RigidBodyBuilder.dynamic().translation(pos).build()
                collider = ColliderBuilder.ball(0.5).build()
                shape = collider.shared_shape()
                handle = state.insert_rigid_body(body, collider)
                viewer.insert_shape(handle, shape, Pose.IDENTITY)

    # Optional: render even before starting the simulation.
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
