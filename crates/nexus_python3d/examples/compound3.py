"""Python port of `crates/examples3d/compound3.rs`.

Port of rapier's `compound3` demo, but every "U"-shaped body is assembled from
THREE separate colliders attached to one rigid body -- exercising
multiple-colliders-per-body support -- instead of a single compound collider.
Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/compound3.py
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


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    # Floor made of large cuboids.
    thick = 50.0
    height = 7.0
    walls_color = Vec4(0.6, 0.8, 1.0, 0.3)
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

    # "U"-shaped bodies, each made of three cuboid colliders: a horizontal base
    # and two vertical walls, forming an upward-opening cup. `(local_pose,
    # half_extents)` for the three parts.
    rad = 0.2
    parts = [
        (Vec3.ZERO, Vec3(rad * 10.0, rad, rad)),
        (Vec3(rad * 10.0, rad * 10.0, 0.0), Vec3(rad, rad * 10.0, rad)),
        (Vec3(-rad * 10.0, rad * 10.0, 0.0), Vec3(rad, rad * 10.0, rad)),
    ]

    num = 10
    numy = 100
    # Each U spans ~4 units in x; space the grid out so they don't start
    # interpenetrating.
    shift = rad * 10.0 * 2.0 + 1.0
    center = shift * num / 2.0

    for j in range(numy):
        for i in range(num):
            for k in range(num):
                x = i * shift - center
                y = j * shift + 5.0
                z = k * shift - center

                body = RigidBodyBuilder.dynamic().translation(Vec3(x, y, z)).build()

                handle = state.insert_body_in(0, body)
                for offset, he in parts:
                    collider = (
                        ColliderBuilder.cuboid(he.x, he.y, he.z)
                        .translation(offset)
                        .build()
                    )
                    viewer.insert_shape(
                        handle, collider.shared_shape(), Pose.from_translation(offset)
                    )
                    state.insert_collider_in(0, collider, parent=handle)

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
