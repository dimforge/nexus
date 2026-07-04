"""Python port of `crates/examples3d/primitives3.rs`.

A grid of falling dynamic objects cycling through every primitive shape
(cylinder, cuboid, cone, capsule, ball) plus 5 randomly-generated convex
polyhedra, dropping into a walled box. Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/primitives3.py
"""

import random

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

NXZ = 20
NY = 40


def make_polyhedron_points():
    """Build 5 point lists (10 points each, scaled by 2.0) for convex hulls.

    Unlike the Rust source we keep the POINT LISTS (there is no
    `ColliderBuilder.new(shape)` to reuse a `SharedShape`), and rebuild the
    convex hull on each reuse. The center-of-mass recentring step is skipped.
    """
    rng = random.Random(42)
    polyhedra = []
    for _ in range(5):
        points = []
        scale = 2.0
        for _ in range(10):
            points.append(
                [
                    (rng.random() - 0.5) * scale,
                    (rng.random() - 0.5) * scale,
                    (rng.random() - 0.5) * scale,
                ]
            )
        polyhedra.append(points)
    return polyhedra


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    # Create 5 predefined convex polyhedron point sets (so we can render them
    # efficiently with instancing).
    polyhedron_points = make_polyhedron_points()

    for j in range(NY):
        max_ik = NXZ // 2
        for i in range(-max_ik, max_ik):
            for k in range(-max_ik, max_ik):
                x = i * 1.1 + j * 0.01
                y = j * 1.6 + 1.0
                z = k * 1.1 + j * 0.01
                pos = Vec3(x, y, z)

                jm = j % 6
                if jm == 0:
                    cb = ColliderBuilder.cylinder(0.5, 0.5)
                elif jm == 1:
                    cb = ColliderBuilder.cuboid(0.5, 0.5, 0.5)
                elif jm == 2:
                    cb = ColliderBuilder.cone(0.5, 0.5)
                elif jm == 3:
                    cb = ColliderBuilder.capsule_y(0.4, 0.4)
                elif jm == 4:
                    cb = ColliderBuilder.ball(0.5)
                else:
                    if i % 2 == 0 or k % 2 == 0:
                        continue
                    # Reuse one of the 5 predefined polyhedron shapes.
                    shape_idx = ((i + max_ik) + (k + max_ik)) % len(polyhedron_points)
                    cb = ColliderBuilder.convex_hull(polyhedron_points[shape_idx])

                collider = cb.build()
                body = RigidBodyBuilder.dynamic().translation(pos).build()
                shape = collider.shared_shape()
                handle = state.insert_rigid_body(body, collider)
                viewer.insert_shape(handle, shape, Pose.IDENTITY)

    # Floor made of large cuboids.
    thick = NXZ * 1.3
    height = 5.0
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
