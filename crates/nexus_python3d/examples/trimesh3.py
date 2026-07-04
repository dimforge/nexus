"""Python port of `crates/examples3d/trimesh3.rs`.

The same grid of falling primitives and random convex polyhedra as
`primitives3`, but dropping onto a trimesh floor built from a heightfield.
Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/trimesh3.py
"""

import math
import random

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

NXZ = 20
NY = 40


def heightfield_trimesh(nrows, ncols, height_fn, scale):
    """Grid trimesh centered at origin. scale=(sx,sy,sz); y = height_fn(i,j)*sy."""
    sx, sy, sz = scale
    verts = []
    for i in range(nrows):
        for j in range(ncols):
            x = (i / (nrows - 1) - 0.5) * sx
            z = (j / (ncols - 1) - 0.5) * sz
            verts.append([x, height_fn(i, j) * sy, z])
    idx = []
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            a = i * ncols + j
            idx.append([a, a + 1, a + ncols])
            idx.append([a + 1, a + ncols + 1, a + ncols])
    return verts, idx


def make_polyhedron_points():
    """Build 5 point lists (10 points each, scaled by 2.0) for convex hulls.

    Keep the POINT LISTS (there is no `ColliderBuilder.new(shape)` to reuse a
    `SharedShape`) and rebuild the convex hull on each reuse. The center-of-mass
    recentring step is skipped.
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

    # Create 5 predefined convex polyhedron point sets.
    polyhedron_points = make_polyhedron_points()

    for j in range(NY):
        max_ik = NXZ // 2
        for i in range(-max_ik, max_ik):
            for k in range(-max_ik, max_ik):
                x = i * 1.1 + j * 0.01
                y = j * 1.6 + 2.0
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
                    shape_idx = ((i + max_ik) + (k + max_ik)) % len(polyhedron_points)
                    cb = ColliderBuilder.convex_hull(polyhedron_points[shape_idx])

                collider = cb.build()
                body = RigidBodyBuilder.dynamic().translation(pos).build()
                shape = collider.shared_shape()
                handle = state.insert_rigid_body(body, collider)
                viewer.insert_shape(handle, shape, Pose.IDENTITY)

    # A trimesh floor built from the mesh representation of a heightfield.
    ground_size = (100.0, 1.0, 100.0)
    nsubdivs = 20

    def height_fn(i, j):
        if i == 0 or i == nsubdivs or j == 0 or j == nsubdivs:
            return 10.0
        x = i * ground_size[0] / nsubdivs
        z = j * ground_size[2] / nsubdivs
        return math.sin(x) + math.cos(z)

    vertices, indices = heightfield_trimesh(
        nsubdivs + 1, nsubdivs + 1, height_fn, ground_size
    )

    body = RigidBodyBuilder.fixed().build()
    collider = ColliderBuilder.trimesh(vertices, indices).build()
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
    # The work is done once the window closes. Exit without running interpreter
    # finalization: some GPU/windowing thread-locals abort if destroyed during
    # Python teardown (an upstream ordering quirk, harmless to skip).
    import os

    os._exit(0)
