"""Python port of `crates/examples3d/elastic_cut3.rs`.

An elastic MPM block falls onto a floor through three tilted cutting planes
(each a flat heightfield trimesh rotated about X and offset).
"""

import math

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RbdCoupling,
    BoundaryCondition,
    RigidBodyBuilder,
    ColliderBuilder,
    GpuTimestamps,
    SimulationParams,
    ParticleModel,
    Particle,
    Vec3,
    Pose,
    vec3,
)


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


def rotate_x(p, angle):
    """Rotate point [x, y, z] about the X axis by `angle` radians."""
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = p
    return [x, y * c - z * s, y * s + z * c]


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()
    coupling = RbdCoupling.mpm_one_way(BoundaryCondition.separate(1.0))

    nxz = 50
    cell_width = 1.0

    particles = []
    for i in range(nxz):
        for j in range(30):
            for k in range(nxz):
                position = vec3(
                    i + 0.5 - nxz / 2.0,
                    j + 0.5 + 60.0,
                    k + 0.5 - nxz / 2.0,
                ) * (cell_width / 2.0)
                density = 2700.0
                radius = cell_width / 4.0
                model = ParticleModel.elastic(1.0e7, 0.2)
                particles.append(Particle(position, radius, density, model))

    params = SimulationParams(vec3(0.0, -9.81, 0.0) * 4.0, 1.0 / 60.0)
    state.set_mpm_params(viewer, params, cell_width)
    state.set_mpm_substeps(20)
    state.add_particles(viewer, particles)

    # Floor
    body = RigidBodyBuilder.fixed().translation(vec3(0.0, -4.0, 0.0)).build()
    collider = ColliderBuilder.cuboid(100.0, 1.0, 100.0).build()
    shape = collider.shared_shape()
    handle = state.insert_rigid_body(body, collider, coupling)
    viewer.insert_shape(handle, shape, Pose.IDENTITY)

    # Cutting planes (3 heightfield trimeshes), each tilted about X by 1.3 rad
    # and offset. The rotation + translation are baked into the vertices here.
    for k in range(3):
        vtx, idx = heightfield_trimesh(
            10, 10, lambda i, j: 0.0, (35.0, 1.0, 10.0)
        )
        offset = [0.0, 10.0, k * 10.0 - 10.0]
        vtx = [
            [
                r[0] + offset[0],
                r[1] + offset[1],
                r[2] + offset[2],
            ]
            for r in (rotate_x(pt, 1.3) for pt in vtx)
        ]
        body = RigidBodyBuilder.fixed().build()
        collider = ColliderBuilder.trimesh(vtx, idx).build()
        shape = collider.shared_shape()
        handle = state.insert_rigid_body(body, collider, coupling)
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
