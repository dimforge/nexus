"""Python port of `crates/examples3d/heightfield3.rs`.

MPM sand poured onto a sinusoidal heightfield terrain (rendered as a trimesh).
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


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()
    coupling = RbdCoupling.mpm_one_way(BoundaryCondition.separate(1.0))

    nxz = 45
    cell_width = 1.0

    particles = []
    for i in range(nxz):
        for j in range(100):
            for k in range(nxz):
                position = vec3(
                    i + 0.5 - nxz / 2.0,
                    j + 0.5 + 14.0,
                    k + 0.5 - nxz / 2.0,
                ) * (cell_width / 2.0)
                density = 2700.0
                radius = cell_width / 4.0
                model = ParticleModel.sand(2.0e9, 0.2)
                particles.append(Particle(position, radius, density, model))

    params = SimulationParams(vec3(0.0, -9.81, 0.0), 1.0 / 60.0)
    state.set_mpm_params(viewer, params, cell_width)
    state.set_mpm_substeps(20)
    state.add_particles(viewer, particles)

    # Sinusoidal heightfield terrain (rendered as the converted trimesh).
    vtx, idx = heightfield_trimesh(
        200,
        200,
        lambda i, j: math.sin(i / 10.0) * math.cos(j / 10.0),
        (100.0, 5.0, 100.0),
    )
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
