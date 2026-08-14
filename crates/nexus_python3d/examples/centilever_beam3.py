"""Python port of `crates/examples3d/centilever_beam3.rs`.

A Neo-Hookean elastic MPM beam clamped at one end by a fixed cuboid, sagging
under gravity (an MPM example despite the name).
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


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()
    coupling = RbdCoupling.mpm_one_way(BoundaryCondition.stick())

    width = 10.0
    height = 2.0
    fixed_part = 1.0
    cell_width = 0.2
    particle_per_cell_dim = 2
    young_modulus = 1.0e7
    poisson_ratio = 0.3

    diameter = cell_width / particle_per_cell_dim
    ni = int(math.ceil((width + fixed_part) / diameter))
    njk = int(math.ceil(height / diameter))

    particles = []
    for i in range(ni):
        for j in range(njk):
            for k in range(njk):
                position = vec3(float(i), float(j), float(k)) * diameter
                density = 1000.0
                radius = diameter / 2.0
                model = ParticleModel.elastic_neo_hookean(young_modulus, poisson_ratio)
                particle = Particle(position, radius, density, model)
                particle.set_damping(2.0)
                particles.append(particle)

    params = SimulationParams(vec3(0.0, -9.81, 0.0), 1.0 / 60.0)
    state.set_mpm_params(viewer, params, cell_width)
    state.set_mpm_substeps(20)
    state.add_particles(viewer, particles)

    # Fixed block that clamps one end of the beam.
    body = RigidBodyBuilder.fixed().translation(
        vec3(0.0, height / 2.0, height / 2.0)
    ).build()
    collider = ColliderBuilder.cuboid(fixed_part, height, height).build()
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
