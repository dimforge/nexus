"""Python port of `crates/examples3d/mpm_emitter3.rs`.

A dynamic emitter spawns a stream of sand that orbits the center of a walled box.
"""

import math
from collections import deque

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
    Vec4,
    Pose,
    vec3,
)

DENSITY = 2700.0
YOUNG_MODULUS = 2.0e8
POISSON_RATIO = 0.2

EMIT_EVERY = 10
EMIT_BLOCK = 30
EMIT_BLOCK_Y = 2
MAX_PARTICLES = 250_000


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()
    coupling = RbdCoupling.mpm_one_way(BoundaryCondition.separate(1.0))

    cell_width = 1.0
    dt = 1.0 / 60.0

    params = SimulationParams(vec3(0.0, -9.81, 0.0), dt)
    state.set_mpm_params(viewer, params, cell_width)
    state.set_mpm_substeps(10)

    # Boundary colliders: a floor and four walls forming an open box.
    thickness = 0.5
    walls_color = Vec4(0.6, 0.8, 1.0, 0.3)
    walls = [
        (vec3(0.0, -thickness, 0.0), vec3(30.0, thickness, 30.0)),
        (vec3(0.0, 10.0, -30.0), vec3(30.0, 10.0, thickness)),
        (vec3(0.0, 10.0, 30.0), vec3(30.0, 10.0, thickness)),
        (vec3(-30.0, 10.0, 0.0), vec3(thickness, 10.0, 30.0)),
        (vec3(30.0, 10.0, 0.0), vec3(thickness, 10.0, 30.0)),
    ]
    for pos, half_extents in walls:
        body = RigidBodyBuilder.fixed().translation(pos).build()
        collider = ColliderBuilder.cuboid(
            half_extents.x, half_extents.y, half_extents.z
        ).build()
        shape = collider.shared_shape()
        handle = state.insert_rigid_body(body, collider, coupling)
        viewer.insert_shape_with_color(handle, shape, Pose.IDENTITY, walls_color)

    timestamps = GpuTimestamps(viewer, 2048)
    viewer.add_directional_light(Vec3(1.0, -2.0, 3.0))
    state.finalize(viewer)

    # Dynamic emitter: a small cube of sand spawned at an orbiting point.
    radius = cell_width / 4.0
    spacing = radius * 2.0
    model = ParticleModel.sand(YOUNG_MODULUS, POISSON_RATIO)
    emit_height = 40.0
    orbit_radius = 10.0
    angular_speed = 1.5  # rad/s

    chunks: deque = deque()
    total_particles = 0
    t = 0.0
    step = 0

    while viewer.render_frame():
        if viewer.simulating():
            if step % EMIT_EVERY == 0 and total_particles < MAX_PARTICLES:
                angle = t * angular_speed
                center = vec3(
                    orbit_radius * math.cos(angle),
                    emit_height,
                    orbit_radius * math.sin(angle),
                )

                particles = []
                for i in range(EMIT_BLOCK):
                    for j in range(EMIT_BLOCK_Y):
                        for k in range(EMIT_BLOCK):
                            offset = vec3(
                                (i - EMIT_BLOCK // 2) * spacing,
                                (j - EMIT_BLOCK_Y // 2) * spacing,
                                (k - EMIT_BLOCK // 2) * spacing,
                            )
                            particle = Particle(center + offset, radius, DENSITY, model)
                            particle.velocity = vec3(0.0, -8.0, 0.0)
                            particles.append(particle)

                n = len(particles)
                chunk = state.add_particles(viewer, particles)
                chunks.append((chunk, n))
                total_particles += n

            pipeline.simulate(viewer, state, timestamps)
            t += dt
            step += 1
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
