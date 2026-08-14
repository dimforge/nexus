"""Python port of `crates/examples3d/sand3.rs`.

MPM sand poured into a walled box, stirred by a rotating kinematic blade.
"""

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
YOUNG_MODULUS = 2.0e9
POISSON_RATIO = 0.2


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()
    # MPM boundary colliders are inserted as rigid bodies coupled to the
    # continuum; they push the particles but aren't pushed back.
    coupling = RbdCoupling.mpm_one_way(BoundaryCondition.separate(1.0))

    nxz = 45
    cell_width = 1.0

    # Sand particles.
    particles = []
    for i in range(nxz):
        for j in range(100):
            for k in range(nxz):
                position = vec3(
                    i + 0.5 - nxz / 2.0,
                    j + 0.5 + 10.0,
                    k + 0.5 - nxz / 2.0,
                ) * (cell_width / 2.0)
                radius = cell_width / 4.0
                model = ParticleModel.sand(YOUNG_MODULUS, POISSON_RATIO)
                particles.append(Particle(position, radius, DENSITY, model))

    params = SimulationParams(vec3(0.0, -9.81, 0.0), 1.0 / 60.0)
    state.set_mpm_params(viewer, params, cell_width)
    state.set_mpm_substeps(20)
    state.add_particles(viewer, particles)

    # Boundary colliders (floor, walls).
    thickness = 0.5
    walls_color = Vec4(0.6, 0.8, 1.0, 0.3)
    walls = [
        (vec3(0.0, -4.0, 0.0), vec3(100.0, 4.0, 100.0)),
        (vec3(0.0, 5.0, -35.0), vec3(35.0, 5.0, thickness)),
        (vec3(0.0, 5.0, 35.0), vec3(35.0, 5.0, thickness)),
        (vec3(-35.0, 5.0, 0.0), vec3(thickness, 5.0, 35.0)),
        (vec3(35.0, 5.0, 0.0), vec3(thickness, 5.0, 35.0)),
    ]
    for pos, half_extents in walls:
        body = RigidBodyBuilder.fixed().translation(pos).build()
        collider = ColliderBuilder.cuboid(
            half_extents.x, half_extents.y, half_extents.z
        ).build()
        shape = collider.shared_shape()
        handle = state.insert_rigid_body(body, collider, coupling)
        viewer.insert_shape_with_color(handle, shape, Pose.IDENTITY, walls_color)

    # Rotating blade (kinematic).
    body = (
        RigidBodyBuilder.kinematic_velocity_based()
        .translation(vec3(0.0, 2.0, 0.0))
        .rotation(vec3(0.0, 0.0, -0.5))
        .angvel(vec3(0.0, -1.0, 0.0))
        .build()
    )
    collider = ColliderBuilder.cuboid(thickness, 2.0, 30.0).build()
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
