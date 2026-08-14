use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec3};
use rapier3d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

const DENSITY: f32 = 2700.0;
const YOUNG_MODULUS: f32 = 2.0e9;
const POISSON_RATIO: f32 = 0.2;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    // MPM boundary colliders are inserted as rigid bodies coupled to the
    // continuum; they push the particles but aren't pushed back.
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(1.0));

    let nxz = 45;
    let cell_width = 1.0;

    /*
     * Sand particles.
     */
    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..100 {
            for k in 0..nxz {
                let position = vec3(
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 10.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0,
                ) * cell_width
                    / 2.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::sand(YOUNG_MODULUS, POISSON_RATIO);
                particles.push(Particle::new(position, radius, DENSITY, model));
            }
        }
    }

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(20);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Boundary colliders (floor, walls, rotating blade).
     */
    let thickness = 0.5;
    let walls_color = Vec4::new(0.6, 0.8, 1.0, 0.3);
    let walls = [
        (vec3(0.0, -4.0, 0.0), vec3(100.0, 4.0, 100.0)),
        (vec3(0.0, 5.0, -35.0), vec3(35.0, 5.0, thickness)),
        (vec3(0.0, 5.0, 35.0), vec3(35.0, 5.0, thickness)),
        (vec3(-35.0, 5.0, 0.0), vec3(thickness, 5.0, 35.0)),
        (vec3(35.0, 5.0, 0.0), vec3(thickness, 5.0, 35.0)),
    ];
    for (pos, half_extents) in walls {
        let body = RigidBodyBuilder::fixed().translation(pos).build();
        let collider =
            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider, coupling);
        viewer.insert_shape_with_color(handle, &shape, Pose::IDENTITY, walls_color);
    }

    // Rotating blade (kinematic).
    let body = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vec3(0.0, 2.0, 0.0))
        .rotation(vec3(0.0, 0.0, -0.5))
        .angvel(vec3(0.0, -1.0, 0.0))
        .build();
    let collider = ColliderBuilder::cuboid(thickness, 2.0, 30.0).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
