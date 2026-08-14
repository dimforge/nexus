use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::Vec2;
use rapier2d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

/// Inserts a boundary collider coupled (one-way) to the MPM particles and
/// registers it for rendering.
fn insert_boundary(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    body: RigidBody,
    collider: Collider,
) {
    let shape = collider.shared_shape().clone();
    let friction = BoundaryCondition::separate(1.0);
    let handle = state.insert_rigid_body(body, collider, RbdCoupling::MpmOneWay(friction));
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let offset_y = 46.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;

    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position =
                glamx::vec2(i as f32 + 0.5, j as f32 + 0.5) * cell_width / 2.0 + Vec2::Y * offset_y;

            let density = 1000.0;
            let radius = cell_width / 4.0;
            let model = ParticleModel::elastic(5.0e6, 0.2);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(15);
    state.add_particles(viewer.backend(), particles)?;

    // const ANGVEL: f32 = 1.0; // 2.0;

    /*
     * Static platforms.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(35.0, 20.0))
            .build(),
        ColliderBuilder::cuboid(70.0, 1.0).build(),
    );

    let mut polyline = vec![];
    let subdivs = 100;
    let length = 84.0;
    let start = glamx::vec2(35.0, 70.0) - glamx::vec2(length / 2.0, 0.0);

    for i in 0..=subdivs {
        let step = length / (subdivs as f32);
        let dx = i as f32 * step;
        polyline.push(start + glamx::vec2(dx, dx.sin()))
    }

    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::polyline(polyline, None).build(),
    );

    for k in 0..6 {
        insert_boundary(
            &mut state,
            viewer,
            RigidBodyBuilder::fixed().build(),
            ColliderBuilder::polyline(
                vec![
                    glamx::vec2(0.0 + k as f32 * 15.0, 20.0),
                    glamx::vec2(-10.0 + k as f32 * 15.0, 45.0),
                ],
                None,
            )
            .build(),
        );
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
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
