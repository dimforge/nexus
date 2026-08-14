use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Pose3, vec3};
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::{ColliderBuilder, HeightField, Pose, RigidBodyBuilder, TriMeshFlags};

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(1.0));

    let nxz = 50;
    let cell_width = 1.0;

    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..30 {
            for k in 0..nxz {
                let position = vec3(
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 60.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0,
                ) * cell_width
                    / 2.0;
                let density = 2700.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::elastic(1.0e7, 0.2);
                particles.push(Particle::new(position, radius, density, model));
            }
        }
    }

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0) * 4.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(20);
    state.add_particles(viewer.backend(), particles)?;

    // Floor
    let body = RigidBodyBuilder::fixed()
        .translation(vec3(0.0, -4.0, 0.0))
        .build();
    let collider = ColliderBuilder::cuboid(100.0, 1.0, 100.0).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    // Cutting planes (3 heightfield trimeshes)
    for k in 0..3 {
        let heights = Array2::zeros(10, 10);
        let heightfield = HeightField::new(heights, vec3(35.0, 1.0, 10.0));
        let (mut vtx, idx) = heightfield.to_trimesh();
        vtx.iter_mut().for_each(|pt| {
            *pt =
                Pose3::rotation(vec3(1.3, 0.0, 0.0)) * *pt + vec3(0.0, 10.0, k as f32 * 10.0 - 10.0)
        });
        let body = RigidBodyBuilder::fixed().build();
        let collider = ColliderBuilder::trimesh_with_flags(vtx, idx, TriMeshFlags::ORIENTED)
            .unwrap()
            .build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider, coupling);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);
    }

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
