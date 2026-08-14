use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::vec3;
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::{ColliderBuilder, HeightField, Pose, RigidBodyBuilder, TriMeshFlags};

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(1.0));

    let nxz = 45;
    let cell_width = 1.0;

    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..100 {
            for k in 0..nxz {
                let position = vec3(
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 14.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0,
                ) * cell_width
                    / 2.0;
                let density = 2700.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::sand(2.0e9, 0.2);
                particles.push(Particle::new(position, radius, density, model));
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

    // Sinusoidal heightfield terrain (rendered as the converted trimesh).
    let heights = Array2::from_fn(200, 200, |i, j| {
        (i as f32 / 10.0).sin() * (j as f32 / 10.0).cos()
    });
    let heightfield = HeightField::new(heights, vec3(100.0, 5.0, 100.0));
    let (vtx, idx) = heightfield.to_trimesh();
    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::trimesh_with_flags(vtx, idx, TriMeshFlags::ORIENTED)
        .unwrap()
        .build();
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
