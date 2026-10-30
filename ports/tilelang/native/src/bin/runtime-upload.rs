//! Generate ordinary Nexus uploads through the production CPU conversion path.
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{pipeline::RbdState, rapier::prelude::*, shaders::dynamics::RbdSimParams};
use nexus_tilelang_native::{
    upload::write_inputs,
    upload_cases::{capacities, scene, small_contact_scene, wide_scene},
};
use std::{fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    futures::executor::block_on(async {
        let output = PathBuf::from(
            std::env::args()
                .nth(1)
                .ok_or("Expected upload output root")?,
        );
        fs::create_dir_all(&output)?;
        let backend = GpuBackend::Cpu;
        let mesh = SharedShape::trimesh_with_flags(
            vec![
                Vec3::new(-2.0, 0.0, -2.0),
                Vec3::new(2.0, 0.0, -2.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
            vec![[0, 1, 2]],
            TriMeshFlags::MERGE_DUPLICATE_VERTICES,
        )?;
        let scenes = [scene(0.0, mesh.clone()), scene(10.0, mesh)];
        let multibodies = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        params.dt = 0.03;
        let refs: Vec<_> = scenes
            .iter()
            .map(|(b, c, j)| (b, c, j, &multibodies, &params))
            .collect();
        let mut state = RbdState::from_rapier(&backend, &refs, capacities(2));
        state.set_gravity(&backend, [0.0, -3.0, 0.5]);
        state.set_deterministic(&backend, true);
        backend.synchronize()?;
        let mut upload = state.read_scene_upload(&backend).await?;
        write_inputs(&upload, &output.join("multi-collider-joint"))?;
        for i in 0..4 {
            upload.body_poses[i].translation.y -= 2.45;
        }
        write_inputs(&upload, &output.join("mesh-contact-joint"))?;
        let scenes = [wide_scene(0.0), wide_scene(10.0), wide_scene(20.0)];
        let refs: Vec<_> = scenes
            .iter()
            .map(|(b, c, j)| (b, c, j, &multibodies, &params))
            .collect();
        let mut caps = capacities(3);
        caps.body_capacity = 66;
        caps.collisions_capacity = 256;
        let mut state = RbdState::from_rapier(&backend, &refs, caps);
        state.set_gravity(&backend, [0.0, -9.81, 0.0]);
        state.set_deterministic(&backend, true);
        backend.synchronize()?;
        write_inputs(
            &state.read_scene_upload(&backend).await?,
            &output.join("wide-contacts"),
        )?;
        let scenes: Vec<_> = (0..24)
            .map(|i| small_contact_scene(i as f32 * 10.0))
            .collect();
        let refs: Vec<_> = scenes
            .iter()
            .map(|(b, c, j)| (b, c, j, &multibodies, &params))
            .collect();
        let mut caps = capacities(24);
        caps.collisions_capacity = 64;
        let mut state = RbdState::from_rapier(&backend, &refs, caps);
        state.set_gravity(&backend, [0.0, -9.81, 0.0]);
        state.set_deterministic(&backend, true);
        backend.synchronize()?;
        write_inputs(
            &state.read_scene_upload(&backend).await?,
            &output.join("many-batches"),
        )?;
        println!("Exported native CPU joint, mesh-contact and multi-workgroup uploads");
        Ok(())
    })
}
