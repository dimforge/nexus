//! Trace the production Nexus pipeline, independently of the composed shader oracle.
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    pipeline::{RbdPipeline, RbdState},
    rapier::prelude::*,
    shaders::dynamics::RbdSimParams,
};
use nexus_tilelang_native::upload_cases::{capacities, scene, small_contact_scene, wide_scene};
use std::{fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    futures::executor::block_on(async {
        if std::env::var("NEXUS_DISABLE_BF").is_err() {
            return Err("Set NEXUS_DISABLE_BF=1 for the LBVH comparison".into());
        }
        let output = PathBuf::from(std::env::args().nth(1).ok_or("Expected output directory")?);
        fs::create_dir(&output)?;
        let backend = GpuBackend::Cpu;
        let mb = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        params.dt = 0.03;
        for label in [
            "multi-collider-joint",
            "mesh-contact-joint",
            "wide-contacts",
            "many-batches",
            "mesh-overlap-reduced",
            "mesh-overlap-unreduced",
        ] {
            let mesh = SharedShape::trimesh_with_flags(
                vec![
                    Vec3::new(-2., 0., -2.),
                    Vec3::new(2., 0., -2.),
                    Vec3::new(0., 0., 2.),
                ],
                vec![
                    [0, 1, 2];
                    if label.starts_with("mesh-overlap") {
                        3
                    } else {
                        1
                    }
                ],
                TriMeshFlags::MERGE_DUPLICATE_VERTICES,
            )?;
            let scenes = if label == "wide-contacts" {
                vec![wide_scene(0.), wide_scene(10.), wide_scene(20.)]
            } else if label == "many-batches" {
                (0..24)
                    .map(|i| small_contact_scene(i as f32 * 10.0))
                    .collect()
            } else {
                vec![scene(0., mesh.clone()), scene(10., mesh)]
            };
            let refs: Vec<_> = scenes
                .iter()
                .map(|(b, c, j)| (b, c, j, &mb, &params))
                .collect();
            let mut caps = capacities(scenes.len() as u32);
            caps.collisions_capacity = if label == "wide-contacts" { 512 } else { 64 };
            if label == "wide-contacts" {
                caps.body_capacity = 66;
            }
            let mut state = RbdState::from_rapier(&backend, &refs, caps);
            state.set_gravity(
                &backend,
                if label == "wide-contacts" || label == "many-batches" {
                    [0., -9.81, 0.]
                } else {
                    [0., -3., 0.5]
                },
            );
            state.set_deterministic(&backend, true);
            if label == "mesh-contact-joint" || label.starts_with("mesh-overlap") {
                let mut poses = backend.slow_read_vec(state.body_poses().buffer()).await?;
                for p in &mut poses[..4] {
                    p.translation.y -= 2.45;
                }
                backend.write_buffer(state.body_poses_mut().buffer_mut(), 0, &poses)?;
            }
            let initial = state.read_scene_upload(&backend).await?;
            let count = initial.batch_indices.bodies_len as usize
                * initial.batch_indices.num_batches as usize;
            let directory = output.join(label);
            fs::create_dir(&directory)?;
            nexus_tilelang_native::upload::write_inputs(&initial, &directory.join("upload"))?;
            let mut pipeline = RbdPipeline::new(&backend)?;
            pipeline.contact_reduction = !label.ends_with("unreduced");
            let mut trace = Vec::<u32>::new();
            for frame in 1..=8 {
                pipeline.step(&backend, &mut state, None)?;
                backend.synchronize()?;
                let poses = backend.slow_read_vec(state.body_poses().buffer()).await?;
                let vels = backend.slow_read_vec(state.vels().buffer()).await?;
                trace.extend_from_slice(bytemuck::cast_slice(&poses[..count]));
                trace.extend_from_slice(bytemuck::cast_slice(&vels[..count]));
                println!("{label}: production frame {frame}");
            }
            fs::write(
                directory.join("Trajectory.bin"),
                bytemuck::cast_slice(&trace),
            )?;
            fs::write(
                directory.join("scene.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"kind":"production_rbd_pipeline", "frames":8,"bodies":count,"batches":scenes.len(),"broadphase":"lbvh","deterministic":true,"contact_reduction":pipeline.contact_reduction}),
                )?,
            )?;
        }
        Ok(())
    })
}
