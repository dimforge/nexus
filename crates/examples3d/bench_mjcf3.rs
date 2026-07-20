//! Headless benchmark for a single MJCF robot (MuJoCo Menagerie).
//!
//! Reproduces the `mujoco_menagerie3` demo setup (robot + floor + zero-ctrl
//! position servos) without the viewer, and measures per-step wall time plus
//! the GPU pass breakdown.
//!
//! Usage:
//!
//! ```text
//! cargo run -p nexus_examples_3d --release --features metal \
//!     --bin bench_mjcf3 -- [robot] [num_warmup] [num_iters]
//! ```
//!
//! - `robot`: menagerie sub-directory (default: `unitree_a1`), resolved under
//!   `MUJOCO_MENAGERIE_DIR` (default: `../mujoco_menagerie` next to the
//!   workspace).
//! - `BACKEND=cpu|webgpu|metal` selects the backend (default: metal).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use khal::backend::{Backend, GpuBackend, GpuTimestamps, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusPipeline, NexusState};
use rapier3d::prelude::*;
use rapier3d_mjcf::{MjcfLoaderOptions, MjcfMultibodyOptions, MjcfRobot};

fn menagerie_root() -> PathBuf {
    if let Ok(dir) = std::env::var("MUJOCO_MENAGERIE_DIR") {
        return PathBuf::from(dir);
    }
    // CARGO_MANIFEST_DIR == <workspace>/crates/examples3d
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../mujoco_menagerie")
}

/// Mirrors `mujoco_menagerie3::load_scene` (minus the viewer): robot inserted
/// as multibodies with self-contacts disabled, zero-ctrl actuators applied,
/// and a floor sized from the model's AABB.
fn build_state(scene: &Path) -> anyhow::Result<NexusState> {
    // `NEXUS_BENCH_COLLISIONS` overrides the per-batch collision capacity
    // (default 4096) — used to isolate capacity-proportional kernel costs.
    let mut state = match std::env::var("NEXUS_BENCH_COLLISIONS") {
        Ok(cap) => {
            let cap: u32 = cap.parse().expect("bad NEXUS_BENCH_COLLISIONS");
            println!("collision capacity override: {cap}");
            NexusState::new(nexus3d::prelude::NexusCapacities::default().rbd_collisions(cap))
        }
        Err(_) => NexusState::default(),
    };

    let options = MjcfLoaderOptions {
        skip_plane_geoms: true,
        make_roots_fixed: false,
        create_colliders_from_visual_shapes: false,
        collider_blueprint: ColliderBuilder::default().density(0.0),
        ..MjcfLoaderOptions::default()
    };

    let (robot, _model) = MjcfRobot::from_file(scene, options)
        .map_err(|e| anyhow::anyhow!("failed to load `{}`: {e}", scene.display()))?;

    let mut floor = None;
    {
        let world = state.rbd_world_mut(0);
        let handles = robot.insert_using_multibody_joints(
            &mut world.bodies,
            &mut world.colliders,
            &mut world.multibody_joints,
            &mut world.impulse_joints,
            MjcfMultibodyOptions::DISABLE_SELF_CONTACTS,
        );
        let ctrl = vec![0.0; handles.actuators.len()];
        handles.apply_controls_multibody(&mut world.bodies, &mut world.multibody_joints, &ctrl);

        let mut aabb = Aabb::new_invalid();
        for (_, collider) in world.colliders.iter() {
            aabb.merge(&collider.compute_aabb());
        }
        if aabb.mins.x <= aabb.maxs.x {
            let center = aabb.center();
            let he = aabb.half_extents();
            let footprint = he.x.max(he.y).max(0.5);
            let floor_thick = 0.1;
            let floor_he = Vec3::new(footprint * 6.0, footprint * 6.0, floor_thick);
            let floor_center = Vec3::new(center.x, center.y, center.z - he.z - floor_thick);
            floor = Some((floor_center, floor_he));
        }

        println!(
            "scene `{}`: {} bodies, {} colliders, {} multibodies ({} dofs max)",
            scene.display(),
            world.bodies.len(),
            world.colliders.len(),
            world.multibody_joints.multibodies().count(),
            world
                .multibody_joints
                .multibodies()
                .map(|mb| mb.ndofs())
                .max()
                .unwrap_or(0),
        );
    }

    if let Some((center, he)) = floor {
        let body = RigidBodyBuilder::fixed().translation(center).build();
        let collider = ColliderBuilder::cuboid(he.x, he.y, he.z).build();
        state.insert_rigid_body(body, collider);
    }

    Ok(state)
}

async fn webgpu_backend() -> GpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_000_000_000,
        max_storage_buffer_binding_size: 1_000_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut webgpu = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("Failed to initialize WebGPU backend");
    webgpu.force_buffer_copy_src = true;
    GpuBackend::WebGpu(webgpu)
}

/// `BACKEND=metal|webgpu` (default: metal).
async fn select_backend() -> GpuBackend {
    match std::env::var("BACKEND").as_deref() {
        Ok("webgpu") => {
            println!("backend: WebGPU");
            webgpu_backend().await
        }
        _ => {
            #[cfg(feature = "metal")]
            {
                println!("backend: Metal");
                let metal = khal::backend::metal::Metal::new().expect("Metal init failed");
                GpuBackend::Metal(metal)
            }
            #[cfg(not(feature = "metal"))]
            {
                println!("backend: WebGPU");
                webgpu_backend().await
            }
        }
    }
}

async fn run(robot: String, n_warmup: usize, n_iters: usize) -> anyhow::Result<()> {
    let scene = menagerie_root().join(&robot).join("scene.xml");
    let backend = select_backend().await;

    let mut state = build_state(&scene)?;
    let mut pipeline = NexusPipeline::default();
    state.finalize(&backend).await?;
    state.set_rbd_gravity(&backend, [0.0, 0.0, -9.81]);
    // `NEXUS_BENCH_NO_CORIOLIS=1` switches to the explicit-coriolis mode
    // (MuJoCo/Genesis-like): the mass matrix / LU / gravity solve runs once per
    // step instead of once per substep.
    if std::env::var("NEXUS_BENCH_NO_CORIOLIS").is_ok() {
        println!("implicit coriolis: disabled");
        state
            .rbd
            .as_mut()
            .expect("rbd state missing")
            .multibodies_mut()
            .set_implicit_coriolis(false);
    }
    {
        let rbd = state.rbd.as_ref().expect("rbd state missing");
        println!(
            "solver: {} substeps, max_colors {}",
            rbd.num_solver_iterations(),
            rbd.max_colors(),
        );
    }

    for _ in 0..n_warmup {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;

    // `NEXUS_BENCH_PACE_MS=16` sleeps between steps to reproduce the
    // testbed's frame pacing: at low duty cycle the GPU drops to a lower
    // power state, so each step's wall/GPU time inflates vs back-to-back
    // stepping. Only the stepping time itself is measured either way.
    let pace: u64 = std::env::var("NEXUS_BENCH_PACE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if pace > 0 {
        println!("pacing: sleeping {pace} ms between steps");
    }

    let mut samples = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        if pace > 0 {
            std::thread::sleep(Duration::from_millis(pace));
        }
        let t0 = Instant::now();
        pipeline.simulate(&backend, &mut state, None).await?;
        backend.synchronize()?;
        samples.push(t0.elapsed());
    }
    samples.sort();
    let total: Duration = samples.iter().sum();
    let avg = total / n_iters as u32;
    let p50 = samples[n_iters / 2];
    println!(
        "per step: avg {:.1} µs, p50 {:.1} µs ({} timed steps)",
        avg.as_secs_f64() * 1.0e6,
        p50.as_secs_f64() * 1.0e6,
        n_iters
    );

    // GPU pass breakdown of one instrumented step.
    let mut timestamps = GpuTimestamps::new(&backend, 2048);
    pipeline
        .simulate(&backend, &mut state, Some(&mut timestamps))
        .await?;
    backend.synchronize()?;
    for _ in 0..100 {
        if let Some(results) = timestamps.try_take(&backend) {
            let mut aggregated: Vec<(String, f64, u32)> = Vec::new();
            for r in &results {
                if let Some(existing) =
                    aggregated.iter_mut().find(|(label, _, _)| label == &r.label)
                {
                    existing.1 += r.duration_ms;
                    existing.2 += 1;
                } else {
                    aggregated.push((r.label.clone(), r.duration_ms, 1));
                }
            }
            let gpu_total: f64 = aggregated.iter().map(|e| e.1).sum();
            aggregated
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!(
                "gpu passes ({} labels, {:.3} ms total):",
                aggregated.len(),
                gpu_total
            );
            for (label, ms, count) in &aggregated {
                println!("    {:>9.3} ms  ×{:<4} {}", ms, count, label);
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    // Sanity: the robot must stay put (standing on the floor), not explode.
    // Body 0 is the multibody root (the trunk): its height directly shows
    // whether the position servos hold the stance (~0.25-0.35 m for a1) or the
    // robot collapsed (~0.05-0.1 m) / exploded.
    let rbd = state.rbd.as_ref().expect("rbd state missing");
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let mut nan = 0usize;
    let mut max_pos = 0.0f32;
    let mut checksum = 0.0f64;
    for p in &poses {
        let t = p.translation;
        if !t.is_finite() {
            nan += 1;
        } else {
            max_pos = max_pos.max(t.length());
            checksum += (t.x + t.y + t.z) as f64;
        }
    }
    let trunk = poses[0].translation;
    println!(
        "sanity: non-finite {nan}, max |pos| {max_pos:.3}, trunk z {:.3}, \
         checksum {checksum:.6}, max pairs/env {}",
        trunk.z,
        state.counts().collision_pairs
    );
    if nan > 0 || max_pos > 1.0e3 {
        anyhow::bail!("simulation diverged (nan={nan}, max|pos|={max_pos:.2})");
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let robot = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "unitree_a1".to_string());
    let n_warmup = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let n_iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    pollster::block_on(async {
        if let Err(e) = run(robot, n_warmup, n_iters).await {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    });
}
