//! Headless benchmark mirroring the `urdf3` demo.
//!
//! Same setup as `crates/examples3d/urdf3.rs`: load a URDF (defaulting to the
//! openarm_v10 robot), insert it as a multibody with `make_roots_fixed = true`
//! and self-contacts disabled, switch every joint to acceleration-based motors,
//! and step the simulation. The `apply_random_ang_motors` tick is replicated so
//! every motor's target velocity is re-randomised every 5 simulated seconds.
//!
//! Usage:
//!
//! ```text
//! cargo run -p nexus_examples_3d --release \
//!     --bin bench_urdf3 -- [num_batches] [num_warmup] [num_iters] [num_substeps] [path]
//! ```
//!
//! Defaults: 1 batch, 10 warmup steps, 200 timed steps, 4 substeps, openarm path.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use khal::backend::GpuBackend as KhalGpuBackend;
use khal::backend::WebGpu;
use khal::re_exports::wgpu;
use nexus_viewer3d::SimulationState;
use nexus_viewer3d::nexus::rbd::dynamics::RbdSimParams;
use nexus_viewer3d::rbd::BatchEnvironment;
use nexus_viewer3d::rbd::GpuBackend;
use nexus_viewer3d::rbd::backend::SimulationBackend;
use rapier3d::prelude::*;
use rapier3d_urdf::{UrdfLoaderOptions, UrdfMultibodyOptions, UrdfRobot};

fn default_urdf_path() -> PathBuf {
    // Mirrors `urdf3.rs`. The benchmark exits cleanly with an error if missing,
    // so users can pass their own URDF as the last CLI argument.
    PathBuf::from("/Users/sebcrozet/work/nexus-demos/XoQ/js/examples/assets/openarm_v10.urdf")
}

fn build_one_batch(path: &PathBuf, num_substeps: u32) -> Option<BatchEnvironment> {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    let scale = 40.0;
    let options = UrdfLoaderOptions {
        create_colliders_from_collision_shapes: true,
        create_colliders_from_visual_shapes: true,
        apply_imported_mass_props: true,
        make_roots_fixed: true,
        scale,
        mesh_converter: None,
        shift: Pose::from_parts(
            Vec3::new(0.0, scale, 0.0),
            Rotation::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        ),
        collider_blueprint: ColliderBuilder::ball(0.5).collision_groups(InteractionGroups::none()),
        ..UrdfLoaderOptions::default()
    };

    let (mut robot, _) = match UrdfRobot::from_file(path, options, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load URDF file at {}: {e}", path.display());
            return None;
        }
    };

    for urdf_joint in &mut robot.joints {
        urdf_joint
            .joint
            .set_motor_model(JointAxis::AngX, MotorModel::AccelerationBased);
        urdf_joint
            .joint
            .set_motor_velocity(JointAxis::AngX, 0.0, 1.0);
    }

    let _ = robot.insert_using_multibody_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
    );

    let mut sim_params = RbdSimParams::default();
    sim_params.num_solver_iterations = num_substeps;
    Some(BatchEnvironment {
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        sim_params,
        visuals: HashMap::new(),
    })
}

fn build_scene(path: &PathBuf, num_batches: usize, num_substeps: u32) -> Option<SimulationState> {
    let env = build_one_batch(path, num_substeps)?;
    let mut envs = Vec::with_capacity(num_batches.max(1));
    envs.push(env);
    for _ in 1..num_batches.max(1) {
        envs.push(build_one_batch(path, num_substeps)?);
    }
    Some(SimulationState::from_environments(envs))
}

struct Sample {
    label: &'static str,
    per_step_avg: Duration,
    per_step_p50: Duration,
    per_step_min: Duration,
    per_step_max: Duration,
}

impl Sample {
    fn fmt_us(d: Duration) -> String {
        format!("{:>10.2} µs", d.as_secs_f64() * 1.0e6)
    }

    fn print(&self) {
        println!(
            "  {:<10}  avg {}  p50 {}  min {}  max {}",
            self.label,
            Self::fmt_us(self.per_step_avg),
            Self::fmt_us(self.per_step_p50),
            Self::fmt_us(self.per_step_min),
            Self::fmt_us(self.per_step_max),
        );
    }
}

async fn bench_backend(
    label: &'static str,
    backend: &KhalGpuBackend,
    state: &SimulationState,
    n_warmup: usize,
    n_iters: usize,
    num_links: u32,
) -> Sample {
    let mut phys = GpuBackend::try_new(backend, state)
        .await
        .unwrap_or_else(|e| panic!("{label} backend init failed: {e}"));

    // Mirrors the urdf3 demo's `apply_random_ang_motors` tick: every 5 simulated
    // seconds, push a fresh random AngX motor target velocity to every link.
    use rand::RngExt;
    let mut rng = rand::rng();
    let dt = 1.0 / 60.0_f64;
    let mut next_change_at = 0.0_f64;
    let interval = 5.0;
    let n_batches = phys.num_batches() as u32;

    let mut do_tick = |phys: &mut GpuBackend, sim_time: f64| {
        if sim_time < next_change_at {
            return;
        }
        next_change_at = sim_time + interval;
        for batch in 0..n_batches {
            for link_id in 0..num_links {
                let target_vel: f32 = rng.random_range(-0.6f32..=0.6);
                phys.set_multibody_motor_velocity(batch, link_id, JointAxis::AngX, target_vel);
            }
        }
    };

    // Warmup.
    let mut sim_time = 0.0_f64;
    for _ in 0..n_warmup {
        do_tick(&mut phys, sim_time);
        let _ = phys.step(None).await;
        sim_time += dt;
    }

    let mut samples = Vec::with_capacity(n_iters);
    let mut last_stats = None;
    for i in 0..n_iters {
        do_tick(&mut phys, sim_time);
        let stats = phys.step(None).await;
        samples.push(stats.total_simulation_time_without_readback);
        if i == n_iters - 1 {
            last_stats = Some(stats);
        }
        sim_time += dt;
    }
    samples.sort();

    if let Some(stats) = last_stats {
        if !stats.gpu_pass_times.is_empty() {
            let mut passes = stats.gpu_pass_times.clone();
            passes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!(
                "    top passes ({} total, {:.3} ms):",
                passes.len(),
                stats.gpu_total_time
            );
            for (l, ms) in passes.iter().take(15) {
                println!("      {:>9.3} ms  {}", ms, l);
            }
        }
    }

    let total: Duration = samples.iter().sum();
    let per_step_avg = total / (n_iters as u32);
    let per_step_p50 = samples[n_iters / 2];
    let per_step_min = *samples.first().unwrap();
    let per_step_max = *samples.last().unwrap();

    Sample {
        label,
        per_step_avg,
        per_step_p50,
        per_step_min,
        per_step_max,
    }
}

async fn webgpu_backend() -> KhalGpuBackend {
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
    KhalGpuBackend::WebGpu(webgpu)
}

async fn run(
    path: PathBuf,
    num_batches: usize,
    n_warmup: usize,
    n_iters: usize,
    num_substeps: u32,
) {
    println!(
        "URDF3 multibody benchmark — {num_batches} batches, num_substeps={num_substeps}, \
         {n_warmup} warmup + {n_iters} timed steps  (URDF: {})",
        path.display()
    );

    let state = match build_scene(&path, num_batches, num_substeps) {
        Some(s) => s,
        None => {
            eprintln!("Aborting benchmark — URDF could not be loaded.");
            return;
        }
    };
    let webgpu = webgpu_backend().await;
    let num_links = state.environments[0]
        .multibody_joints
        .multibodies()
        .next()
        .map(|mb| mb.num_links())
        .unwrap_or(0) as u32;

    let s = bench_backend("WebGPU", &webgpu, &state, n_warmup, n_iters, num_links).await;
    s.print();
}

fn main() {
    // Args:
    //   bench_urdf3 [num_batches] [num_warmup] [num_iters] [num_substeps] [path]
    let args: Vec<String> = std::env::args().collect();

    let num_batches = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_warmup = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let num_substeps = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4u32);
    let path = args
        .get(5)
        .map(PathBuf::from)
        .unwrap_or_else(default_urdf_path);

    pollster::block_on(run(path, num_batches, n_warmup, n_iters, num_substeps));
}
