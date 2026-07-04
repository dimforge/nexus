//! Headless benchmark for the 3D multibody pendulum scene.
//!
//! Runs the same scene as `multibody_pendulum3` (a 20-link revolute-joint
//! pendulum) on both the WebGPU and the nexus-CPU backends, with no graphics,
//! and reports per-step wall-clock times so we can track the perf gap and
//! verify that shader optimisations are landing.
//!
//! Usage (the `cpu` feature wires the nexus-CPU backend through nexus_viewer3d):
//!
//! ```text
//! cargo run -p nexus_examples_3d --release \
//!     --features cpu \
//!     --bin bench_multibody_pendulum3 -- [num_links] [num_warmup] [num_iters]
//! ```
//!
//! Defaults: 20 links, 10 warmup steps, 200 timed steps.

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
use std::collections::HashMap;

fn build_one_batch(num_links: usize) -> BatchEnvironment {
    build_one_batch_with_substeps(num_links, 4)
}

fn build_one_batch_with_substeps(num_links: usize, num_substeps: u32) -> BatchEnvironment {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    let rad = 0.4;
    let link_len = 2.0;

    let root_body = RigidBodyBuilder::fixed();
    let mut parent_handle = bodies.insert(root_body);
    let root_collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(root_collider, parent_handle, &mut bodies);

    for i in 0..num_links {
        let x = (i as f32 + 1.0) * link_len;
        let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, 0.0, 0.0));
        let handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(link_len * 0.5, rad, rad);
        colliders.insert_with_parent(collider, handle, &mut bodies);

        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            .build();
        multibody_joints.insert(parent_handle, handle, joint, true);

        parent_handle = handle;
    }

    let mut sim_params = RbdSimParams::default();
    sim_params.num_solver_iterations = num_substeps;
    BatchEnvironment {
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        sim_params,
        visuals: HashMap::new(),
    }
}

fn build_scene(num_links: usize, num_batches: usize) -> SimulationState {
    build_scene_with_substeps(num_links, num_batches, 4)
}

fn build_scene_with_substeps(
    num_links: usize,
    num_batches: usize,
    num_substeps: u32,
) -> SimulationState {
    let environments = (0..num_batches.max(1))
        .map(|_| build_one_batch_with_substeps(num_links, num_substeps))
        .collect();
    SimulationState::from_environments(environments)
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
) -> Sample {
    bench_backend_inner(label, backend, state, n_warmup, n_iters, true).await
}

async fn bench_backend_inner(
    label: &'static str,
    backend: &KhalGpuBackend,
    state: &SimulationState,
    n_warmup: usize,
    n_iters: usize,
    print_passes: bool,
) -> Sample {
    let mut phys = GpuBackend::try_new(backend, state)
        .await
        .unwrap_or_else(|e| panic!("{label} backend init failed: {e}"));

    // Warmup — first steps include shader compilation and pipeline cache fill,
    // we don't want those skewing the measurement.
    for _ in 0..n_warmup {
        let _ = phys.step(None).await;
    }

    // Measure `total_simulation_time_without_readback` instead of wall-clocking
    // the whole `phys.step()` — the post-step `auto_resize_buffers` + pose
    // readback + timestamp readback each add a CPU-GPU round trip whose latency
    // is dominated by driver/OS scheduler jitter, not by the pipeline itself.
    let mut samples = Vec::with_capacity(n_iters);
    let mut last_stats = None;
    for i in 0..n_iters {
        let stats = phys.step(None).await;
        samples.push(stats.total_simulation_time_without_readback);
        if i == n_iters - 1 {
            last_stats = Some(stats);
        }
    }
    samples.sort();

    // Print the top per-pass GPU timings from the last iteration (so the
    // user can see which kernel still dominates after warmup). The CPU
    // backend reports an empty list — guard with `is_empty()`.
    if print_passes {
        if let Some(stats) = last_stats {
            if !stats.gpu_pass_times.is_empty() {
                let mut passes = stats.gpu_pass_times.clone();
                passes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                println!(
                    "    top passes ({} total, {:.3} ms):",
                    passes.len(),
                    stats.gpu_total_time
                );
                for (l, ms) in passes.iter().take(8) {
                    println!("      {:>9.3} ms  {}", ms, l);
                }
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
    // The pendulum scene's narrow-phase shader needs more storage buffers
    // and a larger workgroup-storage budget than wgpu's defaults — mirror
    // the limits the viewer requests.
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
    num_links: usize,
    num_batches: usize,
    n_warmup: usize,
    n_iters: usize,
    num_substeps: u32,
) {
    println!(
        "Multibody pendulum benchmark — {num_links} links × {num_batches} batches, \
         num_substeps={num_substeps}, {n_warmup} warmup + {n_iters} timed steps"
    );

    let state = build_scene_with_substeps(num_links, num_batches, num_substeps);
    let webgpu = webgpu_backend().await;

    let webgpu_sample = {
        let s = bench_backend("WebGPU", &webgpu, &state, n_warmup, n_iters).await;
        s.print();
        s
    };

    // Nexus-CPU backend (same pipeline, executed on CPU). Only available when
    // built with `--features cpu`.
    #[cfg(feature = "cpu")]
    let cpu_sample = {
        let backend = KhalGpuBackend::Cpu;
        let s = bench_backend("Nexus-CPU", &backend, &state, n_warmup, n_iters).await;
        s.print();
        s
    };

    #[cfg(feature = "cpu")]
    {
        let ratio =
            webgpu_sample.per_step_avg.as_secs_f64() / cpu_sample.per_step_avg.as_secs_f64();
        println!(
            "  → WebGPU/CPU ratio (avg): {:.2}× {}",
            ratio,
            if ratio > 1.0 {
                "(GPU slower)"
            } else {
                "(GPU faster)"
            }
        );
    }
    #[cfg(not(feature = "cpu"))]
    {
        let _ = webgpu_sample;
        println!("  (rebuild with --features cpu to also benchmark the nexus CPU backend)");
    }
}

/// Sweep mode: hold `num_links` fixed and vary `num_batches` over a power-of-two
/// range, reporting the crossover where GPU catches up to CPU.
#[cfg(feature = "cpu")]
async fn sweep(num_links: usize, max_batches: usize, n_warmup: usize, n_iters: usize) {
    println!(
        "Multibody pendulum sweep — {num_links} links, batches 1..={max_batches} (×2 each step), \
         {n_warmup} warmup + {n_iters} timed steps each"
    );
    println!(
        "{:>7}  {:>14}  {:>14}  {:>10}  {}",
        "batches", "WebGPU avg", "CPU avg", "ratio", "verdict"
    );

    let webgpu = webgpu_backend().await;
    let cpu = KhalGpuBackend::Cpu;

    let mut bs = 1;
    while bs <= max_batches {
        let state = build_scene(num_links, bs);
        let g = bench_backend_inner("WebGPU", &webgpu, &state, n_warmup, n_iters, false).await;
        let c = bench_backend_inner("Nexus-CPU", &cpu, &state, n_warmup, n_iters, false).await;
        let ratio = g.per_step_avg.as_secs_f64() / c.per_step_avg.as_secs_f64();
        let verdict = if ratio < 1.0 {
            "GPU faster"
        } else if ratio < 4.0 {
            "GPU within 4×"
        } else {
            "GPU slower"
        };
        println!(
            "{:>7}  {:>14}  {:>14}  {:>9.2}×  {}",
            bs,
            Sample::fmt_us(g.per_step_avg),
            Sample::fmt_us(c.per_step_avg),
            ratio,
            verdict,
        );
        bs *= 2;
    }
}

fn main() {
    // Args:
    //   bench_multibody_pendulum3 [num_links] [num_batches] [num_warmup] [num_iters]
    //   bench_multibody_pendulum3 sweep [num_links] [max_batches] [num_warmup] [num_iters]
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some("sweep") {
        #[cfg(feature = "cpu")]
        {
            let num_links = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            let max_batches = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
            let n_warmup = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
            let n_iters = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30);
            pollster::block_on(sweep(num_links, max_batches, n_warmup, n_iters));
        }
        #[cfg(not(feature = "cpu"))]
        eprintln!("sweep mode requires the `cpu` feature");
        return;
    }

    let num_links = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let num_batches = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_warmup = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_iters = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let num_substeps = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4u32);

    pollster::block_on(run(num_links, num_batches, n_warmup, n_iters, num_substeps));
}
