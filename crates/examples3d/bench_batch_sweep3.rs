//! Headless benchmark for BATCHED simulation with many small environments.
//!
//! Measures how per-step time scales with the number of environments when each
//! environment is tiny (a few boxes, or a short multibody chain). This is the
//! RL-style regime: thousands of independent envs, each with a single robot.
//!
//! Usage:
//!
//! ```text
//! cargo run -p nexus_examples_3d --release --features metal \
//!     --bin bench_batch_sweep3 -- [scene] [size] [max_batches] [num_warmup] [num_iters]
//! ```
//!
//! - `scene`: `boxes` (ground + size×size box pile per env, contact path),
//!   `chain` (size-link revolute multibody pendulum per env, robot path), or
//!   `chain-nsc` (same with multibody self-contacts disabled, as MJCF robots).
//! - Defaults: boxes, size 3, max_batches 1024, 20 warmup, 100 timed steps.
//! - `BACKEND=cpu|webgpu|metal` selects the backend (default: metal).
//!
//! For each batch count (1, 4, 16, ..., max_batches) it rebuilds the scene,
//! steps it, and prints per-step wall time + per-env throughput. At the last
//! sweep point it prints the GPU pass breakdown.

use std::time::{Duration, Instant};

use khal::backend::{Backend, GpuBackend, GpuTimestamps, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

fn build_boxes_env(state: &mut NexusState, env: usize, size: usize) {
    // Small local ground so envs don't interact.
    state.insert_rigid_body_in(
        env,
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -0.1, 0.0))
            .build(),
        ColliderBuilder::cuboid(10.0, 0.1, 10.0).build(),
    );

    let rad = 0.5;
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                state.insert_rigid_body_in(
                    env,
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(
                            i as f32 * rad * 2.2 - size as f32 * rad,
                            rad * 1.1 + j as f32 * rad * 2.2,
                            k as f32 * rad * 2.2 - size as f32 * rad,
                        ))
                        .build(),
                    ColliderBuilder::cuboid(rad, rad, rad).build(),
                );
            }
        }
    }
}

fn build_chain_env(state: &mut NexusState, env: usize, num_links: usize, self_contacts: bool) {
    let rad = 0.4;
    let link_len = 2.0;

    let mut parent = state.insert_rigid_body_in(
        env,
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(rad, rad, rad).build(),
    );

    let mut first_joint = None;
    for i in 0..num_links {
        let x = (i as f32 + 1.0) * link_len;
        let handle = state.insert_rigid_body_in(
            env,
            RigidBodyBuilder::dynamic()
                .translation(Vec3::new(x, 0.0, 0.0))
                .build(),
            // Slightly longer than the link spacing so adjacent links overlap
            // at the joints (like real robot collision meshes) — this makes
            // the self-contact broad-phase pairs real rather than borderline.
            ColliderBuilder::cuboid(link_len * 0.55, rad, rad).build(),
        );

        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            .build();
        let jh = state
            .insert_multibody_joint_in(env, parent, handle, joint)
            .expect("invalid multibody chain");
        first_joint.get_or_insert(jh);
        parent = handle;
    }

    // `chain-nsc`: robot-style setup (MJCF `DISABLE_SELF_CONTACTS`) — links of
    // the same multibody never collide with each other.
    if !self_contacts {
        let jh = first_joint.expect("chain has at least one joint");
        let world = state.rbd_world_mut(env);
        let (mb, _) = world
            .multibody_joints
            .get_mut(jh)
            .expect("chain multibody missing");
        mb.set_self_contacts_enabled(false);
    }
}

fn build_state(scene: &str, size: usize, num_batches: usize) -> NexusState {
    // `rbd_collisions` is a PER-BATCH capacity; keep it small for tiny envs
    // (the Grow resize policy will bump it if a scene needs more).
    let capacities = NexusCapacities::default().rbd_collisions(256);
    let mut state = NexusState::new(capacities);

    for b in 0..num_batches {
        let env = if b == 0 { 0 } else { state.add_environment() };
        match scene {
            "chain" => build_chain_env(&mut state, env, size, true),
            "chain-nsc" => build_chain_env(&mut state, env, size, false),
            _ => build_boxes_env(&mut state, env, size),
        }
    }
    state
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

fn fmt_us(d: Duration) -> String {
    format!("{:>9.1} µs", d.as_secs_f64() * 1.0e6)
}

async fn bench_point(
    backend: &GpuBackend,
    scene: &str,
    size: usize,
    num_batches: usize,
    n_warmup: usize,
    n_iters: usize,
    print_passes: bool,
) -> anyhow::Result<(Duration, Duration)> {
    let mut state = build_state(scene, size, num_batches);
    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await?;

    // `NEXUS_EXPLICIT_CORIOLIS=1`: MuJoCo/Genesis-style explicit coriolis —
    // the mass matrix / LU / gravity solve runs once per step instead of
    // once per substep (different integration semantics, so checksums are
    // not comparable with the implicit default).
    if std::env::var("NEXUS_EXPLICIT_CORIOLIS").is_ok()
        && let Some(rbd) = state.rbd.as_mut()
    {
        rbd.multibodies_mut().set_implicit_coriolis(false);
    }

    for _ in 0..n_warmup {
        pipeline.simulate(backend, &mut state, None).await?;
    }
    backend.synchronize()?;

    let mut samples = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        pipeline.simulate(backend, &mut state, None).await?;
        backend.synchronize()?;
        samples.push(t0.elapsed());
    }
    samples.sort();
    let total: Duration = samples.iter().sum();
    let avg = total / n_iters as u32;
    let p50 = samples[n_iters / 2];

    if print_passes {
        let mut timestamps = GpuTimestamps::new(backend, 2048);
        pipeline
            .simulate(backend, &mut state, Some(&mut timestamps))
            .await?;
        backend.synchronize()?;
        for _ in 0..100 {
            if let Some(results) = timestamps.try_take(backend) {
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
                aggregated.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                println!(
                    "  gpu passes at {num_batches} batches ({} labels, {:.3} ms total):",
                    aggregated.len(),
                    gpu_total
                );
                for (label, ms, count) in aggregated.iter().take(25) {
                    println!("    {:>9.3} ms  ×{:<4} {}", ms, count, label);
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // Sanity + regression signal: poses must stay finite and bounded, and the
    // checksum makes behavior drift visible across optimization patches (the
    // step count is deterministic, so unchanged math ⇒ unchanged checksum).
    let rbd = state.rbd.as_ref().expect("rbd state missing");
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let mut nan = 0usize;
    let mut max_pos = 0.0f32;
    let mut sum = 0.0f64;
    for p in &poses {
        let t = p.translation;
        if !t.is_finite() {
            nan += 1;
        } else {
            max_pos = max_pos.max(t.length());
            sum += (t.x + t.y + t.z) as f64;
        }
    }
    println!(
        "  sanity at {num_batches} batches: non-finite {nan}, max |pos| {max_pos:.3}, \
         checksum {sum:.6}, max pairs/env {}",
        state.counts().collision_pairs
    );
    if nan > 0 || max_pos > 1.0e4 {
        anyhow::bail!("simulation diverged (nan={nan}, max|pos|={max_pos:.2})");
    }

    Ok((avg, p50))
}

async fn run(
    scene: String,
    size: usize,
    max_batches: usize,
    n_warmup: usize,
    n_iters: usize,
) -> anyhow::Result<()> {
    println!(
        "Batch sweep — scene `{scene}` size {size}, batches 1..={max_batches} (×4), \
         {n_warmup} warmup + {n_iters} timed steps each"
    );
    let backend = select_backend().await;

    println!(
        "{:>8}  {:>13}  {:>13}  {:>16}  {:>13}",
        "batches", "avg/step", "p50/step", "env·steps/s", "µs/env/step"
    );

    let mut bs = 1usize;
    while bs <= max_batches {
        let last = bs * 4 > max_batches;
        let (avg, p50) =
            bench_point(&backend, &scene, size, bs, n_warmup, n_iters, last).await?;
        let env_steps_per_s = bs as f64 / avg.as_secs_f64();
        let us_per_env_step = avg.as_secs_f64() * 1.0e6 / bs as f64;
        println!(
            "{:>8}  {:>13}  {:>13}  {:>16.0}  {:>13.2}",
            bs,
            fmt_us(avg),
            fmt_us(p50),
            env_steps_per_s,
            us_per_env_step
        );
        bs *= 4;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scene = args.get(1).cloned().unwrap_or_else(|| "boxes".to_string());
    let size = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_batches = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_warmup = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_iters = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100);
    pollster::block_on(async {
        if let Err(e) = run(scene, size, max_batches, n_warmup, n_iters).await {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    });
}
