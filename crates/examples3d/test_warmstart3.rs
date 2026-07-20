//! Headless regression test for contact-warmstart energy injection.
//!
//! Builds a cuboid pyramid on a ground plane and lets it settle, then checks
//! that the highest cube stops rising. A correct solver reaches a resting height
//! and holds it to ~1e-3; a warmstart that applies impulses from the wrong
//! constraints pumps energy into the stack and the cubes creep back upwards
//! (and, in bigger scenes, get launched).
//!
//! The failure is nondeterministic — it depends on whatever a previous frame left
//! in the manifold slots above `contacts_len`, and it only fires while the contact
//! count is still churning — so this runs several independent trials and fails if
//! any one of them misbehaves. Roughly a quarter of the trials caught the original
//! defect (the worst observed launched a cube 10 m up), so the default of 12 trials
//! detects it ~97% of the time. It is a probabilistic guard, not a proof: the tight
//! check is that `count`/`sort` iterate the same manifolds.
//!
//! This guards the `body_constraint_ids` CSR that the gather (colorless)
//! warmstart gathers each body's constraints from: `gpu_solver_count_constraints`
//! sizes each body's range and `gpu_solver_sort_constraints` fills it, so the two
//! must iterate exactly the same manifolds. When they disagree, each body's range
//! starts inside an unwritten hole still holding a stale constraint id, and the
//! warmstart silently applies an unrelated contact's impulse.
//!
//! Usage:
//!
//! ```text
//! cargo run -p nexus_examples_3d --release --features metal --bin test_warmstart3
//! ```
//!
//! `BACKEND=webgpu` selects the WebGPU backend (default: Metal).

use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

/// Steps before the stack is assumed settled; only later steps are checked.
const SETTLE_STEPS: usize = 80;
/// How far the highest cube may rise after settling, in metres. A healthy stack
/// holds its resting height to well under a millimetre.
///
/// Only the *rise* is checked, never the absolute resting height: the pyramid has
/// 0.5 m gaps between cubes, so a cube can legitimately come to rest perched on a
/// neighbour's edge a few decimetres high, and GPU atomics make contact ordering
/// vary run to run. Height gained by an already-settled stack has no such
/// innocent explanation.
const MAX_RISE: f32 = 0.02;

fn build_state(stack_height: usize) -> (NexusState, usize) {
    let num_cubes = (stack_height * (stack_height + 1) * (2 * stack_height + 1) / 6) as u32;
    let capacities = NexusCapacities::default()
        .rbd_bodies((num_cubes + 1).next_power_of_two().max(4096))
        .rbd_collisions((num_cubes * 16).max(4096));
    let mut state = NexusState::new(capacities);

    let ground_height = 0.1;
    state.insert_rigid_body(
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -ground_height, 0.0))
            .build(),
        ColliderBuilder::cuboid(200.0, ground_height, 200.0).build(),
    );

    let hext = Vec3::splat(1.0);
    let shift = hext * 2.5;
    let offset = Vec3::new(0.0, 1.0, 0.0);
    let mut num_bodies = 0;
    for i in 0usize..stack_height {
        for j in i..stack_height {
            for k in i..stack_height {
                let (fi, fj, fk) = (i as f32, j as f32, k as f32);
                let x = (fi * shift.x / 2.0) + (fk - fi) * shift.x + offset.x
                    - stack_height as f32 * hext.x;
                let y = fi * shift.y + offset.y;
                let z = (fi * shift.z / 2.0) + (fj - fi) * shift.z + offset.z
                    - stack_height as f32 * hext.z;
                state.insert_rigid_body(
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .build(),
                    ColliderBuilder::cuboid(hext.x, hext.y, hext.z).build(),
                );
                num_bodies += 1;
            }
        }
    }
    (state, num_bodies)
}

async fn select_backend() -> GpuBackend {
    if std::env::var("BACKEND").as_deref() == Ok("webgpu") {
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
    } else {
        GpuBackend::Metal(khal::backend::metal::Metal::new().expect("Metal init failed"))
    }
}

/// Height of the highest cube, or an error if any pose went non-finite.
async fn highest_cube(
    backend: &GpuBackend,
    state: &NexusState,
    num_bodies: usize,
) -> anyhow::Result<f32> {
    backend.synchronize()?;
    let rbd = state.rbd.as_ref().expect("rbd state missing");
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let mut max_y = f32::MIN;
    for p in poses.iter().take(num_bodies + 1) {
        if !p.translation.is_finite() {
            anyhow::bail!("non-finite pose");
        }
        max_y = max_y.max(p.translation.y);
    }
    Ok(max_y)
}

/// Runs one settle-and-hold trial. Returns the reason it failed, if it did.
async fn trial(
    backend: &GpuBackend,
    stack_height: usize,
    n_steps: usize,
) -> anyhow::Result<Option<String>> {
    let (mut state, num_bodies) = build_state(stack_height);
    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await?;

    for _ in 0..SETTLE_STEPS {
        pipeline.simulate(backend, &mut state, None).await?;
    }
    let settled = highest_cube(backend, &state, num_bodies).await?;

    let mut peak = settled;
    for _ in SETTLE_STEPS..n_steps {
        pipeline.simulate(backend, &mut state, None).await?;
        peak = peak.max(highest_cube(backend, &state, num_bodies).await?);
    }

    let rise = peak - settled;
    println!("    settled {settled:.4}, peak {peak:.4} ({rise:+.4}, limit {MAX_RISE})");
    if rise > MAX_RISE {
        return Ok(Some(format!("gained {rise:.4} m of height after settling")));
    }
    Ok(None)
}

async fn run(stack_height: usize, n_steps: usize, n_trials: usize) -> anyhow::Result<()> {
    println!("pyramid stack {stack_height}, {n_steps} steps x {n_trials} trials");
    let backend = select_backend().await;

    let mut failures = Vec::new();
    for i in 0..n_trials {
        println!("  trial {}/{n_trials}:", i + 1);
        match trial(&backend, stack_height, n_steps).await {
            Ok(Some(why)) => failures.push(format!("trial {}: {why}", i + 1)),
            Ok(None) => {}
            Err(e) => failures.push(format!("trial {}: {e}", i + 1)),
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {f}");
        }
        anyhow::bail!(
            "{}/{n_trials} trials injected energy into the stack",
            failures.len()
        );
    }
    println!("  OK: {n_trials}/{n_trials} trials settled and held");
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stack_height = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let n_steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let n_trials = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);
    pollster::block_on(async {
        if let Err(e) = run(stack_height, n_steps, n_trials).await {
            eprintln!("FAIL: {e}");
            std::process::exit(1);
        }
    });
}
