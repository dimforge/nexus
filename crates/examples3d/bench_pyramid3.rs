//! Headless rigid-body benchmark mirroring the `pyramid3` demo.
//!
//! Builds the same cuboid-pyramid-on-a-ground scene as
//! `crates/examples3d/pyramid3.rs`, steps it with no window, and reports
//! per-step wall timing plus the GPU pass breakdown. Used to validate RBD
//! pipeline optimizations.
//!
//! Usage:
//!
//! ```text
//! cargo run -p nexus_examples_3d --release --features metal \
//!     --bin bench_pyramid3 -- [stack_height] [num_warmup] [num_iters]
//! ```
//!
//! Defaults: stack height 20 (2870 cubes), 20 warmup steps, 200 timed steps.
//! `BACKEND=cpu|webgpu|metal` selects the backend (default: webgpu).

use std::time::{Duration, Instant};

use khal::backend::{Backend, GpuBackend, GpuTimestamps, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

fn create_pyramid(
    state: &mut NexusState,
    offset: Vector,
    stack_height: usize,
    half_extents: Vector,
) -> usize {
    let shift = half_extents * 2.5;
    let mut num_bodies = 0;
    for i in 0usize..stack_height {
        for j in i..stack_height {
            for k in i..stack_height {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                let x = (fi * shift.x / 2.0) + (fk - fi) * shift.x + offset.x
                    - stack_height as f32 * half_extents.x;
                let y = fi * shift.y + offset.y;
                let z = (fi * shift.z / 2.0) + (fj - fi) * shift.z + offset.z
                    - stack_height as f32 * half_extents.z;

                state.insert_rigid_body(
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .build(),
                    ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
                        .build(),
                );
                num_bodies += 1;
            }
        }
    }
    num_bodies
}

fn build_state(stack_height: usize) -> (NexusState, usize) {
    // Generous fixed sizing: the pyramid produces up to ~12 manifolds per cube.
    let num_cubes =
        (stack_height * (stack_height + 1) * (2 * stack_height + 1) / 6) as u32;
    let capacities = NexusCapacities::default()
        .rbd_bodies((num_cubes + 1).next_power_of_two().max(4096))
        .rbd_collisions((num_cubes * 16).max(4096));
    let mut state = NexusState::new(capacities);

    // Ground.
    let ground_size = 200.0;
    let ground_height = 0.1;
    state.insert_rigid_body(
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -ground_height, 0.0))
            .build(),
        ColliderBuilder::cuboid(ground_size, ground_height, ground_size).build(),
    );

    let cube_size = 1.0;
    let hext = Vec3::splat(cube_size);
    let num_bodies = create_pyramid(
        &mut state,
        Vec3::new(0.0, cube_size, 0.0),
        stack_height,
        hext,
    );
    (state, num_bodies)
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

/// `BACKEND=cpu|metal|webgpu` (default: webgpu).
async fn select_backend() -> GpuBackend {
    match std::env::var("BACKEND").as_deref() {
        Ok("cpu") => {
            println!("backend: CPU");
            GpuBackend::Cpu
        }
        #[cfg(feature = "metal")]
        Ok("metal") => {
            println!("backend: Metal");
            let metal = khal::backend::metal::Metal::new().expect("Metal init failed");
            GpuBackend::Metal(metal)
        }
        _ => {
            println!("backend: WebGPU");
            webgpu_backend().await
        }
    }
}

fn fmt_us(d: Duration) -> String {
    format!("{:>9.1} µs", d.as_secs_f64() * 1.0e6)
}

async fn run(stack_height: usize, n_warmup: usize, n_iters: usize) -> anyhow::Result<()> {
    let (mut state, num_bodies) = build_state(stack_height);
    println!(
        "Pyramid3 benchmark — stack height {stack_height} ({num_bodies} cubes + ground), \
         {n_warmup} warmup + {n_iters} timed steps"
    );

    let backend = select_backend().await;
    let mut pipeline = NexusPipeline::default();
    state.finalize(&backend).await?;

    // Warmup (includes shader compilation on the first step).
    for _ in 0..n_warmup {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;

    // Timed steps, each synchronized so wall time covers the full GPU frame.
    let mut samples = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        pipeline.simulate(&backend, &mut state, None).await?;
        backend.synchronize()?;
        samples.push(t0.elapsed());
    }
    samples.sort();

    let total: Duration = samples.iter().sum();
    let avg = total / n_iters as u32;
    let p50 = samples[n_iters / 2];
    let min = *samples.first().unwrap();
    let max = *samples.last().unwrap();
    println!(
        "  per-step  avg {}  p50 {}  min {}  max {}",
        fmt_us(avg),
        fmt_us(p50),
        fmt_us(min),
        fmt_us(max)
    );

    // One extra profiled step for the GPU pass breakdown.
    let mut timestamps = GpuTimestamps::new(&backend, 2048);
    pipeline
        .simulate(&backend, &mut state, Some(&mut timestamps))
        .await?;
    backend.synchronize()?;
    for _ in 0..100 {
        if let Some(results) = timestamps.try_take(&backend) {
            let mut aggregated: Vec<(String, f64)> = Vec::new();
            for r in &results {
                if let Some(existing) =
                    aggregated.iter_mut().find(|(label, _)| label == &r.label)
                {
                    existing.1 += r.duration_ms;
                } else {
                    aggregated.push((r.label.clone(), r.duration_ms));
                }
            }
            let gpu_total: f64 = aggregated.iter().map(|e| e.1).sum();
            aggregated
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("  gpu passes ({} labels, {:.3} ms total):", aggregated.len(), gpu_total);
            for (label, ms) in aggregated.iter().take(20) {
                println!("    {:>9.3} ms  {}", ms, label);
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    // Sanity check: read back poses; the pyramid must neither explode nor NaN.
    let rbd = state.rbd.as_ref().expect("rbd state missing");
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let mut nan = 0usize;
    let mut max_pos = 0.0f32;
    let mut min_y = f32::MAX;
    for p in poses.iter().take(num_bodies + 1) {
        let t = p.translation;
        if !t.is_finite() {
            nan += 1;
        } else {
            max_pos = max_pos.max(t.length());
            min_y = min_y.min(t.y);
        }
    }
    println!("  sanity: non-finite poses {nan}, max |pos| {max_pos:.2}, min y {min_y:.2}");
    if nan > 0 || max_pos > 1.0e3 {
        anyhow::bail!("simulation diverged (nan={nan}, max|pos|={max_pos:.2})");
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stack_height = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_warmup = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    pollster::block_on(async {
        if let Err(e) = run(stack_height, n_warmup, n_iters).await {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    });
}
