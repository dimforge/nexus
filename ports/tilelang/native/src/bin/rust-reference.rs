//! Execute the existing production Rust GPU kernel on the exported input buffers.
use khal::backend::{Backend, Encoder, GpuBackend, Metal};
use nexus_rbd3d::{
    SPIRV_DIR,
    shaders::{
        Pose,
        dynamics::{
            RbdSimParams, Velocity,
            coulomb_tiles::{CoulombTile, GpuSolveConstraints},
        },
        utils::BatchIndices,
    },
};
use std::{
    fs,
    path::{Path, PathBuf},
};
use vortx::tensor::Tensor;
fn read<T: bytemuck::Pod>(path: &Path) -> Vec<T> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len() % size_of::<T>(), 0);
    bytes
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    futures::executor::block_on(run())
}
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args().nth(1).ok_or("Expected fixture root")?);
    let suite: serde_json::Value = serde_json::from_slice(&fs::read(root.join("suite.json"))?)?;
    let backend = GpuBackend::Metal(Metal::new()?);
    let kernel = GpuSolveConstraints::from_dir(&backend, &SPIRV_DIR)?;
    let storage = khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC;
    let uniform = khal::BufferUsages::UNIFORM;
    for case in suite["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bias = case["use_bias"].as_bool().unwrap();
        let friction = case["solve_friction"].as_bool().unwrap();
        if !bias && !friction {
            println!("{name}: unavailable (production kernel has no neither-row mode)");
            continue;
        }
        let dir = root.join(name);
        let n = case["count"].as_u64().unwrap() as u32;
        let mut tiles = Tensor::vector(
            &backend,
            read::<CoulombTile>(&dir.join("Tiles.bin")),
            storage,
        )?;
        let mut velocities = Tensor::vector(
            &backend,
            read::<Velocity>(&dir.join("Velocities.bin")),
            storage,
        )?;
        let poses = Tensor::vector(&backend, read::<Pose>(&dir.join("Poses.bin")), storage)?;
        let params = Tensor::scalar(
            &backend,
            read::<RbdSimParams>(&dir.join("RbdParams.bin"))[0],
            uniform,
        )?;
        let buckets = Tensor::vector(&backend, vec![0, n], storage)?;
        let color = Tensor::scalar(&backend, 1u32, uniform)?;
        let ids = Tensor::scalar(
            &backend,
            BatchIndices {
                num_batches: 1,
                solver_color_buckets_stride: 2,
                ..Default::default()
            },
            uniform,
        )?;
        let mode = Tensor::scalar(
            &backend,
            if bias {
                if friction { 2u32 } else { 1 }
            } else {
                0
            },
            uniform,
        )?;
        for _ in 0..case["iterations"].as_u64().unwrap() {
            let mut encoder = backend.begin_encoding();
            {
                let mut pass = encoder.begin_pass("rust-reference", None);
                kernel.call(
                    &mut pass,
                    n.max(1),
                    &mut tiles,
                    &mut velocities,
                    &buckets,
                    &poses,
                    &color,
                    &ids,
                    &mode,
                    &params,
                )?;
            }
            backend.submit(encoder)?;
        }
        backend.synchronize()?;
        for (name, words) in [
            (
                "Tiles",
                bytemuck::cast_slice::<CoulombTile, u8>(
                    &backend.slow_read_vec::<CoulombTile>(tiles.buffer()).await?,
                )
                .to_vec(),
            ),
            (
                "Velocities",
                bytemuck::cast_slice::<Velocity, u8>(
                    &backend
                        .slow_read_vec::<Velocity>(velocities.buffer())
                        .await?,
                )
                .to_vec(),
            ),
        ] {
            fs::write(dir.join(format!("{name}.rust-metal.bin")), words)?;
        }
        println!("{name}: executed");
    }
    Ok(())
}
