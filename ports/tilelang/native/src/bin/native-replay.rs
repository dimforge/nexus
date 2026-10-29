use khal::backend::{Backend, Metal};
use nexus_tilelang_native::{Manifest, NativeKernel};
use std::{collections::BTreeMap, fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("Expected fixture directory")?,
    );
    let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("bindings.json"))?)?;
    manifest.validate()?;
    let backend = Metal::new()?;
    let kernel = NativeKernel::new(
        &backend,
        &fs::read_to_string(root.join("kernel.metal"))?,
        manifest.clone(),
    )?;
    let mut storage = BTreeMap::new();
    for name in &manifest.storage_buffers {
        let bytes = fs::read(root.join(format!("{name}.bin")))?;
        if bytes.len() as u64 != manifest.buffer_bytes[name] {
            return Err("Invalid fixture buffer".into());
        }
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        storage.insert(
            name.clone(),
            backend.init_buffer(
                &words,
                khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC,
            )?,
        );
    }
    let buffers = storage
        .iter()
        .map(|(name, buffer)| (name.clone(), buffer.raw().as_ref()))
        .collect();
    // The replay driver validates indices, sizes and disjoint writes before launch.
    unsafe {
        kernel.dispatch(&backend, &buffers)?;
    }
    for (name, buffer) in storage {
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.raw().contents().cast::<u8>(), buffer.byte_len())
        };
        fs::write(root.join(format!("{name}.actual.bin")), bytes)?;
    }
    println!(
        "{}",
        serde_json::json!({"device": backend.device().name(), "runtime": "native Rust/khal Metal"})
    );
    Ok(())
}
