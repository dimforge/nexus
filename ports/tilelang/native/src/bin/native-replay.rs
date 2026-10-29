use khal::backend::{Backend, Metal};
use nexus_tilelang_native::{Manifest, NativeKernel, NativeProgram, ProgramManifest};
use std::{collections::BTreeMap, fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("Expected fixture directory")?,
    );
    let value: serde_json::Value = serde_json::from_slice(&fs::read(root.join("bindings.json"))?)?;
    let backend = Metal::new()?;
    let (kernel, program, storage_buffers, buffer_bytes) = if value["schema"] == 2 {
        let manifest: ProgramManifest = serde_json::from_value(value)?;
        manifest.validate()?;
        let sources = manifest
            .stages
            .iter()
            .map(|stage| {
                Ok((
                    stage.source.clone(),
                    fs::read_to_string(root.join(&stage.source))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, std::io::Error>>()?;
        let program = NativeProgram::new(&backend, manifest.clone(), &sources)?;
        (
            None,
            Some(program),
            manifest.storage_buffers,
            manifest.buffer_bytes,
        )
    } else {
        let manifest: Manifest = serde_json::from_value(value)?;
        let kernel = NativeKernel::new(
            &backend,
            &fs::read_to_string(root.join("kernel.metal"))?,
            manifest.clone(),
        )?;
        (
            Some(kernel),
            None,
            manifest.storage_buffers,
            manifest.buffer_bytes,
        )
    };
    let mut storage = BTreeMap::new();
    for name in &storage_buffers {
        let bytes = fs::read(root.join(format!("{name}.bin")))?;
        if bytes.len() as u64 != buffer_bytes[name] {
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
        if let Some(kernel) = kernel {
            kernel.dispatch(&backend, &buffers)?;
        }
        if let Some(program) = program {
            program.dispatch(&backend, &buffers)?;
        }
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
