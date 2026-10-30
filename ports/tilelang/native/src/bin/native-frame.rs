//! Execute a trusted exported capacity ladder with resident native frames.
use khal::backend::Metal;
use nexus_tilelang_native::frame::{FrameManifest, FrameSession, FrameVariant};
use std::{collections::BTreeMap, fs, path::Path};
fn load(root: &Path) -> Result<FrameVariant, Box<dyn std::error::Error>> {
    let manifest: FrameManifest = serde_json::from_slice(&fs::read(root.join("frame.json"))?)?;
    manifest.validate()?;
    let sources = manifest
        .program
        .stages
        .iter()
        .map(|s| Ok((s.source.clone(), fs::read_to_string(root.join(&s.source))?)))
        .collect::<Result<BTreeMap<_, _>, std::io::Error>>()?;
    let initial = manifest
        .program
        .storage_buffers
        .iter()
        .map(|n| {
            let bytes = fs::read(root.join(format!("{n}.bin")))?;
            if !bytes.len().is_multiple_of(4) {
                return Err(std::io::Error::other("Misaligned upload"));
            }
            Ok((
                n.clone(),
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect(),
            ))
        })
        .collect::<Result<_, std::io::Error>>()?;
    Ok(FrameVariant {
        manifest,
        sources,
        initial,
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(args.get(1).ok_or("Expected frame bundle directory")?);
    let names: Vec<String> = serde_json::from_slice(&fs::read(root.join("variants.json"))?)?;
    if names.is_empty()
        || names.iter().any(|n| {
            n.is_empty()
                || n == "."
                || n == ".."
                || !n
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
        })
    {
        return Err("Invalid capacity variant directories".into());
    }
    let variants = names
        .iter()
        .map(|n| load(&root.join(n)))
        .collect::<Result<Vec<_>, _>>()?;
    let storage = variants
        .last()
        .unwrap()
        .manifest
        .program
        .storage_buffers
        .clone();
    let frames = variants[0]
        .manifest
        .max_steps
        .ok_or("Replay CLI requires a finite frame count")?;
    let backend = Metal::new()?;
    // This CLI is for trusted exporter/replay bundles. The exporter is responsible
    // for index bounds, race freedom and placing every gate before persistent writes.
    let mut runtime = unsafe { FrameSession::new(&backend, variants) }?;
    let mut stats = Vec::new();
    for _ in 0..frames {
        let result = runtime.step(&backend).map_err(|e| format!("{e:?}"))?;

        stats.push(serde_json::json!({"step":result.step,"retries":result.retries,"variant":result.variant,"counts":result.counts}));
    }
    for name in storage {
        let words = runtime.read(&backend, &name)?;
        fs::write(
            root.join(format!("{name}.actual.bin")),
            bytemuck::cast_slice(&words),
        )?;
    }
    fs::write(root.join("stats.json"), serde_json::to_vec_pretty(&stats)?)?;
    println!(
        "{}",
        serde_json::json!({"device":backend.device().name(),"runtime":"native Rust/khal Metal resident frames","steps":frames})
    );
    Ok(())
}
