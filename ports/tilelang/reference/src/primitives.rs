//! Live reference for irregular integer GPU operations, without GPU emulation.
use nexus_rbd_shaders3d::{
    dynamics::gpu_stabilize_body_constraint_ids, glamx::UVec3, utils::BatchIndices,
};
use serde_json::json;
use std::{fs, path::Path};
fn write(root: &Path, name: &str, words: &[u32]) {
    let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(root.join(name), bytes).unwrap();
}
fn main() {
    let output = std::env::args().nth(1).expect("Expected fixture directory");
    let output = Path::new(&output);
    fs::create_dir_all(output).unwrap();
    fs::write(
        output.join("layout.json"),
        "{\"version\":1,\"word_bytes\":4}\n",
    )
    .unwrap();
    let root = output.join("stabilize");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for (name, lengths, duplicates) in [
        ("empty", vec![], false),
        ("empty-rows", vec![0, 0, 0], false),
        ("single", vec![1], false),
        ("boundaries", vec![31, 32, 33, 63, 64, 65, 127, 129], false),
        ("duplicates", vec![65, 129, 33], true),
        ("skewed", vec![0, 1, 513, 0, 2], false),
    ] {
        let mut sum = 0u32;
        let ends: Vec<u32> = lengths
            .iter()
            .map(|n| {
                sum += n;
                sum
            })
            .collect();
        let mut state = 42u32;
        let mut ids: Vec<u32> = (0..sum)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if duplicates { state % 5 } else { state }
            })
            .collect();
        ids.extend([u32::MAX, 0xabcdef01, 17]);
        let mut sorted = vec![0xa5a5a5a5; ids.len()];
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        write(
            &dir,
            "Counts.bin",
            if ends.is_empty() { &[0] } else { &ends },
        );
        write(&dir, "Ids.bin", &ids);
        write(&dir, "Sorted.bin", &sorted);
        let batch_ids = BatchIndices {
            colliders_len: lengths.len() as u32,
            num_batches: 1,
            ..Default::default()
        };
        for body in 0..lengths.len() {
            for lane in 0..64 {
                gpu_stabilize_body_constraint_ids(
                    UVec3::new(body as u32, 0, 0),
                    UVec3::new(lengths.len().max(1) as u32, 1, 1),
                    lane,
                    &ends,
                    &ids,
                    &mut sorted,
                    &batch_ids,
                );
            }
        }
        write(&dir, "Sorted.expected.bin", &sorted);
        cases.push(json!({"name":name, "kind":"stabilize", "rows":lengths.len(), "count":sum, "storage":ids.len(), "iterations":1}));
    }
    fs::write(
        root.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}
