//! Native resident LBVH composition shared by isolated and full-frame oracles.
use super::{BUFFERS, Data, radix};
use khal::backend::GpuBuffer;
use serde_json::{Value, json};
use std::{fs, path::Path};
type Workspace = std::collections::BTreeMap<String, Vec<u32>>;
pub(super) fn workspace(d: &Data) -> Workspace {
    let n = d.batch.colliders_len as usize;
    let nb = d.batch.num_batches as usize;
    let cap = d.batch.colliders_batch_capacity as usize;
    let guard = 0xdeadbeefu32;
    let mut extra = std::collections::BTreeMap::<String, Vec<u32>>::new();
    for key in [
        "Identity",
        "SortKeys",
        "PongKeys",
        "PongValues",
        "RadixAux",
        "RadixOutAux",
        "RadixBatchIds",
    ] {
        extra.insert(key.into(), vec![guard; cap * nb + 3]);
    }
    for b in 0..nb {
        for i in 0..n {
            extra.get_mut("Identity").unwrap()[b * cap + i] = i as u32;
        }
    }
    for (key, len) in [
        ("RadixCounts", 16 * cap.div_ceil(1024) * nb + 3),
        ("Reduced", 1024 * nb + 3),
        ("SortGrid", 6),
        ("ReduceGrid", 6),
    ] {
        extra.insert(key.into(), vec![guard; len]);
    }
    extra.insert("NumKeys".into(), vec![n as u32; nb]);
    for (key, shift) in [("Config".into(), 0)]
        .into_iter()
        .chain((0..8).map(|p| (format!("Config{p}"), 4 * p)))
    {
        extra.insert(key, vec![shift, cap as u32, 0]);
    }
    extra
}
pub(super) fn execute(d: &mut Data, extra: &mut Workspace, g: u32) {
    let cap = d.batch.colliders_batch_capacity as usize;
    let nb = d.batch.num_batches as usize;
    for op in ["lbvh_domain", "lbvh_domain_merge", "lbvh_morton"] {
        d.run(op, g);
    }
    extra.insert("Keys".into(), d.keys.clone());
    extra.insert("Sorted".into(), d.sorted.clone());
    for p in 0..8 {
        let mut mapping = std::collections::BTreeMap::<String, String>::new();
        for key in ["NumKeys", "Reduced", "SortGrid", "ReduceGrid", "Config"] {
            mapping.insert(key.into(), key.into());
        }
        mapping.insert("Counts".into(), "RadixCounts".into());
        mapping.insert("Aux".into(), "RadixAux".into());
        mapping.insert("OutAux".into(), "RadixOutAux".into());
        mapping.insert("BatchIds".into(), "RadixBatchIds".into());
        let (keys, values) = if p == 0 {
            ("Keys", "Identity")
        } else if p % 2 == 1 {
            ("PongKeys", "PongValues")
        } else {
            ("SortKeys", "Sorted")
        };
        let (out_keys, out_values) = if p % 2 == 1 {
            ("SortKeys", "Sorted")
        } else {
            ("PongKeys", "PongValues")
        };
        for (k, v) in [
            ("Keys", keys),
            ("Values", values),
            ("OutKeys", out_keys),
            ("OutValues", out_values),
        ] {
            mapping.insert(k.into(), v.into());
        }
        let mut arrays = radix::Arrays::new();
        for (k, v) in &mapping {
            arrays.insert(k.clone(), GpuBuffer::Cpu(extra[v].clone()));
        }
        arrays.insert(
            format!("Config{p}"),
            GpuBuffer::Cpu(extra[&format!("Config{p}")].clone()),
        );
        if p == 0 {
            radix::exec(&mut arrays, "radix_dispatch", cap as u32, nb as u32, None);
        }
        for stage in radix::STAGES {
            radix::exec(&mut arrays, stage, cap as u32, nb as u32, Some(p));
        }
        for (k, v) in &mapping {
            extra.insert(v.clone(), arrays[k].unwrap_slice().to_vec());
        }
    }
    d.keys = extra["SortKeys"].clone();
    d.sorted = extra.remove("Sorted").unwrap();
    d.run("lbvh_build", g);
    d.keys = extra.remove("Keys").unwrap();
    for op in [
        "lbvh_escapes",
        "lbvh_leaves",
        "lbvh_refit_chunks",
        "lbvh_refit_plan",
        "lbvh_refit_frontier",
        "lbvh_reset",
        "lbvh_pairs",
        "flat_list_dispatch",
    ] {
        d.run(op, g);
    }
}
pub(crate) fn fixture(root: &Path, before: &Data, g: u32, mode: &str, name: &str) -> Value {
    let cap = before.batch.colliders_batch_capacity as usize;
    let n = before.batch.colliders_len as usize;
    let nb = before.batch.num_batches as usize;
    let mut d = before.clone();
    d.frontier_len[..nb].fill(0);
    let mut extra = workspace(&d);
    let dir = root.join("lbvh_program").join(name);
    fs::create_dir_all(&dir).unwrap();
    d.write(&dir, "");
    for (k, v) in &extra {
        fs::write(dir.join(format!("{k}.bin")), bytemuck::cast_slice(v)).unwrap();
    }
    execute(&mut d, &mut extra, g);
    d.write(&dir, ".expected");
    for (k, v) in &extra {
        fs::write(
            dir.join(format!("{k}.expected.bin")),
            bytemuck::cast_slice(v),
        )
        .unwrap();
    }
    let sizes = fs::read_dir(&dir)
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|r| {
            let name = r.file_name().to_string_lossy().into_owned();
            let stem = name.trim_end_matches(".bin");
            if name.ends_with(".bin") && (BUFFERS.contains(&stem) || extra.contains_key(stem)) {
                Some((
                    name.trim_end_matches(".bin").to_owned(),
                    json!(r.metadata().unwrap().len() / 4),
                ))
            } else {
                None
            }
        })
        .collect::<serde_json::Map<_, _>>();
    let c = json!({"name":name,"kind":"lbvh_program","count":n*nb,"colliders":n,"batches":nb,"capacity":cap,"storage":d.poses.len(),"groups":g,"pair_capacity":before.batch.collision_pairs_capacity,"iterations":1,"mode":mode,"buffer_words":sizes});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&c).unwrap(),
    )
    .unwrap();
    c
}
