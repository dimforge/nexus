//! Cooperative native radix shader oracle, preserving every intermediate buffer.
#[path = "dispatch.rs"]
mod dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::utils::radix_sort::{
    GpuInitSortBatched, GpuSortCount, GpuSortReduce, GpuSortScan, GpuSortScanAdd, GpuSortScatter,
    SortUniforms, gpu_init_sort_dispatch,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
pub(crate) type Arrays = BTreeMap<String, GpuBuffer<u32>>;
fn write(root: &Path, name: &str, words: &[u32]) {
    fs::write(root.join(name), bytemuck::cast_slice(words)).unwrap();
}
pub(crate) fn exec(a: &mut Arrays, stage: &str, stride: u32, batches: u32, pass: Option<u32>) {
    let cfg = if stage == "radix_init_batched" {
        "InitConfig".to_string()
    } else {
        pass.map(|p| format!("Config{p}"))
            .unwrap_or("Config".to_string())
    };
    let v = a[&cfg].unwrap_slice();
    let config = GpuBuffer::Cpu(vec![SortUniforms {
        shift: v[0],
        max_keys_per_batch: v[1],
        has_aux: v[2],
    }]);
    let num = GpuBuffer::Cpu(a["NumKeys"].unwrap_slice().to_vec());
    let keys = GpuBuffer::Cpu(a["Keys"].unwrap_slice().to_vec());
    let vals = GpuBuffer::Cpu(a["Values"].unwrap_slice().to_vec());
    let aux = GpuBuffer::Cpu(a["Aux"].unwrap_slice().to_vec());
    let counts = GpuBuffer::Cpu(a["Counts"].unwrap_slice().to_vec());
    let reduced = GpuBuffer::Cpu(a["Reduced"].unwrap_slice().to_vec());
    let grid = dispatch::workgroups([stride.div_ceil(1024).max(1) + 1, batches, 1]);
    let reduce = dispatch::workgroups([16 * stride.div_ceil(1024).div_ceil(1024) + 1, batches, 1]);
    let mut gp = GpuPass::Cpu(None);
    match stage {
        "radix_dispatch" => {
            let mut sort = [0; 3];
            let mut red = [0; 3];
            gpu_init_sort_dispatch(num.unwrap_slice(), &mut sort, &mut red);
            a.get_mut("SortGrid").unwrap().unwrap_slice_mut()[..3].copy_from_slice(&sort);
            a.get_mut("ReduceGrid").unwrap().unwrap_slice_mut()[..3].copy_from_slice(&red);
        }
        "radix_init_batched" => {
            let mut out = a.remove("OutKeys").unwrap();
            let mut ov = a.remove("OutValues").unwrap();
            let mut ids = a.remove("BatchIds").unwrap();
            GpuInitSortBatched::from_bytes(&GpuBackend::Cpu, &[], GpuInitSortBatched::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    dispatch::workgroups([(stride * batches).div_ceil(256).max(1) + 1, 1, 1]),
                    &config,
                    &num,
                    &keys,
                    &vals,
                    &mut out,
                    &mut ov,
                    &mut ids,
                )
                .unwrap();
            a.insert("OutKeys".into(), out);
            a.insert("OutValues".into(), ov);
            a.insert("BatchIds".into(), ids);
        }
        "radix_count" => {
            GpuSortCount::from_bytes(&GpuBackend::Cpu, &[], GpuSortCount::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    grid,
                    &config,
                    &num,
                    &keys,
                    a.get_mut("Counts").unwrap(),
                )
                .unwrap();
        }
        "radix_reduce" => {
            GpuSortReduce::from_bytes(&GpuBackend::Cpu, &[], GpuSortReduce::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    reduce,
                    &config,
                    &num,
                    &counts,
                    a.get_mut("Reduced").unwrap(),
                )
                .unwrap();
        }
        "radix_scan" => {
            GpuSortScan::from_bytes(&GpuBackend::Cpu, &[], GpuSortScan::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    dispatch::workgroups([1, batches, 1]),
                    &num,
                    a.get_mut("Reduced").unwrap(),
                )
                .unwrap();
        }
        "radix_scan_add" => {
            GpuSortScanAdd::from_bytes(&GpuBackend::Cpu, &[], GpuSortScanAdd::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    reduce,
                    &config,
                    &num,
                    &reduced,
                    a.get_mut("Counts").unwrap(),
                )
                .unwrap();
        }
        "radix_scatter" => {
            let mut out = a.remove("OutKeys").unwrap();
            let mut ov = a.remove("OutValues").unwrap();
            let mut oa = a.remove("OutAux").unwrap();
            GpuSortScatter::from_bytes(&GpuBackend::Cpu, &[], GpuSortScatter::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp, grid, &config, &num, &keys, &vals, &counts, &mut out, &mut ov, &aux,
                    &mut oa,
                )
                .unwrap();
            a.insert("OutKeys".into(), out);
            a.insert("OutValues".into(), ov);
            a.insert("OutAux".into(), oa);
        }
        _ => panic!("unknown stage {stage}"),
    }
}
fn swap_io(a: &mut Arrays) {
    for (x, y) in [
        ("Keys", "OutKeys"),
        ("Values", "OutValues"),
        ("Aux", "OutAux"),
    ] {
        let left = a.remove(x).unwrap();
        let right = a.remove(y).unwrap();
        a.insert(x.into(), right);
        a.insert(y.into(), left);
    }
}
pub(crate) const STAGES: [&str; 5] = [
    "radix_count",
    "radix_reduce",
    "radix_scan",
    "radix_scan_add",
    "radix_scatter",
];
fn fixture(
    root: &Path,
    stride: u32,
    batches: u32,
    shift: u32,
    has_aux: u32,
    pattern: &str,
    op: &str,
) -> Value {
    let name = format!("n{stride}-b{batches}-s{shift}-a{has_aux}-{pattern}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let mut a = Arrays::new();
    let guard = 0xdeadbeefu32;
    let synthetic = pattern == "histograms";
    let n: Vec<u32> = (0..batches)
        .map(|b| {
            if b == 0 {
                stride
            } else if b % 2 == 1 {
                stride.saturating_sub(17)
            } else {
                0
            }
        })
        .collect();
    a.insert("NumKeys".into(), GpuBuffer::Cpu(n.clone()));
    for (key, s) in [("Config".into(), shift), ("InitConfig".into(), batches)]
        .into_iter()
        .chain((0..8).map(|p| (format!("Config{p}"), p * 4)))
    {
        a.insert(key, GpuBuffer::Cpu(vec![s, stride, has_aux]));
    }
    for key in [
        "Keys",
        "Values",
        "Aux",
        "OutKeys",
        "OutValues",
        "OutAux",
        "BatchIds",
    ] {
        a.insert(
            key.into(),
            GpuBuffer::Cpu(vec![
                guard;
                if synthetic {
                    3
                } else {
                    (stride * batches) as usize + 3
                }
            ]),
        );
    }
    for (key, len) in [
        ("Counts", 16 * stride.div_ceil(1024) * batches + 3),
        ("Reduced", 1024 * batches + 3),
        ("SortGrid", 6),
        ("ReduceGrid", 6),
    ] {
        a.insert(key.into(), GpuBuffer::Cpu(vec![guard; len as usize]));
    }
    if !synthetic {
        for b in 0..batches {
            for i in 0..n[b as usize] {
                let j = (b * stride + i) as usize;
                let key = match pattern {
                    "equal" => u32::MAX,
                    "reverse" => stride - i,
                    "digits" => ((i * 7 + b) % 16) << shift,
                    _ => i.wrapping_mul(2654435761).wrapping_add(b * 71) ^ (i >> 3),
                };
                a.get_mut("Keys").unwrap().unwrap_slice_mut()[j] =
                    if pattern == "mixed" && i % 7 == 0 {
                        u32::MAX
                    } else {
                        key
                    };
                a.get_mut("Values").unwrap().unwrap_slice_mut()[j] = i;
                a.get_mut("Aux").unwrap().unwrap_slice_mut()[j] = i.wrapping_mul(11) ^ b;
            }
        }
    }
    if synthetic {
        for (i, v) in a
            .get_mut("Counts")
            .unwrap()
            .unwrap_slice_mut()
            .iter_mut()
            .enumerate()
        {
            *v = ((i * 13 + 17) % 211) as u32;
        }
    }
    if let Some(at) = STAGES.iter().position(|&s| s == op) {
        for s in &STAGES[..at] {
            if !(synthetic && *s == "radix_count") {
                exec(&mut a, s, stride, batches, None);
            }
        }
    }
    for (k, v) in &a {
        write(&dir, &format!("{k}.bin"), v.unwrap_slice());
    }
    if op == "radix_sort" {
        for p in 0..8 {
            if p % 2 == 1 {
                swap_io(&mut a);
            }
            for s in STAGES {
                exec(&mut a, s, stride, batches, Some(p));
            }
            if p % 2 == 1 {
                swap_io(&mut a);
            }
        }
    } else if op == "radix_pass" {
        for s in STAGES {
            exec(&mut a, s, stride, batches, None);
        }
    } else {
        exec(&mut a, op, stride, batches, None);
    }
    for (k, v) in &a {
        write(&dir, &format!("{k}.expected.bin"), v.unwrap_slice());
    }
    let words: BTreeMap<_, _> = a
        .iter()
        .map(|(k, v)| (k.clone(), v.unwrap_slice().len()))
        .collect();
    json!({"name":name,"kind":op,"count":n.iter().sum::<u32>(),"capacity":stride,"batches":batches,"shift":shift,"has_aux":has_aux,"synthetic_counts":synthetic,"iterations":1,"buffer_words":words})
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    let layout = json!({"word_bytes":4,"sort_uniforms_words":size_of::<SortUniforms>()/4,"sort_uniforms":{"shift":offset_of!(SortUniforms,shift)/4,"max_keys_per_batch":offset_of!(SortUniforms,max_keys_per_batch)/4,"has_aux":offset_of!(SortUniforms,has_aux)/4}});
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&layout).unwrap(),
    )
    .unwrap();
    for op in [
        "radix_dispatch",
        "radix_init_batched",
        "radix_count",
        "radix_reduce",
        "radix_scan",
        "radix_scan_add",
        "radix_scatter",
        "radix_pass",
        "radix_sort",
    ] {
        let out = root.join(op);
        fs::create_dir_all(&out).unwrap();
        let mut cases = Vec::new();
        for (n, b, s, a, p) in [
            (0, 3, 0, 0, "mixed"),
            (1, 1, 0, 1, "mixed"),
            (31, 3, 4, 1, "mixed"),
            (255, 1, 28, 0, "mixed"),
            (256, 2, 0, 1, "equal"),
            (257, 3, 12, 1, "digits"),
            (1023, 1, 0, 1, "reverse"),
            (1024, 2, 28, 1, "mixed"),
            (1025, 3, 4, 0, "mixed"),
            (2049, 2, 16, 1, "mixed"),
        ] {
            cases.push(fixture(&out, n, b, s, a, p, op));
        }
        if ["radix_reduce", "radix_scan", "radix_scan_add"].contains(&op) {
            cases.push(fixture(&out, 1048577, 3, 0, 0, "histograms", op));
        }
        fs::write(
            out.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
