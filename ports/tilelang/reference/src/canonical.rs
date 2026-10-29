//! Native canonical key/permutation shaders with the native cooperative radix pipeline.
#[allow(dead_code)]
mod radix;
use khal::backend::GpuBuffer;
#[allow(dead_code)]
mod contacts;
use contacts::pilot;
use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan,
    dynamics::{gpu_contact_sort_keys, gpu_permute_contacts},
    glamx::{UVec2, UVec3},
    queries::IndexedManifold,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn layout() -> Value {
    let mut l = contacts::layout();
    l["plan_words"] = json!(size_of::<ContactPlan>() / 4);
    l["plan"] = json!({"bound":offset_of!(ContactPlan,bound)/4,"pfm_base":offset_of!(ContactPlan,pfm_base)/4,"pfm_len":offset_of!(ContactPlan,pfm_len)/4});
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = json!({"contact_sort_collider_shift":offset_of!(BatchIndices,contact_sort_collider_shift)/4});
    l["sort_uniforms_words"] =
        json!(size_of::<nexus_rbd_shaders3d::utils::radix_sort::SortUniforms>() / 4);
    l["sort_uniforms"] = json!({"shift":offset_of!(nexus_rbd_shaders3d::utils::radix_sort::SortUniforms,shift)/4,"max_keys_per_batch":offset_of!(nexus_rbd_shaders3d::utils::radix_sort::SortUniforms,max_keys_per_batch)/4,"has_aux":offset_of!(nexus_rbd_shaders3d::utils::radix_sort::SortUniforms,has_aux)/4});
    let default = IndexedManifold::default();
    l["default_contact"] = json!(bytemuck::cast_slice::<_, u32>(&[default]));
    l
}
fn fixture(root: &Path, n: usize, op: &str, mode: &str, selector: u32) -> Value {
    let name = format!("n{n}-{mode}-s{selector}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let storage = n.max(1) + 3;
    let mut src = vec![IndexedManifold::default(); storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut src).fill(0xdeadbeef);
    for (i, c) in src.iter_mut().take(n).enumerate() {
        c.contact.len = if i % 5 == 0 { 0 } else { (i % 4 + 1) as u32 };
        c.colliders = UVec2::new(((i * 13 + 11) % 19) as u32, ((i * 17 + 2) % 23) as u32);
        c.subshape = ((i * 31) % 7) as u32;
        if mode == "duplicates" {
            c.colliders = UVec2::new(7, 11);
            c.subshape = 3;
        }
        if mode == "full-u32" {
            c.colliders = UVec2::new(u32::MAX - (i % 3) as u32, u32::MAX - (i % 7) as u32);
            c.subshape = if i % 2 == 0 { u32::MAX } else { 0x80000000 };
        }
        if mode == "empty" {
            c.contact.len = 0;
        }
        c.bodies = UVec2::new(i as u32, 0x80000000 + i as u32);
    }
    let mut dst = vec![IndexedManifold::default(); storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut dst).fill(0xabcdef12);
    let mut order: Vec<u32> = (0..storage)
        .map(|i| {
            if i < n {
                (n - 1 - i) as u32
            } else {
                0xdeadbeef
            }
        })
        .collect();
    let mut keys = vec![0xdeadbeefu32; storage];
    let mut values = keys.clone();
    let mut count = [0xabcdef12u32; 1];
    let mut sorting = radix::Arrays::new();
    for key in [
        "SortKeys",
        "PongKeys",
        "PongValues",
        "RadixAux",
        "RadixOutAux",
        "RadixBatchIds",
    ] {
        sorting.insert(key.into(), GpuBuffer::Cpu(vec![0xdeadbeef; storage]));
    }
    for (key, len) in [
        ("Counts", 16 * storage.div_ceil(1024) + 3),
        ("Reduced", 1024 + 3),
        ("SortGrid", 6),
        ("ReduceGrid", 6),
    ] {
        sorting.insert(key.into(), GpuBuffer::Cpu(vec![0xdeadbeef; len]));
    }
    for (key, shift) in [("Config".into(), 0), ("InitConfig".into(), 1)]
        .into_iter()
        .chain((0..8).map(|i| (format!("Config{i}"), i * 4)))
    {
        sorting.insert(key, GpuBuffer::Cpu(vec![shift, storage as u32, 0]));
    }

    let plan = [
        ContactPlan {
            bound: n as u32,
            pfm_base: n as u32,
            pfm_len: 0,
        },
        ContactPlan {
            bound: 0xdeadbeef,
            pfm_base: 0xabcdef12,
            pfm_len: 0x12345678,
        },
    ];
    let ids = BatchIndices {
        contact_sort_collider_shift: if mode == "full-u32" { 0 } else { 5 },
        ..Default::default()
    };
    let selectors = [selector, 0xdeadbeef];
    let g = if n > 64 { 1 } else { 2 };
    macro_rules! buffers {
        ($suffix:expr) => {{
            macro_rules! w {
                ($name:expr,$v:expr) => {
                    pilot::write_buffer(&dir, &format!("{}{}", $name, $suffix), $v)
                };
            }
            w!("Contacts", &src);
            w!("Output", &dst);
            w!("InOrder", &order);
            w!("Keys", &keys);
            w!("Values", &values);
            w!("SortCount", &count);
            w!("Plan", &plan);
            w!("BatchIds", &[ids]);
            w!("Selector", &selectors);
            for (k, v) in &sorting {
                w!(k, v.unwrap_slice());
            }
        }};
    }
    buffers!(".bin");
    let passes = if op == "canonical_contacts" {
        if mode == "packed" {
            vec![7]
        } else if mode == "two-fields" {
            vec![5, 2]
        } else {
            vec![4, 1, 2]
        }
    } else {
        vec![selector]
    };
    if op != "permute_contacts" {
        for sel in &passes {
            for gid in 0..g * 64 {
                gpu_contact_sort_keys(
                    UVec3::new(gid, 0, 0),
                    UVec3::new(g, 1, 1),
                    &src,
                    &plan[0],
                    &order,
                    &mut keys,
                    &mut values,
                    sel,
                    &mut count,
                    &ids,
                );
            }
            if op == "canonical_contacts" {
                sorting.insert("Keys".into(), GpuBuffer::Cpu(keys.clone()));
                sorting.insert("Values".into(), GpuBuffer::Cpu(values.clone()));
                sorting.insert("InOrder".into(), GpuBuffer::Cpu(order.clone()));
                sorting.insert("SortCount".into(), GpuBuffer::Cpu(count.to_vec()));
                for pass in 0..8 {
                    let (input_k, input_v) = if pass == 0 {
                        ("Keys", "Values")
                    } else if pass % 2 == 1 {
                        ("PongKeys", "PongValues")
                    } else {
                        ("SortKeys", "InOrder")
                    };
                    let (output_k, output_v) = if pass % 2 == 0 {
                        ("PongKeys", "PongValues")
                    } else {
                        ("SortKeys", "InOrder")
                    };
                    let mapping = [
                        ("Config", "Config"),
                        ("NumKeys", "SortCount"),
                        ("Keys", input_k),
                        ("Values", input_v),
                        ("OutKeys", output_k),
                        ("OutValues", output_v),
                        ("Aux", "RadixAux"),
                        ("OutAux", "RadixOutAux"),
                        ("BatchIds", "RadixBatchIds"),
                        ("Counts", "Counts"),
                        ("Reduced", "Reduced"),
                        ("SortGrid", "SortGrid"),
                        ("ReduceGrid", "ReduceGrid"),
                    ];
                    let mut local = radix::Arrays::new();
                    for (name, physical) in mapping {
                        local.insert(
                            name.into(),
                            GpuBuffer::Cpu(sorting[physical].unwrap_slice().to_vec()),
                        );
                    }
                    for i in 0..8 {
                        let k = format!("Config{i}");
                        local.insert(
                            k.clone(),
                            GpuBuffer::Cpu(sorting[&k].unwrap_slice().to_vec()),
                        );
                    }
                    if pass == 0 {
                        radix::exec(&mut local, "radix_dispatch", storage as u32, 1, None);
                    }
                    for stage in radix::STAGES {
                        radix::exec(&mut local, stage, storage as u32, 1, Some(pass));
                    }
                    for (name, physical) in mapping {
                        sorting.insert(physical.into(), local.remove(name).unwrap());
                    }
                }
                order.copy_from_slice(sorting["InOrder"].unwrap_slice());
                for name in ["Keys", "Values", "InOrder", "SortCount"] {
                    sorting.remove(name);
                }
            }
        }
    }
    if op != "contact_keys" {
        for gid in 0..g * 64 {
            gpu_permute_contacts(
                UVec3::new(gid, 0, 0),
                UVec3::new(g, 1, 1),
                &src,
                &mut dst,
                &order,
                &plan[0],
            );
        }
    }
    buffers!(".expected.bin");
    let names = [
        "Contacts",
        "Output",
        "InOrder",
        "Keys",
        "Values",
        "SortCount",
        "Plan",
        "BatchIds",
        "Selector",
    ];
    let sizes: serde_json::Map<String, Value> = names
        .into_iter()
        .chain(sorting.keys().map(String::as_str))
        .map(|n| {
            (
                n.to_owned(),
                json!(fs::metadata(dir.join(format!("{n}.bin"))).unwrap().len() / 4),
            )
        })
        .collect();
    json!({"name":name,"kind":op,"count":n,"storage":storage,"groups":g,"mode":mode,"selector":selector,"passes":passes,"iterations":1,"buffer_words":sizes})
}
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in ["contact_keys", "permute_contacts", "canonical_contacts"] {
        let dir = root.join(op);
        let mut cases = Vec::new();
        for n in [0, 1, 31, 32, 33, 63, 64, 65, 129] {
            cases.push(fixture(&dir, n, op, "mixed", 7));
        }
        if op == "contact_keys" {
            for sel in 0..8 {
                cases.push(fixture(&dir, 65, op, "full-u32", sel));
            }
        }
        if op == "canonical_contacts" {
            for mode in ["duplicates", "full-u32", "empty", "packed", "two-fields"] {
                cases.push(fixture(&dir, 65, op, mode, 7));
            }
        }
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}
