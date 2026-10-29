//! Native narrow-phase counters, publication and indirect-grid oracle.
#[allow(dead_code)]
#[path = "main.rs"]
mod pilot;
use nexus_rbd_shaders3d::{
    broad_phase::{ContactPlan, gpu_contact_plan, gpu_reset_narrow_phase},
    glamx::UVec3,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn layout() -> Value {
    let mut l = pilot::layout();
    l["plan_words"] = json!(size_of::<ContactPlan>() / 4);
    l["plan"] = json!({"bound":offset_of!(ContactPlan,bound)/4,"pfm_base":offset_of!(ContactPlan,pfm_base)/4,"pfm_len":offset_of!(ContactPlan,pfm_len)/4});
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = json!({"collision_pairs_capacity":offset_of!(BatchIndices,collision_pairs_capacity)/4,"contacts_capacity":offset_of!(BatchIndices,contacts_capacity)/4,"multibodies_batch_capacity":offset_of!(BatchIndices,multibodies_batch_capacity)/4,"num_batches":offset_of!(BatchIndices,num_batches)/4});
    l
}
fn fixture(
    root: &Path,
    op: &str,
    name: &str,
    count: usize,
    counters: [u32; 2],
    cap: u32,
    batching: [u32; 2],
) -> Value {
    let [pairs, pfm] = counters;
    let [batches, mb] = batching;
    let dir = root.join(name);
    fs::create_dir_all(&dir).unwrap();
    let mut pair_count = [pairs, 0xdeadbeef, 0xabcdef12];
    let mut pfm_count = vec![
        0xdeadbeef;
        if op == "reset_narrow" {
            count.max(1) + 3
        } else {
            3
        }
    ];
    pfm_count[0] = pfm;
    let mut plan = [ContactPlan {
        bound: 0xdeadbeef,
        pfm_base: 0xabcdef12,
        pfm_len: 0x12345678,
    }; 2];
    let mut sort_len = [0xdeadbeef; 3];
    let mut contacts_grid = [0xdeadbeef; 6];
    let mut pfm_grid = [0xdeadbeef; 6];
    let mut mb_grid = [0xdeadbeef; 6];
    let ids = BatchIndices {
        num_batches: batches,
        collision_pairs_capacity: cap,
        contacts_capacity: 2 * cap,
        multibodies_batch_capacity: mb,
        ..Default::default()
    };
    macro_rules! buffers {
        ($suffix:expr) => {{
            macro_rules! w {
                ($n:expr,$v:expr) => {
                    pilot::write_buffer(&dir, &format!("{}{}", $n, $suffix), $v)
                };
            }
            w!("PairCount", &pair_count);
            w!("PfmCount", &pfm_count);
            w!("Plan", &plan);
            w!("PfmSortLen", &sort_len);
            w!("ContactsIndirect", &contacts_grid);
            w!("PfmIndirect", &pfm_grid);
            w!("MbIndirect", &mb_grid);
            w!("BatchIds", &[ids]);
        }};
    }
    buffers!(".bin");
    let groups = if op == "reset_narrow" {
        count.div_ceil(64).max(1)
    } else {
        1
    };
    let threads = if op == "reset_narrow" { 64 } else { 1 };
    if op == "reset_narrow" {
        for i in 0..groups * threads {
            gpu_reset_narrow_phase(UVec3::new(i as u32, 0, 0), &mut pfm_count[..count]);
        }
    } else {
        gpu_contact_plan(
            &mut pair_count,
            &mut pfm_count,
            &mut plan[0],
            &mut sort_len,
            (&mut contacts_grid[..3]).try_into().unwrap(),
            (&mut pfm_grid[..3]).try_into().unwrap(),
            (&mut mb_grid[..3]).try_into().unwrap(),
            &ids,
        );
    }
    buffers!(".expected.bin");
    let sizes: serde_json::Map<String, Value> = [
        "PairCount",
        "PfmCount",
        "Plan",
        "PfmSortLen",
        "ContactsIndirect",
        "PfmIndirect",
        "MbIndirect",
        "BatchIds",
    ]
    .into_iter()
    .map(|n| {
        (
            n.to_owned(),
            json!(fs::metadata(dir.join(format!("{n}.bin"))).unwrap().len() / 4),
        )
    })
    .collect();
    json!({"name":name,"kind":op,"count":count,"groups":groups,"threads":threads,"iterations":1,"buffer_words":sizes})
}
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in ["reset_narrow", "contact_plan"] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let mut cases = Vec::new();
        if op == "reset_narrow" {
            for n in [0, 1, 31, 32, 33, 63, 64, 65, 129] {
                cases.push(fixture(&dir, op, &format!("n{n}"), n, [17, 23], 65, [3, 9]));
            }
        } else {
            for (name, a, b, c, nb, mb) in [
                ("empty", 0, 0, 100, 3, 9),
                ("zero-capacity", 99, 23, 0, 3, 9),
                ("pairs-only", 65, 0, 100, 1, 0),
                ("pfm-only", 0, 65, 100, 3, 9),
                ("one", 1, 0, 100, 1, 1),
                ("31", 15, 16, 100, 2, 7),
                ("32", 16, 16, 100, 2, 7),
                ("63", 31, 32, 100, 3, 9),
                ("64", 32, 32, 100, 3, 9),
                ("65", 33, 32, 100, 3, 9),
                ("129", 65, 64, 100, 3, 9),
                ("clamp-pairs", u32::MAX, 3, 33, 3, 9),
                ("clamp-pfm", 3, u32::MAX, 33, 3, 9),
                ("clamp-both", u32::MAX, u32::MAX, 65, 3, 9),
                (
                    "max-capacity",
                    u32::MAX,
                    u32::MAX,
                    u32::MAX / 2,
                    u32::MAX,
                    u32::MAX,
                ),
                ("no-batches", 17, 23, 65, 0, 9),
                ("no-multibodies", 17, 23, 65, 3, 0),
            ] {
                cases.push(fixture(&dir, op, name, 1, [a, b], c, [nb, mb]));
            }
        }
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}
