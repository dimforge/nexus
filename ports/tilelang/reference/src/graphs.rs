//! Live original count/scan/scatter adjacency construction, without a host graph fallback.
#[allow(dead_code)]
#[path = "contacts.rs"]
mod contacts;
mod dispatch;
mod prefix_program;
use contacts::pilot;
use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan,
    dynamics::{WorldMassProperties, gpu_solver_count_constraints, gpu_solver_sort_constraints},
    glamx::{UVec2, UVec3, Vec3},
    queries::IndexedManifold,
};
use prefix_program::scan;
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn layout() -> Value {
    let mut l = contacts::layout();
    l["plan_words"] = json!(size_of::<ContactPlan>() / 4);
    l["plan"] = json!({"bound":offset_of!(ContactPlan,bound)/4});
    l
}

fn fixture(root: &Path, n: usize, bodies: usize, batches: usize, op: &str) -> Value {
    let name = format!("n{n}-r{bodies}-b{batches}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let guard = 0xDEADBEEF;
    let body_storage = bodies + 3;
    let mut ims = vec![IndexedManifold::default(); n + 3];
    let mut mass = vec![WorldMassProperties::default(); body_storage];
    let mut multibody = vec![0u32; body_storage];
    for i in 0..body_storage {
        mass[i].inv_mass = if i % batches == 0 && (i / batches).is_multiple_of(7) {
            Vec3::ZERO
        } else if i % 5 == 0 {
            Vec3::new(0.0, 2.0, 0.0)
        } else {
            Vec3::ONE
        };
        multibody[i] = if i % 17 == 16 { 0x80000000 } else { 0 };
    }
    for (i, im) in ims.iter_mut().enumerate() {
        im.bodies = UVec2::splat(u32::MAX);
        if i < n && i % 11 != 0 {
            let batch = i % batches;
            let per_batch = bodies / batches;
            let a = (i * 7 % per_batch) * batches + batch;
            let b = if i % 9 == 0 {
                a
            } else {
                (i * 13 + 1) % per_batch * batches + batch
            };
            im.bodies = UVec2::new(a as u32, b as u32);
            im.contact.len = (i % 4 + 1) as u32;
        }
    }
    let plan = ContactPlan {
        bound: n as u32,
        pfm_base: n as u32 / 2,
        pfm_len: n as u32 - n as u32 / 2,
    };
    let groups = if n > 192 { 1 } else { 3 };
    let mut counts = vec![guard; body_storage];
    counts[..bodies].fill(0);
    let mut ids = vec![guard; 2 * n + 3];
    let mut strides = [0u32; 5];
    strides[0] = bodies as u32;
    for i in 1..5 {
        strides[i] = strides[i - 1].div_ceil(256);
    }
    let levels = (1..5).find(|&i| strides[i] <= 1).unwrap();
    let mut aux: Vec<_> = (0..4)
        .map(|i| vec![guard; strides[i + 1] as usize + 3])
        .collect();
    if op == "adjacency_scatter" {
        for i in 0..groups * 64 {
            gpu_solver_count_constraints(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &ims,
                &mut counts,
                &multibody,
                &mass,
                &plan,
            );
        }
        let mut sum = 0;
        for c in &mut counts[..bodies] {
            let old = *c;
            *c = sum;
            sum += old;
        }
    }
    pilot::write_buffer(&dir, "Contacts.bin", &ims);
    pilot::write_buffer(&dir, "WorldMass.bin", &mass);
    pilot::write_buffer(&dir, "IsMultibody.bin", &multibody);
    pilot::write_buffer(&dir, "Plan.bin", &[plan]);
    pilot::write_buffer(&dir, "Counts.bin", &counts);
    pilot::write_buffer(&dir, "Ids.bin", &ids);
    for i in 0..4 {
        pilot::write_buffer(&dir, &format!("Aux{i}.bin"), &aux[i]);
        pilot::write_buffer(&dir, &format!("Stride{i}.bin"), &[strides[i], guard]);
    }
    if op != "adjacency_scatter" {
        for i in 0..groups * 64 {
            gpu_solver_count_constraints(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &ims,
                &mut counts,
                &multibody,
                &mass,
                &plan,
            );
        }
    }
    if op == "adjacency_build" {
        scan(&mut counts, &mut aux, &strides, levels);
    }
    if op != "adjacency_count" {
        for i in 0..groups * 64 {
            gpu_solver_sort_constraints(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &mut counts,
                &mass,
                &ims,
                &plan,
                &mut ids,
                &multibody,
            );
        }
    }
    pilot::write_buffer(&dir, "Counts.expected.bin", &counts);
    pilot::write_buffer(&dir, "Ids.expected.bin", &ids);
    for (i, a) in aux.iter().enumerate() {
        pilot::write_buffer(&dir, &format!("Aux{i}.expected.bin"), a);
    }
    let l = layout();
    let mut sizes = json!({"Contacts":ims.len()*l["contact_words"].as_u64().unwrap() as usize,
        "Counts":counts.len(),"Ids":ids.len(),"WorldMass":mass.len()*l["world_mass_words"].as_u64().unwrap() as usize,
        "IsMultibody":multibody.len(),"Plan":l["plan_words"].as_u64().unwrap()});
    for i in 0..4 {
        sizes[format!("Aux{i}")] = json!(aux[i].len());
        sizes[format!("Stride{i}")] = json!(2);
    }
    json!({"name":name,"kind":op,"count":n,"bodies":bodies,"batches":batches,"body_storage":body_storage,"contact_storage":ims.len(),"groups":groups,"iterations":1,"strides":&strides[..4],"levels":levels,"buffer_words":sizes})
}
fn main() {
    let path = std::env::args().nth(1).expect("Expected output directory");
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in ["adjacency_count", "adjacency_scatter", "adjacency_build"] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let cases: Vec<_> = [
            (0, 0, 1),
            (0, 16, 2),
            (1, 8, 1),
            (31, 30, 3),
            (32, 32, 1),
            (33, 66, 2),
            (63, 128, 2),
            (64, 256, 2),
            (65, 258, 3),
            (257, 514, 2),
            (513, 1026, 3),
        ]
        .into_iter()
        .map(|(n, r, b)| fixture(&dir, n, r, b, op))
        .collect();
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}
