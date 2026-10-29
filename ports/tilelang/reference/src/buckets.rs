//! Live original color buckets, including the single-batch shared-memory fast path.
#[allow(dead_code)]
#[path = "contacts.rs"]
mod contacts;
mod dispatch;
mod prefix_program;
use contacts::pilot;
use khal::backend::GpuBuffer;
use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan,
    dynamics::{ColorStats, ContactLink},
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
    l["plan"] = json!({"bound":offset_of!(ContactPlan,bound)/4});
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = json!({"num_batches":offset_of!(BatchIndices,num_batches)/4,"solver_color_buckets_stride":offset_of!(BatchIndices,solver_color_buckets_stride)/4});
    l["stats_words"] = json!(size_of::<ColorStats>() / 4);
    l["stats"] = json!({"sizes":offset_of!(ColorStats,sizes)/4,"max_bucket":offset_of!(ColorStats,max_bucket)/4,"highest_color":offset_of!(ColorStats,highest_color)/4});
    l
}
mod bucket_program;
use bucket_program::State;
impl State {
    fn write(&self, dir: &Path, expected: bool) {
        macro_rules! w {
            ($name:literal,$buf:expr,$changed:expr) => {
                if !expected || $changed {
                    pilot::write_buffer(
                        dir,
                        &format!("{}{}.bin", $name, if expected { ".expected" } else { "" }),
                        $buf.unwrap_slice(),
                    );
                }
            };
        }
        w!("Colors", self.colors, false);
        w!("Links", self.links, false);
        w!("Plan", self.plan, false);
        w!("Buckets", self.buckets, true);
        w!("BatchIds", self.batch, false);
        w!("SortedLinks", self.sorted, true);
        w!("Indices", self.indices, true);
        w!("Stats", self.stats, true);
        w!("Indirect", self.indirect, true);
        for i in 0..4 {
            pilot::write_buffer(
                dir,
                &format!("Aux{i}{}.bin", if expected { ".expected" } else { "" }),
                &self.aux[i],
            );
            if !expected {
                pilot::write_buffer(
                    dir,
                    &format!("Stride{i}.bin"),
                    &[self.strides[i], 0xDEADBEEF],
                );
            }
        }
    }
}
fn fixture(
    root: &Path,
    n: usize,
    nb: u32,
    stride: u32,
    groups: u32,
    op: &str,
    clip: bool,
) -> Value {
    let name = format!(
        "n{n}-b{nb}-s{stride}-g{groups}{}",
        if clip { "-clip" } else { "" }
    );
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let guard = 0xDEADBEEF;
    let storage = n + 3;
    let mut links = vec![ContactLink::default(); storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut links).fill(0xa5a5a5a5);
    let mut colors = vec![guard; storage];
    for i in 0..n {
        links[i].len = if i % 7 == 0 { 0 } else { (i % 4 + 1) as u32 };
        links[i].solver_body_a = if links[i].len == 0 {
            u32::MAX
        } else {
            (i as u32 * 5 % 11) * nb + i as u32 % nb
        };
        links[i].solver_body_b = links[i].solver_body_a;
        links[i].previous_constraint = i as u32; // Unique copied identity for permutations.
        colors[i] = [
            0,
            1,
            2,
            31,
            32,
            63,
            64,
            stride - 2,
            stride - 1,
            stride,
            u32::MAX,
        ][i % 11];
    }
    let mut sorted = vec![ContactLink::default(); storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut sorted).fill(0xa3a3a3a3);
    let mut strides = [0u32; 5];
    strides[0] = nb * stride;
    for i in 1..5 {
        strides[i] = strides[i - 1].div_ceil(256);
    }
    let levels = (1..5).find(|&i| strides[i] <= 1).unwrap();
    let mut state = State {
        colors: GpuBuffer::Cpu(colors),
        links: GpuBuffer::Cpu(links),
        plan: GpuBuffer::Cpu(vec![ContactPlan {
            bound: n as u32,
            pfm_base: n as u32,
            pfm_len: 0,
        }]),
        buckets: GpuBuffer::Cpu(vec![guard; (nb * stride) as usize]),
        batch: GpuBuffer::Cpu(vec![BatchIndices {
            num_batches: nb,
            solver_color_buckets_stride: stride,
            ..Default::default()
        }]),
        sorted: GpuBuffer::Cpu(sorted),
        indices: GpuBuffer::Cpu(vec![guard; storage]),
        stats: GpuBuffer::Cpu(vec![
            ColorStats {
                max_bucket: guard,
                highest_color: guard,
                sizes: [guard; 64]
            };
            2
        ]),
        indirect: GpuBuffer::Cpu(vec![[guard; 3]; 3]),
        aux: (0..4)
            .map(|i| vec![guard; strides[i + 1] as usize + 3])
            .collect(),
        strides,
        levels,
        groups,
    };
    if !matches!(op, "bucket_reset" | "bucket_build") {
        state.reset();
    }
    if matches!(op, "bucket_scatter" | "bucket_dispatch") {
        state.count();
        state.scan();
    }
    if op == "bucket_dispatch" {
        state.scatter();
    }
    if clip {
        // Synthetic large current bucket: dispatch sizing is independent of link storage.
        let buckets = state.buckets.unwrap_slice_mut();
        buckets.fill(0);
        buckets[(nb * 2 - 1) as usize..].fill(20000);
    }
    state.write(&dir, false);
    match op {
        "bucket_reset" => state.reset(),
        "bucket_count" => state.count(),
        "bucket_scatter" => state.scatter(),
        "bucket_dispatch" => state.dispatch(),
        "bucket_build" => {
            state.reset();
            state.count();
            state.scan();
            state.scatter();
            state.dispatch();
        }
        _ => unreachable!(),
    }
    state.write(&dir, true);
    let l = layout();
    let mut sizes = json!({"Colors":storage,"Links":storage*l["link_words"].as_u64().unwrap() as usize,"Plan":l["plan_words"],"Buckets":nb*stride,"BatchIds":l["batch_words"],"SortedLinks":storage*l["link_words"].as_u64().unwrap() as usize,"Indices":storage,"Stats":l["stats_words"].as_u64().unwrap()*2,"Indirect":9});
    for i in 0..4 {
        sizes[format!("Aux{i}")] = json!(state.aux[i].len());
        sizes[format!("Stride{i}")] = json!(2);
    }
    json!({"name":name,"kind":op,"count":n,"storage":storage,"batches":nb,"color_stride":stride,"bucket_count":nb*stride,"groups":groups,"strides":&strides[..4],"levels":levels,"iterations":1,"clip":clip,"buffer_words":sizes})
}
fn main() {
    let path = std::env::args().nth(1).expect("Expected output directory");
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in [
        "bucket_reset",
        "bucket_count",
        "bucket_scatter",
        "bucket_dispatch",
        "bucket_build",
    ] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let mut cases = Vec::new();
        for (n, nb, stride, groups) in [
            (0, 1, 3, 1),
            (1, 1, 5, 1),
            (31, 1, 65, 1),
            (32, 2, 66, 1),
            (33, 3, 67, 1),
            (63, 1, 128, 1),
            (64, 1, 129, 1),
            (65, 1, 65, 2),
            (65, 2, 130, 1),
            (257, 1, 67, 1),
            (513, 3, 130, 2),
        ] {
            cases.push(fixture(&dir, n, nb, stride, groups, op, false));
        }
        if op == "bucket_dispatch" {
            cases.push(fixture(&dir, 33, 2, 130, 1, op, true));
        }
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}
