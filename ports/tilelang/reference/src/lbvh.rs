//! Native LBVH domain, key, topology, escape and leaf-refit checkpoints.
mod broadphase_shapes;
#[allow(dead_code)]
#[path = "collision_layout.rs"]
mod collision_layout;
mod dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    bounding_volumes::Aabb,
    broad_phase::*,
    glamx::{Quat, UVec3, Vec3},
    shapes::Shape,
    utils::BatchIndices,
};
use nexus_rbd_shaders3d::{
    glamx::UVec2,
    rapier::geometry::{Group, InteractionGroups, InteractionTestMode},
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
const OPS: [&str; 12] = [
    "lbvh_domain",
    "lbvh_domain_merge",
    "lbvh_morton",
    "lbvh_build",
    "lbvh_escapes",
    "lbvh_leaves",
    "lbvh_refit_chunks",
    "lbvh_refit_frontier",
    "lbvh_pairs",
    "lbvh_refit_plan",
    "lbvh_reset",
    "flat_list_dispatch",
];
#[derive(Clone)]
struct Data {
    pairs: Vec<CollisionPair>,
    groups: Vec<InteractionGroups>,
    filters: Vec<[u32; 2]>,
    poses: Vec<Pose>,
    shapes: Vec<Shape>,
    vertices: Vec<PaddedVector>,
    partials: Vec<Aabb>,
    domain: Vec<Aabb>,
    keys: Vec<u32>,
    sorted: Vec<u32>,
    tree: Vec<LbvhNode>,
    frontier: Vec<u32>,
    frontier_len: Vec<u32>,
    rounds: Vec<u32>,
    pair_count: Vec<u32>,
    indirect: Vec<u32>,
    batch: BatchIndices,
}
impl Data {
    fn write(&self, d: &Path, suffix: &str) {
        macro_rules! w {
            ($name:literal,$v:expr) => {
                fs::write(
                    d.join(format!("{}{}.bin", $name, suffix)),
                    bytemuck::cast_slice($v),
                )
                .unwrap();
            };
        }
        w!("Pairs", &self.pairs);
        w!("Groups", &self.groups);
        w!("Filters", &self.filters);
        w!("Poses", &self.poses);
        w!("Shapes", &self.shapes);
        w!("Vertices", &self.vertices);
        w!("Partials", &self.partials);
        w!("Domain", &self.domain);
        w!("Keys", &self.keys);
        w!("Sorted", &self.sorted);
        w!("Tree", &self.tree);
        w!("Frontier", &self.frontier);
        w!("FrontierLen", &self.frontier_len);
        w!("Rounds", &self.rounds);
        w!("PairCount", &self.pair_count);
        w!("Indirect", &self.indirect);
        w!("BatchIds", &[self.batch]);
    }
    fn run(&mut self, op: &str, g: u32) {
        let nb = self.batch.num_batches;
        match op {
            "lbvh_domain" => {
                let kernel = GpuLbvhComputeDomain::from_bytes(
                    &GpuBackend::Cpu,
                    &[],
                    GpuLbvhComputeDomain::ENTRY_POINT,
                )
                .unwrap();
                let mut partials = GpuBuffer::Cpu(self.partials.clone());
                kernel
                    .call(
                        &mut GpuPass::Cpu(None),
                        dispatch::workgroups([g, nb, 1]),
                        &GpuBuffer::Cpu(self.poses.clone()),
                        &mut partials,
                        &GpuBuffer::Cpu(vec![self.batch]),
                    )
                    .unwrap();
                self.partials = partials.unwrap_slice().to_vec();
            }
            "lbvh_domain_merge" => {
                let kernel = GpuLbvhDomainMerge::from_bytes(
                    &GpuBackend::Cpu,
                    &[],
                    GpuLbvhDomainMerge::ENTRY_POINT,
                )
                .unwrap();
                let mut domain = GpuBuffer::Cpu(self.domain.clone());
                kernel
                    .call(
                        &mut GpuPass::Cpu(None),
                        dispatch::workgroups([1, nb, 1]),
                        &GpuBuffer::Cpu(self.partials.clone()),
                        &mut domain,
                    )
                    .unwrap();
                self.domain = domain.unwrap_slice().to_vec();
            }
            "lbvh_refit_chunks" => {
                let kernel = GpuLbvhRefitChunks::from_bytes(
                    &GpuBackend::Cpu,
                    &[],
                    GpuLbvhRefitChunks::ENTRY_POINT,
                )
                .unwrap();
                let mut tree = GpuBuffer::Cpu(self.tree.clone());
                let mut frontier = GpuBuffer::Cpu(self.frontier.clone());
                let mut len = GpuBuffer::Cpu(self.frontier_len.clone());
                kernel
                    .call(
                        &mut GpuPass::Cpu(None),
                        dispatch::workgroups([
                            self.batch.colliders_len.div_ceil(256).max(1),
                            nb,
                            1,
                        ]),
                        &mut tree,
                        &mut frontier,
                        &mut len,
                        &GpuBuffer::Cpu(vec![self.batch]),
                    )
                    .unwrap();
                self.tree = tree.unwrap_slice().to_vec();
                self.frontier = frontier.unwrap_slice().to_vec();
                self.frontier_len = len.unwrap_slice().to_vec();
            }
            "lbvh_refit_frontier" => {
                let kernel = GpuLbvhRefitFrontier::from_bytes(
                    &GpuBackend::Cpu,
                    &[],
                    GpuLbvhRefitFrontier::ENTRY_POINT,
                )
                .unwrap();
                let mut tree = GpuBuffer::Cpu(self.tree.clone());
                kernel
                    .call(
                        &mut GpuPass::Cpu(None),
                        dispatch::workgroups([1, nb, 1]),
                        &mut tree,
                        &GpuBuffer::Cpu(self.frontier.clone()),
                        &GpuBuffer::Cpu(self.frontier_len.clone()),
                        &GpuBuffer::Cpu(vec![self.batch]),
                        &GpuBuffer::Cpu(self.rounds.clone()),
                    )
                    .unwrap();
                self.tree = tree.unwrap_slice().to_vec();
            }
            "lbvh_pairs" => {
                if self.batch.colliders_len == 0 {
                    return;
                }
                let kernel = GpuLbvhFindCollisionPairs::from_bytes(
                    &GpuBackend::Cpu,
                    &[],
                    GpuLbvhFindCollisionPairs::ENTRY_POINT,
                )
                .unwrap();
                let mut pairs = GpuBuffer::Cpu(self.pairs.clone());
                let mut count = GpuBuffer::Cpu(self.pair_count.clone());
                kernel
                    .call(
                        &mut GpuPass::Cpu(None),
                        dispatch::workgroups([g, nb, 1]),
                        &GpuBuffer::Cpu(self.tree.clone()),
                        &mut pairs,
                        &mut count,
                        &GpuBuffer::Cpu(self.groups.clone()),
                        &GpuBuffer::Cpu(vec![self.batch]),
                        &GpuBuffer::Cpu(self.filters.clone()),
                    )
                    .unwrap();
                self.pairs = pairs.unwrap_slice().to_vec();
                self.pair_count = count.unwrap_slice().to_vec();
            }
            "lbvh_refit_plan" => {
                gpu_lbvh_refit_plan(&self.frontier_len, &mut self.rounds[0], &self.batch)
            }
            "lbvh_reset" => {
                for i in 0..64 {
                    gpu_lbvh_reset_collision_pairs(UVec3::new(i, 0, 0), &mut self.pair_count);
                }
            }
            "flat_list_dispatch" => {
                let dst: &mut [u32; 3] = (&mut self.indirect[..3]).try_into().unwrap();
                gpu_flat_list_dispatch(&mut self.pair_count, dst, &self.batch);
            }
            _ => {
                for b in 0..nb {
                    for i in 0..g * 64 {
                        let id = UVec3::new(i, b, 0);
                        let ng = UVec3::new(g, nb, 1);
                        match op {
                            "lbvh_morton" => gpu_lbvh_compute_morton(
                                id,
                                ng,
                                &self.poses,
                                &self.domain,
                                &mut self.keys,
                                &self.batch,
                            ),
                            "lbvh_build" => {
                                if self.batch.colliders_len > 0 {
                                    gpu_lbvh_build(id, ng, &self.keys, &mut self.tree, &self.batch)
                                }
                            }
                            "lbvh_escapes" => gpu_lbvh_escapes(id, ng, &mut self.tree, &self.batch),
                            "lbvh_leaves" => {
                                if self.batch.colliders_len > 0 {
                                    gpu_lbvh_refit_leaves(
                                        id,
                                        ng,
                                        &self.poses,
                                        &self.shapes,
                                        &self.sorted,
                                        &mut self.tree,
                                        &self.batch,
                                        &mut self.frontier_len,
                                        &self.vertices,
                                    )
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
    }
}
fn case(root: &Path, n: usize, nb: usize, g: u32, mode: &str, suites: &mut [Vec<Value>]) {
    let cap = n + 3;
    let storage = cap * nb + 3;
    let guard = 0xdeadbeef;
    let aabb = Aabb {
        mins: Vec3::splat(-17.),
        maxs: Vec3::splat(23.),
        padding0: guard,
        padding1: guard,
    };
    let mut node = LbvhNode::default();
    bytemuck::cast_slice_mut::<_, u32>(std::slice::from_mut(&mut node)).fill(guard);
    let pair_capacity = if mode == "overflow" { 3 } else { n * n * nb };
    let mut d = Data {
        pairs: vec![
            CollisionPair {
                colliders: UVec2::new(guard, guard)
            };
            pair_capacity + 3
        ],
        groups: vec![InteractionGroups::all(); storage],
        filters: (0..storage).map(|i| [i as u32, 0]).collect(),
        poses: vec![Pose::IDENTITY; storage],
        shapes: vec![Shape::ball(0.); storage],
        vertices: vec![
            PaddedVector::new(Vec3::ZERO),
            PaddedVector::new(Vec3::new(-0.7, -0.3, 0.2)),
            PaddedVector::new(Vec3::new(0.6, 0.1, -0.8)),
            PaddedVector::new(Vec3::new(0.2, 0.8, 0.3)),
            PaddedVector::new(Vec3::new(-0.1, 0.2, 0.9)),
            PaddedVector::new(Vec3::new(0.3, -0.6, -0.2)),
        ],
        partials: vec![aabb; 64 * nb + 3],
        domain: vec![aabb; nb + 3],
        keys: vec![guard; cap * nb + 3],
        sorted: vec![guard; cap * nb + 3],
        tree: vec![node; 2 * cap * nb + 3],
        frontier: vec![guard; cap * nb + 3],
        frontier_len: (0..nb + 3)
            .map(|i| {
                if i == nb - 1 {
                    u32::MAX
                } else {
                    (i * 257) as u32
                }
            })
            .collect(),
        rounds: vec![guard; 3],
        pair_count: vec![(n * n * nb + 17) as u32, guard, guard],
        indirect: vec![guard; 6],
        batch: BatchIndices {
            num_batches: nb as u32,
            colliders_len: n as u32,
            colliders_batch_capacity: cap as u32,
            collision_pairs_capacity: pair_capacity as u32,
            ..Default::default()
        },
    };
    for b in 0..nb {
        for i in 0..n {
            let index = i * nb + b;
            let t = if mode == "duplicates" { i / 5 } else { i };
            let xyz = if mode == "collapsed" {
                Vec3::new(2., -3., 1.)
            } else {
                Vec3::new(
                    ((t * 17) % 31) as f32 * 0.125 - 2.,
                    ((t * 7) % 19) as f32 * 0.25 + b as f32,
                    if mode == "flat" {
                        -0.3
                    } else {
                        ((t * 13) % 23) as f32 * 0.1
                    },
                )
            };
            d.poses[index] = Pose::from_parts(
                xyz,
                Quat::from_scaled_axis(Vec3::new(0.17 * i as f32, -0.3, 0.2)),
            );
            d.shapes[index] = broadphase_shapes::shape((i % 9) as u32);
        }
    }
    if mode == "filters" {
        for i in 0..n {
            for b in 0..nb {
                let at = i * nb + b;
                d.filters[at] = [(i / 2) as u32, if i % 3 == 0 { 7 } else { 0 }];
                d.groups[at] = InteractionGroups::new(
                    Group::from_bits_retain(1u32 << (i % 32)),
                    Group::from_bits_retain(if i % 4 == 0 {
                        u32::MAX
                    } else {
                        1u32 << ((i + 1) % 32)
                    }),
                    if (i / 2) % 2 == 0 {
                        InteractionTestMode::Or
                    } else {
                        InteractionTestMode::And
                    },
                );
            }
        }
    }
    let name = format!("n{n}-b{nb}-g{g}-{mode}");
    for (ix, op) in OPS.iter().enumerate() {
        if *op == "lbvh_build" {
            // Topology stage consumes already sorted Morton keys, with stable collider IDs.
            for b in 0..nb {
                let start = b * cap;
                let mut rows = (0..n)
                    .map(|i| (d.keys[start + i], i as u32))
                    .collect::<Vec<_>>();
                rows.sort_by_key(|r| r.0);
                for (i, (key, id)) in rows.into_iter().enumerate() {
                    d.keys[start + i] = key;
                    d.sorted[start + i] = id;
                }
            }
        }
        if *op == "lbvh_pairs" {
            d.pair_count[0] = 0;
        }
        if *op == "flat_list_dispatch" {
            d.pair_count[0] = (n * n * nb + 17) as u32;
        }
        if *op == "lbvh_refit_chunks" && n == 0 {
            d.frontier_len[..nb].fill(0);
        }
        if *op == "lbvh_refit_plan" {
            for b in 0..nb {
                d.frontier_len[b] = if b == nb - 1 {
                    u32::MAX
                } else {
                    (b * 257) as u32
                };
            }
        }
        let dir = root.join(op).join(&name);
        fs::create_dir_all(&dir).unwrap();
        d.write(&dir, "");
        let mut out = d.clone();
        out.run(op, g);
        out.write(&dir, ".expected");
        let sizes = fs::read_dir(&dir)
            .unwrap()
            .map(|r| r.unwrap())
            .filter_map(|r| {
                let name = r.file_name().to_string_lossy().into_owned();
                if name.ends_with(".bin") && !name.contains(".expected") {
                    Some((
                        name.trim_end_matches(".bin").to_owned(),
                        json!(r.metadata().unwrap().len() / 4),
                    ))
                } else {
                    None
                }
            })
            .collect::<serde_json::Map<_, _>>();
        let c = json!({"name":name,"kind":op,"count":n*nb,"colliders":n,"batches":nb,"capacity":cap,"storage":storage,"groups":g,"pair_capacity":pair_capacity,"iterations":1,"mode":mode,"buffer_words":sizes});
        fs::write(
            dir.join("case.json"),
            serde_json::to_vec_pretty(&c).unwrap(),
        )
        .unwrap();
        suites[ix].push(c);
        if ix < 9 {
            d = out;
        }
    }
}
fn main() {
    let output = std::env::args().nth(1).unwrap();
    let root = Path::new(&output);
    fs::create_dir_all(root).unwrap();
    let mut l = collision_layout::layout();
    macro_rules! fields {($ty:ty,$($f:ident),+)=>{json!({$(stringify!($f):offset_of!($ty,$f)/4),+})}}
    l["max_flt_bits"] = json!(nexus_rbd_shaders3d::MAX_FLT.to_bits());
    l["lbvh_node_words"] = json!(size_of::<LbvhNode>() / 4);
    l["lbvh_node"] = fields!(
        LbvhNode,
        aabb,
        left,
        right,
        parent,
        refit_count,
        first_leaf,
        last_leaf,
        escape,
        max_extent
    );
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&l).unwrap(),
    )
    .unwrap();
    let mut suites = vec![Vec::new(); OPS.len()];
    for (n, nb, g, mode) in [
        (0, 1, 1, "mixed"),
        (1, 2, 1, "collapsed"),
        (2, 1, 1, "mixed"),
        (33, 3, 1, "mixed"),
        (65, 2, 2, "flat"),
        (129, 1, 1, "duplicates"),
        (257, 2, 3, "mixed"),
        (513, 1, 2, "collapsed"),
        (33, 2, 1, "overflow"),
        (65, 2, 2, "filters"),
    ] {
        case(root, n, nb, g, mode, &mut suites);
    }
    for (op, cases) in OPS.iter().zip(suites) {
        fs::write(
            root.join(op).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
