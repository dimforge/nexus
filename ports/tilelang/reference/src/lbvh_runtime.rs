//! Shared native LBVH stage execution and resident scene workspace.
use crate::{dispatch, radix};
#[path = "lbvh_program.rs"]
pub(crate) mod lbvh_program;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    bounding_volumes::Aabb,
    broad_phase::*,
    glamx::{UVec2, UVec3, Vec3},
    rapier::geometry::InteractionGroups,
    shapes::Shape,
    utils::BatchIndices,
};
use std::{fs, path::Path};
pub(crate) const BUFFERS: [&str; 17] = [
    "Pairs",
    "Groups",
    "Filters",
    "Poses",
    "Shapes",
    "Vertices",
    "Partials",
    "Domain",
    "Keys",
    "Sorted",
    "Tree",
    "Frontier",
    "FrontierLen",
    "Rounds",
    "PairCount",
    "Indirect",
    "BatchIds",
];
#[derive(Clone)]
pub(crate) struct Data {
    pub(crate) pairs: Vec<CollisionPair>,
    pub(crate) groups: Vec<InteractionGroups>,
    pub(crate) filters: Vec<[u32; 2]>,
    pub(crate) poses: Vec<Pose>,
    pub(crate) shapes: Vec<Shape>,
    pub(crate) vertices: Vec<PaddedVector>,
    pub(crate) partials: Vec<Aabb>,
    pub(crate) domain: Vec<Aabb>,
    pub(crate) keys: Vec<u32>,
    pub(crate) sorted: Vec<u32>,
    pub(crate) tree: Vec<LbvhNode>,
    pub(crate) frontier: Vec<u32>,
    pub(crate) frontier_len: Vec<u32>,
    pub(crate) rounds: Vec<u32>,
    pub(crate) pair_count: Vec<u32>,
    pub(crate) indirect: Vec<u32>,
    pub(crate) batch: BatchIndices,
}
impl Data {
    pub(crate) fn write(&self, d: &Path, suffix: &str) {
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
    pub(crate) fn run(&mut self, op: &str, g: u32) {
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
/// Persistent native LBVH workspace used by full-frame reference execution.
#[allow(dead_code)]
pub(crate) struct SceneLbvh {
    data: Data,
    extra: std::collections::BTreeMap<String, Vec<u32>>,
}
#[allow(dead_code)]
impl SceneLbvh {
    pub(crate) fn new(batch: BatchIndices, body_storage: usize) -> Self {
        let nb = batch.num_batches as usize;
        let cap = batch.colliders_batch_capacity as usize;
        let guard = 0xdeadbeefu32;
        let aabb = Aabb {
            mins: Vec3::splat(-17.),
            maxs: Vec3::splat(23.),
            padding0: guard,
            padding1: guard,
        };
        let mut node = LbvhNode::default();
        bytemuck::cast_slice_mut::<_, u32>(std::slice::from_mut(&mut node)).fill(guard);
        let mut data = Data {
            pairs: vec![
                CollisionPair {
                    colliders: UVec2::splat(guard)
                };
                batch.collision_pairs_capacity as usize + 3
            ],
            groups: vec![InteractionGroups::all(); body_storage],
            filters: vec![[guard; 2]; body_storage],
            poses: vec![Pose::IDENTITY; body_storage],
            shapes: vec![Shape::ball(0.); body_storage],
            vertices: vec![],
            partials: vec![aabb; 64 * nb + 3],
            domain: vec![aabb; nb + 3],
            keys: vec![guard; cap * nb + 3],
            sorted: vec![guard; cap * nb + 3],
            tree: vec![node; 2 * cap * nb + 3],
            frontier: vec![guard; cap * nb + 3],
            frontier_len: vec![guard; nb + 3],
            rounds: vec![guard; 3],
            pair_count: vec![0; 3],
            indirect: vec![guard; 6],
            batch,
        };
        data.frontier_len[..nb].fill(0);
        let extra = lbvh_program::workspace(&data);
        Self { data, extra }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        poses: &[Pose],
        shapes: &[Shape],
        vertices: &[PaddedVector],
        groups: &[InteractionGroups],
        filters: &[[u32; 2]],
        pairs: &mut [CollisionPair],
        count: &mut [u32],
    ) {
        self.data.poses.clone_from_slice(poses);
        self.data.shapes.clone_from_slice(shapes);
        self.data.groups.clone_from_slice(groups);
        self.data.filters.clone_from_slice(filters);
        self.data.vertices = vertices.to_vec();
        self.data.pairs.clone_from_slice(pairs);
        self.data.pair_count.clone_from_slice(count);
        lbvh_program::execute(&mut self.data, &mut self.extra, 1);
        pairs.clone_from_slice(&self.data.pairs);
        count.clone_from_slice(&self.data.pair_count);
    }
    pub(crate) fn write(&self, directory: &Path, suffix: &str) {
        macro_rules! w {
            ($name:literal, $value:expr) => {
                fs::write(
                    directory.join(format!("Lbvh{}{}.bin", $name, suffix)),
                    bytemuck::cast_slice($value),
                )
                .unwrap();
            };
        }
        w!("Partials", &self.data.partials);
        w!("Domain", &self.data.domain);
        w!("Keys", &self.data.keys);
        w!("Sorted", &self.data.sorted);
        w!("Tree", &self.data.tree);
        w!("Frontier", &self.data.frontier);
        w!("FrontierLen", &self.data.frontier_len);
        w!("Rounds", &self.data.rounds);
        w!("Indirect", &self.data.indirect);
        for (key, value) in &self.extra {
            fs::write(
                directory.join(format!("Lbvh{key}{suffix}.bin")),
                bytemuck::cast_slice(value),
            )
            .unwrap();
        }
    }
}
