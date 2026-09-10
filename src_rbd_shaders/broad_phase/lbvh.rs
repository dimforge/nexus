//! Linear Bounding Volume Hierarchy (LBVH)
//!
//! GPU-based LBVH construction and traversal for broad-phase collision
//! detection, based on the Karras algorithm:
//! <https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf>.
//! O(n log n) construction and query.

use crate::bounding_volumes::Aabb;
use crate::broad_phase::CollisionPair;
use crate::shapes::Shape;
use crate::utils::{BatchIndices, Slice, SliceMut, div_ceil};
use crate::{MAX_FLT, PaddedVector, Pose, Vector};
use glamx::UVec2;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::{
    atomic_add_u32, atomic_load_u32, control_barrier, workgroup_memory_barrier_with_group_sync,
};
use rapier::geometry::InteractionGroups;

const WORKGROUP_SIZE: u32 = 64;
const REDUCTION_WORKGROUP_SIZE: u32 = 128;

/// A node in the Linear BVH tree.
///
/// The tree has n-1 internal nodes (indices `[0..n-1[`) and n leaf nodes
/// (indices `[n-1..2n-1[`). To store multiple trees in the same buffer (batch
/// dimensions), 2n node slots are reserved per tree (the last is unused) so the
/// tree's start offset depends only on the total node count.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct LbvhNode {
    /// Axis-aligned bounding box for this node's subtree.
    pub aabb: Aabb,
    /// Left child index (internal) or collider index (leaf).
    pub left: u32,
    /// Right child index (internal nodes only).
    pub right: u32,
    /// Parent node index.
    pub parent: u32,
    /// Bottom-up refit arrival counter (internal nodes): each child's thread atomically
    /// increments it and only the second one continues upward (both children ready).
    pub refit_count: u32,
    /// The first sorted leaf index in this node's subtree.
    pub first_leaf: u32,
    /// The last sorted leaf index in this node's subtree. The pair traversal uses it to prune
    /// subtrees that can only produce duplicate pairs.
    pub last_leaf: u32,
    /// The node following this one's subtree in depth-first order ([`NO_ESCAPE`] for the
    /// last one), for stackless traversals.
    pub escape: u32,
    /// The largest AABB extent of the leaves in this node's subtree.
    pub max_extent: f32,
}

/// The largest extent of an AABB.
#[inline(always)]
fn max_extent(aabb: &Aabb) -> f32 {
    let e = aabb.maxs - aabb.mins;
    #[cfg(feature = "dim3")]
    return e.x.max(e.y).max(e.z);
    #[cfg(feature = "dim2")]
    return e.x.max(e.y);
}

/// The escape index of the nodes ending a depth-first traversal.
pub const NO_ESCAPE: u32 = u32::MAX;

/// Resets the (single, global) collision pairs counter.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_reset_collision_pairs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs_len: &mut [u32],
) {
    let i = invocation_id.x as usize;
    if i < collision_pairs_len.len() {
        collision_pairs_len.write(i, 0);
    }
}

/// Writes the flat 1-D indirect grid for a kernel iterating a flat work-list
/// (collision pairs or PFM pairs): `[ceil(min(len, capacity) / 64), 1, 1]`.
///
/// NOTE: the load must be atomic or it occasionally reads stale data (breaks
/// Windows+Nvidia+wgpu, see <https://github.com/gfx-rs/wgpu/issues/9221>).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_flat_list_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    // Loop shell per the `gpu_reset_completion_flag_topo_gc` rustgpu-triviality
    // workaround (too-trivial kernels don't get their spirv generated).
    for _ in 0..1 {
        let total = atomic_load_u32(len.at_mut(0)).min(batch_ids.collision_pairs_capacity);
        *indirect_args.at_mut(0) = total.div_ceil(WORKGROUP_SIZE);
        *indirect_args.at_mut(1) = 1;
        *indirect_args.at_mut(2) = 1;
    }
}

/// Most workgroups per batch of [`gpu_lbvh_compute_domain`].
pub const DOMAIN_WORKGROUPS: u32 = 64;

/// Min/max reduction of the collider positions, first pass: each of the batch's workgroups
/// reduces a share of them into `partials`, merged by [`gpu_lbvh_domain_merge`].
#[spirv_bindgen]
#[spirv(compute(threads(128)))]
pub fn gpu_lbvh_compute_domain(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] partials: &mut [Aabb],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] workspace_mins: &mut [Vector; 128],
    #[spirv(workgroup)] workspace_maxs: &mut [Vector; 128],
) {
    let batch_id = workgroup_id.y;
    let thread_id = local_id.x;
    let group = workgroup_id.x;
    let num_groups = num_workgroups.x.min(DOMAIN_WORKGROUPS);
    let first = group * REDUCTION_WORKGROUP_SIZE + thread_id;
    let stride = num_groups * REDUCTION_WORKGROUP_SIZE;
    *workspace_mins.at_mut(thread_id as usize) = Vector::splat(MAX_FLT);
    *workspace_maxs.at_mut(thread_id as usize) = Vector::splat(-MAX_FLT);

    for i in StepRng::new(first..batch_ids.colliders_len, stride) {
        let val_i = poses.at(batch_ids.body_global(batch_id, i)).translation;
        *workspace_mins.at_mut(thread_id as usize) =
            workspace_mins.at(thread_id as usize).min(val_i);
        *workspace_maxs.at_mut(thread_id as usize) =
            workspace_maxs.at(thread_id as usize).max(val_i);
    }

    workgroup_memory_barrier_with_group_sync();

    // Reduction steps
    macro_rules! step_reduce(
        ($stride: expr) => {
            if thread_id < $stride {
                *workspace_mins.at_mut(thread_id as usize) = workspace_mins.at(thread_id as usize)
                    .min(*workspace_mins.at((thread_id + $stride) as usize));
                *workspace_maxs.at_mut(thread_id as usize) = workspace_maxs.at(thread_id as usize)
                    .max(*workspace_maxs.at((thread_id + $stride) as usize));
            }
            workgroup_memory_barrier_with_group_sync();
        }
    );
    step_reduce!(64);
    step_reduce!(32);
    step_reduce!(16);
    step_reduce!(8);
    step_reduce!(4);
    step_reduce!(2);
    step_reduce!(1);

    let base = (batch_id * DOMAIN_WORKGROUPS) as usize;
    if thread_id == 0 && group < num_groups {
        partials.at_mut(base + group as usize).mins = *workspace_mins.at(0);
        partials.at_mut(base + group as usize).maxs = *workspace_maxs.at(0);
    }
    // The first workgroup empties the slots no workgroup fills.
    if group == 0 && thread_id >= num_groups && thread_id < DOMAIN_WORKGROUPS {
        partials.at_mut(base + thread_id as usize).mins = Vector::splat(MAX_FLT);
        partials.at_mut(base + thread_id as usize).maxs = Vector::splat(-MAX_FLT);
    }
}

/// Min/max reduction of the collider positions, second pass: one workgroup per batch merges
/// the partial domains of [`gpu_lbvh_compute_domain`].
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_domain_merge(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partials: &[Aabb],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] domain_aabb: &mut [Aabb],
    #[spirv(workgroup)] workspace_mins: &mut [Vector; 64],
    #[spirv(workgroup)] workspace_maxs: &mut [Vector; 64],
) {
    let batch_id = workgroup_id.y;
    let thread_id = local_id.x;
    let partial = partials.at((batch_id * DOMAIN_WORKGROUPS + thread_id) as usize);
    *workspace_mins.at_mut(thread_id as usize) = partial.mins;
    *workspace_maxs.at_mut(thread_id as usize) = partial.maxs;
    workgroup_memory_barrier_with_group_sync();

    macro_rules! step_reduce(
        ($stride: expr) => {
            if thread_id < $stride {
                *workspace_mins.at_mut(thread_id as usize) = workspace_mins.at(thread_id as usize)
                    .min(*workspace_mins.at((thread_id + $stride) as usize));
                *workspace_maxs.at_mut(thread_id as usize) = workspace_maxs.at(thread_id as usize)
                    .max(*workspace_maxs.at((thread_id + $stride) as usize));
            }
            workgroup_memory_barrier_with_group_sync();
        }
    );
    step_reduce!(32);
    step_reduce!(16);
    step_reduce!(8);
    step_reduce!(4);
    step_reduce!(2);
    step_reduce!(1);

    if thread_id == 0 {
        domain_aabb.at_mut(batch_id as usize).mins = *workspace_mins.at(0);
        domain_aabb.at_mut(batch_id as usize).maxs = *workspace_maxs.at(0);
    }
}

/// Computes Morton codes for all colliders.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_compute_morton(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] domain_aabb: &[Aabb],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] morton_keys: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    // NOTE: for simplicity we compute the morton key of the collider position instead of
    //       the collider shape's AABB center. We might want to revisit that in the future
    //       once we start adding more complex shapes.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let domain_aabb = domain_aabb.read(batch_id as usize);
    let scratch = scratch_start(batch_ids, batch_id);

    for i in StepRng::new(invocation_id.x..batch_ids.colliders_len, num_threads) {
        let center = poses.at(batch_ids.body_global(batch_id, i)).translation;
        let normalized = (center - domain_aabb.mins) / (domain_aabb.maxs - domain_aabb.mins);
        let morton_key = morton(normalized);
        morton_keys.write((scratch + i) as usize, morton_key);
    }
}

/// Builds each node of the tree in parallel.
///
/// This only computes the tree topology (children and parent pointers).
/// This doesn't update the bounding boxes. Call `refit` for updating bounding boxes!
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_build(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] morton_keys: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let colliders_start = scratch_start(batch_ids, batch_id);
    let num_bodies = batch_ids.colliders_len;
    let num_internal_nodes = num_bodies - 1;
    let first_leaf_id = num_internal_nodes;

    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);
    let morton_keys = Slice(morton_keys, colliders_start as usize);

    for i in StepRng::new(invocation_id.x..num_internal_nodes, num_threads) {
        // Determine the direction of the range (+1 or -1).
        let ii = i as i32;
        let curr_key = morton_keys.read(i as usize);
        let diff = prefix_len(curr_key, ii, ii + 1, num_bodies, &morton_keys)
            - prefix_len(curr_key, ii, ii - 1, num_bodies, &morton_keys);
        let d = if diff > 0 {
            1
        } else if diff < 0 {
            -1
        } else {
            0
        }; // Not using `diff.signum()` since it fail spirv compilation without `Int8` capabilities.

        // Compute upper bound for the length of the range.
        let delta_min = prefix_len(curr_key, ii, ii - d, num_bodies, &morton_keys);
        let mut lmax = 2;

        for _ in 0..32u32 {
            if prefix_len(curr_key, ii, ii + lmax * d, num_bodies, &morton_keys) > delta_min {
                lmax *= 2;
            } else {
                break;
            }
        }

        // Find the other end using binary search.
        let mut l = 0;
        let mut t = lmax / 2;

        // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
        //       Running up to 32 loops is always correct since we can’t have more than 2^30 morton
        //       keys.
        for _ in 0..32u32 {
            if t < 1 {
                break;
            }
            if prefix_len(curr_key, ii, ii + (l + t) * d, num_bodies, &morton_keys) > delta_min {
                l += t;
            }
            t /= 2;
        }
        let j = ii + l * d;

        // Find the split position using binary search.
        let delta_node = prefix_len(curr_key, ii, j, num_bodies, &morton_keys);
        let mut s = 0;
        let mut t = div_ceil(l, 2);

        for _ in 0..32u32 {
            if t < 1 {
                break;
            }
            if prefix_len(curr_key, ii, ii + (s + t) * d, num_bodies, &morton_keys) > delta_node {
                s += t;
            }
            t = div_ceil(t, 2);
        }

        let gamma = ii + s * d + d.min(0);

        // Output child and parent pointers.
        let left = if ii.min(j) == gamma {
            first_leaf_id as i32 + gamma
        } else {
            gamma
        };
        let right = if ii.max(j) == gamma + 1 {
            first_leaf_id as i32 + gamma + 1
        } else {
            gamma + 1
        };
        let node_id = i;

        tree.at_mut(node_id as usize).left = left as u32;
        tree.at_mut(node_id as usize).right = right as u32;
        tree.at_mut(node_id as usize).refit_count = 0;
        tree.at_mut(node_id as usize).first_leaf = ii.min(j) as u32;
        tree.at_mut(node_id as usize).last_leaf = ii.max(j) as u32;
        tree.at_mut(left as usize).parent = node_id;
        tree.at_mut(right as usize).parent = node_id;
    }
}

/// Computes the escape index of every node: the right sibling of its first ancestor (itself
/// included) that is a left child.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_escapes(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let colliders_start = scratch_start(batch_ids, batch_id);
    let num_nodes = (2 * batch_ids.colliders_len).max(2) - 1;
    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);

    for i in StepRng::new(invocation_id.x..num_nodes, num_threads) {
        let mut escape = NO_ESCAPE;
        let mut node = i;
        for _ in 0..REFIT_MAX_DEPTH {
            if node == 0 {
                break;
            }
            let parent = tree.at(node as usize).parent;
            if tree.at(parent as usize).left == node {
                escape = tree.at(parent as usize).right;
                break;
            }
            node = parent;
        }
        tree.at_mut(i as usize).escape = escape;
    }
}

/// Computes leaf AABBs from shapes.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_refit_leaves(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] refit_frontier_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[PaddedVector],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let colliders_start = scratch_start(batch_ids, batch_id);
    let num_colliders = batch_ids.colliders_len;
    let first_leaf_id = num_colliders - 1;

    let poses = batch_ids.ib(batch_id, poses);
    let shapes = batch_ids.ib(batch_id, shapes);
    let sorted_colliders = Slice(sorted_colliders, colliders_start as usize);
    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);

    for i in StepRng::new(invocation_id.x..num_colliders, num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders[i as usize];
        let leaf_pose = poses[leaf_collider as usize];
        let leaf_shape = &shapes[leaf_collider as usize];

        let aabb = leaf_shape.compute_aabb(leaf_pose, vertices);
        tree.at_mut(curr_leaf_id as usize).aabb = aabb;
        tree.at_mut(curr_leaf_id as usize).max_extent = max_extent(&aabb);
        tree.at_mut(curr_leaf_id as usize).left = leaf_collider;
        tree.at_mut(curr_leaf_id as usize).first_leaf = i;
        tree.at_mut(curr_leaf_id as usize).last_leaf = i;
    }

    // The bottom-up refit appends to this batch's frontier list.
    if invocation_id.x == 0 {
        refit_frontier_len.write(batch_id as usize, 0);
    }
}

/// Number of sorted leaves refit by one workgroup of [`gpu_lbvh_refit_chunks`].
pub const REFIT_CHUNK: u32 = 256;
/// Bound on the depth of the tree walked by the bottom-up refits.
const REFIT_MAX_DEPTH: u32 = 64;

/// Workgroup barrier making storage writes of the workgroup visible to its other lanes.
#[inline(always)]
fn refit_barrier() {
    control_barrier::<
        { khal_std::memory::Scope::Workgroup as u32 },
        { khal_std::memory::Scope::QueueFamily as u32 },
        {
            khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
        },
    >();
}

/// Whether any lane of the workgroup is still `active`. Must be reached by every lane.
#[cfg(not(feature = "web-compat"))]
#[inline(always)]
fn any_lane_active(lane: u32, active: bool, flag: &mut [u32; 1]) -> bool {
    if lane == 0 {
        flag.write(0, 0);
    }
    workgroup_memory_barrier_with_group_sync();
    if active {
        // Every active lane stores the same value.
        flag.write(0, 1);
    }
    workgroup_memory_barrier_with_group_sync();
    let any = flag.read(0) != 0;
    // Lane 0 resets the flag at the next call: keep it from racing with these reads.
    workgroup_memory_barrier_with_group_sync();
    any
}

/// Merges the AABBs of `node`'s children into its own.
#[inline(always)]
fn refit_node(tree: &mut SliceMut<LbvhNode>, node: u32) {
    let left = tree.at(node as usize).left;
    let right = tree.at(node as usize).right;
    let aabb = tree
        .at(left as usize)
        .aabb
        .merged(&tree.at(right as usize).aabb);
    let extent = tree
        .at(left as usize)
        .max_extent
        .max(tree.at(right as usize).max_extent);
    tree.at_mut(node as usize).aabb = aabb;
    tree.at_mut(node as usize).max_extent = extent;
}

/// First phase of the bottom-up refit: one workgroup per chunk of [`REFIT_CHUNK`] sorted
/// leaves refits every node whose leaves all lie in its chunk.
///
/// Both children of such a node are refit by lanes of the same workgroup, so the usual
/// arrival counter only needs workgroup barriers. A lane stops at a node whose parent's leaves
/// cross the chunk boundary, and appends it to the batch's frontier list, which
/// [`gpu_lbvh_refit_frontier`] finishes.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_lbvh_refit_chunks(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &mut [LbvhNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] refit_frontier: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] refit_frontier_len: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] any_active_flag: &mut [u32; 1],
) {
    let lane = local_id.x;
    #[cfg(feature = "web-compat")]
    let _ = any_active_flag;
    let chunk = workgroup_id.x;
    let batch_id = workgroup_id.y;
    let colliders_start = scratch_start(batch_ids, batch_id);
    let num_leaves = batch_ids.colliders_len;
    if num_leaves < 2 {
        return;
    }
    // Uniform across the workgroup, but derived from `workgroup_id`, which the WGSL uniformity
    // analysis doesn't trust: on the web, out-of-range chunks run the loop inactive instead.
    #[cfg(not(feature = "web-compat"))]
    if chunk * REFIT_CHUNK >= num_leaves {
        return;
    }
    let first_leaf_id = num_leaves - 1;
    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);

    let leaf = chunk * REFIT_CHUNK + lane;
    let mut active = leaf < num_leaves;
    let mut child = first_leaf_id + leaf;
    let mut node = 0u32;
    if active {
        node = tree.at(child as usize).parent;
    }

    for _ in 0..REFIT_MAX_DEPTH {
        // On the web, every level runs: the early exit reads workgroup memory, which the WGSL
        // uniformity analysis can't prove uniform (no `workgroupUniformLoad` in rust-gpu).
        #[cfg(not(feature = "web-compat"))]
        if !any_lane_active(lane, active, any_active_flag) {
            break;
        }

        if active {
            let first = tree.at(node as usize).first_leaf;
            let last = tree.at(node as usize).last_leaf;
            if first / REFIT_CHUNK != chunk || last / REFIT_CHUNK != chunk {
                // `child` is complete but its parent spans several chunks.
                let k = atomic_add_u32(refit_frontier_len.at_mut(batch_id as usize), 1);
                refit_frontier.write((colliders_start + k) as usize, child);
                active = false;
            } else if atomic_add_u32(&mut tree.at_mut(node as usize).refit_count, 1) == 0 {
                // The sibling isn't ready: its lane continues when it arrives.
                active = false;
            } else {
                refit_node(&mut tree, node);
                if node == 0 {
                    // The root: the whole tree fits in this chunk.
                    active = false;
                } else {
                    child = node;
                    node = tree.at(node as usize).parent;
                }
            }
        }

        refit_barrier();
    }
}

/// Reduce the actual frontier sizes to a uniform loop bound for the web refit.
/// A storage load indexed by `workgroup_id` does not pass WGSL uniformity analysis;
/// a separate uniform buffer does, without iterating over every possible leaf.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_lbvh_refit_plan(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] frontier_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rounds: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let mut largest = 0;
    for batch in 0..batch_ids.num_batches {
        largest = largest.max(
            frontier_len
                .read(batch as usize)
                .min(batch_ids.colliders_len),
        );
    }
    *rounds = largest.div_ceil(REFIT_CHUNK);
}

/// Second phase of the bottom-up refit: one workgroup per batch refits the nodes spanning
/// several chunks, starting from the frontier left by [`gpu_lbvh_refit_chunks`].
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_lbvh_refit_frontier(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &mut [LbvhNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] refit_frontier: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] refit_frontier_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] refit_rounds: &u32,
    #[spirv(workgroup)] any_active_flag: &mut [u32; 1],
) {
    let lane = local_id.x;
    #[cfg(feature = "web-compat")]
    let _ = any_active_flag;
    let batch_id = workgroup_id.y;
    let colliders_start = scratch_start(batch_ids, batch_id);
    let num_leaves = batch_ids.colliders_len;
    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);
    // Uniform across the workgroup: every lane reads the same count.
    let frontier_len = refit_frontier_len.read(batch_id as usize).min(num_leaves);
    // The web bound is the largest actual frontier over the batches, uploaded by
    // the preceding GPU planning pass. Every lane reaches the same barriers.
    #[cfg(not(feature = "web-compat"))]
    let num_rounds = frontier_len.div_ceil(REFIT_CHUNK);
    #[cfg(feature = "web-compat")]
    let num_rounds = *refit_rounds;
    #[cfg(not(feature = "web-compat"))]
    let _ = refit_rounds;

    for round in 0..num_rounds {
        let k = round * REFIT_CHUNK + lane;
        let mut active = k < frontier_len;
        let mut node = 0u32;
        if active {
            let child = refit_frontier.read((colliders_start + k) as usize);
            node = tree.at(child as usize).parent;
        }

        for _ in 0..REFIT_MAX_DEPTH {
            // See `gpu_lbvh_refit_chunks`: every level runs on the web.
            #[cfg(not(feature = "web-compat"))]
            if !any_lane_active(lane, active, any_active_flag) {
                break;
            }

            if active {
                if atomic_add_u32(&mut tree.at_mut(node as usize).refit_count, 1) == 0 {
                    active = false;
                } else {
                    refit_node(&mut tree, node);
                    if node == 0 {
                        // The root.
                        active = false;
                    } else {
                        node = tree.at(node as usize).parent;
                    }
                }
            }

            refit_barrier();
        }
    }
}

/// `if c { a } else { b }` as a select instead of a branch.
#[inline(always)]
fn select_u32(c: bool, a: u32, b: u32) -> u32 {
    let mask = 0u32.wrapping_sub(c as u32);
    (a & mask) | (b & !mask)
}

/// Pairs each thread of [`gpu_lbvh_find_collision_pairs`] buffers in workgroup memory.
const PAIRS_PER_THREAD: usize = 8;

/// Finds collision pairs by traversing the LBVH tree.
///
/// A pair belongs to its smaller leaf when one leaf is more than twice the other's size, and to
/// its first sorted leaf otherwise: a large leaf (e.g. the ground) then prunes the tree right
/// away instead of visiting every small leaf it overlaps on a single thread.
///
/// Each workgroup gathers its pairs in workgroup memory and reserves their range of the pair
/// buffer with one atomic, so a workgroup's pairs (of nearby leaves) stay contiguous.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_find_collision_pairs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &[LbvhNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    collision_pairs: &mut [CollisionPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collision_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    collision_groups: &[InteractionGroups],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] pair_filter: &[[u32; 2]],
    #[spirv(workgroup)] wg_pairs: &mut [UVec2; PAIRS_PER_THREAD * WORKGROUP_SIZE as usize],
    #[spirv(workgroup)] wg_scan: &mut [u32; WORKGROUP_SIZE as usize],
    #[spirv(workgroup)] wg_base: &mut [u32; 1],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let tid = local_id.x as usize;
    let mut num_buffered = 0usize;
    let batch_id = invocation_id.y;
    let colliders_start = scratch_start(batch_ids, batch_id);
    let num_bodies = batch_ids.colliders_len;
    let first_leaf_id = num_bodies - 1;

    let tree = Slice(tree, root_id(colliders_start) as usize);
    let collision_groups = batch_ids.ib(batch_id, collision_groups);
    let pair_filter = batch_ids.ib(batch_id, pair_filter);

    for leaf_i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let i = tree.at((first_leaf_id + leaf_i) as usize).left;
        let groups_i = collision_groups[i as usize];
        let filter_i = pair_filter[i as usize];
        let mut aabb1 = tree.at((first_leaf_id + leaf_i) as usize).aabb;
        let extent1 = max_extent(&aabb1);
        let prediction = 2.0e-3; // TODO: should be configurable.
        let dilation = Vector::splat(prediction);
        aabb1.mins -= dilation;
        aabb1.maxs += dilation;

        // Stackless depth-first traversal: a node's escape index skips its subtree.
        let mut curr_id = 0u32;
        // NOTE: we use a fixed-size for loop to avoid miscompilation issues of
        //       while loops on MacOs. Each node is visited at most once per
        //       traversal, so `2 * num_bodies` (≥ node count) bounds the loop.
        for _ in 0..2 * num_bodies {
            if curr_id == NO_ESCAPE {
                break;
            }
            let node = tree.at(curr_id as usize);
            let is_leaf = curr_id >= first_leaf_id;

            // Only subtrees with a leaf this one owns a pair with: a much larger leaf, or a
            // similar one after `leaf_i`. For a leaf, that's whether it owns the pair. The next
            // node is selected without branching, so the node's fields load at once.
            let owned = (node.max_extent > 2.0 * extent1)
                | ((leaf_i < node.last_leaf) & (node.max_extent >= 0.5 * extent1));
            let hit = owned & aabb1.intersects(&node.aabb);
            let descend = hit & !is_leaf;
            curr_id = select_u32(descend, node.left, node.escape);
            if !(hit & is_leaf) {
                continue;
            }

            // We reached a leaf, register a collision pair.
            let j = node.left;
            let groups_j = collision_groups[j as usize];

            // Skip pairs whose collision groups don't authorize an interaction.
            if !groups_i.test(groups_j) {
                continue;
            }

            // Apply computed filters (same-body and self-contact).
            let filter_j = pair_filter[j as usize];
            if filter_i[0] == filter_j[0] || (filter_i[1] != 0 && filter_i[1] == filter_j[1]) {
                continue;
            }

            let (ci, cj) = if i < j { (i, j) } else { (j, i) };
            let pair = UVec2::new(
                batch_ids.body_global(batch_id, ci) as u32,
                batch_ids.body_global(batch_id, cj) as u32,
            );
            // Slot-major, so that consecutive threads hit consecutive words.
            if num_buffered < PAIRS_PER_THREAD {
                wg_pairs.write(num_buffered * WORKGROUP_SIZE as usize + tid, pair);
                num_buffered += 1;
                continue;
            }

            // Single global counter: every batch appends to the same flat
            // buffer (pairs from different batches interleave freely).
            let target_pair_index = atomic_add_u32(collision_pairs_len.at_mut(0), 1);

            // NOTE: if the index is out-of-bounds (meaning the `collision_pairs` isn't
            //       big enough), don't write. But keep traversing so we get the exact count we need
            //       for reallocating the buffers.
            if target_pair_index < batch_ids.collision_pairs_capacity {
                // NOTE: we only store the collider pair here, with global
                //       collider ids (the owning batch is recovered from the
                //       id downstream). The parent body ids are resolved
                //       lazily, at the very last moment, when the
                //       narrow-phase writes the `IndexedManifold` consumed
                //       by the solver — keeping this hot buffer (and the
                //       intermediate pfm-pair buffer) narrow, and keeping
                //       `collider_parent` out of the broad phase entirely.
                collision_pairs.write(
                    target_pair_index as usize,
                    CollisionPair { colliders: pair },
                );
            }
        }
    }

    // Offsets of the buffered pairs in the workgroup's range (inclusive Hillis-Steele scan).
    wg_scan.write(tid, num_buffered as u32);
    let mut offset = 1usize;
    for _ in 0..6u32 {
        workgroup_memory_barrier_with_group_sync();
        let v = if tid >= offset {
            wg_scan.read(tid - offset)
        } else {
            0
        };
        workgroup_memory_barrier_with_group_sync();
        let s = wg_scan.read(tid) + v;
        wg_scan.write(tid, s);
        offset *= 2;
    }
    workgroup_memory_barrier_with_group_sync();
    if tid == WORKGROUP_SIZE as usize - 1 {
        let total = wg_scan.read(tid);
        wg_base.write(0, atomic_add_u32(collision_pairs_len.at_mut(0), total));
    }
    workgroup_memory_barrier_with_group_sync();

    let first = wg_base.read(0) + wg_scan.read(tid) - num_buffered as u32;
    for k in 0..PAIRS_PER_THREAD {
        let target_pair_index = first + k as u32;
        if k < num_buffered && target_pair_index < batch_ids.collision_pairs_capacity {
            collision_pairs.write(
                target_pair_index as usize,
                CollisionPair {
                    colliders: wg_pairs.read(k * WORKGROUP_SIZE as usize + tid),
                },
            );
        }
    }
}

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit (3D).
#[cfg(feature = "dim3")]
pub fn expand_bits_3d(v: u32) -> u32 {
    let mut vv = v.wrapping_mul(0x00010001) & 0xFF0000FF;
    vv = vv.wrapping_mul(0x00000101) & 0x0F00F00F;
    vv = vv.wrapping_mul(0x00000011) & 0xC30C30C3;
    vv = vv.wrapping_mul(0x00000005) & 0x49249249;
    vv
}

/// Calculates a 30-bit Morton code for the given 3D point located within the unit cube \[0,1\].
#[cfg(feature = "dim3")]
pub fn morton_3d(v: Vector) -> u32 {
    let scaled_x = v.x.clamp(0.0, 1023.0 / 1024.0) * 1024.0;
    let scaled_y = v.y.clamp(0.0, 1023.0 / 1024.0) * 1024.0;
    let scaled_z = v.z.clamp(0.0, 1023.0 / 1024.0) * 1024.0;
    let xx = expand_bits_3d(scaled_x as u32);
    let yy = expand_bits_3d(scaled_y as u32);
    let zz = expand_bits_3d(scaled_z as u32);
    xx * 4 + yy * 2 + zz
}

/// Expands a 16-bit integer into 32 bits by inserting 1 zero after each bit (2D).
#[cfg(feature = "dim2")]
pub fn expand_bits_2d(v: u32) -> u32 {
    let mut x = v & 0x0000ffff;
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Calculates a 32-bit Morton code for the given 2D point located within the unit square \[0,1\].
#[cfg(feature = "dim2")]
pub fn morton_2d(v: Vector) -> u32 {
    let scaled_x = (v.x * 65536.0).clamp(0.0, 65535.0);
    let scaled_y = (v.y * 65536.0).clamp(0.0, 65535.0);
    let xx = expand_bits_2d(scaled_x as u32);
    let yy = expand_bits_2d(scaled_y as u32);
    xx | (yy << 1)
}

/// Calculates a Morton code for the given point located within the unit hypercube \[0,1\].
#[cfg(feature = "dim2")]
pub fn morton(v: Vector) -> u32 {
    morton_2d(v)
}

/// Calculates a Morton code for the given point located within the unit hypercube \[0,1\].
#[cfg(feature = "dim3")]
pub fn morton(v: Vector) -> u32 {
    morton_3d(v)
}

/// Computes the common prefix length between two Morton keys.
pub fn prefix_len(
    curr_key: u32,
    curr_index: i32,
    other_index: i32,
    num_colliders: u32,
    morton_keys: &Slice<u32>,
) -> i32 {
    if other_index < 0 || other_index > num_colliders as i32 - 1 {
        return -1;
    }

    let other_key = morton_keys.read(other_index as usize);
    let morton_prefix_len = ((curr_key as i32) ^ (other_key as i32)).leading_zeros() as i32;
    // Fallback to indices if the morton keys are equal.
    let fallback_prefix_len = 32 + (curr_index ^ other_index).leading_zeros() as i32;
    if curr_key != other_key {
        morton_prefix_len
    } else {
        fallback_prefix_len
    }
}

/// Start of `batch_id`'s segment in the broad-phase internal scratch buffers
/// (morton keys, sorted collider ids, tree, brute-force AABBs). These stay
/// batch-major: the radix sort works on contiguous per-batch key segments.
/// Only the shared per-collider state (poses, shapes, groups, filters) is
/// batch-interleaved.
#[inline]
pub fn scratch_start(batch_ids: &BatchIndices, batch_id: u32) -> u32 {
    batch_id * batch_ids.colliders_batch_capacity
}

fn root_id(collider_start_id: u32) -> u32 {
    // Every LBVH tree contains `n - 1` internal nodes and `n` leaves, where
    // `n` is its number of colliders. This is a total of `2n - 1`, but to
    // simplify calculations we allocate `2n` nodes per tree.
    //
    // Before the batch dimension with collider id starting at `collider_start_id`,
    // there are `collider_start_id` colliders for other batch dimensions, so they
    // require a total of `2 * colliders_start_id` nodes for their LBVH; so the root
    // of the current LBVH is `2 * collider_start_id`.
    //
    // NOTE: if we allocated `2n - 1` node per LBVH instead of `2n`, then the root
    //       id for the current LBVH would be `2n - b` where `b` is the current batch
    //       id. We don’t do this for the simplicity of not having to deal with the
    //       `- b`.
    collider_start_id * 2
}

#[cfg(all(test, not(target_arch_is_gpu)))]
mod tests {
    use super::*;

    #[test]
    fn web_refit_plan_covers_all_batches_and_clamps_stale_counts() {
        for (counts, leaves, expected) in [
            (vec![], 513, 0),
            (vec![0, 0, 0], 513, 0),
            (vec![1, 256, 0], 513, 1),
            (vec![0, 257, 512], 513, 2),
            (vec![513, 1, 0], 513, 3),
            (vec![u32::MAX, 1], 256, 1),
        ] {
            let mut rounds = 999;
            gpu_lbvh_refit_plan(
                &counts,
                &mut rounds,
                &BatchIndices {
                    num_batches: counts.len() as u32,
                    colliders_len: leaves,
                    ..Default::default()
                },
            );
            assert_eq!(rounds, expected);
        }
    }
}
