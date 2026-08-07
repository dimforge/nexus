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
    /// Counter for bottom-up refitting (0, 1, or 2).
    pub refit_count: u32,
}

/// Resets the collision pairs counter.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_lbvh_reset_collision_pairs(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs_len: &mut [u32],
) {
    let batch_id = workgroup_id.y as usize;

    // NOTE: this `for` loop is silly. It doesn’t do anything
    //       more than a `*collision_pairs_len = 0` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        collision_pairs_len.write(batch_id, k);
    }
}

/// Initializes indirect dispatch arguments for narrow phase.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_lbvh_init_dispatch(
    // TODO: take the batch dimension as argument (instead of relying on the len of `collision_pairs_len`)?
    // NOTE: the `collision_pairs_len` is mutable here even though we don’t modify it. That’s
    //       because we access it with an atomic load otherwise it would occasionally read
    //       stale data (on Windows+Nvidia+wgpu backend). This might be caused by:
    //       https://github.com/gfx-rs/wgpu/issues/9221
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
) {
    // For indirect dispatch, get the largest length along all batch dimensions.
    let num_batches = collision_pairs_len.len();
    let mut highest_pairs_len = 0;
    for batch_id in 0..num_batches {
        // NOTE: atomic_load is needed for correctness on some platforms (see comment above `collision_pairs_len`).
        highest_pairs_len =
            highest_pairs_len.max(atomic_load_u32(collision_pairs_len.at_mut(batch_id)));
    }

    *indirect_args.at_mut(0) = highest_pairs_len.div_ceil(WORKGROUP_SIZE);
    *indirect_args.at_mut(1) = num_batches as u32;
    *indirect_args.at_mut(2) = 1;
}

/// Runs a reduction to compute the AABB of the collider positions.
/// Needs to be called with a single workgroup.
#[spirv_bindgen]
#[spirv(compute(threads(128)))]
pub fn gpu_lbvh_compute_domain(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] domain_aabb: &mut [Aabb],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] workspace_mins: &mut [Vector; 128],
    #[spirv(workgroup)] workspace_maxs: &mut [Vector; 128],
) {
    let batch_id = global_id.y;
    let thread_id = global_id.x;
    *workspace_mins.at_mut(thread_id as usize) = Vector::splat(MAX_FLT);
    *workspace_maxs.at_mut(thread_id as usize) = Vector::splat(-MAX_FLT);
    let colliders_start = batch_ids.coll_start(batch_id) as u32;
    let colliders_end = colliders_start + batch_ids.colliders_len;

    for i in StepRng::new(
        colliders_start + thread_id..colliders_end,
        REDUCTION_WORKGROUP_SIZE,
    ) {
        let val_i = poses.at(i as usize).translation;
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
    let colliders_start = batch_ids.coll_start(batch_id) as u32;
    let colliders_end = colliders_start + batch_ids.colliders_len;

    for i in StepRng::new(
        colliders_start + invocation_id.x..colliders_end,
        num_threads,
    ) {
        let center = poses.at(i as usize).translation;
        let normalized = (center - domain_aabb.mins) / (domain_aabb.maxs - domain_aabb.mins);
        let morton_key = morton(normalized);
        morton_keys.write(i as usize, morton_key);
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
    let colliders_start = batch_ids.coll_start(batch_id) as u32;
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
        tree.at_mut(node_id as usize).refit_count = 0; // Might as well reset the refit count here.
        tree.at_mut(left as usize).parent = node_id;
        tree.at_mut(right as usize).parent = node_id;
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
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[PaddedVector],
) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let num_colliders = batch_ids.colliders_len;
    let first_leaf_id = num_colliders - 1;

    // Flat over all batches' leaves: the per-batch collider count is uniform,
    // so batch/index recover by division — no per-batch workgroup rounding
    // (a 14-collider robot env used 14 of 64 lanes per workgroup, and the
    // per-leaf work is mesh-AABB computation, which is genuinely expensive).
    let num_batches = (sorted_colliders.len() / batch_ids.colliders_batch_capacity as usize) as u32;
    let total = num_colliders * num_batches;

    for t in StepRng::new(invocation_id.x..total, num_threads) {
        let batch_id = t / num_colliders;
        let i = t % num_colliders;
        let colliders_start = batch_ids.coll_start(batch_id) as u32;
        let poses = batch_ids.coll_batch(batch_id, poses);
        let shapes = batch_ids.coll_batch(batch_id, shapes);
        let sorted_colliders = batch_ids.coll_batch(batch_id, sorted_colliders);
        let mut tree = SliceMut(&mut *tree, root_id(colliders_start) as usize);
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders[i as usize];
        let leaf_pose = poses[leaf_collider as usize];
        let leaf_shape = &shapes[leaf_collider as usize];

        tree.at_mut(curr_leaf_id as usize).aabb = leaf_shape.compute_aabb(leaf_pose, vertices);
        tree.at_mut(curr_leaf_id as usize).left = leaf_collider;
    }
}

/// Bottom-up AABB propagation using atomic synchronization.
/// This version uses uniform control flow for web compatibility.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_lbvh_refit_internal(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] batch_ids: &BatchIndices,
) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = 256u32;
    let batch_id = workgroup_id.y;
    let colliders_start = batch_ids.coll_start(batch_id) as u32;
    let num_bodies = batch_ids.colliders_len;
    let first_leaf_id = num_bodies - 1;

    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);

    // All threads must execute the same number of outer loop iterations for uniform control flow.
    // NOTE: we calculate the interation count based on `colliders_batch_capacity` instead of
    //       `num_bodies` since the latter is non-uniform because it originates from a storage
    //       buffer.
    let num_iterations = batch_ids.colliders_batch_capacity.div_ceil(num_threads);

    // NOTE: using unchecked indexing (via MaybeIndexUnchecked) because otherwise the bounds
    //        checking inserted by rustgpu breaks the shader when targeting some NVidia graphics
    //        (works fine on AMD).
    for iter in 0..num_iterations {
        let i = local_id.x + iter * num_threads;
        let mut thread_is_active = i < num_bodies;

        let mut curr_id = 0u32;
        if thread_is_active {
            let curr_leaf_id = first_leaf_id + i;
            curr_id = tree.at(curr_leaf_id as usize).parent;
        }

        // Process the tree level by level with uniform barriers.
        // Maximum tree depth is log2(num_colliders), but we use 32 as a safe upper bound.
        for _level in 0..32u32 {
            if thread_is_active {
                let refit_count = atomic_add_u32(&mut tree.at_mut(curr_id as usize).refit_count, 1);

                if refit_count == 0 {
                    // If `refit_count` was 0 then the other thread hasn't reached this node
                    // yet and the sibling aabb might not be available yet.
                    // Stop the propagation to the parents here, the other thread will do it.
                    thread_is_active = false;
                } else {
                    // If `refit_count` was 1 then the other thread has already reached this node
                    // and we know the sibblings aabb is available. So we can continue the propagation.

                    // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
                    //            previous loop so we don't have to re-fetch one of the two aabbs.
                    let left_idx = tree.at(curr_id as usize).left;
                    let right_idx = tree.at(curr_id as usize).right;
                    let left = tree.at(left_idx as usize).aabb;
                    let right = tree.at(right_idx as usize).aabb;
                    tree.at_mut(curr_id as usize).aabb = left.merged(&right);

                    if curr_id == 0 {
                        // We reached the root, can't go higher.
                        thread_is_active = false;
                    } else {
                        curr_id = tree.at(curr_id as usize).parent;
                    }
                }
            }

            // workgroup_memory_barrier_with_group_sync();
            // Barrier ensures all AABB writes (to device/storage buffer memory) are complete
            // before the next iteration's atomics. Uses QueueFamily scope (device-equivalent
            // under the Vulkan memory model) with UNIFORM_MEMORY to cover storage buffers.
            control_barrier::<
                { khal_std::memory::Scope::Workgroup as u32 },
                { khal_std::memory::Scope::QueueFamily as u32 },
                {
                    khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                        | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
                },
            >();
        }
    }
}

/// Full refit: computes leaf AABBs and propagates to ancestors.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_refit(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[PaddedVector],
) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let batch_id = invocation_id.y;
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let colliders_start = batch_ids.coll_start(batch_id) as u32;
    let num_bodies = batch_ids.colliders_len;
    let first_leaf_id = num_bodies - 1;

    let poses = batch_ids.coll_batch(batch_id, poses);
    let shapes = batch_ids.coll_batch(batch_id, shapes);
    let sorted_colliders = batch_ids.coll_batch(batch_id, sorted_colliders);
    let mut tree = SliceMut(tree, root_id(colliders_start) as usize);

    for i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders[i as usize];
        let leaf_pose = poses[leaf_collider as usize];
        let leaf_shape = &shapes[leaf_collider as usize];

        tree.at_mut(curr_leaf_id as usize).aabb = leaf_shape.compute_aabb(leaf_pose, vertices);
        tree.at_mut(curr_leaf_id as usize).left = leaf_collider;

        // Propagate to ancestors.
        let mut curr_id = tree.at(curr_leaf_id as usize).parent;

        loop {
            let refit_count = atomic_add_u32(&mut tree.at_mut(curr_id as usize).refit_count, 1);

            if refit_count == 0 {
                // If `refit_count` was 0 then the other thread hasn't reached this node
                // yet and the sibling aabb might not be available yet.
                // Stop the propagation to the parents here, the other thread will do it.
                break;
            }

            // If `refit_count` was 1 then the other thread has already reached this node,
            // and we know the siblings aabb is available. So we can continue the propagation.

            // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
            //            previous loop so we don't have to re-fetch one of the two aabbs.
            let left_idx = tree.at(curr_id as usize).left;
            let right_idx = tree.at(curr_id as usize).right;
            let left = tree.at(left_idx as usize).aabb;
            let right = tree.at(right_idx as usize).aabb;
            tree.at_mut(curr_id as usize).aabb = left.merged(&right);

            if curr_id == 0 {
                // We reached the root, can't go higher.
                break;
            }

            curr_id = tree.at(curr_id as usize).parent;
        }
    }
}

/// Finds collision pairs by traversing the LBVH tree.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_find_collision_pairs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &[LbvhNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    collision_pairs: &mut [CollisionPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collision_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    collision_groups: &[InteractionGroups],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let colliders_start = batch_ids.coll_start(batch_id) as u32;
    let num_bodies = batch_ids.colliders_len;
    let first_leaf_id = num_bodies - 1;

    let mut collision_pairs = batch_ids.collision_pairs_batch_mut(batch_id, collision_pairs);
    let tree = Slice(tree, root_id(colliders_start) as usize);
    let collision_groups = batch_ids.coll_batch(batch_id, collision_groups);

    for leaf_i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let i = tree.at((first_leaf_id + leaf_i) as usize).left;
        let groups_i = collision_groups[i as usize];
        let mut aabb1 = tree.at((first_leaf_id + leaf_i) as usize).aabb;
        let prediction = 2.0e-3; // TODO: should be configurable.
        let dilation = Vector::splat(prediction);
        aabb1.mins -= dilation;
        aabb1.maxs += dilation;

        // Traverse the tree using a stack.
        let mut stack = [0u32; 64];
        let mut stack_len = 1u32;
        stack.write(0, 0);

        while stack_len != 0 {
            stack_len -= 1;
            let curr_id = stack.read(stack_len as usize);
            let node = tree.at(curr_id as usize);

            if curr_id >= first_leaf_id {
                // We reached a leaf, register a collision pair.
                let j = node.left;
                let groups_j = collision_groups[j as usize];

                // Skip pairs whose collision groups don't authorize an interaction.
                // NOTE: same-body collider pairs are *not* filtered here — that
                //       skip is deferred to the narrow-phase so the broad phase
                //       never has to touch `collider_parent`.
                if !groups_i.test(groups_j) {
                    continue;
                }

                // NOTE: we don't have to compare i < j to avoid duplicates since that comparison already happened
                //       alongside the AABB check.
                let target_pair_index =
                    atomic_add_u32(collision_pairs_len.at_mut(batch_id as usize), 1);

                // NOTE: if the index is out-of-bounds (meaning the `collision_pairs` isn't
                //       big enough), don't write. But keep traversing so we get the exact count we need
                //       for reallocating the buffers.
                if target_pair_index < batch_ids.collision_pairs_batch_capacity {
                    // NOTE: we only store the collider pair here. The parent body
                    //       ids are resolved lazily, at the very last moment, when
                    //       the narrow-phase writes the `IndexedManifold` consumed
                    //       by the solver — keeping this hot buffer (and the
                    //       intermediate pfm-pair buffer) narrow, and keeping
                    //       `collider_parent` out of the broad phase entirely.
                    collision_pairs[target_pair_index as usize] = CollisionPair {
                        colliders: UVec2::new(i, j),
                    };
                }
            } else {
                let left = node.left;
                let right = node.right;

                // Go on the child only if the AABB intersects and either the child isn't a leaf, or it is a leaf with associated collider
                // smaller than `i` (to avoid duplicate pairs).
                if (left < first_leaf_id || i < tree.at(left as usize).left)
                    && aabb1.intersects(&tree.at(left as usize).aabb)
                    && stack_len < 64
                {
                    stack.write(stack_len as usize, node.left);
                    stack_len += 1;
                }

                // NOTE: on leaves (including tree[right]), the collider id is stored as the left child index.
                if (right < first_leaf_id || i < tree.at(right as usize).left)
                    && aabb1.intersects(&tree.at(right as usize).aabb)
                    && stack_len < 64
                {
                    stack.write(stack_len as usize, node.right);
                    stack_len += 1;
                }
            }
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
