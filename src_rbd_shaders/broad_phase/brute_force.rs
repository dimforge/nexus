//! Brute-force O(n²) broad phase for tiny environments.
//!
//! When each batch holds only a handful of colliders (which happens often
//! in single-robot many-batches simulations), the BVH broad-phase is significantly
//! slower than a naive brute-force approach.
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::atomic_add_u32;

use crate::bounding_volumes::Aabb;
use crate::broad_phase::CollisionPair;
use crate::shapes::Shape;
use crate::utils::BatchIndices;
use crate::{PaddedVector, Pose, Vector};
use glamx::UVec2;
use rapier::geometry::InteractionGroups;

use super::narrow_phase::PREDICTION;

/// Computes every active collider's world AABB.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_bf_compute_aabbs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] aabbs: &mut [Aabb],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[PaddedVector],
) {
    let n = batch_ids.colliders_len;
    if invocation_id.x >= n * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / n;
    let i = invocation_id.x % n;

    let poses = batch_ids.coll_batch(batch_id, poses);
    let shapes = batch_ids.coll_batch(batch_id, shapes);
    let out = batch_ids.coll_start(batch_id) + i as usize;
    aabbs.write(out, shapes[i as usize].compute_aabb(poses[i as usize], vertices));
}

/// Tests every collider pair of every batch and appends the intersecting,
/// unfiltered ones to `collision_pairs`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_bf_find_pairs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] aabbs: &[Aabb],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    collision_pairs: &mut [CollisionPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collision_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    collision_groups: &[InteractionGroups],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] pair_filter: &[[u32; 2]],
) {
    let n = batch_ids.colliders_len;
    let nn = n * n;
    if invocation_id.x >= nn * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / nn;
    let r = invocation_id.x % nn;
    let i = r / n;
    let j = r % n;
    if i >= j {
        return;
    }

    let collision_groups = batch_ids.coll_batch(batch_id, collision_groups);
    let pair_filter = batch_ids.coll_batch(batch_id, pair_filter);

    // Skip pairs whose collision groups don't authorize an interaction.
    if !collision_groups[i as usize].test(collision_groups[j as usize]) {
        return;
    }

    // Built-in filters (same-body and adjacent-multibody-links).
    let filter_i = pair_filter[i as usize];
    let filter_j = pair_filter[j as usize];
    if filter_i[0] == filter_j[0] || (filter_i[1] != 0 && filter_i[1] == filter_j[1]) {
        return;
    }

    let coll_start = batch_ids.coll_start(batch_id);
    // Dilate one side by the contact prediction distance.
    let mut aabb_i = aabbs.read(coll_start + i as usize);
    let dilation = Vector::splat(PREDICTION);
    aabb_i.mins -= dilation;
    aabb_i.maxs += dilation;
    let aabb_j = aabbs.read(coll_start + j as usize);
    if !aabb_i.intersects(&aabb_j) {
        return;
    }

    let target_pair_index = atomic_add_u32(collision_pairs_len.at_mut(batch_id as usize), 1);

    // If we exceed capacity, keep counting the pairs but don’t store any more to avoid overflow.
    if target_pair_index < batch_ids.collision_pairs_batch_capacity {
        let mut collision_pairs = batch_ids.collision_pairs_batch_mut(batch_id, collision_pairs);
        collision_pairs[target_pair_index as usize] = CollisionPair {
            colliders: UVec2::new(i, j),
        };
    }
}
