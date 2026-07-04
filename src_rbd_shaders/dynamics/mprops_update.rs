//! Mass properties update compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for mass properties update.

use khal_std::glamx::UVec3;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::Pose;

use super::body::{LocalMassProperties, WorldMassProperties};
use crate::utils::BatchIndices;

const WORKGROUP_SIZE: u32 = 64;

/// Updates world-space mass properties for all rigid bodies.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_mprops(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mprops: &mut [WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] poses: &[Pose],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;

    let num_bodies = batch_ids.bodies_len;
    let mut mprops = batch_ids.coll_batch_mut(batch_id, mprops);
    let local_mprops = batch_ids.coll_batch(batch_id, local_mprops);
    let poses = batch_ids.coll_batch(batch_id, poses);

    for i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let idx = i as usize;
        let new_mprops = local_mprops[idx].to_world(&poses[idx]);
        mprops[idx] = new_mprops;
    }
}

/// Recomputes the world pose of every collider from the body world pose and the
/// collider's body-local offset. Mirrors rapier's `RigidBody → Collider` pose
/// propagation: `collider.position = body.position * collider.position_wrt_parent`.
///
/// Run after each integration substep (and after multibody forward-kinematics)
/// so the broad-phase, narrow-phase and contact-to-constraint pipeline see
/// up-to-date collider world poses.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_sync_collider_poses(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] collider_local_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collider_world_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] collider_parent: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let num_colliders = batch_ids.colliders_len;

    let body_poses = batch_ids.coll_batch(batch_id, body_poses);
    let collider_local_poses = batch_ids.coll_batch(batch_id, collider_local_poses);
    let mut collider_world_poses = batch_ids.coll_batch_mut(batch_id, collider_world_poses);
    let collider_parent = batch_ids.coll_batch(batch_id, collider_parent);

    for i in StepRng::new(invocation_id.x..num_colliders, num_threads) {
        let idx = i as usize;
        let body = collider_parent[idx] as usize;
        collider_world_poses[idx] = body_poses[body] * collider_local_poses[idx];
    }
}
