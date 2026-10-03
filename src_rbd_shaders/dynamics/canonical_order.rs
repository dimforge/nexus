//! Kernels that put the contacts and constraint lists in a fixed order, so two runs
//! of the same scene give identical results. Only used in deterministic mode.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::broad_phase::ContactPlan;
use crate::dynamics::{MbContactIndexEntry, MultibodyInfo};
use crate::queries::IndexedManifold;
use crate::utils::{BatchIndices, Slice, SliceMut};

const WORKGROUP_SIZE: u32 = 64;

/// Sort key: the sub-shape index within the collider pair.
pub const CONTACT_KEY_SUB_SHAPE: u32 = 0;
/// Sort key: the second collider of the pair.
pub const CONTACT_KEY_COLLIDER_B: u32 = 1;
/// Sort key: the first collider of the pair.
pub const CONTACT_KEY_COLLIDER_A: u32 = 2;
/// Sort key: both colliders in one value, `a << shift | b`, sorted in a single pass.
/// Only usable if both ids fit and there are no composite shapes.
pub const CONTACT_KEY_PAIR: u32 = 3;
/// Flag for the first pass, where contact `i` is still at index `i`.
pub const CONTACT_KEY_IDENTITY: u32 = 4;

/// Builds the `(key, value)` buffers of one pass of the contact sort.
/// Empty slots (`len == 0`) get the largest key so they go last.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_contact_sort_keys(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] in_order: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out_keys: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out_values: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] key_selector: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] n_sort: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let len = contact_plan.bound;
    if invocation_id.x == 0 {
        n_sort.write(0, len);
    }

    let selector = *key_selector;
    let identity = selector >= CONTACT_KEY_IDENTITY;
    let field = selector % CONTACT_KEY_IDENTITY;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        let contact_id = if identity { i } else { in_order.read(idx) };
        let contact = contacts.at(contact_id as usize);
        let key = if contact.contact.len == 0 {
            u32::MAX
        } else if field == CONTACT_KEY_PAIR {
            (contact.colliders.x << batch_ids.contact_sort_collider_shift) | contact.colliders.y
        } else if field == CONTACT_KEY_COLLIDER_A {
            contact.colliders.x
        } else if field == CONTACT_KEY_COLLIDER_B {
            contact.colliders.y
        } else {
            contact.subshape
        };
        out_keys.write(idx, key);
        out_values.write(idx, contact_id);
    }
}

/// Writes the contacts in the order given by `order`. The caller then swaps `src` and `dst`.
/// Empty slots are written as blank manifolds.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_permute_contacts(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] src: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dst: &mut [IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] order: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contact_plan: &ContactPlan,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let len = contact_plan.bound;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        let contact = src.read(order.read(idx) as usize);
        if contact.contact.len == 0 {
            dst.write(idx, IndexedManifold::default());
        } else {
            dst.write(idx, contact);
        }
    }
}

/// Sorts the constraint list of each body by constraint index (the lists are filled with
/// atomics). One workgroup per body, looping if there are more than 65535 bodies.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_stabilize_body_constraint_ids(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(local_invocation_index)] lane: u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    stable_body_constraint_ids: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let body_constraint_counts = Slice(body_constraint_counts, 0);
    let ids = Slice(body_constraint_ids, 0);
    let mut stable_ids = SliceMut(stable_body_constraint_ids, 0);
    let num_bodies = batch_ids.colliders_len * batch_ids.num_batches;

    for body in StepRng::new(workgroup_id.x..num_bodies, num_workgroups.x) {
        let start = if body != 0 {
            body_constraint_counts[body as usize - 1]
        } else {
            0
        };
        let end = body_constraint_counts[body as usize];

        for e in StepRng::new(start + lane..end, WORKGROUP_SIZE) {
            let id = ids.read(e as usize);
            let mut rank = 0u32;

            for q in start..end {
                let id_q = ids.read(q as usize);
                if id_q < id || (id_q == id && q < e) {
                    rank += 1;
                }
            }

            stable_ids.write((start + rank) as usize, id);
        }
    }
}

/// Sorts each multibody contact-index segment by contact slot (they are filled with atomics).
/// One workgroup per `(multibody, batch)`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_stabilize_mb_contact_index(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(local_invocation_index)] lane: u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    mb_contact_index: &[MbContactIndexEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    stable_mb_contact_index: &mut [MbContactIndexEntry],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let total_infos = batch_ids.multibodies_len * batch_ids.num_batches;

    for info in StepRng::new(workgroup_id.x..total_infos, num_workgroups.x) {
        let mb = multibody_info.read(info as usize);
        let start = mb.contact_index_start;
        let end = start + mb.contact_index_len;

        for e in StepRng::new(start + lane..end, WORKGROUP_SIZE) {
            let entry = mb_contact_index.read(e as usize);
            let mut rank = 0u32;

            for q in start..end {
                if mb_contact_index.read(q as usize).contact_slot < entry.contact_slot {
                    rank += 1;
                }
            }

            stable_mb_contact_index.write((start + rank) as usize, entry);
        }
    }
}
