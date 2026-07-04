//! Constraint warmstarting (impulse caching across frames).
//!
//! Transfers impulses from frame `n-1` to frame `n` as initial guesses for the solver.
//! Contacts are matched by proximity in local coordinates (threshold: 10cm).

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use super::constraint::{TwoBodyConstraint, TwoBodyConstraintBuilder};
use crate::utils::{BatchIndices, Slice, SliceMut};
use khal_std::index::MaybeIndexUnchecked;

/// Transfers warmstart impulses from previous frame to current frame.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_transfer_warmstart_impulses(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] old_body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] old_body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    old_constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    old_constraint_builders: &[TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    new_constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    new_constraint_builders: &[TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let contacts_start = batch_ids.contacts_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);
    let bci_start = batch_id as usize * 2 * batch_ids.contacts_batch_capacity as usize;

    let old_body_constraint_counts = Slice(old_body_constraint_counts, colliders_start);
    let old_body_constraint_ids = Slice(old_body_constraint_ids, bci_start);
    let old_constraints = Slice(old_constraints, contacts_start);
    let old_constraint_builders = Slice(old_constraint_builders, contacts_start);
    let mut new_constraints = SliceMut(new_constraints, contacts_start);
    let new_constraint_builders = Slice(new_constraint_builders, contacts_start);

    let len = contacts_len.read(batch_id as usize);
    let cid_new = invocation_id.x;

    if cid_new < len {
        transfer_warmstart_impulses(
            cid_new,
            &old_body_constraint_counts,
            &old_body_constraint_ids,
            &old_constraints,
            &old_constraint_builders,
            &mut new_constraints,
            &new_constraint_builders,
        );
    }
}

/// Transfers warmstart impulses from previous frame to current frame.
///
/// NOTE: this assumes that the solver body ids in the constraints match the index of the body itself.
///       This also assumes that bodies in a given constraint pair are always in the same order (they don't
///       get swapped from one frame to another).
pub fn transfer_warmstart_impulses(
    cid_new: u32,
    old_body_constraint_counts: &Slice<u32>,
    old_body_constraint_ids: &Slice<u32>,
    old_constraints: &Slice<TwoBodyConstraint>,
    old_constraint_builders: &Slice<TwoBodyConstraintBuilder>,
    new_constraints: &mut SliceMut<TwoBodyConstraint>,
    new_constraint_builders: &Slice<TwoBodyConstraintBuilder>,
) {
    let i = cid_new as usize;

    // Get the two bodies involved in this new constraint
    let body_a = new_constraints[i].solver_body_a;
    let body_b = new_constraints[i].solver_body_b;

    // Find the range of old constraints involving body_a
    // old_body_constraint_counts is a prefix sum, so the range is [counts[i-1], counts[i])
    let first_constraint_id_a = if body_a != 0 {
        old_body_constraint_counts[body_a as usize - 1] as usize
    } else {
        0
    };
    let last_constraint_id_a = old_body_constraint_counts[body_a as usize] as usize;

    // Find the range of old constraints involving body_b
    let first_constraint_id_b = if body_b != 0 {
        old_body_constraint_counts[body_b as usize - 1] as usize
    } else {
        0
    };
    let last_constraint_id_b = old_body_constraint_counts[body_b as usize] as usize;

    let len_a = last_constraint_id_a - first_constraint_id_a;
    let len_b = last_constraint_id_b - first_constraint_id_b;

    // Optimization: search the smaller constraint list to minimize iterations
    // Also avoid static bodies which may have zero-length lists
    // Select the smallest list with a nonzero size (for example static bodies would have
    // a zero-length list despite having some constraints).
    // TODO: compare this approach with just using a hashmap.
    let (first_constraint_id_ref, last_constraint_id_ref) = if len_a != 0 && len_a < len_b {
        (first_constraint_id_a, last_constraint_id_a)
    } else {
        (first_constraint_id_b, last_constraint_id_b)
    };

    // Search through old constraints for matching body pair
    for j in first_constraint_id_ref..last_constraint_id_ref {
        let cid_old = old_body_constraint_ids[j] as usize;

        // Check if this old constraint involves the same body pair
        if old_constraints[cid_old].solver_body_a == body_a
            && old_constraints[cid_old].solver_body_b == body_b
        {
            // Body pair match found! Now match individual contact points.
            // We don't have feature IDs, so matching is done by proximity in local space.

            // Distance threshold for matching contact points (10cm)
            let dist_threshold = 1.0e-1; // 10cm
            let sq_threshold = dist_threshold * dist_threshold;

            // Try to match each new contact point with old contact points
            for k_new in 0..(new_constraints[i].len as usize) {
                let pt_new_a = new_constraint_builders[i].infos.at(k_new).local_pt_a;
                let pt_new_b = new_constraint_builders[i].infos.at(k_new).local_pt_b;

                // Search through old contact points for a match
                for k_old in 0..(old_constraints[cid_old].len as usize) {
                    let pt_old_a = old_constraint_builders[cid_old].infos.at(k_old).local_pt_a;
                    let pt_old_b = old_constraint_builders[cid_old].infos.at(k_old).local_pt_b;

                    // Compute distance between contact points in local space
                    let dpt_a = pt_old_a - pt_new_a;
                    let dpt_b = pt_old_b - pt_new_b;

                    // If both points are close enough, consider it a match
                    if dpt_a.dot(dpt_a) < sq_threshold && dpt_b.dot(dpt_b) < sq_threshold {
                        // Contact point match found! Transfer the accumulated impulse.
                        // The impulse field contains the last substep's impulse, which serves
                        // as the warmstart value for this frame.
                        // NOTE: we sum the impulse + impulse_accumulator since the accumulator contains the
                        //       accumulated impulse for all the substeps except the last one.
                        // TODO: what if we have multiple matches? (currently uses first match)
                        new_constraints[i]
                            .elements
                            .at_mut(k_new)
                            .normal_part
                            .impulse = old_constraints[cid_old]
                            .elements
                            .at(k_old)
                            .normal_part
                            .impulse;
                        new_constraints[i]
                            .elements
                            .at_mut(k_new)
                            .tangent_part
                            .impulse = old_constraints[cid_old]
                            .elements
                            .at(k_old)
                            .tangent_part
                            .impulse;
                    }
                }
            }

            // Since we found a matching body pair, no need to search further
            break;
        }
    }
}
