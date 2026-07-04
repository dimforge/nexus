//! Graph coloring for parallel constraint solving.
//!
//! Assigns colors to constraints so that no two constraints sharing a body get the
//! same color. Implements Jones-Plassmann-Luby and Topo-GC algorithms.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{
    iter::StepRng,
    sync::{atomic_add_u32, atomic_max_u32},
};

use crate::utils::{BatchIndices, Slice};
use khal_std::index::MaybeIndexUnchecked;

use super::constraint::TwoBodyConstraint;

const WORKGROUP_SIZE: u32 = 64;

/// Maximum u32 value (used to mark uncolored constraints in Luby algorithm).
pub const MAX_U32: u32 = u32::MAX;

/// Hash function for generating random weights.
///
/// Uses a variant of the Murmur3 hash function to generate pseudo-random
/// weights from constraint indices.
#[inline]
fn hash(packed_key: u32) -> u32 {
    let mut key = packed_key;
    key *= 0xcc9e2d51;
    key = key.rotate_left(15);
    key *= 0x1b873593;
    key
}

/*
 * Jones-Plassmann-Luby Graph Coloring Algorithm
 *
 * Randomized parallel graph coloring: colors constraints so no two sharing a
 * body get the same color.
 */

/// Initializes Luby algorithm state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset_luby(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints_rands: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mut constraints_colors = batch_ids.contact_batch_mut(batch_id, constraints_colors);
    let mut constraints_rands = batch_ids.contact_batch_mut(batch_id, constraints_rands);
    let len = contacts_len.read(batch_id as usize);

    let i = invocation_id.x;

    if i < len {
        let idx = i as usize;
        // Mark as uncolored
        constraints_colors[idx] = MAX_U32;
        // Assign random weight
        constraints_rands[idx] = hash(i);
    }
}

/// Performs one iteration of Luby's graph coloring algorithm.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_graph_coloring_luby(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] constraints_rands: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] uncolored: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] curr_color: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let bci_start = batch_id as usize * 2 * batch_ids.contacts_batch_capacity as usize;

    let body_constraint_counts = batch_ids.coll_batch(batch_id, body_constraint_counts);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let body_group = batch_ids.coll_batch(batch_id, body_group);
    let constraints = batch_ids.contact_batch(batch_id, constraints);
    let mut constraints_colors = batch_ids.contact_batch_mut(batch_id, constraints_colors);
    let constraints_rands = batch_ids.contact_batch(batch_id, constraints_rands);

    let len = contacts_len.read(batch_id as usize);
    let color = *curr_color;

    for constraint_i in StepRng::new(invocation_id.x..len, num_threads) {
        let i = constraint_i as usize;

        if constraints_colors[i] == MAX_U32 {
            // This constraint doesn't have a color yet.
            let rand_i = constraints_rands[i];
            // Map raw body ids to graph-coloring GROUP ids (multibody-aware).
            let body_a = body_group[constraints[i].solver_body_a as usize];
            let body_b = body_group[constraints[i].solver_body_b as usize];

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts[body_a as usize - 1] as usize
            } else {
                0
            };
            let last_constraint_id_a = body_constraint_counts[body_a as usize] as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts[body_b as usize - 1] as usize
            } else {
                0
            };
            let last_constraint_id_b = body_constraint_counts[body_b as usize] as usize;

            let mut is_greatest = true;

            // Traverse all constraints from body A.
            for j in first_constraint_id_a..last_constraint_id_a {
                if !is_greatest {
                    break;
                }
                let constraint_j = body_constraint_ids[j];
                let rand_j = constraints_rands[constraint_j as usize];
                let color_j = constraints_colors[constraint_j as usize];
                // NOTE: there is a very rare case both constraints got assigned the same random number.
                //       in that case, we define the "greatest" comparison based on the constraint's array index.
                // NOTE: the equality in i >= j is important here to account for the fact we will iterate
                //       through the current constraint's index too.
                is_greatest = is_greatest
                    && (color_j != MAX_U32
                        || rand_i > rand_j
                        || (rand_i == rand_j && constraint_i >= constraint_j));
            }

            // Traverse all constraints from body B.
            for j in first_constraint_id_b..last_constraint_id_b {
                if !is_greatest {
                    break;
                }
                let cid = body_constraint_ids[j];
                let rand_j = constraints_rands[cid as usize];
                let color_j = constraints_colors[cid as usize];
                // NOTE: there is a very rare case both constraints got assigned the same random number.
                //       in that case, we define the "greatest" comparison based on the constraint's array index.
                // NOTE: the equality in i >= j is important here to account for the fact we will iterate
                //       through the current constraint's index too.
                is_greatest = is_greatest
                    && (color_j != MAX_U32
                        || rand_j < rand_i
                        || (rand_i == rand_j && constraint_i >= cid));
            }

            if is_greatest {
                constraints_colors[i] = color;
            } else {
                // Still uncolored
                atomic_add_u32(uncolored, 1);
            }
        }
    }
}

/*
 * Topo-GC (Topological Graph Coloring) Algorithm
 *
 * Parallel graph coloring alternative to Luby. Limited to 63 colors (2x u32
 * bitmask representation). Color indices start at 1; index 0 is reserved.
 *
 * Reference: https://people.csail.mit.edu/xchen/docs/ipdpsw-2016.pdf
 */

/// Initializes Topo-GC algorithm state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mut constraints_colors = batch_ids.contact_batch_mut(batch_id, constraints_colors);
    let mut colored = batch_ids.contact_batch_mut(batch_id, colored);
    let len = contacts_len.read(batch_id as usize);

    let i = invocation_id.x;

    if i < len {
        let idx = i as usize;
        // Color 0 is reserved for "uncolored" state
        constraints_colors[idx] = 0;
        colored[idx] = 0;
    }
}

/// Resets the convergence flag for Topo-GC.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_completion_flag_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_colors: &mut u32,
) {
    if invocation_id.x == 0 {
        // NOTE: this `for` loop is silly. It doesn’t do anything
        //       more than a `*num_colors = 1` in a convoluted
        //       way because otherwise rustgpu apparently does not generate
        //       the spirv for this kernel (seems to happen if the kernel is
        //       too trivial.
        for k in 0..1 {
            // Non-zero value indicates algorithm should continue
            *num_colors = k + 1;
        }
    }
}

/// Performs one iteration of Topo-GC coloring.
///
/// Generates up to 63 colors (color 0 = uncolored).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_graph_coloring_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let bci_start = batch_id as usize * 2 * batch_ids.contacts_batch_capacity as usize;

    let body_constraint_counts = batch_ids.coll_batch(batch_id, body_constraint_counts);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let body_group = batch_ids.coll_batch(batch_id, body_group);
    let constraints = batch_ids.contact_batch(batch_id, constraints);
    let mut constraints_colors = batch_ids.contact_batch_mut(batch_id, constraints_colors);
    let mut colored = batch_ids.contact_batch_mut(batch_id, colored);

    let len = contacts_len.read(batch_id as usize);

    for constraint_i in StepRng::new(invocation_id.x..len, num_threads) {
        let i = constraint_i as usize;

        if colored[i] == 0 {
            // This constraint doesn't have a color yet.
            // NOTE: generates up to 63 colors.
            // Note that we always mark the color 0 as occupied (cf. paper using i > 0).
            let mut color_mask = (1u32, 0u32);

            // Map raw body ids to graph-coloring GROUP ids (multibody-aware).
            let body_a = body_group[constraints[i].solver_body_a as usize];
            let body_b = body_group[constraints[i].solver_body_b as usize];

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts[body_a as usize - 1] as usize
            } else {
                0
            };
            let last_constraint_id_a = body_constraint_counts[body_a as usize] as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts[body_b as usize - 1] as usize
            } else {
                0
            };
            let last_constraint_id_b = body_constraint_counts[body_b as usize] as usize;

            // Traverse all constraints from body A.
            for j in first_constraint_id_a..last_constraint_id_a {
                let constraint_j = body_constraint_ids[j];

                if constraint_j != constraint_i {
                    let color_j = constraints_colors[constraint_j as usize];
                    if color_j < 32 {
                        color_mask.0 |= 1u32 << color_j;
                    } else {
                        color_mask.1 |= 1u32 << (color_j - 32);
                    }
                }
            }

            // Traverse all constraints from body B.
            for j in first_constraint_id_b..last_constraint_id_b {
                let constraint_j = body_constraint_ids[j];

                if constraint_j != constraint_i {
                    let color_j = constraints_colors[constraint_j as usize];
                    if color_j < 32 {
                        color_mask.0 |= 1u32 << color_j;
                    } else {
                        color_mask.1 |= 1u32 << (color_j - 32);
                    }
                }
            }

            let my_color = (!color_mask.0).trailing_zeros() + (!color_mask.1).trailing_zeros();
            constraints_colors[i] = my_color;
            colored[i] = 1;
            // We are not finished coloring. 0 indicates the algorithm must continue.
            *num_colors = 0;
        }
    }
}

/// Fixes conflicts in Topo-GC coloring.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_fix_conflicts_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let bci_start = batch_id as usize * 2 * batch_ids.contacts_batch_capacity as usize;

    let body_constraint_counts = batch_ids.coll_batch(batch_id, body_constraint_counts);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let body_group = batch_ids.coll_batch(batch_id, body_group);
    let constraints = batch_ids.contact_batch(batch_id, constraints);
    let constraints_colors = batch_ids.contact_batch(batch_id, constraints_colors);
    let mut colored = batch_ids.contact_batch_mut(batch_id, colored);

    let len = contacts_len.read(batch_id as usize);

    for constraint_i in StepRng::new(invocation_id.x..len, num_threads) {
        let i = constraint_i as usize;
        let color_i = constraints_colors[i];

        // NOTE: this `num_colors` read doesn't need to be atomic. Any non-zero value is indicative of a finished
        //       algorithm.
        // If `num_colors > 0u` then we know that the coloring algorithm has converged. So we use this dispatch
        // as an opportunity to compute the colors count that will be ready back to the CPU side.
        if *num_colors > 0 {
            // TODO PERF: not sure if that would have a significant impact but we could keep track of
            //            whether the last iteration of the TOPO-GC algorithm already finished, in which
            //            case we can skip the atomic max entirely and just early-exist.

            atomic_max_u32(num_colors, color_i);
        } else {
            // Map raw body ids to graph-coloring GROUP ids (multibody-aware).
            let body_a = body_group[constraints[i].solver_body_a as usize];
            let body_b = body_group[constraints[i].solver_body_b as usize];

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts[body_a as usize - 1] as usize
            } else {
                0
            };
            let last_constraint_id_a = body_constraint_counts[body_a as usize] as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts[body_b as usize - 1] as usize
            } else {
                0
            };
            let last_constraint_id_b = body_constraint_counts[body_b as usize] as usize;

            // Traverse all constraints from body A.
            for j in first_constraint_id_a..last_constraint_id_a {
                let constraint_j = body_constraint_ids[j];

                if constraint_j != constraint_i {
                    let color_j = constraints_colors[constraint_j as usize];
                    if color_i == color_j && constraint_i < constraint_j {
                        // Found a conflict, uncolor this node.
                        colored[i] = 0;
                    }
                }
            }

            // Traverse all constraints from body B.
            for j in first_constraint_id_b..last_constraint_id_b {
                let constraint_j = body_constraint_ids[j];

                if constraint_j != constraint_i {
                    let color_j = constraints_colors[constraint_j as usize];
                    if color_i == color_j && constraint_i < constraint_j {
                        // Found a conflict, uncolor this node.
                        colored[i] = 0;
                    }
                }
            }
        }
    }
}
