//! Graph coloring for parallel constraint solving.
//!
//! Assigns colors to constraints so that no two constraints sharing a body get the
//! same color. Implements Jones-Plassmann-Luby and Topo-GC algorithms.

use crate::broad_phase::ContactPlan;
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{iter::StepRng, sync::atomic_add_u32};

use crate::utils::{BatchIndices, Slice, SliceMut};
use khal_std::index::MaybeIndexUnchecked;

use super::ContactLink;

const WORKGROUP_SIZE: u32 = 64;

/// Maximum u32 value (used to mark uncolored constraints in Luby algorithm).
pub const MAX_U32: u32 = u32::MAX;

/// Deterministic mode: the constraint picked a color this round and has no conflict yet.
/// Becomes 1 (kept) or 0 (lost) in the same `gpu_fix_conflicts_topo_gc` dispatch.
const PENDING_COLORED: u32 = 2;

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

/// Lowest free color. The upper word only contributes when the lower word is full.
/// Adding both words' trailing-zero counts can select an occupied color when a previous
/// frame seeded colors above 31, repeatedly recreating the same conflict.
#[inline(always)]
pub(super) fn first_free_color(mask: (u32, u32)) -> u32 {
    let low = (!mask.0).trailing_zeros();
    if low < 32 {
        low
    } else {
        32 + (!mask.1).trailing_zeros()
    }
}

/// Returns the `n`-th free color of a 64-bit mask (`n = 0` is the lowest), or 63 if none.
/// Spreads the picks of the same round over different colors to avoid conflicts.
#[inline]
fn nth_free_color(mask: (u32, u32), n: u32) -> u32 {
    let mut result = 63u32;

    for c in 0..64u32 {
        let (free, below) = if c < 32 {
            let occupied_below = (mask.0 & ((1u32 << c) - 1)).count_ones();
            ((mask.0 >> c) & 1 == 0, c - occupied_below)
        } else {
            let bit = c - 32;
            let occupied_below = mask.0.count_ones() + (mask.1 & ((1u32 << bit) - 1)).count_ones();
            ((mask.1 >> bit) & 1 == 0, c - occupied_below)
        };

        if free && below == n && c < result {
            result = c;
        }
    }

    result
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[ContactLink],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contact_plan: &ContactPlan,
) {
    let total = contact_plan.bound;
    let i = invocation_id.x;

    if i < total {
        let idx = i as usize;
        if constraints.at(idx).len == 0 {
            // Gap / inert slot: pre-colored with 0 so the Luby steps skip it.
            constraints_colors.write(idx, 0);
        } else {
            // Mark as uncolored
            constraints_colors.write(idx, MAX_U32);
        }
        // Assign random weight
        constraints_rands.write(idx, hash(i));
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] constraints_rands: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] uncolored: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] curr_color: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] contact_plan: &ContactPlan,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let body_constraint_counts = Slice(body_constraint_counts, 0);
    let body_constraint_ids = Slice(body_constraint_ids, 0);
    let body_group = Slice(body_group, 0);
    let constraints = Slice(constraints, 0);
    let mut constraints_colors = SliceMut(constraints_colors, 0);
    let constraints_rands = Slice(constraints_rands, 0);

    let color = *curr_color;

    for constraint_i in StepRng::new(invocation_id.x..total, num_threads) {
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[ContactLink],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] pending_colors: &mut [u32],
) {
    let total = contact_plan.bound;
    let i = invocation_id.x;

    if i == 0 {
        // Not converged: the first iteration runs.
        *num_colors = 0;
    }

    if i < total {
        let idx = i as usize;
        // Color 0 is reserved for "uncolored" state
        constraints_colors.write(idx, 0);
        // Gaps / inert slots are pre-marked colored so the topo-gc iterations
        // skip them (and converge).
        let inert = if constraints.at(idx).len == 0 { 1 } else { 0 };
        colored.write(idx, inert);
        // No pick yet, so the first fix pass only reads committed colors.
        pending_colors.write(idx, MAX_U32);
    }
}

/// Resets the convergence flag for Topo-GC, and sizes this iteration's dispatches: none once
/// the coloring has converged (the flag stayed non-zero through the previous iteration).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_completion_flag_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_indirect: &[[u32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] coloring_indirect: &mut [[u32; 3]],
) {
    if invocation_id.x == 0 {
        let converged = *num_colors != 0;
        let grid = if converged {
            [0, 1, 1]
        } else {
            contacts_indirect.read(0)
        };
        coloring_indirect.write(0, grid);
        *num_colors = 1;
    }
}

/// Clears the convergence flag and opens the full coloring grid so the first fix-conflicts pass
/// validates every color.
///
/// Colors seeded from the previous frame are marked colored, so the first step may color
/// nothing and leave the flag set, which would skip the validation of the seeds.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_clear_completion_flag_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_indirect: &[[u32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] coloring_indirect: &mut [[u32; 3]],
) {
    if invocation_id.x == 0 {
        coloring_indirect.write(0, contacts_indirect.read(0));
        *num_colors = 0;
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colors: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] pending_colors: &mut [u32],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let body_constraint_counts = Slice(body_constraint_counts, 0);
    let body_constraint_ids = Slice(body_constraint_ids, 0);
    let body_group = Slice(body_group, 0);
    let constraints = Slice(constraints, 0);
    let mut constraints_colors = SliceMut(constraints_colors, 0);
    let mut colored = SliceMut(colored, 0);
    let mut pending_colors = SliceMut(pending_colors, 0);

    // Deterministic mode: the pick goes to `pending_colors` and the fix pass commits it.
    // So this pass only reads colors from the previous rounds, whatever the thread order.
    let deterministic = batch_ids.deterministic != 0;

    for constraint_i in StepRng::new(invocation_id.x..total, num_threads) {
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
            let mut last_constraint_id_a = body_constraint_counts[body_a as usize] as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts[body_b as usize - 1] as usize
            } else {
                0
            };
            let mut last_constraint_id_b = body_constraint_counts[body_b as usize] as usize;

            // A split body's constraints each act on their own sub-body: they don't conflict.
            // The split points them at a sub-body velocity slot instead of the body's own.
            if constraints[i].vel_slot_a != constraints[i].solver_body_a {
                last_constraint_id_a = first_constraint_id_a;
            }
            if constraints[i].vel_slot_b != constraints[i].solver_body_b {
                last_constraint_id_b = first_constraint_id_b;
            }

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

            if deterministic {
                // Rank among the neighbors picking in this round, so they get different colors.
                // Counted in separate loops: the SPIR-V backend dislikes loops with many variables.
                let mut rank = 0u32;

                for j in first_constraint_id_a..last_constraint_id_a {
                    let constraint_j = body_constraint_ids[j];
                    if constraint_j < constraint_i && colored[constraint_j as usize] == 0 {
                        rank += 1;
                    }
                }

                for j in first_constraint_id_b..last_constraint_id_b {
                    let constraint_j = body_constraint_ids[j];
                    if constraint_j < constraint_i && colored[constraint_j as usize] == 0 {
                        rank += 1;
                    }
                }

                pending_colors[i] = nth_free_color(color_mask, rank);
                // `colored` is not written here since the rank loops read it.
                // The fix pass sets it, based on `pending_colors[i]`.
            } else {
                constraints_colors[i] = first_free_color(color_mask);
                colored[i] = 1;
            }
            // We are not finished coloring. 0 indicates the algorithm must continue.
            *num_colors = 0;
        } else if deterministic {
            // No pick in this round: the fix pass uses the committed color.
            pending_colors[i] = MAX_U32;
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colors: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] pending_colors: &[u32],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let body_constraint_counts = Slice(body_constraint_counts, 0);
    let body_constraint_ids = Slice(body_constraint_ids, 0);
    let body_group = Slice(body_group, 0);
    let constraints = Slice(constraints, 0);
    let mut constraints_colors = SliceMut(constraints_colors, 0);
    let mut colored = SliceMut(colored, 0);
    let pending_colors = Slice(pending_colors, 0);

    // In deterministic mode, this pass also commits the picks that have no conflict.
    // It never reads and writes `constraints_colors` at the same slot.
    let deterministic = batch_ids.deterministic != 0;

    for constraint_i in StepRng::new(invocation_id.x..total, num_threads) {
        let i = constraint_i as usize;
        // Gap / inert slot: its stale body ids must not be dereferenced.
        if constraints[i].len == 0 {
            continue;
        }
        let pending_i = if deterministic {
            pending_colors[i]
        } else {
            MAX_U32
        };
        // The color of the constraint in this round: its new pick, or its previous color.
        let color_i = if pending_i != MAX_U32 {
            pending_i
        } else {
            constraints_colors[i]
        };

        // Deterministic mode: mark the picks of this round. The loops below clear the
        // mark on the ones in conflict.
        if pending_i != MAX_U32 {
            colored[i] = PENDING_COLORED;
        }

        // A non-zero `num_colors` means the previous step iteration colored nothing new: the
        // coloring has converged and there is nothing left to fix. (The CPU only reads it as a
        // convergence flag.)
        if *num_colors == 0 {
            // Map raw body ids to graph-coloring GROUP ids (multibody-aware).
            let body_a = body_group[constraints[i].solver_body_a as usize];
            let body_b = body_group[constraints[i].solver_body_b as usize];

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts[body_a as usize - 1] as usize
            } else {
                0
            };
            let mut last_constraint_id_a = body_constraint_counts[body_a as usize] as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts[body_b as usize - 1] as usize
            } else {
                0
            };
            let mut last_constraint_id_b = body_constraint_counts[body_b as usize] as usize;

            // A split body's constraints each act on their own sub-body: they don't conflict.
            // The split points them at a sub-body velocity slot instead of the body's own.
            if constraints[i].vel_slot_a != constraints[i].solver_body_a {
                last_constraint_id_a = first_constraint_id_a;
            }
            if constraints[i].vel_slot_b != constraints[i].solver_body_b {
                last_constraint_id_b = first_constraint_id_b;
            }

            // Traverse all constraints from body A. On a conflict, the largest index keeps
            // its color, the others pick again in the next round.
            for j in first_constraint_id_a..last_constraint_id_a {
                let constraint_j = body_constraint_ids[j];

                if constraint_i < constraint_j {
                    let cj = constraint_j as usize;
                    let pending_j = if deterministic {
                        pending_colors[cj]
                    } else {
                        MAX_U32
                    };
                    let color_j = if pending_j != MAX_U32 {
                        pending_j
                    } else {
                        constraints_colors[cj]
                    };
                    if color_i == color_j {
                        // Found a conflict, uncolor this node.
                        colored[i] = 0;
                        break;
                    }
                }
            }

            if colored[i] != 0 {
                // Traverse all constraints from body B.
                for j in first_constraint_id_b..last_constraint_id_b {
                    let constraint_j = body_constraint_ids[j];

                    if constraint_i < constraint_j {
                        let cj = constraint_j as usize;
                        let pending_j = if deterministic {
                            pending_colors[cj]
                        } else {
                            MAX_U32
                        };
                        let color_j = if pending_j != MAX_U32 {
                            pending_j
                        } else {
                            constraints_colors[cj]
                        };
                        if color_i == color_j {
                            // Found a conflict, uncolor this node.
                            colored[i] = 0;
                            break;
                        }
                    }
                }
            }

            // Deterministic mode: still marked, so there is no conflict and the pick is committed.
            // Read from memory instead of a register (loop variables can miscompile).
            if pending_i != MAX_U32 && colored[i] == PENDING_COLORED {
                constraints_colors[i] = pending_i;
                colored[i] = 1;
            }
        }
    }
}

#[cfg(all(test, not(target_arch_is_gpu)))]
#[path = "../tests/coloring.rs"]
mod tests;
