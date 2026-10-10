//! Contact-tile kernels shared by the friction models: building the constraints, solving them
//! and warmstarting from them.
//!
//! Included by each model's module, which provides `ContactTile`, `TwoBodyConstraint` (its
//! constraint type) and the tile accessors (`header`, `read_constraint`, `write_constraint`,
//! `solve_constraint`, `tile_warmstart`, `constraint_warmstart`). It also provides
//! `model_kernel!`, which tags each kernel with the model: CUDA names entry points after a hash
//! of the kernel's tokens, which would otherwise be the same for every model.

use super::*;
use crate::broad_phase::ContactPlan;
use crate::dynamics::contact_tiles::{TAIL_COLOR, tile_lane};
use crate::dynamics::mass_splitting::{HubCounts, NOT_A_HUB};
use crate::dynamics::{
    ContactLink, ContactRecycleOffsets, ContactRecycleState, WorldMassProperties, decode_bias_mode,
};
use crate::queries::IndexedManifold;
use crate::utils::BatchIndices;
use khal_std::glamx::UVec3;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::control_barrier;

const WORKGROUP_SIZE: u32 = 64;

/// The number of constraints in the tiles: the colored ones, then the unsolved ones (in the
/// last color bucket).
#[inline(always)]
fn constraint_count(buckets: &[u32], ids: &BatchIndices) -> u32 {
    buckets.read((ids.solver_color_buckets_stride * ids.num_batches - 1) as usize)
}

/// The constraints of `color`, over every batch. Buckets are color-major and hold exclusive
/// ends.
#[inline(always)]
fn color_constraints(buckets: &[u32], ids: &BatchIndices, color: u32) -> (u32, u32) {
    let nb = ids.num_batches;
    (
        buckets.read((color * nb - 1) as usize),
        buckets.read(((color + 1) * nb - 1) as usize),
    )
}

/// Makes every lane's storage writes visible to the workgroup before the next color.
#[inline(always)]
fn color_barrier() {
    control_barrier::<
        { khal_std::memory::Scope::Workgroup as u32 },
        { khal_std::memory::Scope::QueueFamily as u32 },
        {
            khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
        },
    >();
}

/// Caches the warmstart of the constraint at `index`: the velocity changes of its bodies,
/// tagged with their index.
#[inline(always)]
fn write_warmstart(tiles: &mut [ContactTile], index: usize, (a, b): (Velocity, Velocity)) {
    let (tile, lane) = tile_lane(index);
    tiles.at_mut(tile).warmstart.bodies.write(lane, [a, b]);
}

model_kernel! {
    /// Builds every constraint in color order, gives it the impulses of its previous-frame
    /// constraint (or that constraint's contacts when they are recycled), and caches its warmstart
    /// for the first substep. Runs after the links are sorted by color.
    #[spirv_bindgen]
    #[spirv(compute(threads(64)))]
    pub fn gpu_prepare_constraints(
        #[spirv(global_invocation_id)] gid: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] sorted_links: &[ContactLink],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts: &[IndexedManifold],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mprops: &[WorldMassProperties],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] solver_poses: &[Pose],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] vels: &[Velocity],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] states: &[ContactRecycleState],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] old_tiles: &[ContactTile],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] tiles: &mut [ContactTile],
        #[spirv(uniform, descriptor_set = 0, binding = 8)] plan: &ContactPlan,
        #[spirv(uniform, descriptor_set = 0, binding = 9)] recycle_offsets: &ContactRecycleOffsets,
    ) {
        let index = gid.x as usize;
        if gid.x >= plan.bound {
            return;
        }
        // The links past the last constraint are inactive (see `gpu_color_buckets_count`).
        let link = sorted_links.read(index);
        if link.len == 0 {
            return;
        }
        let i = link.contact as usize;
        let im = contacts.at(i);
        let mut c = TwoBodyConstraint::default();
        c.init_header(im, &Slice(mprops, 0));
        c.apply_link(&link);
        if link.recycled == 0 {
            let pose_a = states.read(recycle_offsets.new_base as usize + i).pose_a;
            c.init_points_with_poses(
                im,
                pose_a,
                solver_poses.read(link.solver_body_a as usize),
                solver_poses.read(link.solver_body_b as usize),
                &Slice(vels, 0),
            );
        }
        if link.previous_constraint != u32::MAX {
            let old = read_constraint(old_tiles, link.previous_constraint_index as usize);
            if link.recycled != 0 {
                c.recycle_from(
                    &old,
                    vels.at(link.solver_body_a as usize),
                    vels.at(link.solver_body_b as usize),
                );
            } else {
                transfer_contact_impulses(&old, &mut c);
            }
        }
        write_constraint(tiles, index, &c);
        write_warmstart(tiles, index, constraint_warmstart(&c));
    }
}

/// Gives each new contact point the impulses of the nearest previous one (by local anchors,
/// within 10cm).
#[inline(always)]
fn transfer_contact_impulses(old: &TwoBodyConstraint, new: &mut TwoBodyConstraint) {
    let sq_threshold = 1.0e-1 * 1.0e-1;

    // The points of one small manifold are all within 10cm of each other: taking the first
    // candidate instead of the nearest would hand every point the same impulse.
    for k_new in 0..(new.len as usize) {
        let (pt_new_a, pt_new_b) = new.local_anchors(k_new);
        let mut best_k_old = usize::MAX;
        let mut best_sq = sq_threshold;
        for k_old in 0..(old.len as usize) {
            let (pt_old_a, pt_old_b) = old.local_anchors(k_old);
            let dpt_a = pt_old_a - pt_new_a;
            let dpt_b = pt_old_b - pt_new_b;
            let sq = dpt_a.dot(dpt_a).max(dpt_b.dot(dpt_b));
            if sq < best_sq {
                best_sq = sq;
                best_k_old = k_old;
            }
        }
        if best_k_old != usize::MAX {
            new.points.at_mut(k_new).normal_impulse = old.points.at(best_k_old).normal_impulse;
            new.transfer_friction_point(old.friction_warmstart(best_k_old), k_new);
        }
    }
    new.finish_friction_transfer();
}

model_kernel! {
    /// Scales the accumulated impulses of every constraint by the warmstart coefficient and caches
    /// their warmstart (only dispatched when that coefficient isn't 1).
    #[spirv_bindgen]
    #[spirv(compute(threads(64)))]
    pub fn gpu_scale_constraint_impulses(
        #[spirv(global_invocation_id)] gid: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buckets: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] tiles: &mut [ContactTile],
        #[spirv(uniform, descriptor_set = 0, binding = 2)] ids: &BatchIndices,
        #[spirv(uniform, descriptor_set = 0, binding = 3)] params: &RbdSimParams,
    ) {
        let index = gid.x as usize;
        if gid.x >= constraint_count(buckets, ids) {
            return;
        }
        let mut c = read_constraint(tiles, index);
        c.scale_impulses(params.warmstart_coefficient);
        write_constraint(tiles, index, &c);
        write_warmstart(tiles, index, constraint_warmstart(&c));
    }
}

/// Solves the constraints of one color, or of every color from [`TAIL_COLOR`] for the tail
/// variants (one workgroup, with a barrier between colors). The caching variants also store
/// each constraint's warmstart for the next substep.
macro_rules! tile_solver_kernel {
    ($name:ident, $mode:ident, $mode_value:expr, $cache:expr, $tail:expr) => {
        model_kernel! {
            #[spirv_bindgen]
            #[spirv(compute(threads(64)))]
            pub fn $name(
                #[spirv(global_invocation_id)] gid: UVec3,
                #[spirv(num_workgroups)] num_workgroups: UVec3,
                #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tiles: &mut [ContactTile],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
                solver_vels: &mut [Velocity],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] buckets: &[u32],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
                solver_body_poses: &[Pose],
                #[spirv(uniform, descriptor_set = 0, binding = 4)] curr_color: &u32,
                #[spirv(uniform, descriptor_set = 0, binding = 5)] ids: &BatchIndices,
                #[spirv(uniform, descriptor_set = 0, binding = 6)] $mode: &u32,
                #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &RbdSimParams,
            ) {
                let _ = $mode;
                let (use_bias, solve_friction) = $mode_value;
                let poses = Slice(solver_body_poses, 0);
                let first = if $tail { TAIL_COLOR } else { *curr_color };
                for color in first..=*curr_color {
                    let (start, end) = color_constraints(buckets, ids, color);
                    let stride = num_workgroups.x * WORKGROUP_SIZE;
                    for index in StepRng::new(start + gid.x..end, stride) {
                        let index = index as usize;
                        let h = header(tiles, index);
                        let mut a = solver_vels.read(h.vel_slot_a as usize);
                        let mut b = solver_vels.read(h.vel_slot_b as usize);
                        solve_constraint(
                            &h,
                            tiles,
                            index,
                            &poses,
                            params,
                            &mut a,
                            &mut b,
                            use_bias,
                            solve_friction,
                        );
                        solver_vels.write(h.vel_slot_a as usize, a);
                        solver_vels.write(h.vel_slot_b as usize, b);
                        if $cache {
                            let warmstart = tile_warmstart(tiles, index, &h);
                            write_warmstart(tiles, index, warmstart);
                        }
                    }
                    if $tail {
                        color_barrier();
                    }
                }
            }
        }
    };
}

tile_solver_kernel!(
    gpu_solve_constraints,
    mode,
    decode_bias_mode(*mode),
    false,
    false
);
tile_solver_kernel!(
    gpu_solve_constraints_biased,
    mode,
    (true, false),
    false,
    false
);
tile_solver_kernel!(
    gpu_solve_constraints_unbiased,
    mode,
    (false, true),
    false,
    false
);
tile_solver_kernel!(
    gpu_solve_constraints_unbiased_cached,
    mode,
    (false, true),
    true,
    false
);
tile_solver_kernel!(
    gpu_solve_constraints_tail,
    mode,
    decode_bias_mode(*mode),
    false,
    true
);
tile_solver_kernel!(
    gpu_solve_constraints_tail_biased,
    mode,
    (true, false),
    false,
    true
);
tile_solver_kernel!(
    gpu_solve_constraints_tail_unbiased,
    mode,
    (false, true),
    false,
    true
);
tile_solver_kernel!(
    gpu_solve_constraints_tail_unbiased_cached,
    mode,
    (false, true),
    true,
    true
);

/// Solves every color with one 64-lane workgroup per batch, with a barrier between colors.
/// Used for small scenes where the contact count is small wrt. the environment count.
macro_rules! fused_tile_solver_kernel {
    ($name:ident, $cache:expr) => {
        model_kernel! {
            #[spirv_bindgen]
            #[spirv(compute(threads(64)))]
            pub fn $name(
                #[spirv(global_invocation_id)] invocation_id: UVec3,
                #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tiles: &mut [ContactTile],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
                solver_vels: &mut [Velocity],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] buckets: &[u32],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
                solver_body_poses: &[Pose],
                #[spirv(uniform, descriptor_set = 0, binding = 4)] num_colors: &u32,
                #[spirv(uniform, descriptor_set = 0, binding = 5)] ids: &BatchIndices,
                #[spirv(uniform, descriptor_set = 0, binding = 6)] mode: &u32,
                #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &RbdSimParams,
            ) {
                let lane = invocation_id.x;
                let batch = invocation_id.y;
                let nb = ids.num_batches;
                let poses = Slice(solver_body_poses, 0);
                let (use_bias, solve_friction) = decode_bias_mode(*mode);
                for color in 1..=*num_colors {
                    let bucket = (color * nb + batch) as usize;
                    let start = buckets.read(bucket - 1);
                    let end = buckets.read(bucket);
                    #[cfg(not(feature = "web-compat"))]
                    if start == end {
                        continue;
                    }
                    for index in StepRng::new(start + lane..end, WORKGROUP_SIZE) {
                        let index = index as usize;
                        let h = header(tiles, index);
                        let mut a = solver_vels.read(h.vel_slot_a as usize);
                        let mut b = solver_vels.read(h.vel_slot_b as usize);
                        solve_constraint(
                            &h,
                            tiles,
                            index,
                            &poses,
                            params,
                            &mut a,
                            &mut b,
                            use_bias,
                            solve_friction,
                        );
                        solver_vels.write(h.vel_slot_a as usize, a);
                        solver_vels.write(h.vel_slot_b as usize, b);
                        if $cache {
                            let warmstart = tile_warmstart(tiles, index, &h);
                            write_warmstart(tiles, index, warmstart);
                        }
                    }
                    color_barrier();
                }
            }
        }
    };
}

fused_tile_solver_kernel!(gpu_solve_constraints_fused, false);
fused_tile_solver_kernel!(gpu_solve_constraints_fused_cached, true);

/// Adds a constraint's warmstart to the velocities of its bodies.
#[inline(always)]
fn apply_warmstart(tiles: &[ContactTile], solver_vels: &mut [Velocity], index: usize) {
    let h = header(tiles, index);
    let (da, db) = tile_warmstart(tiles, index, &h);
    let mut a = solver_vels.read(h.vel_slot_a as usize);
    let mut b = solver_vels.read(h.vel_slot_b as usize);
    a.linear += da.linear;
    a.angular += da.angular;
    b.linear += db.linear;
    b.angular += db.angular;
    solver_vels.write(h.vel_slot_a as usize, a);
    solver_vels.write(h.vel_slot_b as usize, b);
}

model_kernel! {
    /// Applies the warmstart of one color's constraints to their bodies (scatter-style, for scenes
    /// with multibodies, whose warmstart can't be gathered per body).
    #[spirv_bindgen]
    #[spirv(compute(threads(64)))]
    pub fn gpu_warmstart_constraints(
        #[spirv(global_invocation_id)] gid: UVec3,
        #[spirv(num_workgroups)] num_workgroups: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tiles: &[ContactTile],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] buckets: &[u32],
        #[spirv(uniform, descriptor_set = 0, binding = 3)] curr_color: &u32,
        #[spirv(uniform, descriptor_set = 0, binding = 4)] ids: &BatchIndices,
    ) {
        let (start, end) = color_constraints(buckets, ids, *curr_color);
        for index in StepRng::new(start + gid.x..end, num_workgroups.x * WORKGROUP_SIZE) {
            apply_warmstart(tiles, solver_vels, index as usize);
        }
    }
}

model_kernel! {
    /// [`gpu_warmstart_constraints`] for every color, with one 64-lane workgroup per batch.
    #[spirv_bindgen]
    #[spirv(compute(threads(64)))]
    pub fn gpu_warmstart_constraints_fused(
        #[spirv(global_invocation_id)] invocation_id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tiles: &[ContactTile],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] buckets: &[u32],
        #[spirv(uniform, descriptor_set = 0, binding = 3)] num_colors: &u32,
        #[spirv(uniform, descriptor_set = 0, binding = 4)] ids: &BatchIndices,
    ) {
        let lane = invocation_id.x;
        let batch = invocation_id.y;
        let nb = ids.num_batches;
        for color in 1..=*num_colors {
            let bucket = (color * nb + batch) as usize;
            let start = buckets.read(bucket - 1);
            let end = buckets.read(bucket);
            #[cfg(not(feature = "web-compat"))]
            if start == end {
                continue;
            }
            for index in StepRng::new(start + lane..end, WORKGROUP_SIZE) {
                apply_warmstart(tiles, solver_vels, index as usize);
            }
            color_barrier();
        }
    }
}

model_kernel! {
    /// Sums the cached warmstart of each body's constraints, in its adjacency-list order (split
    /// bodies are warmstarted through their sub-bodies).
    #[spirv_bindgen]
    #[spirv(compute(threads(64)))]
    pub fn gpu_gather_warmstart_velocities(
        #[spirv(global_invocation_id)] gid: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] counts: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraint_ids: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraint_indices: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tiles: &[ContactTile],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] solver_vels: &mut [Velocity],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] hub_first_slot: &[u32],
        #[spirv(uniform, descriptor_set = 0, binding = 6)] ids: &BatchIndices,
    ) {
        let body = gid.x as usize;
        if gid.x < ids.bodies_len * ids.num_batches && hub_first_slot.read(body) == NOT_A_HUB {
            let start = if body == 0 { 0 } else { counts.read(body - 1) };
            let end = counts.read(body);
            let mut vel = solver_vels.read(body);
            for entry in start..end {
                let index =
                    constraint_indices.read(constraint_ids.read(entry as usize) as usize) as usize;
                let (tile, lane) = tile_lane(index);
                let [a, b] = tiles.at(tile).warmstart.bodies.read(lane);
                let dv = if a.padding1 == gid.x { a } else { b };
                vel.linear += dv.linear;
                vel.angular += dv.angular;
            }
            solver_vels.write(body, vel);
        }
    }
}

model_kernel! {
    /// Applies the warmstart of each split body's constraint to its sub-body (the body's own
    /// warmstart gather skips them).
    #[spirv_bindgen]
    #[spirv(compute(threads(64)))]
    pub fn gpu_hub_warmstart_constraints(
        #[spirv(global_invocation_id)] invocation_id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] tiles: &[ContactTile],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_slot_constraint: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] hub_counts: &HubCounts,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] constraint_indices: &[u32],
    ) {
        let slot = invocation_id.x;
        if slot < hub_counts.live_slots() {
            let vel_slot = hub_counts.base + slot;
            let index =
                constraint_indices.read(hub_slot_constraint.read(slot as usize) as usize) as usize;
            let h = header(tiles, index);
            let (da, db) = tile_warmstart(tiles, index, &h);
            let d = if h.vel_slot_a == vel_slot { da } else { db };
            let mut vel = solver_vels.read(vel_slot as usize);
            vel.linear += d.linear;
            vel.angular += d.angular;
            solver_vels.write(vel_slot as usize, vel);
        }
    }
}
