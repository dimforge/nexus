//! The contact-order links of the contact constraints: their topology for mass splitting and
//! coloring, and their match with the previous frame's constraints (warmstarting and contact
//! recycling).
//!
//! A contact is matched with the previous constraint of the same body pair, collider pair and
//! sub-shape. A pair that barely moved since its contacts were computed keeps them instead of
//! recomputing them (rapier's contact recycling).

use crate::broad_phase::ContactPlan;
use crate::queries::IndexedManifold;
use crate::{Pose, Rotation};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[allow(unused_imports)] // Needed on the GPU for `sqrt`.
use khal_std::num_traits::Float;

use super::RbdSimParams;
use super::constraint::{ContactLink, ContactRecycleOffsets, ContactRecycleState};

/// Initializes the link of every contact. Manifolds touching a multibody link belong to the
/// multibody contact solver: their link stays inactive, like gap and inert slots.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_contact_links(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links: &mut [ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_is_multibody: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] plan: &ContactPlan,
) {
    let i = gid.x as usize;
    if gid.x >= plan.bound {
        return;
    }
    let im = contacts.at(i);
    let multibody =
        body_is_multibody.read(im.bodies.x as usize) | body_is_multibody.read(im.bodies.y as usize);
    let len = if multibody != 0 { 0 } else { im.contact.len };
    links.write(
        i,
        ContactLink {
            len,
            solver_body_a: im.bodies.x,
            solver_body_b: im.bodies.y,
            vel_slot_a: im.bodies.x,
            vel_slot_b: im.bodies.y,
            warmstart_collider_a: im.colliders.x,
            warmstart_collider_b: im.colliders.y,
            warmstart_subshape: im.subshape,
            previous_constraint: u32::MAX,
            previous_constraint_index: u32::MAX,
            recycled: 0,
            mass_scale_a: 1.0,
            mass_scale_b: 1.0,
            restitution: im.restitution,
            contact: i as u32,
            _padding: 0,
        },
    );
}

/// Matches every active link with the previous frame's, decides whether its contacts are
/// recycled, and records the recycling state of its contacts.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_match_contact_links(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links: &mut [ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] old_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] old_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] old_links: &[ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] old_constraint_indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] states: &mut [ContactRecycleState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] collider_poses: &[Pose],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] plan: &ContactPlan,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] params: &RbdSimParams,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] recycle_offsets: &ContactRecycleOffsets,
) {
    let i = gid.x as usize;
    if gid.x >= plan.bound {
        return;
    }
    let link = links.read(i);
    // Gap slots may hold stale body ids: they must not be dereferenced.
    if link.len == 0 {
        return;
    }
    let im = contacts.at(i);
    let previous = find_previous_constraint(old_counts, old_ids, old_links, &link);
    let pose_a = collider_poses.read(im.colliders.x as usize);
    let pose_b = collider_poses.read(im.colliders.y as usize);
    let new_state = recycle_offsets.new_base as usize + i;
    let mut recycled = 0;
    let mut previous_constraint_index = u32::MAX;
    if previous != u32::MAX {
        previous_constraint_index = old_constraint_indices.read(previous as usize);
        let old_state = states.read(recycle_offsets.old_base as usize + previous as usize);
        if can_recycle(&old_state, pose_a, pose_b) {
            recycled = 1;
            states.write(new_state, old_state);
        }
    }
    if recycled == 0 {
        states.write(
            new_state,
            ContactRecycleState {
                pose_a,
                pose_b,
                colliders: im.colliders,
                max_extent: im.recycle_extent,
                max_drift: if im.recycle_extent >= 0.0 {
                    params.contact_recycle_distance()
                } else {
                    0.0
                },
            },
        );
    }
    let dst = links.at_mut(i);
    dst.previous_constraint = previous;
    dst.previous_constraint_index = previous_constraint_index;
    dst.recycled = recycled;
}

/// The previous constraint of the same body pair, collider pair and sub-shape, searched in
/// the shorter of the two bodies' previous adjacency lists, or `u32::MAX`.
#[inline(always)]
fn find_previous_constraint(
    counts: &[u32],
    ids: &[u32],
    old_links: &[ContactLink],
    new: &ContactLink,
) -> u32 {
    let a = new.solver_body_a as usize;
    let b = new.solver_body_b as usize;
    let first_a = if a == 0 { 0 } else { counts.read(a - 1) };
    let last_a = counts.read(a);
    let first_b = if b == 0 { 0 } else { counts.read(b - 1) };
    let last_b = counts.read(b);
    let len_a = last_a - first_a;
    let len_b = last_b - first_b;
    // Static bodies have empty lists: search the other one.
    let (first, last) = if len_b == 0 || (len_a != 0 && len_a < len_b) {
        (first_a, last_a)
    } else {
        (first_b, last_b)
    };
    for entry in first..last {
        let cid = ids.read(entry as usize);
        let old = old_links.at(cid as usize);
        if (old.solver_body_a == new.solver_body_a)
            & (old.solver_body_b == new.solver_body_b)
            & (old.warmstart_collider_a == new.warmstart_collider_a)
            & (old.warmstart_collider_b == new.warmstart_collider_b)
            & (old.warmstart_subshape == new.warmstart_subshape)
        {
            return cid;
        }
    }
    u32::MAX
}

/// Seeds the coloring with the colors of the matched previous constraints. Runs after the
/// topo-gc reset, before the coloring iterations.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_seed_colors_from_warmstart(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] links: &[ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] old_constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] colored: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contact_plan: &ContactPlan,
) {
    let i = invocation_id.x as usize;
    if invocation_id.x < contact_plan.bound {
        let link = links.at(i);
        if link.len != 0 && link.previous_constraint != u32::MAX {
            let old_color = old_constraints_colors.read(link.previous_constraint as usize);
            // The conflict pass validates every seed, including stale colors.
            if old_color > 0 && old_color < 64 {
                constraints_colors.write(i, old_color);
                colored.write(i, 1);
            }
        }
    }
}

/// Whether a pair whose contacts were computed at `state` can keep them at the collider poses
/// `pose_a` and `pose_b` (rapier's `relative_pose_drift` and `relative_rot_cos` tests).
#[inline(always)]
fn can_recycle(state: &ContactRecycleState, pose_a: Pose, pose_b: Pose) -> bool {
    let pos12_old = state.pose_a.inverse() * state.pose_b;
    let pos12 = pose_a.inverse() * pose_b;
    let trans = (pos12.translation - pos12_old.translation).length();
    let drot = pos12.rotation * pos12_old.rotation.inverse();
    // Bound on the rotation's displacement of a point within `max_extent` of the origin.
    #[cfg(feature = "dim3")]
    let chord = 2.0 * glamx::Vec3::new(drot.x, drot.y, drot.z).length() * state.max_extent;
    #[cfg(feature = "dim2")]
    let chord = {
        let denom = 2.0 * (1.0 + drot.re);
        let half_sin = if denom > 1.0e-6 {
            drot.im.abs() / denom.sqrt()
        } else {
            1.0
        };
        2.0 * half_sin * state.max_extent
    };
    // The frozen normal and lever arms also bound each collider's own rotation.
    let rot_cos = rot_cos(state.pose_a.rotation, pose_a.rotation)
        .min(rot_cos(state.pose_b.rotation, pose_b.rotation));
    state.max_drift > 0.0 && trans + chord <= state.max_drift && rot_cos > 0.98
}

/// The cosine of the angle between two rotations.
#[inline(always)]
fn rot_cos(a: Rotation, b: Rotation) -> f32 {
    #[cfg(feature = "dim3")]
    {
        let c = a.dot(b);
        2.0 * c * c - 1.0
    }
    #[cfg(feature = "dim2")]
    {
        a.re * b.re + a.im * b.im
    }
}
