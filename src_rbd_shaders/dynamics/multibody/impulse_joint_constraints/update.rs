//! Per-joint (re)build of all axis constraints (`update_one_joint`) and the
//! multibody-side `M⁻¹·Jᵀ` back-solve used by the finalize pass.

use khal_std::index::MaybeIndexUnchecked;

use crate::dynamics::body::WorldMassProperties;
use crate::dynamics::joint::{ANG_AXES_MASK, LIN_AXES_MASK, SPATIAL_DIM};
use crate::utils::linalg::{MatSlice, lu_solve_in_place};
use crate::{DIM, Pose};

use super::super::types::{MultibodyInfo, MultibodyLinkWorkspace};
use super::helper::*;
use super::jacobians::*;
use super::types::*;

/// Back-solve `W·J = M⁻¹·Jᵀ` for one multibody side of a constraint and store
/// it in the constraint's `W·J` jacobian slot. Extracted from the old fused
/// `fill_mb_jacobians` step 2 so it can run in the finalize pass (which binds
/// `mass_matrices` / `lu_pivots`, unlike the build pass).
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_mb_wj(
    jacobians: &mut [f32],
    j_id: u32,
    ndofs: u32,
    mb: &MultibodyInfo,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
) {
    // Copy J into the W·J slot, then LU back-solve in place (matches the old
    // fused path: `wj = M⁻¹·j`).
    let wj_base = wj_id(j_id, ndofs);
    for k in 0..ndofs {
        let v = jacobians.read(j_id as usize + k as usize);
        jacobians.write(wj_base + k as usize, v);
    }
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    let piv_offset = dof_start + mb.first_dof as usize;
    lu_solve_in_place(mass_matrices, m, lu_pivots, piv_offset, jacobians, wj_base);
}

impl MbImpulseJointBuilder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update_one_joint(
        &self,
        constraints: &mut [MbImpulseJointConstraint],
        cons_start: usize,
        jacobians: &mut [f32],
        jac_buf_start: usize,
        multibody_info: &[MultibodyInfo],
        mb_start: usize,
        links_workspace: &[MultibodyLinkWorkspace],
        links_start: usize,
        body_jacobians: &[f32],
        body_jac_start: usize,
        poses: &[Pose],
        colliders_start: usize,
        mprops: &[WorldMassProperties],
        dt: f32,
        lock_erp_inv_dt: f32,
        lock_cfm_coeff: f32,
    ) {
        let cons_base = cons_start + self.constraint_id as usize;
        // Mark all axis-constraint slots inactive up-front; the active branches
        // below overwrite the live ones (rapier rebuilds the entire
        // `out[start..len]` slab each `update` call, so unfilled slots are
        // guaranteed inactive).
        for s in 0..MAX_AXIS_CONSTRAINTS {
            let mut cz = constraints.read(cons_base + s as usize);
            cz.kind = 0;
            cz.impulse = 0.0;
            constraints.write(cons_base + s as usize, cz);
        }

        // Resolve per-side multibody descriptors (read by value to avoid
        // SPIR-V's "pointer to arbitrary element" restriction). Free / fixed
        // sides ignore the read.
        let mb_a = if self.side_a_kind == SIDE_KIND_MB {
            multibody_info.read(mb_start + self.side_a_id as usize)
        } else {
            MultibodyInfo::default()
        };
        let mb_b = if self.side_b_kind == SIDE_KIND_MB {
            multibody_info.read(mb_start + self.side_b_id as usize)
        } else {
            MultibodyInfo::default()
        };

        let pose_a = side_world_pose(
            self.side_a_kind,
            self.side_a_id,
            self.side_a_link,
            &mb_a,
            links_workspace,
            links_start,
            poses,
            colliders_start,
        );
        let pose_b = side_world_pose(
            self.side_b_kind,
            self.side_b_id,
            self.side_b_link,
            &mb_b,
            links_workspace,
            links_start,
            poses,
            colliders_start,
        );

        let frame1 = pose_a * self.joint.local_frame_a;
        let frame2 = pose_b * self.joint.local_frame_b;
        let world_com1 = pose_a.translation;
        let world_com2 = pose_b.translation;

        let helper = new_helper(
            frame1,
            frame2,
            world_com1,
            world_com2,
            self.joint.locked_axes,
        );

        let ndofs_a = if self.side_a_kind == SIDE_KIND_BODY {
            SPATIAL_DIM as u32
        } else if self.side_a_kind == SIDE_KIND_MB {
            mb_a.ndofs
        } else {
            0
        };
        let ndofs_b = if self.side_b_kind == SIDE_KIND_BODY {
            SPATIAL_DIM as u32
        } else if self.side_b_kind == SIDE_KIND_MB {
            mb_b.ndofs
        } else {
            0
        };
        let a_ctx = SideCtx {
            side_kind: self.side_a_kind,
            side_id: self.side_a_id,
            side_link: self.side_a_link,
            ndofs: ndofs_a,
            mb: mb_a,
        };
        let b_ctx = SideCtx {
            side_kind: self.side_b_kind,
            side_id: self.side_b_id,
            side_link: self.side_b_link,
            ndofs: ndofs_b,
            mb: mb_b,
        };
        let stride = axis_stride(ndofs_a, ndofs_b);
        let j_base = jac_buf_start + self.jacobian_offset as usize;

        // `lock_erp_inv_dt` / `lock_cfm_coeff` are passed in from the configurable
        // joint softness (rapier's `joint.softness.{erp_inv_dt,cfm_coeff}(dt)`).
        let locked_axes = self.joint.locked_axes;
        let motor_axes = self.joint.motor_axes & !locked_axes;
        let limit_axes = self.joint.limit_axes & !locked_axes;

        let mut len = 0u32;
        let mut j_off = j_base as u32;

        // Order matches rapier's `lock_axes`: motors → locks → limits.
        // Within each kind: angular axes before linear axes.

        // Angular motors.
        for i in DIM..(SPATIAL_DIM as u32) {
            if (motor_axes & (1 << i)) != 0 {
                if len >= MAX_AXIS_CONSTRAINTS {
                    break;
                }
                let mut c = constraints.read(cons_base + len as usize);
                let j_id_a = j_off;
                let j_id_b = j_off + 2 * ndofs_a;
                motor_angular_generic(
                    &mut c,
                    &helper,
                    self.joint_id,
                    &a_ctx,
                    &b_ctx,
                    (i - DIM) as usize,
                    self.joint.motors.at(i as usize),
                    dt,
                    jacobians,
                    j_id_a,
                    j_id_b,
                    body_jacobians,
                    body_jac_start,
                    mprops,
                    colliders_start,
                );
                constraints.write(cons_base + len as usize, c);
                len += 1;
                j_off += stride;
            }
        }

        // Linear motors.
        for i in 0..(DIM as u32) {
            if (motor_axes & (1 << i)) != 0 {
                if len >= MAX_AXIS_CONSTRAINTS {
                    break;
                }
                let mut c = constraints.read(cons_base + len as usize);
                let j_id_a = j_off;
                let j_id_b = j_off + 2 * ndofs_a;
                motor_linear_generic(
                    &mut c,
                    &helper,
                    self.joint_id,
                    &a_ctx,
                    &b_ctx,
                    i as usize,
                    self.joint.motors.at(i as usize),
                    dt,
                    jacobians,
                    j_id_a,
                    j_id_b,
                    body_jacobians,
                    body_jac_start,
                    mprops,
                    colliders_start,
                );
                constraints.write(cons_base + len as usize, c);
                len += 1;
                j_off += stride;
            }
        }

        // Angular locks.
        for i in DIM..(SPATIAL_DIM as u32) {
            if (locked_axes & (1 << i)) != 0 {
                if len >= MAX_AXIS_CONSTRAINTS {
                    break;
                }
                let mut c = constraints.read(cons_base + len as usize);
                let j_id_a = j_off;
                let j_id_b = j_off + 2 * ndofs_a;
                lock_angular_generic(
                    &mut c,
                    &helper,
                    self.joint_id,
                    &a_ctx,
                    &b_ctx,
                    (i - DIM) as usize,
                    lock_erp_inv_dt,
                    lock_cfm_coeff,
                    jacobians,
                    j_id_a,
                    j_id_b,
                    body_jacobians,
                    body_jac_start,
                    mprops,
                    colliders_start,
                );
                constraints.write(cons_base + len as usize, c);
                len += 1;
                j_off += stride;
            }
        }

        // Linear locks.
        for i in 0..(DIM as u32) {
            if (locked_axes & (1 << i)) != 0 {
                if len >= MAX_AXIS_CONSTRAINTS {
                    break;
                }
                let mut c = constraints.read(cons_base + len as usize);
                let j_id_a = j_off;
                let j_id_b = j_off + 2 * ndofs_a;
                lock_linear_generic(
                    &mut c,
                    &helper,
                    self.joint_id,
                    &a_ctx,
                    &b_ctx,
                    i as usize,
                    lock_erp_inv_dt,
                    lock_cfm_coeff,
                    jacobians,
                    j_id_a,
                    j_id_b,
                    body_jacobians,
                    body_jac_start,
                    mprops,
                    colliders_start,
                );
                constraints.write(cons_base + len as usize, c);
                len += 1;
                j_off += stride;
            }
        }

        // Angular limits.
        for i in DIM..(SPATIAL_DIM as u32) {
            if (limit_axes & (1 << i)) != 0 {
                if len >= MAX_AXIS_CONSTRAINTS {
                    break;
                }
                let mut c = constraints.read(cons_base + len as usize);
                let j_id_a = j_off;
                let j_id_b = j_off + 2 * ndofs_a;
                let lim = self.joint.limits.at(i as usize);
                limit_angular_generic(
                    &mut c,
                    &helper,
                    self.joint_id,
                    &a_ctx,
                    &b_ctx,
                    (i - DIM) as usize,
                    [lim.min, lim.max],
                    lock_erp_inv_dt,
                    lock_cfm_coeff,
                    jacobians,
                    j_id_a,
                    j_id_b,
                    body_jacobians,
                    body_jac_start,
                    mprops,
                    colliders_start,
                );
                constraints.write(cons_base + len as usize, c);
                len += 1;
                j_off += stride;
            }
        }

        // Linear limits.
        for i in 0..(DIM as u32) {
            if (limit_axes & (1 << i)) != 0 {
                if len >= MAX_AXIS_CONSTRAINTS {
                    break;
                }
                let mut c = constraints.read(cons_base + len as usize);
                let j_id_a = j_off;
                let j_id_b = j_off + 2 * ndofs_a;
                let lim = self.joint.limits.at(i as usize);
                limit_linear_generic(
                    &mut c,
                    &helper,
                    self.joint_id,
                    &a_ctx,
                    &b_ctx,
                    i as usize,
                    [lim.min, lim.max],
                    lock_erp_inv_dt,
                    lock_cfm_coeff,
                    jacobians,
                    j_id_a,
                    j_id_b,
                    body_jacobians,
                    body_jac_start,
                    mprops,
                    colliders_start,
                );
                constraints.write(cons_base + len as usize, c);
                len += 1;
                j_off += stride;
            }
        }

        // Silence unused-variable warning on 2D where ANG_AXES_MASK isn't read.
        let _ = ANG_AXES_MASK;
        let _ = LIN_AXES_MASK;
    }
}

/// Look up the world-space pose of a side. Free-body sides read from the
/// shared `poses` buffer (COM-centered solver pose); multibody sides take
/// their link's `local_to_world` from the multibody workspace (which also
/// stores body-origin = COM-centered, since multibody links have a zeroed
/// `local_com`, as set up by the host pipeline). The `mb` argument is read
/// by value to keep SPIR-V happy and is only meaningful when `side_kind ==
/// SIDE_KIND_MB`.
#[inline]
pub(super) fn side_world_pose(
    side_kind: u32,
    side_id: u32,
    side_link: u32,
    mb: &MultibodyInfo,
    links_workspace: &[MultibodyLinkWorkspace],
    links_start: usize,
    poses: &[Pose],
    colliders_start: usize,
) -> Pose {
    if side_kind == SIDE_KIND_FIXED {
        return Pose::IDENTITY;
    }
    if side_kind == SIDE_KIND_BODY {
        return poses.read(colliders_start + side_id as usize);
    }
    let link_global = links_start + mb.first_link as usize + side_link as usize;
    links_workspace.read(link_global).local_to_world
}
