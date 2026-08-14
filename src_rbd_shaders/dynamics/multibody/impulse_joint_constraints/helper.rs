//! `JointConstraintHelper` and the generic per-axis constraint builders
//! (lock / limit / motor, linear and angular) mirroring rapier.

#[cfg(feature = "dim3")]
use glamx::{Mat3, Quat, Vec3};

use khal_std::index::MaybeIndexUnchecked;

use crate::dynamics::body::WorldMassProperties;
use crate::dynamics::joint::JointMotor;
use crate::dynamics::smallest_abs_diff_between_sin_angles;
use crate::{AngVector, MAX_FLT, Pose, Vector, rotation_to_matrix};

use super::super::types::MultibodyInfo;
use super::jacobians::*;
use super::types::*;
use crate::utils::linalg::VSlice;

/// `JointConstraintHelper`-equivalent: precomputed per-joint quantities used
/// by `lock_*`, `limit_*`, `motor_*`. Mirrors the homonymous rapier struct.
pub(super) struct JointConstraintHelper {
    #[cfg(feature = "dim2")]
    basis: glamx::Mat2,
    #[cfg(feature = "dim2")]
    cmat1_basis: [f32; 2],
    #[cfg(feature = "dim2")]
    cmat2_basis: [f32; 2],
    #[cfg(feature = "dim3")]
    basis: Mat3,
    #[cfg(feature = "dim3")]
    cmat1_basis: Mat3,
    #[cfg(feature = "dim3")]
    cmat2_basis: Mat3,
    #[cfg(feature = "dim3")]
    ang_basis: Mat3,
    lin_err: Vector,
    #[cfg(feature = "dim2")]
    ang_err: crate::Rotation,
    #[cfg(feature = "dim3")]
    ang_err: [f32; 3],
    /// Real (scalar) part of the angular error quaternion, sign-matched to
    /// `ang_err`. Needed to recover the re-centered angle of a limit row.
    #[cfg(feature = "dim3")]
    ang_err_w: f32,
}

#[cfg(feature = "dim3")]
pub(super) fn quat_dot(a: Quat, b: Quat) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

#[cfg(feature = "dim3")]
pub(super) fn rotation_diff_conj1_2(q1: Quat, q2: Quat) -> Mat3 {
    let v1 = q1.xyz();
    let v2 = q2.xyz();
    let w1 = q1.w;
    let w2 = q2.w;

    let tensor_product = Mat3::from_cols(v1 * v2.x, v1 * v2.y, v1 * v2.z);
    let diag = Mat3::from_cols(
        Vec3::new(w1 * w2, 0.0, 0.0),
        Vec3::new(0.0, w1 * w2, 0.0),
        Vec3::new(0.0, 0.0, w1 * w2),
    );
    let cross_sum = gcross_matrix3(v1 * w2 + v2 * w1);
    let cross_v1 = gcross_matrix3(v1);
    let cross_v2 = gcross_matrix3(v2);

    (tensor_product + diag - cross_sum + cross_v1 * cross_v2) * 0.5
}

#[cfg(feature = "dim3")]
pub(super) fn gcross_matrix3(r: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, r.z, -r.y),
        Vec3::new(-r.z, 0.0, r.x),
        Vec3::new(r.y, -r.x, 0.0),
    )
}

#[cfg(feature = "dim2")]
pub(super) fn gcross_matrix2(r: glamx::Vec2) -> glamx::Vec2 {
    glamx::Vec2::new(-r.y, r.x)
}

pub(super) fn new_helper(
    frame1_in: Pose,
    frame2: Pose,
    world_com1: Vector,
    world_com2: Vector,
    locked_lin_axes: u32,
) -> JointConstraintHelper {
    let mut frame1 = frame1_in;
    let basis = rotation_to_matrix(frame1.rotation);
    let lin_err = frame2.translation - frame1.translation;

    // Snap the frame1 origin to frame2's center along free axes (rapier's
    // `JointConstraintHelper::new`), so the lock jacobians act at a single
    // point rather than at the now-relatively-displaced anchor1.
    let mut new_center1 = frame2.translation;
    for i in 0..DIM_USIZE {
        if (locked_lin_axes & (1u32 << i)) != 0 {
            let axis = basis.col(i);
            new_center1 -= axis * lin_err.dot(axis);
        }
    }
    frame1.translation = new_center1;

    let r1 = frame1.translation - world_com1;
    let r2 = frame2.translation - world_com2;

    #[cfg(feature = "dim2")]
    {
        let cmat1 = gcross_matrix2(r1);
        let cmat2 = gcross_matrix2(r2);
        let ang_err = frame1.rotation.inverse() * frame2.rotation;
        JointConstraintHelper {
            basis,
            cmat1_basis: [cmat1.dot(basis.col(0)), cmat1.dot(basis.col(1))],
            cmat2_basis: [cmat2.dot(basis.col(0)), cmat2.dot(basis.col(1))],
            lin_err,
            ang_err,
        }
    }
    #[cfg(feature = "dim3")]
    {
        let cmat1 = gcross_matrix3(r1);
        let cmat2 = gcross_matrix3(r2);
        let mut ang_basis = rotation_diff_conj1_2(frame1.rotation, frame2.rotation).transpose();
        let quat_err = frame1.rotation.inverse() * frame2.rotation;
        let sgn = if quat_dot(frame1.rotation, frame2.rotation) > 0.0 {
            1.0
        } else {
            -1.0
        };
        ang_basis *= sgn;
        let ang_err = [quat_err.x * sgn, quat_err.y * sgn, quat_err.z * sgn];
        JointConstraintHelper {
            basis,
            cmat1_basis: cmat1 * basis,
            cmat2_basis: cmat2 * basis,
            ang_basis,
            lin_err,
            ang_err,
            ang_err_w: quat_err.w * sgn,
        }
    }
}

/// Row parameters of an angular limit allowing `[min, max]` (radians): the row
/// measures the angle from the middle of the range against `±half_range`.
pub(super) struct AngularLimitParams {
    /// Middle of the allowed range.
    center: f32,
    /// Half the allowed range. Larger than π means the row is unconstrained.
    half_range: f32,
}

impl AngularLimitParams {
    pub(super) fn new(min: f32, max: f32) -> Self {
        let half_range = (max - min) * 0.5;
        // A range of a full turn or more is indistinguishable from "no limit"
        // for an angle read off a relative rotation. NaN bounds are rejected
        // here too, before they poison the row.
        if half_range.is_nan() || half_range >= core::f32::consts::PI {
            return Self {
                center: 0.0,
                half_range: 10.0,
            };
        }

        Self {
            center: (min + max) * 0.5,
            half_range,
        }
    }

    /// Symmetric bounds the re-centered angle is tested against.
    #[inline]
    pub(super) fn bounds(&self) -> [f32; 2] {
        [-self.half_range, self.half_range]
    }
}

#[inline]
pub(super) fn pseudo_inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

/// Bundle of per-side context the `lock_jacobians_generic` analog needs to
/// fill the jacobians for both sides of a generic constraint and stamp the
/// constraint header.
///
/// `mb` is read by value (not behind a reference / Option) so the SPIR-V
/// backend doesn't need pointers-to-arbitrary-storage. `side_kind` gates
/// whether `mb` carries meaningful data.
#[derive(Clone, Copy)]
pub(super) struct SideCtx {
    pub(super) side_kind: u32,
    pub(super) side_id: u32,
    pub(super) side_link: u32,
    pub(super) ndofs: u32,
    pub(super) mb: MultibodyInfo,
}

/// Mirrors rapier's `JointConstraintHelper::lock_jacobians_generic`: pack
/// per-side `Jᵀ` + `M⁻¹·Jᵀ` into the jacobians buffer and stamp a fresh
/// constraint with the side metadata. Caller fills the rhs / impulse-bound
/// fields afterwards.
pub(super) fn lock_jacobians_generic(
    out: &mut MbImpulseJointConstraint,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    a: &SideCtx,
    b: &SideCtx,
    lin_jac: Vector,
    ang_jac1: AngVector,
    ang_jac2: AngVector,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    // Loop-closure path: both attachment links are in the SAME multibody. Fold
    // the two sides into a single relative jacobian `J_rel = J_b - J_a` on the
    // "B" side and mark side A inactive (`ndofs_a = 0`). The solver's
    // `vel2 - vel1`, the `dot_j_wj` effective mass, and the impulse apply then
    // read the relative quantities directly, with no solver-side changes.
    // Mirrors rapier's `lock_jacobians_generic` same-multibody branch. 3D only;
    // 2D keeps the separate-block path (no 2D loop-closure models in practice).
    #[cfg(feature = "dim3")]
    {
        if a.side_kind == SIDE_KIND_MB && b.side_kind == SIDE_KIND_MB && a.side_id == b.side_id {
            out.side_a_kind = a.side_kind;
            out.side_a_id = a.side_id;
            out.side_a_link = a.side_link;
            out.ndofs_a = 0;
            out.j_id_a = j_id_a;

            out.side_b_kind = b.side_kind;
            out.side_b_id = b.side_id;
            out.side_b_link = b.side_link;
            out.ndofs_b = b.ndofs;
            out.j_id_b = j_id_b;

            fill_relative_mb_jacobians(
                jacobians,
                j_id_b,
                &b.mb,
                a.side_link,
                lin_jac,
                ang_jac1,
                b.side_link,
                lin_jac,
                ang_jac2,
                body_jacobians,
                il,
            );

            out.kind = 1;
            out.impulse = 0.0;
            out.impulse_lo = -MAX_FLT;
            out.impulse_hi = MAX_FLT;
            out.inv_lhs = 0.0;
            out.rhs = 0.0;
            out.rhs_wo_bias = 0.0;
            out.cfm_coeff = 0.0;
            out.cfm_gain = 0.0;
            return;
        }
    }

    out.side_a_kind = a.side_kind;
    out.side_a_id = a.side_id;
    out.side_a_link = a.side_link;
    out.ndofs_a = a.ndofs;
    out.j_id_a = j_id_a;
    if a.side_kind == SIDE_KIND_BODY {
        fill_body_jacobians(
            jacobians,
            j_id_a,
            a.side_id,
            lin_jac,
            ang_jac1,
            mprops,
            colliders_start,
        );
    } else if a.side_kind == SIDE_KIND_MB {
        fill_mb_jacobians(
            jacobians,
            j_id_a,
            &a.mb,
            a.side_link,
            lin_jac,
            ang_jac1,
            body_jacobians,
            il,
        );
    }

    out.side_b_kind = b.side_kind;
    out.side_b_id = b.side_id;
    out.side_b_link = b.side_link;
    out.ndofs_b = b.ndofs;
    out.j_id_b = j_id_b;
    if b.side_kind == SIDE_KIND_BODY {
        fill_body_jacobians(
            jacobians,
            j_id_b,
            b.side_id,
            lin_jac,
            ang_jac2,
            mprops,
            colliders_start,
        );
    } else if b.side_kind == SIDE_KIND_MB {
        fill_mb_jacobians(
            jacobians,
            j_id_b,
            &b.mb,
            b.side_link,
            lin_jac,
            ang_jac2,
            body_jacobians,
            il,
        );
    }

    out.kind = 1;
    out.impulse = 0.0;
    out.impulse_lo = -MAX_FLT;
    out.impulse_hi = MAX_FLT;
    out.inv_lhs = 0.0; // filled by `finalize_generic_constraints`.
    out.rhs = 0.0;
    out.rhs_wo_bias = 0.0;
    out.cfm_coeff = 0.0;
    out.cfm_gain = 0.0;
}

impl MbImpulseJointConstraint {
    /// Compute `J · W·J` over both sides — matches the dot product rapier
    /// performs in `finalize_generic_constraints` to set `inv_lhs`.
    #[inline]
    pub(super) fn dot_j_wj(&self, jacobians: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..self.ndofs_a {
            let j = jacobians.read(self.j_id_a as usize + i as usize);
            let wj = jacobians.read(wj_id(self.j_id_a, self.ndofs_a) + i as usize);
            acc += j * wj;
        }
        for i in 0..self.ndofs_b {
            let j = jacobians.read(self.j_id_b as usize + i as usize);
            let wj = jacobians.read(wj_id(self.j_id_b, self.ndofs_b) + i as usize);
            acc += j * wj;
        }
        acc
    }

    /// Populate `inv_lhs` / `cfm_gain` for each filled constraint. Mirrors
    /// rapier's `finalize_generic_constraints` (orthogonalization branch is
    /// disabled there too — `ORTHOGONALIZE = false`).
    #[inline]
    pub(super) fn finalize_generic_constraint(&mut self, jacobians: &[f32]) {
        let dot_jj = self.dot_j_wj(jacobians);
        let cfm_gain = dot_jj * self.cfm_coeff + self.cfm_gain;
        self.inv_lhs = pseudo_inv(dot_jj + cfm_gain);
        self.cfm_gain = cfm_gain;
    }
}

impl JointConstraintHelper {
    #[inline]
    #[cfg(feature = "dim2")]
    pub(super) fn ang_jac_for_axis(&self, _axis: usize) -> AngVector {
        1.0
    }

    #[inline]
    #[cfg(feature = "dim3")]
    pub(super) fn ang_jac_for_axis(&self, axis: usize) -> AngVector {
        self.ang_basis.col(axis)
    }

    #[inline]
    #[cfg(feature = "dim2")]
    pub(super) fn axis_ang_jac(&self, _axis: usize) -> AngVector {
        1.0
    }

    #[inline]
    #[cfg(feature = "dim3")]
    pub(super) fn axis_ang_jac(&self, axis: usize) -> AngVector {
        self.basis.col(axis)
    }

    /// Angle of the relative rotation about `axis`, measured from the middle of
    /// the limit's range and wrapped to `(-π, π]`. Its gradient is the plain
    /// joint axis, so it pairs with [`Self::axis_ang_jac`].
    #[inline]
    #[cfg(feature = "dim3")]
    pub(super) fn recentered_angle(&self, axis: usize, limit: &AngularLimitParams) -> f32 {
        let c_cos = crate::cos(limit.center * 0.5);
        let c_sin = crate::sin(limit.center * 0.5);
        let x = self.ang_err.read(axis);
        let w = self.ang_err_w;
        let sin_half = c_cos * x - c_sin * w;
        let cos_half = c_cos * w + c_sin * x;
        // The re-centered HALF angle; doubling it must wrap back to `(-π, π]`,
        // which is a ±π shift whenever the half angle leaves `(-π/2, π/2]`.
        let half = crate::atan2(sin_half, cos_half);
        let shift = if half >= 0.0 {
            core::f32::consts::PI
        } else {
            -core::f32::consts::PI
        };
        let wrapped_half = if crate::abs(half) > core::f32::consts::FRAC_PI_2 {
            half - shift
        } else {
            half
        };
        wrapped_half * 2.0
    }

    #[inline]
    #[cfg(feature = "dim2")]
    pub(super) fn recentered_angle(&self, _axis: usize, limit: &AngularLimitParams) -> f32 {
        let d = crate::rotation_angle(self.ang_err) - limit.center;
        // `atan2(sin, cos)` wraps the result to `(-π, π]`.
        crate::atan2(crate::sin(d), crate::cos(d))
    }

    #[inline]
    #[cfg(feature = "dim2")]
    pub(super) fn ang_err_axis(&self, _axis: usize) -> f32 {
        crate::sin(crate::rotation_angle(self.ang_err) * 0.5) * 2.0 * 0.5 // = sin(a/2)
    }

    #[inline]
    #[cfg(feature = "dim3")]
    pub(super) fn ang_err_axis(&self, axis: usize) -> f32 {
        self.ang_err.read(axis)
    }
}

/// Lock one linear axis (`Dof(axis)` writeback). Mirrors rapier's
/// `JointConstraintHelper::lock_linear_generic`.
#[allow(clippy::too_many_arguments)]
pub(super) fn lock_linear_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    locked_axis: usize,
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let lin_jac = helper.basis.col(locked_axis);
    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(locked_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(locked_axis);
    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col(locked_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col(locked_axis);

    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        lin_jac,
        ang_jac1,
        ang_jac2,
        body_jacobians,
        il,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 0;
    out.writeback_axis = locked_axis as u32;
    out.cfm_coeff = cfm_coeff;
    let rhs_bias = lin_jac.dot(helper.lin_err) * erp_inv_dt_val;
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
}

/// Lock one angular axis. Mirrors `lock_angular_generic`.
#[allow(clippy::too_many_arguments)]
pub(super) fn lock_angular_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    locked_axis: usize,
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let ang_jac = helper.ang_jac_for_axis(locked_axis);
    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        Vector::ZERO,
        ang_jac,
        ang_jac,
        body_jacobians,
        il,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 0;
    out.writeback_axis = (DIM_USIZE + locked_axis) as u32;
    out.cfm_coeff = cfm_coeff;
    let rhs_bias = helper.ang_err_axis(locked_axis) * erp_inv_dt_val;
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
}

/// Limit one linear axis. Mirrors `limit_linear_generic`.
#[allow(clippy::too_many_arguments)]
pub(super) fn limit_linear_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    limited_axis: usize,
    limits: [f32; 2],
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    max_corr_velocity: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let lin_jac = helper.basis.col(limited_axis);
    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(limited_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(limited_axis);
    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col(limited_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col(limited_axis);

    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        lin_jac,
        ang_jac1,
        ang_jac2,
        body_jacobians,
        il,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 1;
    out.writeback_axis = limited_axis as u32;
    out.cfm_coeff = cfm_coeff;

    let dist = helper.lin_err.dot(lin_jac);
    let min_enabled = dist <= limits[0];
    let max_enabled = limits[1] <= dist;
    // Cap the bias so a deep violation recovers over a few steps instead of
    // catapulting the bodies.
    let rhs_bias = (((dist - limits[1]).max(0.0) - (limits[0] - dist).max(0.0)) * erp_inv_dt_val)
        .max(-max_corr_velocity)
        .min(max_corr_velocity);
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
    out.impulse_lo = if min_enabled { -MAX_FLT } else { 0.0 };
    out.impulse_hi = if max_enabled { MAX_FLT } else { 0.0 };
}

/// Limit one angular axis. Mirrors `limit_angular_generic`.
#[allow(clippy::too_many_arguments)]
pub(super) fn limit_angular_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    limited_axis: usize,
    limits: [f32; 2],
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    max_corr_velocity: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    // The row measures the wrapped angle from the middle of the allowed range;
    // its gradient is the plain joint axis, like the angular motor row.
    let limit = AngularLimitParams::new(limits[0], limits[1]);
    let ang_jac = helper.axis_ang_jac(limited_axis);
    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        Vector::ZERO,
        ang_jac,
        ang_jac,
        body_jacobians,
        il,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 1;
    out.writeback_axis = (DIM_USIZE + limited_axis) as u32;
    out.cfm_coeff = cfm_coeff;

    let ang_limits = limit.bounds();
    let ang = helper.recentered_angle(limited_axis, &limit);
    let min_enabled = ang <= ang_limits[0];
    let max_enabled = ang_limits[1] <= ang;
    // Cap the bias so a deep violation recovers over a few steps instead of
    // catapulting the bodies.
    let rhs_bias = (((ang - ang_limits[1]).max(0.0) - (ang_limits[0] - ang).max(0.0))
        * erp_inv_dt_val)
        .max(-max_corr_velocity)
        .min(max_corr_velocity);
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
    out.impulse_lo = if min_enabled { -MAX_FLT } else { 0.0 };
    out.impulse_hi = if max_enabled { MAX_FLT } else { 0.0 };
}

/// Linear motor. Mirrors `motor_linear_generic`.
#[allow(clippy::too_many_arguments)]
pub(super) fn motor_linear_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    motor_axis: usize,
    motor: &JointMotor,
    dt: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let mp = motor.motor_params(dt);
    let lin_jac = helper.basis.col(motor_axis);
    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(motor_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(motor_axis);
    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col(motor_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col(motor_axis);

    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        lin_jac,
        ang_jac1,
        ang_jac2,
        body_jacobians,
        il,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 2;
    out.writeback_axis = motor_axis as u32;

    let mut rhs_wo_bias = 0.0;
    if mp.erp_inv_dt != 0.0 {
        let dist = helper.lin_err.dot(lin_jac);
        rhs_wo_bias += (dist - mp.target_pos) * mp.erp_inv_dt;
    }
    rhs_wo_bias += -mp.target_vel;

    out.cfm_coeff = mp.cfm_coeff;
    out.cfm_gain = mp.cfm_gain;
    out.impulse_lo = -mp.max_impulse;
    out.impulse_hi = mp.max_impulse;
    out.rhs = rhs_wo_bias;
    out.rhs_wo_bias = rhs_wo_bias;
}

/// Angular motor. Mirrors `motor_angular_generic`.
#[allow(clippy::too_many_arguments)]
pub(super) fn motor_angular_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    motor_axis: usize,
    motor: &JointMotor,
    dt: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    il: VSlice,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let mp = motor.motor_params(dt);
    let ang_jac = helper.axis_ang_jac(motor_axis);
    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        Vector::ZERO,
        ang_jac,
        ang_jac,
        body_jacobians,
        il,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 2;
    out.writeback_axis = (DIM_USIZE + motor_axis) as u32;

    let mut rhs_wo_bias = 0.0;
    if mp.erp_inv_dt != 0.0 {
        #[cfg(feature = "dim3")]
        let s_ang_dist = helper.ang_err.read(motor_axis);
        #[cfg(feature = "dim2")]
        let s_ang_dist = crate::sin(crate::rotation_angle(helper.ang_err) * 0.5);
        let s_target_ang = crate::sin(mp.target_pos * 0.5);
        rhs_wo_bias +=
            smallest_abs_diff_between_sin_angles(s_ang_dist, s_target_ang) * mp.erp_inv_dt;
    }
    rhs_wo_bias += -mp.target_vel;

    out.cfm_coeff = mp.cfm_coeff;
    out.cfm_gain = mp.cfm_gain;
    out.impulse_lo = -mp.max_impulse;
    out.impulse_hi = mp.max_impulse;
    out.rhs = rhs_wo_bias;
    out.rhs_wo_bias = rhs_wo_bias;
}
