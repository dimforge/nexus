//! Shared plain-old-data types of the multibody impulse-joint pipeline:
//! per-joint builder / axis-constraint records and the side-kind tags.

use crate::dynamics::joint::{GenericJoint, SPATIAL_DIM};

/// Maximum unit-axis constraints any single impulse joint can produce.
///
/// `SPATIAL_DIM * 2` covers a free joint with both limits AND motors enabled
/// on every axis (rapier emits limits and motors as separate constraints).
pub const MAX_AXIS_CONSTRAINTS: u32 = (SPATIAL_DIM as u32) * 2;

/// Sentinel "no body" — used when a side is `Fixed` (rapier `LinkOrBody::Fixed`).
pub const SIDE_FIXED: u32 = u32::MAX;

#[cfg(feature = "dim2")]
pub(super) const DIM_USIZE: usize = 2;
#[cfg(feature = "dim3")]
pub(super) const DIM_USIZE: usize = 3;

/// Tag distinguishing how each side of a generic impulse joint connects
/// to the solver state.
///
/// Mirrors rapier's `LinkOrBody`:
///   * `0` — Free rigid body. `body_id` indexes into the per-batch solver
///     velocity / mprops buffer; `ndofs` is `SPATIAL_DIM`.
///   * `1` — Multibody link. `mb_id` indexes the per-batch
///     `multibody_info`; `link_id` indexes the link within the multibody;
///     `ndofs` is `mb.ndofs`.
///   * `2` — Static fixed pose. No DOFs, no velocity update.
pub const SIDE_KIND_BODY: u32 = 0;
pub const SIDE_KIND_MB: u32 = 1;
pub const SIDE_KIND_FIXED: u32 = 2;

/// Per-impulse-joint static descriptor — the GPU mirror of rapier's
/// `JointGenericExternalConstraintBuilder`.
///
/// One slot per joint that touches at least one multibody. The init kernel
/// reads it to (re)build the joint's axis constraints in the per-batch
/// `constraints` slab.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbImpulseJointBuilder {
    /// Joint description — frames already shifted into solver-body
    /// (COM-centered) space at host time, mirroring
    /// `GenericJoint::transform_to_solver_body_space`.
    pub joint: GenericJoint,

    /// `SIDE_KIND_BODY` / `SIDE_KIND_MB` / `SIDE_KIND_FIXED`.
    pub side_a_kind: u32,
    /// Free-body local id (when `SIDE_KIND_BODY`) or multibody index in
    /// the per-batch `multibody_info` (when `SIDE_KIND_MB`). `SIDE_FIXED`
    /// when `side_a_kind == SIDE_KIND_FIXED`.
    pub side_a_id: u32,
    /// Link index within the multibody (only meaningful for `SIDE_KIND_MB`).
    pub side_a_link: u32,
    /// Source impulse-joint id, for impulse writeback.
    pub joint_id: u32,

    pub side_b_kind: u32,
    pub side_b_id: u32,
    pub side_b_link: u32,
    /// First constraint slot (in the per-batch constraints slab) reserved
    /// for this joint's axis constraints.
    pub constraint_id: u32,

    /// First float index (in the per-batch jacobians buffer) reserved for
    /// this joint.
    pub jacobian_offset: u32,
    /// Total floats reserved for this joint's jacobian block.
    pub jacobian_capacity: u32,
    /// Pad to GenericJoint's alignment (16 bytes in 3D — see ImpulseJoint).
    #[cfg(feature = "dim3")]
    pub _pad0: [u32; 2],
}

/// One unit-axis generic impulse-joint constraint — the GPU mirror of
/// rapier's `GenericJointConstraint`.
///
/// `kind` values: `0` = inactive / unused slot, `1` = active.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbImpulseJointConstraint {
    /// `0` = inactive, `1` = active.
    pub kind: u32,
    /// Joint id of the source impulse joint (for impulse writeback).
    pub joint_id: u32,
    /// Writeback type — mirrors rapier's `WritebackId`:
    ///   * `0` = `Dof(writeback_axis)` (lock)
    ///   * `1` = `Limit(writeback_axis)`
    ///   * `2` = `Motor(writeback_axis)`
    pub writeback_kind: u32,
    /// Axis index for the writeback (0..SPATIAL_DIM).
    pub writeback_axis: u32,

    pub side_a_kind: u32,
    pub side_a_id: u32,
    pub side_a_link: u32,
    pub ndofs_a: u32,

    pub side_b_kind: u32,
    pub side_b_id: u32,
    pub side_b_link: u32,
    pub ndofs_b: u32,

    /// First float of `J_a` in the per-batch jacobians buffer.
    pub j_id_a: u32,
    /// First float of `J_b`.
    pub j_id_b: u32,
    pub _pad0: [u32; 2],

    pub impulse: f32,
    pub impulse_lo: f32,
    pub impulse_hi: f32,
    pub inv_lhs: f32,

    pub rhs: f32,
    pub rhs_wo_bias: f32,
    pub cfm_coeff: f32,
    pub cfm_gain: f32,
}
