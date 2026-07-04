//! Contact constraint data structures for the iterative solver.

use crate::{AngVector, Pad, Vector};

#[cfg(feature = "dim3")]
use glamx::Vec2;

#[cfg(feature = "dim2")]
/// Number of tangent constraint directions (2D: one tangent perpendicular to normal).
pub const SUB_LEN: usize = 1;

#[cfg(feature = "dim3")]
/// Number of tangent constraint directions (3D: two tangents in contact plane).
pub const SUB_LEN: usize = 2;

#[cfg(feature = "dim2")]
/// Maximum number of contact points per contact manifold (2D: typically 2).
pub const MAX_CONSTRAINTS_PER_MANIFOLD: usize = 2;

#[cfg(feature = "dim3")]
/// Maximum number of contact points per contact manifold (3D: up to 4).
pub const MAX_CONSTRAINTS_PER_MANIFOLD: usize = 4;

/// Metadata for building a two-body constraint from a contact point.
///
/// This data is used to initialize and update constraints at each solver substep.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")] // Same as dim3, but with a different fields ordering to limit padding.
pub struct TwoBodyConstraintInfos {
    /// Tangent velocity (for conveyor belt effects, currently unused).
    pub tangent_vel: Vector, // TODO PERF: could be one float less, be shared by both contact point infos?

    /// Contact point in body A's local coordinates (for warmstarting).
    /// Stored in local space to detect matching contacts across frames.
    pub local_pt_a: Vector,

    /// Contact point in body B's local coordinates (for warmstarting).
    pub local_pt_b: Vector,

    /// Normal relative velocity at the contact point (for restitution).
    pub normal_vel: f32,

    /// Penetration depth (negative = penetration, positive = separation).
    pub dist: f32,
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")]
pub struct TwoBodyConstraintInfos {
    /// Tangent velocity (for conveyor belt effects, currently unused).
    pub tangent_vel: Vector, // TODO PERF: could be one float less, be shared by both contact point infos?

    /// Normal relative velocity at the contact point (for restitution).
    pub normal_vel: f32,

    /// Contact point in body A's local coordinates (for warmstarting).
    /// Stored in local space to detect matching contacts across frames.
    pub local_pt_a: Vector,
    pub _padding0: u32,

    /// Contact point in body B's local coordinates (for warmstarting).
    pub local_pt_b: Vector,

    /// Penetration depth (negative = penetration, positive = separation).
    pub dist: f32,
}

/// Builder data for constructing constraints from contact manifolds.
///
/// Stores auxiliary information needed to update constraints at each solver substep.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C, align(16))]
pub struct TwoBodyConstraintBuilder {
    /// Information for each contact point in the manifold.
    pub infos: [TwoBodyConstraintInfos; MAX_CONSTRAINTS_PER_MANIFOLD],
}

/// A contact constraint between two rigid bodies.
///
/// Encodes all the data needed to solve a contact constraint, including:
/// - Constraint directions (normal and tangent).
/// - Effective masses and inverse masses.
/// - Solver coefficients (CFM factor, friction limit).
/// - Per-contact-point constraint elements.
// PERF: differentiate two-bodies and one-body constraints?
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwoBodyConstraint {
    /// Contact normal direction from body A's perspective (points away from A).
    /// Normal impulses are applied along this direction to prevent penetration.
    pub dir_a: Vector, // Non-penetration force direction for the first body.
    #[cfg(feature = "dim3")]
    pub _padding0: f32,

    #[cfg(feature = "dim3")]
    /// First tangent direction (3D only, orthogonal to normal).
    /// Used for friction in the contact plane.
    pub tangent_a: Vector, // One of the friction force directions.
    #[cfg(feature = "dim3")]
    pub _padding1: f32,

    /// Inverse mass of body A along each axis.
    /// Used to compute linear velocity changes from impulses.
    pub im_a: Vector,
    #[cfg(feature = "dim3")]
    pub _padding2: f32,

    /// Inverse mass of body B along each axis.
    pub im_b: Vector,
    #[cfg(feature = "dim3")]
    pub _padding3: f32,

    /// Constraint Force Mixing (CFM) factor for regularization.
    /// Softens the constraint: new_impulse = cfm_factor * (old_impulse - ...)
    pub cfm_factor: f32,

    /// Friction coefficient (μ in Coulomb friction model).
    /// Friction impulse magnitude limited to: |f_tangent| <= limit * f_normal
    pub limit: f32,

    /// Index of body A in the solver arrays.
    pub solver_body_a: u32,

    /// Index of body B in the solver arrays.
    pub solver_body_b: u32,

    /// Per-contact-point constraint data (up to MAX_CONSTRAINTS_PER_MANIFOLD).
    pub elements: [TwoBodyConstraintElement; MAX_CONSTRAINTS_PER_MANIFOLD],

    /// Number of active contact points in this manifold (1-4 in 3D, 1-2 in 2D).
    pub len: u32,
    #[cfg(feature = "dim2")]
    pub _padding: u32,
    #[cfg(feature = "dim3")]
    pub _padding: [u32; 3],
}

/// Constraint data for a single contact point.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwoBodyConstraintElement {
    /// Normal constraint: prevents penetration.
    pub normal_part: TwoBodyConstraintNormalPart,

    /// Tangent constraint(s): models friction.
    pub tangent_part: TwoBodyConstraintTangentPart,
}

/// Normal constraint data (non-penetration).
///
/// Implements the constraint: C >= 0 (bodies cannot interpenetrate)
/// Solved as a unilateral constraint (impulse >= 0).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwoBodyConstraintNormalPart {
    /// Angular contribution for body A: r_a × normal
    /// (In 2D: scalar cross product, in 3D: vector cross product)
    pub torque_dir_a: AngVector,
    #[cfg(feature = "dim3")]
    pub _padding0: f32,
    /// `torque_dir_a` multiplied by the inverse angular inertia tensor.
    pub ii_torque_dir_a: AngVector,
    #[cfg(feature = "dim3")]
    pub _padding1: f32,

    /// Angular contribution for body B: r_b × normal
    pub torque_dir_b: AngVector,
    #[cfg(feature = "dim3")]
    pub _padding2: f32,
    /// `torque_dir_b` multiplied by the inverse angular inertia tensor.
    pub ii_torque_dir_b: AngVector,

    /// Right-hand side: target relative velocity (includes bias for correction).
    /// rhs = desired_velocity + bias_velocity
    pub rhs: f32,

    /// Right-hand side without bias term (used in substep iterations).
    /// rhs_wo_bias = desired_velocity (without position correction)
    pub rhs_wo_bias: f32,

    /// Current impulse magnitude for this iteration.
    /// Updated during solving, used for warmstarting next frame.
    pub impulse: f32,

    /// Inverse effective mass: 1 / (m_eff)
    /// where m_eff = projected mass along constraint direction
    pub r: f32,
    #[cfg(feature = "dim3")]
    pub _padding3: f32,
}

/// Tangent constraint data (friction).
///
/// Implements Coulomb friction: |f_tangent| <= μ * f_normal
/// Solved as a bilateral constraint with limits.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwoBodyConstraintTangentPart {
    /// Angular contributions for body A (one per tangent direction).
    pub torque_dir_a: [Pad<AngVector, u32>; SUB_LEN],
    /// `torque_dir_a` multiplied by the inverse angular inertia tensor.
    pub ii_torque_dir_a: [Pad<AngVector, u32>; SUB_LEN],

    /// Angular contributions for body B (one per tangent direction).
    pub torque_dir_b: [Pad<AngVector, u32>; SUB_LEN],
    /// `torque_dir_b` multiplied by the inverse angular inertia tensor.
    pub ii_torque_dir_b: [Pad<AngVector, u32>; SUB_LEN],

    /// Right-hand sides (one per tangent direction).
    pub rhs: [f32; SUB_LEN],

    /// Right-hand sides without bias (one per tangent direction).
    pub rhs_wo_bias: [f32; SUB_LEN],

    #[cfg(feature = "dim2")]
    /// Current tangent impulse (2D: single scalar for one tangent direction).
    pub impulse: [f32; 1],

    #[cfg(feature = "dim2")]
    /// Inverse effective mass (2D: single value).
    pub r: [f32; 1],

    #[cfg(feature = "dim3")]
    /// Current tangent impulses (3D: vec2 for two tangent directions).
    pub impulse: Vec2,

    #[cfg(feature = "dim3")]
    /// Inverse effective mass components (3D: 3 values for 2x2 mass matrix).
    /// r[0] = r_00, r[1] = r_11, r[2] = r_01 (symmetric, so r_10 = r_01)
    pub r: [f32; 3],
    #[cfg(feature = "dim3")]
    pub _padding0: [u32; 3],
}
