//! Per-link / per-multibody data structures shared across all multibody
//! kernels.

#[cfg(feature = "dim3")]
use glamx::{Quat, Vec3};
#[cfg(feature = "dim2")]
use glamx::{Rot2, Vec2};

use crate::Pose;
use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::dynamics::joint::{GenericJoint, SPATIAL_DIM};

/// Max degrees of freedom any single joint can expose.
///
/// In 3D this is 6 (a free root joint). In 2D it is 3 (2 lin + 1 ang).
/// Equivalent to `SPATIAL_DIM`.
pub const MAX_JOINT_DOFS: usize = SPATIAL_DIM;

/// Maximum number of simultaneously-active multibody contact points per
/// multibody.
/// TODO: make this configurable/auto-resizeable
pub const MAX_MB_CONTACTS_PER_MB: u32 = 128;

/// Number of constraint slots reserved per contact point — one normal +
/// `DIM-1` friction tangents (Coulomb friction). Mirrors rapier's
/// `ContactConstraintNormalPart` + `ContactConstraintTangentPart` layout.
#[cfg(feature = "dim2")]
pub const CONTACT_CONSTRAINTS_PER_POINT: u32 = 2;
/// Number of constraint slots reserved per contact point — one normal +
/// `DIM-1` friction tangents (Coulomb friction). Mirrors rapier's
/// `ContactConstraintNormalPart` + `ContactConstraintTangentPart` layout.
#[cfg(feature = "dim3")]
pub const CONTACT_CONSTRAINTS_PER_POINT: u32 = 3;

/// Total constraint slots reserved per multibody (= contact points × DIM).
pub const MAX_MB_CONTACT_CONSTRAINTS_PER_MB: u32 =
    MAX_MB_CONTACTS_PER_MB * CONTACT_CONSTRAINTS_PER_POINT;

/// `kind` value: inactive / unused slot.
pub const MB_CONTACT_KIND_INACTIVE: u32 = 0;
/// `kind` value: active normal-direction (non-penetration) constraint.
pub const MB_CONTACT_KIND_NORMAL: u32 = 1;
/// `kind` value: active friction tangent constraint. Its impulse is
/// dynamically clamped to `±(friction_coeff · normal.impulse)` at solve
/// time, where `normal` is the constraint at slot
/// `normal_constraint_slot` (relative to the multibody's `cons_base`).
pub const MB_CONTACT_KIND_TANGENT: u32 = 2;

/// Joint-constraint `kind`: unused slot.
pub const MB_JOINT_KIND_INACTIVE: u32 = 0;
/// Joint-constraint `kind`: active limit.
pub const MB_JOINT_KIND_LIMIT: u32 = 1;
/// Joint-constraint `kind`: active motor.
pub const MB_JOINT_KIND_MOTOR: u32 = 2;
/// Joint-constraint `kind`: limit slot that is inactive this substep (its
/// joint coordinate is inside the limits). The solve skips it; the slot
/// still gets a valid M⁻¹ column / `inv_lhs` / `cfm_gain` in case it needs
/// to be enabled at a next substep.
pub const MB_JOINT_KIND_LIMIT_INACTIVE: u32 = 3;
/// Joint-constraint `kind`: holonomic DoF coupling `q2 = coeff·q1 + offset`
/// (rapier's `MultibodyDofCoupling`): one bilateral constraint with jacobian
/// `J = e_{dof_id} − coeff·e_{dof2_id}` and a bias pulling the position drift
/// back to zero.
pub const MB_JOINT_KIND_COUPLING: u32 = 4;
/// Joint-constraint `kind`: dry joint friction (MJCF `frictionloss`). A row
/// with jacobian `J = e_{dof_id}`, zero target velocity and no position
/// residual, whose impulse is clamped to `±frictionloss·dt` and which carries
/// the shared joint CFM softness. MuJoCo models friction loss this way rather
/// than as a `-f·sign(q̇)` force: the bound is load-independent (not Coulomb
/// friction), and only a constraint can hold a DoF at rest instead of
/// chattering around zero velocity.
pub const MB_JOINT_KIND_FRICTION: u32 = 5;

/// Sentinel marking a link with no parent (the root).
pub const MULTIBODY_ROOT: u32 = u32::MAX;

/// Per-link static configuration: backing body, parent, joint definition.
///
/// Written once at init time.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MultibodyLinkStatic {
    // TODO: change the name to `MultibodyLink` ?
    /// Index of the rigid body backing this link in the shared body buffers.
    pub rb_id: u32,
    /// Parent link index within the owning multibody. `MULTIBODY_ROOT` for the root.
    pub parent_link_id: u32,
    /// Index of the owning multibody in the `multibody_info` tensor.
    pub multibody_id: u32,
    /// Starting column (in the jacobian / mass-matrix / gen-force tensors) for this
    /// link's DOFs. Assembly ids are contiguous and parent-before-child.
    pub assembly_id: u32,
    /// Number of DOFs this joint contributes.
    pub ndofs: u32,
    /// 1 if this joint's generalized velocities are user-controlled (ignored by the
    /// LU solve). 0 otherwise.
    pub kinematic: u32,
    /// Pad to 16-byte alignment before `data` in 3D (Pose3 starts with a Quat).
    /// In 2D, Pose2 only needs 4-byte alignment so no extra padding is required.
    #[cfg(feature = "dim3")]
    pub _pad0: [u32; 2],
    /// Joint configuration — reused directly from the impulse-joint infrastructure.
    pub data: GenericJoint,
    /// Per-link mass properties in body-local coordinates. `GenericJoint` ends
    /// 16-byte aligned, so the (Quat-leading, in 3D) `LocalMassProperties` lands
    /// straddle-free.
    pub local_mprops: LocalMassProperties,
}

/// Per-link workspace updated every step.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")]
pub struct MultibodyLinkWorkspace {
    /// Accumulated joint rotation (fed to `body_to_parent`). Quat in 3D.
    pub joint_rot: Quat,
    /// Generalized coordinates for this joint. Only the first `ndofs` entries are
    /// meaningful. Free linear DOFs come first (in axis order), then free angular DOFs.
    pub coords: [f32; MAX_JOINT_DOFS],
    /// Pad: `joint_rot` (16) + `coords` (24) = 40; need 8 more before Pose (align 16).
    pub _pad0: [u32; 2],
    /// Local-to-parent transform.
    pub local_to_parent: Pose,
    /// Local-to-world transform (the link's body pose).
    pub local_to_world: Pose,
    /// Vector (world frame) from the parent COM to the joint frame on the parent side.
    pub shift02: Vec3,
    pub _pad1: u32,
    /// Vector (world frame) from the joint frame on the child side to this link's COM.
    pub shift23: Vec3,
    pub _pad2: u32,
    /// World-space spatial velocity added by this joint (rapier's `link.joint_velocity`).
    pub joint_velocity: Velocity,
    /// World-space total rigid-body velocity (rapier's `rb.vels`). Used by the
    /// Coriolis / gyroscopic assembly. Computed by `gpu_mb_update_velocities`.
    pub rb_vels: Velocity,
    /// Per-link kinematic acceleration (rapier's `workspace.accs[i]`).
    /// Populated by the Coriolis  variant of `apply_gravity`.
    pub kinematic_acc: Velocity,
    /// User-applied force on this link, in world space.
    pub external_force: Vec3,
    /// Per-link multiplier on the global gravity.
    pub gravity_scale: f32,
    /// User-applied torque on this link, in world space.
    pub external_torque: Vec3,
    pub _pad3: u32,
}

/// Per-link workspace updated every step.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")]
pub struct MultibodyLinkWorkspace {
    /// Accumulated joint rotation (fed to `body_to_parent`). Rot2 in 2D.
    pub joint_rot: Rot2,
    /// Generalized coordinates for this joint. Only the first `ndofs` entries are
    /// meaningful. Free linear DOFs come first (in axis order), then the free
    /// angular DOF (only one in 2D).
    pub coords: [f32; MAX_JOINT_DOFS],
    /// Pad: `joint_rot` (8) + `coords` (12) = 20; Pose2 contains a Vec2 which
    /// std430 aligns to 8, so 4 bytes of padding are required here.
    pub _pad0: u32,
    /// Local-to-parent transform.
    pub local_to_parent: Pose,
    /// Local-to-world transform (the link's body pose).
    pub local_to_world: Pose,
    /// Vector (world frame) from the parent COM to the joint frame on the parent side.
    pub shift02: Vec2,
    /// Vector (world frame) from the joint frame on the child side to this link's COM.
    pub shift23: Vec2,
    /// World-space spatial velocity added by this joint (rapier's `link.joint_velocity`).
    pub joint_velocity: Velocity,
    /// World-space total rigid-body velocity (rapier's `rb.vels`).
    pub rb_vels: Velocity,
    /// Per-link kinematic acceleration (rapier's `workspace.accs[i]`).
    pub kinematic_acc: Velocity,
    /// User-applied force on this link, in world space.
    pub external_force: Vec2,
    /// User-applied torque on this link.
    pub external_torque: f32,
    /// Per-link multiplier on the global gravity.
    pub gravity_scale: f32,
}

/// One unit (1-DOF) constraint generated from a multibody joint's limit or
/// motor, exactly mirroring rapier's `unit_joint_*_constraint` output.
///
/// Each constraint targets a single generalized DOF. The "second jacobian row"
/// — the column of `M⁻¹` corresponding to that DOF — lives in a separate flat
/// buffer (`joint_constraint_columns`) so that the solver can update all DOFs of
/// the multibody when applying an impulse.
///
/// `kind` values: 0 = inactive, 1 = limit, 2 = motor.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MultibodyJointConstraint {
    // TODO: rename to MultibodyUnitJointConstraint?
    /// Index of the constrained DOF, relative to the multibody's `first_dof`.
    pub dof_id: u32,
    /// See `MB_JOINT_KIND_*`: 0 = inactive (skipped by the solver), 1 = limit,
    /// 2 = motor, 3 = inactive limit, 4 = DoF coupling.
    pub kind: u32,
    /// Packed `(link_id | axis << 16)` of the constrained DOF, used by the
    /// per-substep refresh to re-read the joint coordinate.
    pub _kind_extra: u32,
    /// Second constrained DOF for coupling rows (`J = e_{dof_id} −
    /// coupling_coeff·e_{dof2_id}`). Zero (together with `coupling_coeff = 0`)
    /// for limits / motors, making the solve's generalized `J·v` collapse to
    /// the plain single-DOF form.
    pub dof2_id: u32,

    /// `J·v` reference + bias velocity (rapier's `rhs`, includes positional bias).
    pub rhs: f32,
    /// Same as `rhs` minus the positional bias (rapier's `rhs_wo_bias`); used by
    /// the post-substep "remove bias" pass.
    pub rhs_wo_bias: f32,
    /// `1 / (Jᵀ·M⁻¹·J) = 1 / M⁻¹[d, d]`.
    pub inv_lhs: f32,
    /// Accumulated impulse (warmstart-able across substeps).
    pub impulse: f32,

    /// Lower / upper bounds for the impulse clamping.
    pub impulse_lo: f32,
    pub impulse_hi: f32,
    /// Constraint-force-mixing coefficients: `cfm_coeff` is `1 / (1 + cfm_coeff)`
    /// as a multiplier on Δimpulse; `cfm_gain` is subtracted from the rhs.
    /// Matches rapier's `cfm_coeff` / `cfm_gain` fields.
    pub cfm_coeff: f32,
    pub cfm_gain: f32,

    /// Coupling coefficient (`q2 − coeff·q1 − offset = 0`); zero for
    /// limit / motor rows.
    pub coupling_coeff: f32,
    /// Coupling constant offset; zero for limit / motor rows.
    pub coupling_offset: f32,
    /// Packed `(link_id | axis << 16)` of the coupling's first joint
    /// coordinate (`q1`), for the per-substep drift refresh.
    pub _kind_extra2: u32,
    pub _pad1: u32,
}

/// One normal-direction contact constraint between a free rigid body and a
/// link of a multibody.
///
/// Mirrors rapier's pattern of "generic" two-body constraints — one side is a
/// regular rigid body (impulse applied via inv_mass / inv_inertia), the other
/// is a multibody whose impulse is propagated through `M⁻¹ · Jᵀ` (stored as a
/// per-constraint column in `contact_constraint_columns`).
///
/// `kind` values (see `MB_CONTACT_KIND_*`): 0 = inactive (skipped),
/// 1 = active normal (non-penetration) constraint, 2 = active friction
/// tangent constraint. Tangent slots reuse the same struct but treat
/// `lin_jac` / `ang_jac` as the tangent direction; the normal slot's
/// current impulse drives the tangent's clamp limit.
///
/// TODO: handle contact between two multibodies.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")]
pub struct MultibodyContactConstraint {
    /// Multibody index within the batch.
    pub multibody_id: u32,
    /// Link index within `multibody_id`.
    pub link_id: u32,
    /// `MB_CONTACT_KIND_*` discriminant.
    pub kind: u32,
    /// Local body id (in the shared body buffers) of the free-body side.
    pub free_body_id: u32,

    /// Free body's effective inverse mass (scalar — assumes isotropic mass).
    /// Zero for static bodies.
    pub free_body_im: f32,
    /// Coulomb friction coefficient `μ` used by tangent slots; for normal
    /// slots this is propagated forward (the same `μ` covers all of the
    /// contact's tangents).
    pub friction_coeff: f32,
    /// Slot index (relative to the multibody's `cons_base`) of the
    /// associated normal constraint. Tangent slots read
    /// `cons[normal_constraint_slot].impulse` to compute their clamp limit
    /// `±μ · normal_impulse`. For normal slots this is just self.
    pub normal_constraint_slot: u32,
    /// Second touched link for a self-contact, `u32::MAX` otherwise.
    pub link_id_b: u32,

    /// Free-body linear jacobian: `+jac_dir` on body B's side or
    /// `-jac_dir` on body A's side, depending on which side of the contact
    /// pair is the multibody. For normal slots, `jac_dir = world_normal`;
    /// for tangent slots, `jac_dir = world_tangent`.
    pub lin_jac: Vec3,
    pub _pad1: u32,
    /// Free-body angular jacobian (`r_free × jac_dir`).
    pub ang_jac: Vec3,
    pub _pad2: u32,
    /// Same as `ang_jac` but pre-multiplied by the free body's
    /// `effective_world_inv_inertia`. Used to update `solver_vels.angular`
    /// without re-multiplying every PGS iteration.
    pub ii_ang_jac: Vec3,
    pub _pad3: u32,

    /// `1 / (J · M⁻¹ · Jᵀ)`.
    pub inv_lhs: f32,
    /// `J·v_target + bias` — bias from penetration (`erp_inv_dt · depth`)
    /// for normals, surface velocity for tangents.
    pub rhs: f32,
    /// `rhs` without the positional bias (used by the stabilization sweep).
    pub rhs_wo_bias: f32,
    /// Accumulated impulse (warmstart-able).
    pub impulse: f32,

    /// Contact CFM factor `1/(1+cfm_coeff)` — rapier's generic-contact form:
    /// multiplies the impulse each PGS iteration for compliance (replaces the old
    /// rigid `cfm_coeff`/`cfm_gain` generic-joint form, which was always 0).
    pub cfm_factor: f32,
    /// Approaching normal velocity captured at the start of the step, scaled by
    /// the restitution coefficient. Zero on non-bouncy points.
    pub restitution_seed: f32,
    /// Combined restitution coefficient of the two colliders.
    pub restitution: f32,
    pub _pad4: u32,

    /// Torque arm of the `link_id` side about that link's center of mass,
    /// crossed with the multibody-side direction (`-lin_jac`).
    pub torque_a: Vec3,
    pub _pad5: u32,
    /// Same for the `link_id_b` side of a self-contact, crossed with `lin_jac`.
    pub torque_b: Vec3,
    pub _pad6: u32,

    /// Contact anchor frozen in the `link_id` side's body frame.
    pub local_p1: Vec3,
    /// Separation at the substep the anchors were frozen; the live separation
    /// is this plus the drift of the two anchors along the contact normal.
    pub base_dist: f32,
    /// Contact anchor frozen in the other side's frame: the second link's body
    /// frame for a self-contact, the free body's solver (center-of-mass) frame
    /// otherwise.
    pub local_p2: Vec3,
    pub _pad7: u32,
}

/// 2D variant of [`MultibodyContactConstraint`] — angular jacobian collapses
/// to a scalar.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")]
pub struct MultibodyContactConstraint {
    pub multibody_id: u32,
    pub link_id: u32,
    pub kind: u32,
    pub free_body_id: u32,

    /// Slot index (relative to `cons_base`) of the associated normal
    /// constraint. Tangents read `cons[normal_constraint_slot].impulse` to
    /// compute their clamp limit `±μ · normal_impulse`.
    pub normal_constraint_slot: u32,
    /// Second touched link for a self-contact, `u32::MAX` otherwise.
    pub link_id_b: u32,
    /// Free-body linear jacobian.
    pub lin_jac: Vec2,

    /// Contact anchor frozen in the `link_id` side's body frame.
    pub local_p1: Vec2,
    /// Contact anchor frozen in the other side's frame: the second link's body
    /// frame for a self-contact, the free body's solver (center-of-mass) frame
    /// otherwise.
    pub local_p2: Vec2,

    pub free_body_im: f32,
    /// Free-body angular jacobian (`r_free × jac_dir`) — scalar in 2D.
    pub ang_jac: f32,
    /// `ang_jac · effective_world_inv_inertia` (scalar in 2D).
    pub ii_ang_jac: f32,
    /// Coulomb friction coefficient `μ`.
    pub friction_coeff: f32,

    pub inv_lhs: f32,
    pub rhs: f32,
    pub rhs_wo_bias: f32,
    pub impulse: f32,

    /// Contact CFM factor `1/(1+cfm_coeff)` (rapier's generic-contact form).
    pub cfm_factor: f32,
    /// Approaching normal velocity captured at the start of the step, scaled by
    /// the restitution coefficient. Zero on non-bouncy points.
    pub restitution_seed: f32,
    /// Combined restitution coefficient of the two colliders.
    pub restitution: f32,
    /// Separation at the substep the anchors were frozen; the live separation
    /// is this plus the drift of the two anchors along the contact normal.
    pub base_dist: f32,

    /// Torque arm of the `link_id` side about that link's center of mass,
    /// crossed with the multibody-side direction (`-lin_jac`).
    pub torque_a: f32,
    /// Same for the `link_id_b` side of a self-contact, crossed with `lin_jac`.
    pub torque_b: f32,
    /// Pads the struct to a multiple of 16 bytes, which std430 requires of the
    /// array stride given the vector members.
    pub _pad1: [f32; 2],
}

/// Descriptor for one multibody: where its links live, how many DOFs it has, and
/// the offsets into the dense jacobian/mass-matrix/gen-force tensors.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MultibodyInfo {
    /// First link index (relative to this batch's link slice).
    pub first_link: u32,
    /// Number of links in the multibody.
    pub num_links: u32,
    /// First DOF offset (relative to this batch's DOF slice).
    pub first_dof: u32,
    /// Total DOFs (sum of each link's `ndofs`).
    pub ndofs: u32,
    /// Offset (in f32 entries) into the `body_jacobians` tensor; each link has
    /// `SPATIAL_DIM * ndofs` contiguous entries, stacked link-by-link in
    /// assembly order.
    pub jacobian_offset: u32,
    /// Offset (in f32 entries) into the `mass_matrices` tensor. Block size: `ndofs * ndofs`.
    pub mass_matrix_offset: u32,
    /// 0 if the root joint is fixed, 1 if it's a free joint.
    pub root_is_dynamic: u32,
    /// Offset (in f32 entries) into `coriolis_v` (`DIM × ndofs` per link) and
    /// `coriolis_w` (`ANG_DIM × ndofs` per link, stride matches `coriolis_v`'s
    /// `DIM × ndofs` slot allocation in the shared layout). Stacked
    /// link-by-link in assembly order.
    pub coriolis_offset: u32,
    /// Offset (in f32 entries) into `i_coriolis_dt`. One `SPATIAL_DIM × ndofs`
    /// scratch slot per multibody (transient — overwritten per link during
    /// assembly).
    pub i_coriolis_dt_offset: u32,
    /// First constraint index for this multibody in the `joint_constraints`
    /// buffer. Each multibody owns `max_constraints` contiguous slots; the
    /// init kernel marks unused slots with `kind = 0`.
    pub first_constraint: u32,
    /// Maximum constraints this multibody can hold (sum over its joints of
    /// `2 * num_free_axes`). Slots beyond this are not touched.
    pub max_constraints: u32,
    /// `1` if contacts between two links of THIS multibody are allowed, `0` if
    /// disabled (rapier's `Multibody::self_contacts_enabled`, set by MJCF's
    /// `DISABLE_SELF_CONTACTS`). The contact-constraint kernel skips self
    /// contacts when this is `0`.
    pub self_contacts_enabled: u32,
    /// Per-frame count of active multibody contact constraints emitted for this
    /// multibody. Written by `gpu_mb_init_contact_constraints`, read by the
    /// warmstart / finalize / solve / remove-bias contact kernels.
    pub contact_constraint_count: u32,
    /// Per-step copy of `contacts_len[batch]` (to work around the web 8 storage
    /// bindings count limit).
    pub batch_contacts_len: u32,
    /// First entry of this multibody's DoF couplings in the `dof_couplings`
    /// buffer (relative to the batch's coupling slice).
    pub first_coupling: u32,
    /// Number of DoF couplings on this multibody.
    pub num_couplings: u32,
}

/// One holonomic coupling `q2 = coeff·q1 + offset` between two generalized
/// coordinates of the same multibody (rapier's `MultibodyDofCoupling`),
/// converted to the GPU's re-numbered assembly ids at build time.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbDofCoupling {
    /// Absolute DOF index of `q1` (relative to the multibody's `first_dof`).
    pub dof1: u32,
    /// Absolute DOF index of `q2`.
    pub dof2: u32,
    /// Packed `(link_id | axis << 16)` locating `q1`'s joint coordinate.
    pub link_axis1: u32,
    /// Packed `(link_id | axis << 16)` locating `q2`'s joint coordinate.
    pub link_axis2: u32,
    /// Linear coupling coefficient.
    pub coeff: f32,
    /// Constant offset.
    pub offset: f32,
    pub _pad: [u32; 2],
}
