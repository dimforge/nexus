//! The [`GpuMultibodySet`] buffers: struct definition, accessors and
//! runtime-mutation entry points (motors, gravity, dt, softness).

use crate::math::Pose;
use crate::shaders::dynamics::{
    ConstraintSoftness, LocalMassProperties, MbImpulseJointBuilder, MbImpulseJointConstraint,
    MultibodyContactConstraint, MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic,
    MultibodyLinkWorkspace, RbdSimParams,
};
use crate::shaders::utils::BatchIndices;
use glamx::Vec4;
use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuBackendError};
use rapier3d::prelude::JointAxis;
use vortx::tensor::Tensor;

/// Workgroup width for the parallelised LU decompose / solve kernels. Must
/// match the `threads(N, 1, 1)` attribute on `gpu_mb_lu_decompose` and
/// `gpu_mb_lu_solve`.
pub(super) const MB_LU_LANES: u32 = 64;

/// Maximum TOTAL multibody count (capacity × batches) for which the
/// constraint-space (Delassus) contact solve is enabled — each multibody's
/// Delassus block costs `MAX_MB_CONTACT_CONSTRAINTS_PER_MB²` floats (~147 KB
/// in 3D), so huge batched scenes keep the dof-space sweep (~19 MB at this
/// cap).
pub(super) const MAX_DELASSUS_MULTIBODIES: u32 = 128;

use crate::shaders::dynamics::{GenericJoint, JointLimits, JointMotor};

/// GPU-resident articulated multibody set, packed across simulation batches.
///
/// Every buffer is a flat tensor with per-batch capacity (`*_batch_capacity`) and
/// a per-batch length. The multibody/link counts are identical across batches
/// (equal-topology invariant) and read from the `BatchIndices` uniform.
pub struct GpuMultibodySet {
    pub(super) num_batches: u32,
    pub(super) multibodies_per_batch: u32,
    /// Number of *active* multibodies per batch. Identical across batches by
    /// the equal-topology invariant; differs from `multibodies_per_batch` when
    /// the latter is padded to ≥1 to avoid size-zero buffers.
    pub(super) num_active_multibodies: u32,
    pub(super) links_per_batch: u32,
    pub(super) dofs_per_batch: u32,
    pub(super) jacobian_entries_per_batch: u32,
    pub(super) mass_matrix_entries_per_batch: u32,
    pub(super) coriolis_entries_per_batch: u32,
    pub(super) i_coriolis_dt_entries_per_batch: u32,
    /// When `true`, the Coriolis / gyroscopic terms are folded into the mass
    /// matrix (implicit integration). When `false`, they are applied explicitly
    /// as part of the RHS.
    pub(super) implicit_coriolis: bool,
    /// When `false` (no joint limits / motors anywhere), the joint constraint
    /// kernel chain is skipped on the host side.
    pub(super) has_joint_constraints: bool,

    /// Per-batch multibody descriptors.
    pub(super) multibody_info: Tensor<MultibodyInfo>,
    /// Per-batch static link data.
    pub(super) links_static: Tensor<MultibodyLinkStatic>,
    /// CPU-side mirror of [`Self::links_static`] used to support runtime
    /// mutations like motor changes without round-tripping through a GPU read.
    pub(super) links_static_mirror: Vec<MultibodyLinkStatic>,
    /// Per-batch per-step link workspace.
    pub(super) links_workspace: Tensor<MultibodyLinkWorkspace>,
    /// Generalized coordinates (flat).
    pub(super) dof_values: Tensor<f32>,
    /// Packed buffer holding generalized velocities (offset 0) and per-DOF
    /// damping coefficients (offset `damping_section_offset`). Callers reading
    /// velocities should use only the velocity section.
    pub(super) dof_state: Tensor<f32>,
    /// Generalized forces / after solve, generalized accelerations.
    pub(super) gen_forces: Tensor<f32>,
    /// Per-link `6 × ndofs` column-major jacobians.
    pub(super) body_jacobians: Tensor<f32>,
    /// Per-multibody `ndofs × ndofs` mass matrices (also used as LU work buffer).
    pub(super) mass_matrices: Tensor<f32>,
    /// Per-DOF pivot buffer used by LU.
    pub(super) lu_pivots: Tensor<u32>,

    /// Packed buffer holding the three Coriolis scratch sections back-to-back.
    pub(super) coriolis_packed: Tensor<f32>,

    /// Per-multibody flat bank of unit (1-DOF) limit / motor constraints.
    pub(super) joint_constraints: Tensor<MultibodyJointConstraint>,
    /// Per-constraint columns of `M⁻¹` (length `ndofs` each, contiguous per multibody).
    pub(super) joint_constraint_columns: Tensor<f32>,

    /// Per-body lookup `[multibody_idx, link_idx]` (`u32::MAX` sentinel for
    /// free / non-multibody bodies). Indexed by the per-batch local body id.
    pub(super) body_to_link: Tensor<[u32; 2]>,

    /// Per-multibody bank of contact constraints (1 normal + 2 friction per
    /// touched contact point).
    pub(super) contact_constraints: Tensor<MultibodyContactConstraint>,
    /// Per-constraint `Jᵀ` row (length `ndofs`) — the multibody side's
    /// contribution to the constraint Jacobian.
    pub(super) contact_constraint_jacs: Tensor<f32>,
    /// Per-constraint M⁻¹·Jᵀ column (length `ndofs`).
    pub(super) contact_constraint_columns: Tensor<f32>,
    /// Per-multibody Delassus blocks (`MAX_MB_CONTACT_CONSTRAINTS_PER_MB²`
    /// floats each) for the constraint-space contact sweep. Only allocated
    /// when the total multibody count is at most
    /// [`MAX_DELASSUS_MULTIBODIES`] (the blocks are ~147 KB each in 3D);
    /// `None` selects the dof-space solve path.
    pub(super) contact_delassus: Option<Tensor<f32>>,

    /// Per-batch number of multibody-touching impulse joints (body1 OR body2
    /// part of any multibody).
    pub(super) mb_imp_joint_count: Tensor<u32>,
    /// Per-batch slab of impulse-joint builder descriptors.
    pub(super) mb_imp_joint_builders: Tensor<MbImpulseJointBuilder>,
    /// Per-batch slab of axis constraints.
    pub(super) mb_imp_joint_constraints: Tensor<MbImpulseJointConstraint>,
    /// Per-batch flat jacobians buffer — stores `J / W·J` for both sides
    /// of every axis constraint of every joint.
    pub(super) mb_imp_joint_jacobians: Tensor<f32>,

    /// Capacities (per-batch strides) for the impulse-joint slabs above.
    /// Mirrored into `BatchIndices` via [`Self::fill_batch_indices`].
    pub(super) mb_imp_joints_per_batch: u32,
    pub(super) mb_imp_joint_constraints_per_batch: u32,
    pub(super) mb_imp_joint_jacobians_per_batch: u32,

    /// Per-batch prefix-sum over the color-sorted `mb_imp_joint_builders`.
    /// Built at init time by `set_impulse_joints` (greedy graph coloring).
    pub(super) mb_imp_joint_color_groups: Tensor<u32>,
    /// Number of colors (per-batch stride of `mb_imp_joint_color_groups`,
    /// and the host color-loop trip count). CPU mirror.
    pub(crate) mb_imp_joint_num_colors: u32,
    /// Max `ndofs` across every multibody in every batch (CPU mirror of
    /// `BatchIndices::mb_max_ndofs`).
    pub(super) max_ndofs: u32,
    /// Max link count across every multibody in every batch (CPU mirror of
    /// `BatchIndices::mb_max_links`).
    pub(super) max_links: u32,
    /// Largest color group across batches — the per-color dispatch width.
    pub(super) mb_imp_joint_max_color_group_len: u32,
    /// Per-batch capacities of the joint / contact constraint slabs (CPU-side
    /// mirror). Stored so `RbdState` can rebuild its `BatchIndices` when caps change.
    pub(super) joint_constraints_per_batch: u32,
    pub(super) joint_constraint_columns_per_batch: u32,
    pub(super) contact_constraints_per_batch: u32,
    pub(super) contact_constraint_columns_per_batch: u32,

    /// Number of solver iterations to run on `joint_constraints` per `step()`.
    pub(super) num_solver_iterations: u32,

    /// Gravity vector (only the first 3 components are read by the shaders).
    pub(super) gravity: Tensor<Vec4>,
    /// Current integration timestep.
    pub(super) dt: Tensor<f32>,
    /// Precomputed soft-constraint coefficients (contact + joint, rapier
    /// TGS-soft).
    pub(super) constraint_softness: Tensor<ConstraintSoftness>,
}

impl GpuMultibodySet {
    /// Number of simulation batches.
    pub fn num_batches(&self) -> u32 {
        self.num_batches
    }

    /// Capacity (max multibodies) per batch.
    pub fn multibodies_per_batch(&self) -> u32 {
        self.multibodies_per_batch
    }

    /// Thread-count grid for the per-multibody kernels, with `(multibody,
    /// batch)` flattened into X. A 2D `[per_batch, num_batches]` grid gives
    /// every batch its own workgroup — mostly idle lanes when each
    /// environment holds a single robot. The kernels decode
    /// `batch_id = x / multibodies_len`, `mb_idx = x % multibodies_len`.
    pub(crate) fn flat_mb_dispatch(&self) -> [u32; 3] {
        [self.num_active_multibodies * self.num_batches, 1, 1]
    }

    /// Lanes per multibody for the packed per-multibody dynamics kernels
    /// (`compute_dynamics_pre`, `gravity_and_lu`) — mirrored into
    /// `BatchIndices::mb_pack_lanes`.
    ///
    /// `1` selects the SERIAL tier (Genesis-style): one thread runs its
    /// multibody's whole FK/CRBA/LU chain with no barriers at all, 64
    /// multibodies per workgroup with every lane busy. For small robots this
    /// beats lane-parallelism — whose ~60-barrier dependency chain caps how
    /// fast one multibody can finish — but ONLY once there are enough
    /// multibodies for the thread count to hide the long serial chain's
    /// latency (measured crossover between 1024 and 4096 on Apple M-series;
    /// below that, spreading each robot across 8 lanes wins despite the
    /// barriers).
    pub(crate) fn pack_lanes(&self) -> u32 {
        let total_mb = self.num_active_multibodies * self.num_batches;
        if self.max_ndofs <= 8 && total_mb >= 2048 {
            1
        } else {
            self.max_ndofs.next_power_of_two().clamp(8, MB_LU_LANES)
        }
    }

    /// Thread-count grid for the packed per-multibody WORKGROUP kernels
    /// (`compute_dynamics_pre`, `gravity_and_lu`): `64 / pack_lanes`
    /// multibodies per 64-lane workgroup, flattened `(multibody, batch)`.
    pub(crate) fn packed_wg_dispatch(&self) -> [u32; 3] {
        let slots = MB_LU_LANES / self.pack_lanes();
        let total = self.num_active_multibodies * self.num_batches;
        [total.div_ceil(slots) * MB_LU_LANES, 1, 1]
    }

    /// True if the set contains no multibodies in any batch.
    ///
    /// Uses the *active* count: the per-batch capacity is padded to >= 1 to
    /// avoid zero-sized buffers, so testing it would run the whole multibody
    /// kernel chain every step for scenes without any multibody.
    pub fn is_empty(&self) -> bool {
        self.num_active_multibodies == 0 || self.links_per_batch == 0
    }

    /// Number of colors used by the colored multibody impulse-joint sweeps.
    pub fn mb_imp_joint_num_colors(&self) -> u32 {
        self.mb_imp_joint_num_colors
    }

    /// GPU buffer holding generalized velocities followed by per-DOF damping.
    /// The velocity section is `[0, dof_batch_capacity * num_batches)`; the
    /// damping section follows. Callers reading velocities should use only the
    /// first half.
    pub fn dof_state(&self) -> &Tensor<f32> {
        &self.dof_state
    }

    /// GPU buffer for generalized coordinates.
    pub fn dof_values(&self) -> &Tensor<f32> {
        &self.dof_values
    }

    /// GPU buffer for the last-computed generalized accelerations (populated by
    /// `GpuMultibodySolver::solve_gravity`).
    pub fn gen_accelerations(&self) -> &Tensor<f32> {
        &self.gen_forces
    }

    /// Enable implicit integration of the Coriolis / gyroscopic terms. Implicit
    /// treatment stabilizes the integrator at large time-steps; the explicit
    /// form is slightly cheaper but can become unstable for fast rotations.
    pub fn set_implicit_coriolis(&mut self, enabled: bool) {
        self.implicit_coriolis = enabled;
    }

    /// Whether the Coriolis / gyroscopic terms are folded into the mass matrix
    /// (implicit integration) in the next `step()`.
    pub fn implicit_coriolis(&self) -> bool {
        self.implicit_coriolis
    }

    /// Number of TGS-soft substeps per visible step (matches rapier's
    /// `num_solver_iterations`).
    pub fn num_solver_iterations(&self) -> u32 {
        self.num_solver_iterations
    }

    /// Set the number of TGS-soft substeps (default `4`). Note: this does not
    /// re-upload `dt`; call [`set_visible_dt`](Self::set_visible_dt) afterwards
    /// to refresh the GPU substep-dt buffer.
    pub fn set_num_solver_iterations(&mut self, n: u32) {
        self.num_solver_iterations = n;
    }

    /// Upload the visible-frame `dt`. Internally divides by `num_solver_iterations`
    /// and stores the *substep* dt (which is what the GPU kernels read).
    pub fn set_visible_dt(&mut self, backend: &GpuBackend, visible_dt: f32) {
        let n = self.num_solver_iterations.max(1) as f32;
        self.dt = Tensor::scalar(
            backend,
            visible_dt / n,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Upload the soft contact-constraint coefficients, computed from the
    /// (substep) sim params. Must be called whenever the contact softness /
    /// timestep changes.
    pub fn set_constraint_softness(&mut self, backend: &GpuBackend, params: &RbdSimParams) {
        self.constraint_softness = Tensor::scalar(
            backend,
            ConstraintSoftness::from_params(params),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Sets a motor's target velocity on a multibody joint and uploads the
    /// updated link to the GPU. `link_id` is the global link id within the
    /// batch (matches the body / collider index that was given to
    /// [`from_rapier`](Self::from_rapier)). `axis` is the joint axis index
    /// (0..=2 for linear, 3..=5 for angular).
    ///
    /// The motor is also auto-enabled (its bit is set in `motor_axes`) so the
    /// solver actually drives the joint at the requested velocity.
    pub fn set_motor_velocity(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) -> Result<(), GpuBackendError> {
        let stride = self.links_per_batch;
        let global_idx = (batch * stride + link_id) as usize;
        let axis_id = axis as usize;
        let entry = match self.links_static_mirror.get_mut(global_idx) {
            Some(e) => e,
            None => return Ok(()),
        };
        entry.data.motors[axis_id].target_vel = target_vel;
        entry.data.motor_axes |= 1u32 << axis_id;
        let snapshot = *entry;
        backend.write_buffer(
            self.links_static.buffer_mut(),
            global_idx as u64,
            std::slice::from_ref(&snapshot),
        )
    }

    /// Upload a new gravity vector.
    pub fn set_gravity(&mut self, backend: &GpuBackend, g: [f32; 3]) {
        self.gravity = Tensor::scalar(
            backend,
            Vec4::new(g[0], g[1], g[2], 0.0),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Number of multibody-touching impulse joints in any batch.
    pub fn mb_impulse_joints_per_batch(&self) -> u32 {
        self.mb_imp_joints_per_batch
    }

    /// Populate the multibody-owned fields of `BatchIndices`. Leaves the
    /// RBD-side fields (`colliders_batch_capacity`, `contacts_batch_capacity`,
    /// `collision_pairs_batch_capacity`, `impulse_joints_batch_capacity`,
    /// `color_groups_batch_capacity`) untouched — the caller fills those.
    pub(crate) fn fill_batch_indices(&self, dst: &mut BatchIndices) {
        dst.multibodies_batch_capacity = self.multibodies_per_batch;
        dst.multibodies_len = self.num_active_multibodies;
        dst.links_batch_capacity = self.links_per_batch;
        dst.jacobians_batch_capacity = self.jacobian_entries_per_batch;
        dst.mass_matrix_batch_capacity = self.mass_matrix_entries_per_batch;
        dst.coriolis_batch_capacity = self.coriolis_entries_per_batch;
        dst.i_coriolis_dt_batch_capacity = self.i_coriolis_dt_entries_per_batch;
        dst.dof_batch_capacity = self.dofs_per_batch;
        dst.mb_joint_constraints_batch_capacity = self.joint_constraints_per_batch;
        dst.mb_joint_constraint_columns_batch_capacity = self.joint_constraint_columns_per_batch;
        dst.mb_contact_constraints_batch_capacity = self.contact_constraints_per_batch;
        dst.mb_contact_constraint_columns_batch_capacity =
            self.contact_constraint_columns_per_batch;
        dst.mb_imp_joints_batch_capacity = self.mb_imp_joints_per_batch.max(1);
        dst.mb_imp_joint_constraints_batch_capacity = self.mb_imp_joint_constraints_per_batch;
        dst.mb_imp_joint_jacobians_batch_capacity = self.mb_imp_joint_jacobians_per_batch;
        dst.mb_imp_joint_color_groups_batch_capacity = self.mb_imp_joint_num_colors.max(1);
        dst.mb_max_ndofs = self.max_ndofs;
        dst.mb_max_links = self.max_links;
        dst.mb_pack_lanes = self.pack_lanes();
        dst.coriolis_w_section_offset = self.coriolis_entries_per_batch * self.num_batches;
        dst.i_coriolis_dt_section_offset = 2 * self.coriolis_entries_per_batch * self.num_batches;
        dst.dof_damping_section_offset = self.dofs_per_batch * self.num_batches;
    }

    /// Upload a new integration timestep.
    pub fn set_dt(&mut self, backend: &GpuBackend, dt: f32) {
        self.dt = Tensor::scalar(
            backend,
            dt,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }
}

pub(super) fn convert_link_mprops(
    m: &crate::rapier::prelude::MassProperties,
) -> LocalMassProperties {
    LocalMassProperties {
        inertia_ref_frame: m.principal_inertia_local_frame,
        inv_principal_inertia: m.inv_principal_inertia,
        padding0: 0,
        inv_mass: glamx::Vec3::splat(m.inv_mass),
        padding1: 0,
        com: m.local_com,
        padding2: 0,
    }
}

pub(super) fn convert_generic_joint(j: crate::rapier::dynamics::GenericJoint) -> GenericJoint {
    GenericJoint {
        local_frame_a: j.local_frame1,
        local_frame_b: j.local_frame2,
        locked_axes: j.locked_axes.bits() as u32,
        limit_axes: j.limit_axes.bits() as u32,
        motor_axes: j.motor_axes.bits() as u32,
        coupled_axes: j.coupled_axes.bits() as u32,
        limits: j.limits.map(|l| JointLimits {
            min: l.min,
            max: l.max,
            impulse: l.impulse,
        }),
        motors: j.motors.map(|m| JointMotor {
            target_vel: m.target_vel,
            target_pos: m.target_pos,
            stiffness: m.stiffness,
            damping: m.damping,
            max_force: m.max_force,
            impulse: m.impulse,
            model: match m.model {
                crate::rapier::prelude::MotorModel::AccelerationBased => 0,
                crate::rapier::prelude::MotorModel::ForceBased => 1,
            },
        }),
    }
}

//
// Zero-initialised workspaces would leave `joint_rot` as the all-zero quaternion, which
// is not a valid rotation — seed it with the identity instead.
//
pub(super) fn make_workspace_init() -> MultibodyLinkWorkspace {
    let mut w: MultibodyLinkWorkspace = bytemuck::Zeroable::zeroed();
    w.joint_rot = glamx::Quat::IDENTITY;
    w.local_to_parent = Pose::default();
    w.local_to_world = Pose::default();
    w
}
