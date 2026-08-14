//! The [`GpuMultibodySet`] buffers: struct definition, accessors and
//! runtime-mutation entry points (motors, dt, softness).

use crate::math::Pose;
use crate::shaders::dynamics::{
    ConstraintSoftness, LocalMassProperties, MbDofCoupling, MbImpulseJointBuilder,
    MbImpulseJointConstraint, MultibodyContactConstraint, MultibodyInfo, MultibodyJointConstraint,
    MultibodyLinkStatic, MultibodyLinkWorkspace, RbdSimParams,
};
use crate::shaders::utils::BatchIndices;
use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuBackendError};
use rapier3d::prelude::JointAxis;
use vortx::tensor::Tensor;

/// Workgroup width for the parallelised LU decompose / solve kernels. Must
/// match the `threads(N, 1, 1)` attribute on `gpu_mb_lu_decompose` and
/// `gpu_mb_lu_solve`.
pub(super) const MB_LU_LANES: u32 = 64;

/// Maximum total multibody count (capacity × batches) for which the
/// constraint-space (Delassus) contact solve is enabled: each multibody's
/// Delassus block costs `MAX_MB_CONTACT_CONSTRAINTS_PER_MB²` floats (~147 KB
/// in 3D), so huge batched scenes would run out of memory.
pub(super) const MAX_DELASSUS_MULTIBODIES: u32 = 0; // 128;

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
    pub(super) implicit_coriolis: bool,
    /// When `false` (no joint limits / motors anywhere), the joint constraint
    /// kernel chain is skipped on the host side.
    pub(super) has_joint_constraints: bool,

    /// Per-batch multibody descriptors.
    pub(super) multibody_info: Tensor<MultibodyInfo>,
    /// Max `contact_constraint_count` across every multibody, written each
    /// step by `gpu_mb_compute_solve_bounds`.
    pub(super) max_contact_constraints: Tensor<u32>,
    /// Per-batch static link data.
    pub(super) links_static: Tensor<MultibodyLinkStatic>,
    /// CPU-side mirror of [`Self::links_static`] used to support runtime
    /// mutations like motor changes without round-tripping through a GPU read.
    pub(super) links_static_mirror: Vec<MultibodyLinkStatic>,
    /// Host copy of the per-multibody descriptors, batch-major (before the
    /// batch interleave), indexed `batch * multibodies_per_batch + mb_idx`.
    pub(super) info_mirror: Vec<MultibodyInfo>,
    /// Per-batch per-step link workspace, SoA quad layout.
    pub(super) links_workspace: Tensor<glamx::Vec4>,
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
    /// Per-batch slab of DoF couplings (rapier's `MultibodyDofCoupling`),
    /// batch-major; each multibody's slice is
    /// `[first_coupling, first_coupling + num_couplings)`.
    pub(super) dof_couplings: Tensor<MbDofCoupling>,
    /// Per-batch capacity (stride) of [`Self::dof_couplings`].
    pub(super) couplings_per_batch: u32,

    /// Per-body lookup `[multibody_idx, link_idx]` (`u32::MAX` sentinel for
    /// free / non-multibody bodies). Indexed by the per-batch local body id.
    pub(super) body_to_link: Tensor<[u32; 2]>,

    /// Per-multibody bank of contact constraints (1 normal + 2 friction per
    /// touched contact point).
    pub(super) contact_constraints: Tensor<MultibodyContactConstraint>,
    /// Snapshot of `contact_constraints` taken at the start of the step; the
    /// warmstart transfer matches this frame's slots against it.
    pub(super) old_contact_constraints: Tensor<MultibodyContactConstraint>,
    /// Per-constraint `Jᵀ` row (length `ndofs`) — the multibody side's
    /// contribution to the constraint Jacobian.
    pub(super) contact_constraint_jacs: Tensor<f32>,
    /// Per-constraint M⁻¹·Jᵀ column (length `ndofs`).
    pub(super) contact_constraint_columns: Tensor<f32>,
    /// Per-multibody Delassus blocks (`MAX_MB_CONTACT_CONSTRAINTS_PER_MB²`
    /// floats each) only allocated when the total multibody count is at most
    /// [`MAX_DELASSUS_MULTIBODIES`].
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
    /// Max joint-constraint slot count across every multibody in every batch
    /// (CPU mirror of `BatchIndices::mb_max_joint_constraints`).
    pub(super) max_joint_constraints: u32,
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
    /// PGS iterations over the joint + contact constraints per substep, in the
    /// biased pass. One is enough for simple articulations; servo-driven robots
    /// resting on contacts need several to stop the motor and contact rows
    /// fighting each other.
    pub(super) num_internal_pgs_iterations: u32,

    /// Current integration timestep.
    pub(super) dt: Tensor<f32>,
    /// Precomputed soft-constraint coefficients (contact + joint, rapier
    /// TGS-soft).
    pub(super) constraint_softness: Tensor<ConstraintSoftness>,
    /// CPU mirror of `ConstraintSoftness::warmstart_coefficient`, so the solver
    /// can skip the warmstart passes entirely when it is zero.
    pub(super) warmstart_coefficient: f32,
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
    /// batch)` flattened into X. The kernels decode
    /// `batch_id = x / multibodies_len`, `mb_idx = x % multibodies_len`.
    pub(crate) fn flat_mb_dispatch(&self) -> [u32; 3] {
        [self.num_active_multibodies * self.num_batches, 1, 1]
    }

    /// Lanes per multibody for the packed per-multibody dynamics kernels
    /// (`compute_dynamics_pre`, `gravity_and_lu`).
    pub(crate) fn pack_lanes(&self) -> u32 {
        let total_mb = self.num_active_multibodies * self.num_batches;
        if self.max_ndofs <= 8 && total_mb >= 1024 {
            1
        } else {
            self.max_ndofs.next_power_of_two().clamp(8, MB_LU_LANES)
        }
    }

    /// Thread-count grid for the packed per-multibody workgroup kernels
    /// (`compute_dynamics_pre`, `gravity_and_lu`): `64 / pack_lanes`
    /// multibodies per 64-lane workgroup, flattened `(multibody, batch)`.
    pub(crate) fn packed_wg_dispatch(&self) -> [u32; 3] {
        let slots = MB_LU_LANES / self.pack_lanes();
        let total = self.num_active_multibodies * self.num_batches;
        [total.div_ceil(slots) * MB_LU_LANES, 1, 1]
    }

    /// True if the set contains no active multibodies in any batch.
    pub fn is_empty(&self) -> bool {
        self.num_active_multibodies == 0 || self.links_per_batch == 0
    }

    /// Number of colors used by the colored multibody impulse-joint sweeps.
    pub fn mb_imp_joint_num_colors(&self) -> u32 {
        self.mb_imp_joint_num_colors
    }

    /// GPU buffer holding six back-to-back per-DOF sections of
    /// `dof_batch_capacity * num_batches` floats each: generalized
    /// velocities, damping, armature, spring stiffness, spring rest position,
    /// and the kinematic-DOF mask. Callers reading velocities should use only
    /// the first section.
    pub fn dof_state(&self) -> &Tensor<f32> {
        &self.dof_state
    }

    /// Per-batch stride of the DoF buffers (the length of each section of
    /// [`Self::dof_state`]).
    pub fn dofs_per_batch(&self) -> u32 {
        self.dofs_per_batch
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

    /// Enables or disables the implicit treatment of coriolis forces.
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

    /// Sets how many PGS iterations the biased pass runs per substep (default 1).
    pub fn set_num_internal_pgs_iterations(&mut self, n: u32) {
        self.num_internal_pgs_iterations = n.max(1);
    }

    /// PGS iterations per substep in the biased pass.
    pub fn num_internal_pgs_iterations(&self) -> u32 {
        self.num_internal_pgs_iterations
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
        self.warmstart_coefficient = params.warmstart_coefficient;
        self.constraint_softness = Tensor::scalar(
            backend,
            ConstraintSoftness::from_params(params),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Overwrites one joint motor of a multibody link and uploads the changed
    /// link to the GPU, enabling the axis so the solver drives it.
    ///
    /// `link_id` is the global link id within the batch (it matches the body
    /// index given to [`from_rapier`](Self::from_rapier)) and `axis` indexes the
    /// 6-DoF spatial layout (`0..DIM` linear, `DIM..` angular). Motors are baked
    /// into the GPU state at finalization, so per-step actuation has to come
    /// through here.
    pub fn set_motor(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        link_id: u32,
        axis: usize,
        motor: JointMotor,
    ) -> Result<(), GpuBackendError> {
        if axis >= 6 {
            return Ok(());
        }
        let global_idx = (link_id * self.num_batches + batch) as usize;
        let entry = match self.links_static_mirror.get_mut(global_idx) {
            Some(e) => e,
            None => return Ok(()),
        };
        // `impulse` is solver state, not configuration: keep the accumulated
        // value so retargeting a servo does not drop its warmstart.
        let impulse = entry.data.motors[axis].impulse;
        entry.data.motors[axis] = motor;
        entry.data.motors[axis].impulse = impulse;
        entry.data.motor_axes |= 1u32 << axis;
        let snapshot = *entry;
        backend.write_buffer(
            self.links_static.buffer_mut(),
            global_idx as u64,
            std::slice::from_ref(&snapshot),
        )
    }

    /// The motor currently configured on `axis` of multibody link `link_id`,
    /// as last uploaded. Use it to adjust one field of a live motor without
    /// rebuilding the rest.
    pub fn motor(&self, batch: u32, link_id: u32, axis: usize) -> Option<JointMotor> {
        if axis >= 6 {
            return None;
        }
        let global_idx = (link_id * self.num_batches + batch) as usize;
        self.links_static_mirror
            .get(global_idx)
            .map(|e| e.data.motors[axis])
    }

    /// Batched [`Self::set_motor`]: applies every `(link_id, axis, motor)` and
    /// uploads each touched link once, rather than once per axis.
    ///
    /// This is the entry point for per-step actuation (a position-servo robot
    /// re-targets several axes of many links every frame), so it is worth
    /// keeping the upload count down to one per link.
    pub fn set_motors(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        updates: &[(u32, usize, JointMotor)],
    ) -> Result<(), GpuBackendError> {
        let mut touched: Vec<usize> = Vec::with_capacity(updates.len());
        for &(link_id, axis, motor) in updates {
            if axis >= 6 {
                continue;
            }
            let global_idx = (link_id * self.num_batches + batch) as usize;
            let Some(entry) = self.links_static_mirror.get_mut(global_idx) else {
                continue;
            };
            // `impulse` is solver state, not configuration: keep the accumulated
            // value so retargeting a servo does not drop its warmstart.
            let impulse = entry.data.motors[axis].impulse;
            entry.data.motors[axis] = motor;
            entry.data.motors[axis].impulse = impulse;
            entry.data.motor_axes |= 1u32 << axis;
            touched.push(global_idx);
        }
        touched.sort_unstable();
        touched.dedup();
        for global_idx in touched {
            let snapshot = self.links_static_mirror[global_idx];
            backend.write_buffer(
                self.links_static.buffer_mut(),
                global_idx as u64,
                std::slice::from_ref(&snapshot),
            )?;
        }
        Ok(())
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
        // Batch-interleaved links layout: element `link_id` of batch `batch`
        // lives at `link_id · num_batches + batch` (mirror included).
        let global_idx = (link_id * self.num_batches + batch) as usize;
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

    /// Per-batch per-step link workspace (generalized coordinates, joint
    /// rotations, world-space link velocities). Read it back with
    /// `slow_read_buffer` for joint/base state observation; entries are laid out
    /// `env * links_per_batch + link`, in [`from_rapier`](Self::from_rapier)'s
    /// link traversal order.
    pub fn links_workspace(&self) -> &Tensor<MultibodyLinkWorkspace> {
        &self.links_workspace
    }

    /// Number of link slots per environment (the stride of
    /// [`Self::links_workspace`] and `links_static`).
    pub fn links_per_batch(&self) -> u32 {
        self.links_per_batch
    }

    /// Refreshes every link's joint parameters (motor targets/gains, limits) of
    /// environment `env` from a rapier multibody set laid out identically to the
    /// one this GPU set was built from (same multibody/link traversal order as
    /// [`from_rapier`](Self::from_rapier)), then uploads the `links_static`
    /// buffer in one write.
    ///
    /// This is the per-step control path for actuated robots: mutate the motors
    /// on the CPU rapier joints (e.g. via `rapier3d-mjcf`'s
    /// `apply_controls_multibody`, which implements the MJCF actuator
    /// semantics), then call this to push the new motor state to the GPU. Only
    /// joint data is refreshed — coordinates, velocities and mass properties are
    /// untouched, so this cannot be used to teleport links.
    pub fn sync_joint_data_from_rapier(
        &mut self,
        backend: &GpuBackend,
        env: u32,
        set: &crate::rapier::dynamics::MultibodyJointSet,
        bodies: &crate::rapier::dynamics::RigidBodySet,
    ) -> Result<(), GpuBackendError> {
        let base = (env * self.links_per_batch) as usize;
        let mut offset = 0usize;
        for mb in set.multibodies() {
            // Mirror `from_rapier`'s fixed-root handling: a non-dynamic root has
            // all 6 DOFs locked on the GPU even though rapier models it as free.
            let root_is_dynamic = mb
                .link(0)
                .and_then(|r| bodies.get(r.rigid_body_handle()))
                .map(|rb| rb.is_dynamic())
                .unwrap_or(false);
            for (link_idx, link) in mb.links().enumerate() {
                let Some(entry) = self.links_static_mirror.get_mut(base + offset) else {
                    return Ok(());
                };
                let mut data = convert_generic_joint(link.joint().data);
                if link_idx == 0 && !root_is_dynamic {
                    data.locked_axes = 0x3f;
                }
                entry.data = data;
                offset += 1;
            }
        }
        backend.write_buffer(
            self.links_static.buffer_mut(),
            0,
            &self.links_static_mirror,
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
        dst.mb_max_joint_constraints = self.max_joint_constraints;
        dst.mb_pack_lanes = self.pack_lanes();
        dst.coriolis_w_section_offset = self.coriolis_entries_per_batch * self.num_batches;
        dst.i_coriolis_dt_section_offset = 2 * self.coriolis_entries_per_batch * self.num_batches;
        dst.dof_damping_section_offset = self.dofs_per_batch * self.num_batches;
        // Implicit coriolis needs two matrices: the coriolis-augmented one (acc
        // section) for the acceleration solve, the plain one for constraints.
        // With the flag off, a single plain matrix serves both.
        dst.mass_matrix_acc_section_offset = if self.implicit_coriolis {
            self.mass_matrix_entries_per_batch
        } else {
            0
        };
        dst.mb_dof_couplings_batch_capacity = self.couplings_per_batch;
    }

    /// Sets the world-space force and torque applied to `link_id` of multibody
    /// `mb_idx` in `batch_id`, plus that link's multiplier on the global
    /// gravity. They stay applied until overwritten.
    pub fn set_link_external_wrench(
        &mut self,
        backend: &GpuBackend,
        batch_id: u32,
        mb_idx: u32,
        link_id: u32,
        force: crate::math::Vector,
        torque: crate::math::AngVector,
        gravity_scale: f32,
    ) -> Result<(), khal::backend::GpuBackendError> {
        use crate::shaders::dynamics::{WS_EXT_FORCE, WS_EXT_TORQUE, WsAddr};

        let info = self.info_mirror[(batch_id * self.multibodies_per_batch + mb_idx) as usize];
        let k = info.first_link + link_id;
        let a = WsAddr::new(0, self.num_batches, batch_id);

        #[cfg(feature = "dim3")]
        {
            let f = glamx::Vec4::new(force.x, force.y, force.z, gravity_scale);
            let t = glamx::Vec4::new(torque.x, torque.y, torque.z, 0.0);
            backend.write_buffer(
                self.links_workspace.buffer_mut(),
                a.at(k, WS_EXT_FORCE) as u64,
                &[f],
            )?;
            backend.write_buffer(
                self.links_workspace.buffer_mut(),
                a.at(k, WS_EXT_TORQUE) as u64,
                &[t],
            )?;
        }
        #[cfg(feature = "dim2")]
        {
            let _ = WS_EXT_TORQUE;
            let f = glamx::Vec4::new(force.x, force.y, torque, gravity_scale);
            backend.write_buffer(
                self.links_workspace.buffer_mut(),
                a.at(k, WS_EXT_FORCE) as u64,
                &[f],
            )?;
        }
        Ok(())
    }

    /// Per-multibody descriptors (contact counts, dof offsets, ...).
    pub fn multibody_info(&self) -> &Tensor<MultibodyInfo> {
        &self.multibody_info
    }

    /// The per-multibody contact-constraint slabs.
    pub fn contact_constraints(&self) -> &Tensor<MultibodyContactConstraint> {
        &self.contact_constraints
    }

    /// Coefficient applied to a contact impulse before it is re-used as the
    /// next substep's or next frame's initial guess. Zero disables warmstarting.
    pub fn warmstart_coefficient(&self) -> f32 {
        self.warmstart_coefficient
    }

    /// Per-batch stride of [`Self::contact_constraints`].
    pub fn contact_constraints_per_batch(&self) -> u32 {
        self.contact_constraints_per_batch
    }

    /// Per-constraint `Jᵀ` rows of the contact constraints (`ndofs` floats each,
    /// laid out like [`Self::contact_constraints`]).
    pub fn contact_constraint_jacs(&self) -> &Tensor<f32> {
        &self.contact_constraint_jacs
    }

    /// Per-constraint `M⁻¹·Jᵀ` columns of the contact constraints, laid out
    /// like [`Self::contact_constraint_jacs`].
    pub fn contact_constraint_columns(&self) -> &Tensor<f32> {
        &self.contact_constraint_columns
    }

    /// Per-link `SPATIAL_DIM × ndofs` column-major body jacobians, indexed from
    /// each multibody's [`MultibodyInfo::jacobian_offset`].
    pub fn body_jacobians(&self) -> &Tensor<f32> {
        &self.body_jacobians
    }

    /// Per-multibody `ndofs × ndofs` mass matrices, indexed from each
    /// multibody's [`MultibodyInfo::mass_matrix_offset`]. Doubles as the LU work
    /// buffer, so after a step this holds the factorization, not `M` itself.
    pub fn mass_matrices(&self) -> &Tensor<f32> {
        &self.mass_matrices
    }

    /// Reads back the generalized coordinate of every DoF of batch `batch_id`,
    /// in assembly order (the same order as [`Self::dof_state`]'s velocity
    /// section). The coordinates live in the link workspace, so this unpacks the
    /// SoA layout for callers.
    pub async fn read_dof_coords(
        &self,
        backend: &GpuBackend,
        batch_id: u32,
    ) -> Result<Vec<f32>, khal::backend::GpuBackendError> {
        use crate::shaders::dynamics::{WsAddr, ws_coord};

        let ws: Vec<glamx::Vec4> = backend.slow_read_vec(self.links_workspace.buffer()).await?;
        let a = WsAddr::new(0, self.num_batches, batch_id);
        let mut out = Vec::new();
        for k in 0..self.links_per_batch {
            let stat = &self.links_static_mirror[(batch_id * self.links_per_batch + k) as usize];
            let locked = stat.data.locked_axes;
            for axis in 0..6u32 {
                if locked & (1 << axis) == 0 {
                    out.push(ws_coord(&ws, a, k, axis));
                }
            }
        }
        Ok(out)
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

pub(super) fn make_workspace_init() -> MultibodyLinkWorkspace {
    let mut w: MultibodyLinkWorkspace = bytemuck::Zeroable::zeroed();
    w.joint_rot = glamx::Quat::IDENTITY;
    w.gravity_scale = 1.0;
    w.local_to_parent = Pose::default();
    w.local_to_world = Pose::default();
    w
}
