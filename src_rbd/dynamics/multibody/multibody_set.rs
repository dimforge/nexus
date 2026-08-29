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
use khal::Shader;
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

/// The motor-target scatter entry point, loaded once per backend.
#[derive(Shader)]
pub(super) struct MotorScatterBundle {
    scatter: crate::shaders::dynamics::GpuScatterMotorTargets,
}

/// Shader plus the constant tensors of one motor-target scatter configuration.
/// The link ids, counts and axis do not change between steps, so a per-call
/// `from_backend` and four allocations would cost more than the dispatch.
pub(super) struct MotorScatterCache {
    shader: MotorScatterBundle,
    t_links: Tensor<u32>,
    u_na: Tensor<u32>,
    u_ne: Tensor<u32>,
    u_ax: Tensor<u32>,
    num_actuated: u32,
    axis: u32,
    link_ids: Vec<u32>,
}

/// The on-device delay-state refresh entry point.
#[derive(Shader)]
pub(super) struct DelayUpdateBundle {
    kernel: crate::shaders::dynamics::GpuMbDelayStateUpdate,
}

/// Shader plus the constant tensors of the delay-state refresh. The link ids
/// and counts do not change between steps, so a per-call `from_backend` plus
/// uniform allocations would cost more than the upload this path removes.
pub(super) struct DelayUpdateCache {
    shader: DelayUpdateBundle,
    t_links: Tensor<u32>,
    params: Tensor<glamx::UVec4>,
    num_actuated: u32,
}

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
    /// What [`Self::implicit_coriolis`] was the last time the `batch_indices`
    /// uniform was built. The kernels read the flag from that uniform, so the
    /// two drifting apart silently changes which dynamics path runs; the
    /// dispatch guard compares them.
    pub(super) coriolis_in_uniform: bool,
    /// Rebuild the joint and contact constraints from scratch every substep.
    /// On (the default) this matches a per-substep constraint refresh; off, the
    /// full build runs once per step and each later substep only refreshes the
    /// joint rhs / limit activity, which is cheaper and closer to how MuJoCo
    /// and Genesis step.
    pub(super) substep_refresh: bool,
    /// Split cadence: refresh the constraints every substep but keep the mass
    /// matrix and its LU factors per step. Ignored when `substep_refresh` is on.
    pub(super) substep_refresh_light: bool,
    /// When `false` (no joint limits / motors anywhere), the joint constraint
    /// kernel chain is skipped on the host side.
    pub(super) has_joint_constraints: bool,
    /// `true` once the joint-constraint bank has been grown to hold a
    /// dry-friction row per DoF; see `reserve_frictionloss_slots`.
    pub(super) frictionloss_slots_reserved: bool,
    /// Set when a capacity edit here has invalidated the shared `BatchIndices`
    /// uniform. The next `RbdPipeline` step re-uploads it and clears this;
    /// without that the kernels would index the resized buffers with stale
    /// per-batch capacities.
    pub(crate) constraint_caps_dirty: bool,

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
    /// Actuator-delay state, per batch `[tick, k, prev_target x
    /// links_per_batch]`. All zeros (the default) means no delay.
    pub(super) motor_delay_state: Tensor<f32>,
    /// `(num_batches, stride, 0, 0)` uniform for the delay tick dispatch.
    pub(super) motor_delay_params: Tensor<glamx::UVec4>,
    /// Cached shader and constants for the on-device delay-state refresh.
    pub(super) delay_update_cache: Option<DelayUpdateCache>,
    /// The sensed multibody link ids, `MAX_CONTACT_SENSORS` slots padded with
    /// `u32::MAX`. The same set is sensed on every multibody in every batch.
    pub(super) contact_sensor_links: Tensor<u32>,
    /// Per-(multibody, slot) summed normal-contact impulse, written once per
    /// step by `gpu_mb_sense_contact_impulses`.
    pub(super) contact_sensor_out: Tensor<f32>,
    /// Number of configured contact sensors; 0 skips the readout dispatch.
    pub(super) num_contact_sensors: u32,
    /// Shader plus staging buffers for the single-env reset scatter, created
    /// on first use.
    pub(super) env_reset: Option<super::env_reset::EnvResetBundle>,
    /// GPU-resident reset templates, published by `publish_reset_templates`.
    pub(super) reset_templates: Option<super::env_reset::ResetTemplatesMb>,
    /// One entry per (axis, actuated link set) the caller has scattered motor
    /// targets for. Grown on demand by [`Self::encode_scatter_motor_targets`].
    pub(super) scatter_caches: Vec<MotorScatterCache>,
    /// CPU mirror of [`Self::body_to_link`], batch-major with a
    /// `body_to_link_cap` stride. Backs [`Self::link_of_body`].
    pub(super) body_to_link_host: Vec<[u32; 2]>,
    /// Per-batch stride of [`Self::body_to_link_host`] (colliders per batch).
    pub(super) body_to_link_cap: u32,

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
    /// the kinematic-DOF mask, and Coulomb joint friction. Callers reading
    /// velocities should use only the first section.
    pub fn dof_state(&self) -> &Tensor<f32> {
        &self.dof_state
    }

    /// Sets whether the joint and contact constraints are rebuilt from scratch
    /// every substep. See [`Self::substep_refresh`]; on by default.
    pub fn set_substep_refresh(&mut self, enabled: bool) {
        self.substep_refresh = enabled;
    }

    /// Whether the per-substep constraint rebuild is enabled.
    pub fn substep_refresh(&self) -> bool {
        self.substep_refresh
    }

    /// Sets the split cadence: constraints per substep, mass matrix and LU per
    /// step. Ignored while [`Self::substep_refresh`] is on.
    pub fn set_substep_refresh_light(&mut self, enabled: bool) {
        self.substep_refresh_light = enabled;
    }

    /// Mutable view of [`Self::dof_state`], for callers that push generalized
    /// velocities straight into the buffer (e.g. an external RL env resetting
    /// one environment). Section offsets are the ones documented on
    /// [`Self::dof_state`]; writing past the velocity section overwrites the
    /// damping, armature and spring parameters.
    pub fn dof_state_mut(&mut self) -> &mut Tensor<f32> {
        &mut self.dof_state
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

    /// Scatters per-(actuated joint, env) motor target positions into
    /// `links_static` on the GPU, reading the targets from a GPU buffer so a
    /// GPU-resident policy can drive the motors with no host round-trip.
    ///
    /// `targets` is row-major `[num_actuated x num_batches]` (element
    /// `(j, env)` at `j · num_batches + env`) and `actuated_link_ids[j]` is the
    /// link index of actuated joint `j`. Sets `motors[axis].target_pos` and the
    /// matching `motor_axes` bit.
    ///
    /// This bypasses `links_static_mirror`: the scattered targets live only on
    /// the GPU, so do not interleave it with [`Self::set_motor`] /
    /// [`Self::set_motors`] on the same axis.
    pub fn scatter_motor_targets(
        &mut self,
        backend: &GpuBackend,
        targets: &Tensor<f32>,
        actuated_link_ids: &[u32],
        axis: u32,
    ) -> Result<(), GpuBackendError> {
        let mut enc = backend.begin_encoding();
        self.encode_scatter_motor_targets(backend, &mut enc, targets, actuated_link_ids, axis)?;
        backend.submit(enc)
    }

    /// [`Self::scatter_motor_targets`], recorded into an existing encoder so
    /// the control step shares one submit with the caller's other work.
    pub fn encode_scatter_motor_targets(
        &mut self,
        backend: &GpuBackend,
        enc: &mut <GpuBackend as Backend>::Encoder,
        targets: &Tensor<f32>,
        actuated_link_ids: &[u32],
        axis: u32,
    ) -> Result<(), GpuBackendError> {
        use khal::backend::Encoder as _;

        // Take the matching entry out so `self.links_static` can be borrowed
        // mutably for the dispatch; it goes back at the end.
        let hit = self
            .scatter_caches
            .iter()
            .position(|c| c.axis == axis && c.link_ids == actuated_link_ids);
        let cache = match hit {
            Some(i) => self.scatter_caches.swap_remove(i),
            None => {
                let num_actuated = actuated_link_ids.len() as u32;
                let uu = BufferUsages::STORAGE | BufferUsages::UNIFORM;
                MotorScatterCache {
                    shader: MotorScatterBundle::from_backend(backend)?,
                    t_links: Tensor::vector(backend, actuated_link_ids, BufferUsages::STORAGE)?,
                    u_na: Tensor::scalar(backend, num_actuated, uu)?,
                    u_ne: Tensor::scalar(backend, self.num_batches, uu)?,
                    u_ax: Tensor::scalar(backend, axis, uu)?,
                    num_actuated,
                    axis,
                    link_ids: actuated_link_ids.to_vec(),
                }
            }
        };
        {
            let mut pass = enc.begin_pass("[RBD] mb/scatter-motor-targets", None);
            cache.shader.scatter.call(
                &mut pass,
                [cache.num_actuated, self.num_batches, 1],
                targets,
                &mut self.links_static,
                &cache.t_links,
                &cache.u_na,
                &cache.u_ne,
                &cache.u_ax,
            )?;
        }
        self.scatter_caches.push(cache);
        Ok(())
    }

    /// Per-batch per-step link workspace (generalized coordinates, joint
    /// rotations, world-space link velocities), in the batch-interleaved SoA
    /// quad layout the kernels index. Read it back with `slow_read_buffer` for
    /// joint/base state observation and decode it with `ws_soa_to_structs`,
    /// which yields one struct per link laid out `env * links_per_batch + link`
    /// in [`from_rapier`](Self::from_rapier)'s link traversal order.
    pub fn links_workspace(&self) -> &Tensor<glamx::Vec4> {
        &self.links_workspace
    }

    /// Per-batch static link data (joint definitions, motors, limits, mass
    /// properties), batch-interleaved like [`Self::links_workspace`]. Exposed
    /// for diagnostics; use [`Self::set_motor`] / [`Self::set_motors`] to
    /// mutate motors so the CPU mirror stays in sync.
    pub fn links_static(&self) -> &Tensor<MultibodyLinkStatic> {
        &self.links_static
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
    /// on the CPU rapier joints, then call this to push them to the GPU. Only
    /// joint data is refreshed (coordinates, velocities and mass properties are
    /// untouched), so this cannot teleport links.
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
        backend.write_buffer(self.links_static.buffer_mut(), 0, &self.links_static_mirror)
    }

    /// Number of multibody-touching impulse joints in any batch.
    pub fn mb_impulse_joints_per_batch(&self) -> u32 {
        self.mb_imp_joints_per_batch
    }

    /// Populate the multibody-owned fields of `BatchIndices`. Leaves the
    /// RBD-side fields (`colliders_batch_capacity`, `contacts_batch_capacity`,
    /// `collision_pairs_batch_capacity`, `impulse_joints_batch_capacity`,
    /// `color_groups_batch_capacity`) untouched — the caller fills those.
    pub(crate) fn fill_batch_indices(&mut self, dst: &mut BatchIndices) {
        self.coriolis_in_uniform = self.implicit_coriolis;
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

    /// The per-multibody bank of unit (1-DoF) joint limit / motor constraints.
    pub fn joint_constraints(&self) -> &Tensor<MultibodyJointConstraint> {
        &self.joint_constraints
    }

    /// Per-batch stride of [`Self::joint_constraints`].
    pub fn joint_constraints_per_batch(&self) -> u32 {
        self.joint_constraints_per_batch
    }

    /// Per-batch stride of the joint-constraint `M⁻¹` column buffer.
    pub fn joint_constraint_columns_per_batch(&self) -> u32 {
        self.joint_constraint_columns_per_batch
    }

    /// Overwrites the per-DoF armature (reflected rotor inertia) section of
    /// [`Self::dof_state`]. `values` is `dofs_per_batch * num_batches` in
    /// env-major order (env outer, DoF inner).
    pub fn set_dof_armature(&mut self, backend: &GpuBackend, values: &[f32]) {
        self.write_dof_section(backend, 2, values, "armature");
    }

    /// Overwrites the per-DoF dry joint friction coefficients.
    pub fn set_dof_frictionloss(&mut self, backend: &GpuBackend, values: &[f32]) {
        if values.iter().any(|v| *v > 0.0) {
            self.reserve_frictionloss_slots(backend);
        }
        self.write_dof_section(backend, 6, values, "frictionloss");
    }

    /// Grows the joint-constraint bank by one slot per DoF of every multibody,
    /// the worst case for `gpu_mb_init_joint_constraints`' dry-friction rows
    /// (emitted only for DoFs whose `frictionloss` is non-zero). Idempotent,
    /// and never called unless some frictionloss is actually set, so scenes
    /// without joint friction keep the tighter limit/motor-only capacity.
    fn reserve_frictionloss_slots(&mut self, backend: &GpuBackend) {
        if self.frictionloss_slots_reserved {
            return;
        }
        self.frictionloss_slots_reserved = true;

        let mb_cap = self.multibodies_per_batch as usize;
        let nb = self.num_batches as usize;
        let mut cons_cap = 0u32;
        let mut max_constraints = 0u32;
        for b in 0..nb {
            let mut cons_off = 0u32;
            for i in 0..mb_cap {
                let info = &mut self.info_mirror[b * mb_cap + i];
                // Padding slots stay untouched: they hold no links, so the
                // kernels bail out before reading their offsets.
                if info.num_links == 0 {
                    continue;
                }
                info.first_constraint = cons_off;
                info.max_constraints += info.ndofs;
                cons_off += info.max_constraints;
                max_constraints = max_constraints.max(info.max_constraints);
            }
            cons_cap = cons_cap.max(cons_off);
        }
        let cons_cap = cons_cap.max(1);
        let cons_col_cap = cons_cap.saturating_mul(self.dofs_per_batch).max(1);

        // Batch-interleaved (batch-minor) upload, matching the build path.
        let mut interleaved = Vec::with_capacity(mb_cap * nb);
        for k in 0..mb_cap {
            for b in 0..nb {
                interleaved.push(self.info_mirror[b * mb_cap + k]);
            }
        }
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        self.multibody_info = Tensor::vector(backend, &interleaved, storage).unwrap();
        self.joint_constraints = Tensor::vector(
            backend,
            vec![MultibodyJointConstraint::default(); cons_cap as usize * nb],
            storage,
        )
        .unwrap();
        self.joint_constraint_columns =
            Tensor::vector(backend, vec![0.0f32; cons_col_cap as usize * nb], storage).unwrap();
        self.joint_constraints_per_batch = cons_cap;
        self.joint_constraint_columns_per_batch = cons_col_cap;
        self.max_joint_constraints = max_constraints;
        self.has_joint_constraints = max_constraints > 0;
        // `BatchIndices` now disagrees with these capacities. `RbdState::
        // set_dof_frictionloss` rebuilds it immediately; for callers who came
        // in through `GpuMultibodySet` directly, the next step does.
        self.constraint_caps_dirty = true;
    }

    /// Transposes an env-major `dofs_per_batch * num_batches` block into the
    /// batch-interleaved layout and writes it over section `section` of
    /// [`Self::dof_state`].
    fn write_dof_section(
        &mut self,
        backend: &GpuBackend,
        section: u64,
        values: &[f32],
        what: &str,
    ) {
        let cap = self.dofs_per_batch as usize;
        let nb = self.num_batches as usize;
        assert_eq!(
            values.len(),
            cap * nb,
            "{what}: expected dofs_per_batch * num_batches values"
        );
        let mut interleaved = vec![0.0f32; cap * nb];
        for b in 0..nb {
            for k in 0..cap {
                interleaved[k * nb + b] = values[b * cap + k];
            }
        }
        backend
            .write_buffer(
                self.dof_state.buffer_mut(),
                section * (cap * nb) as u64,
                &interleaved,
            )
            .unwrap();
    }

    /// Per-batch stride of the actuator-delay state buffer:
    /// `[tick, k, prev_target x links_per_batch]`.
    pub fn motor_delay_stride(&self) -> u32 {
        2 + self.links_per_batch
    }

    /// Uploads the actuator-delay state for every batch. `data.len()` must be
    /// `motor_delay_stride() * num_batches`; all zeros disables the delay.
    ///
    /// While a control step's substep counter `tick` is below that batch's `k`,
    /// every motor tracks `prev_target[link]` instead of its current target, so
    /// latency costs no mid-step host writes. Call this before the step's
    /// kernels are queued: an upload issued between queued substeps stalls the
    /// stream, which is exactly what the GPU-side delay exists to avoid.
    pub fn write_motor_delay_state(
        &mut self,
        backend: &GpuBackend,
        data: &[f32],
    ) -> Result<(), GpuBackendError> {
        assert_eq!(
            data.len(),
            (self.motor_delay_stride() * self.num_batches) as usize,
            "motor delay state: expected motor_delay_stride() * num_batches values"
        );
        backend.write_buffer(self.motor_delay_state.buffer_mut(), 0, data)
    }

    /// Per-step actuator-delay refresh on device (see
    /// `gpu_mb_delay_state_update`): `tick <- 0`, `k <- k_eff`, and the
    /// actuated links' prev-target lanes copied from `prev_targets`, the motor
    /// target tensor as it stood *before* this step's scatter.
    ///
    /// This replaces the full `stride * num_batches` host rebuild and upload
    /// that [`Self::write_motor_delay_state`] performs.
    pub fn update_motor_delay_state_gpu(
        &mut self,
        backend: &GpuBackend,
        prev_targets: &Tensor<f32>,
        k_eff: &Tensor<f32>,
        actuated_link_ids: &[u32],
    ) -> Result<(), GpuBackendError> {
        let mut enc = backend.begin_encoding();
        self.encode_update_motor_delay_state(
            backend,
            &mut enc,
            prev_targets,
            k_eff,
            actuated_link_ids,
        )?;
        backend.submit(enc)
    }

    /// [`Self::update_motor_delay_state_gpu`], recorded into an existing
    /// encoder so the delay refresh and the target scatter share one submit.
    pub fn encode_update_motor_delay_state(
        &mut self,
        backend: &GpuBackend,
        enc: &mut <GpuBackend as Backend>::Encoder,
        prev_targets: &Tensor<f32>,
        k_eff: &Tensor<f32>,
        actuated_link_ids: &[u32],
    ) -> Result<(), GpuBackendError> {
        use khal::backend::Encoder as _;
        let stride = self.motor_delay_stride();
        let cache = match self.delay_update_cache.take() {
            Some(c) => c,
            None => {
                let num_actuated = actuated_link_ids.len() as u32;
                DelayUpdateCache {
                    shader: DelayUpdateBundle::from_backend(backend)?,
                    t_links: Tensor::vector(backend, actuated_link_ids, BufferUsages::STORAGE)?,
                    params: Tensor::scalar(
                        backend,
                        glamx::UVec4::new(num_actuated, self.num_batches, stride, 0),
                        BufferUsages::STORAGE | BufferUsages::UNIFORM,
                    )?,
                    num_actuated,
                }
            }
        };
        {
            let mut pass = enc.begin_pass("[RBD] mb/delay-state-update", None);
            cache.shader.kernel.call(
                &mut pass,
                [cache.num_actuated, self.num_batches, 1],
                prev_targets,
                k_eff,
                &cache.t_links,
                &mut self.motor_delay_state,
                &cache.params,
            )?;
        }
        self.delay_update_cache = Some(cache);
        Ok(())
    }

    /// Configures the contact force sensor: senses the summed normal-contact
    /// impulse on these multibody links (at most
    /// [`MAX_CONTACT_SENSORS`](crate::shaders::dynamics::MAX_CONTACT_SENSORS);
    /// the same links are sensed on every multibody in every batch). Translate
    /// a local body / collider id with [`Self::link_of_body`] first. An empty
    /// slice disables the readout.
    pub fn set_contact_sensor_links(&mut self, backend: &GpuBackend, links: &[u32]) {
        const MAX: usize = crate::shaders::dynamics::MAX_CONTACT_SENSORS as usize;
        assert!(
            links.len() <= MAX,
            "at most {MAX} contact sensors supported (got {})",
            links.len()
        );
        let mut padded = [u32::MAX; MAX];
        padded[..links.len()].copy_from_slice(links);
        backend
            .write_buffer(self.contact_sensor_links.buffer_mut(), 0, &padded)
            .unwrap();
        self.num_contact_sensors = links.len() as u32;
    }

    /// The contact force-sensor readout, interleaved like the other per-mb
    /// buffers: slot `s` of multibody `m` in batch `b` at
    /// `(m · num_batches + b) · MAX_CONTACT_SENSORS + s`. Read it after a step;
    /// the values are accumulated normal impulses, so divide by the step `dt`
    /// for an average force.
    pub fn contact_sensor_out(&self) -> &Tensor<f32> {
        &self.contact_sensor_out
    }

    /// Number of configured contact sensors (0 means sensing is disabled).
    pub fn num_contact_sensors(&self) -> u32 {
        self.num_contact_sensors
    }

    /// `[multibody_idx, link_idx]` of the local body / collider id
    /// `local_body_id` within `batch`, or `[u32::MAX; 2]` when that body is not
    /// a multibody link. Resolved on the CPU mirror, so it costs no readback.
    pub fn link_of_body(&self, batch: u32, local_body_id: u32) -> [u32; 2] {
        let idx = batch as usize * self.body_to_link_cap as usize + local_body_id as usize;
        self.body_to_link_host
            .get(idx)
            .copied()
            .unwrap_or([u32::MAX; 2])
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
