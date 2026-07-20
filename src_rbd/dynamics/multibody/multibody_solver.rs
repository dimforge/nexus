//! The [`GpuMultibodySolver`] shader bundle and its per-substep dispatch phases.

use super::multibody_set::*;
use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::dynamics::{
    GpuMbBuildContactDelassus, GpuMbComputeDynamicsPre,
    GpuMbComputeDynamicsWithoutCoriolisPre,
    GpuMbFinalizeContactConstraints, GpuMbGravityAndLu, GpuMbGravityAndLuT1, GpuMbGravityAndLuT8,
    GpuMbGravityAndLuT16, GpuMbGravityAndLuT32, GpuMbInitContactConstraints,
    GpuMbInitJointConstraints, GpuMbIntegrate, GpuMbIntegrateVelocities,
    GpuMbRefreshJointConstraints, GpuMbRemoveImpulseJointConstraintBias,
    GpuMbResetContactWarmstart, GpuMbStashContactsLen, GpuMbWarmstartContactConstraints,
    GpuMbSolveConstraints, GpuMbSolveContactsDelassus, GpuMbSolveImpulseJointConstraints,
    GpuMbSolveJoints,
    GpuMbFinalizeImpulseJointConstraints,
    GpuMbUpdateImpulseJointConstraints, Velocity, WorldMassProperties,
};
use crate::shaders::utils::BatchIndices;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader bundle for multibody dynamics.
#[derive(Shader)]
pub struct GpuMultibodySolver {
    gravity_and_lu: GpuMbGravityAndLu,
    /// Packed tiers of `gravity_and_lu` — `64/T` multibodies per workgroup
    /// with a `T×T` shared tile each, selected by `max_ndofs`. The fallback
    /// `gravity_and_lu` (one multibody per workgroup, 64×64 tile) only runs
    /// for `max_ndofs > 32`.
    /// Serial tier (one thread per multibody, no barriers — see the kernel
    /// docs), selected for `max_ndofs ≤ 8`.
    gravity_and_lu_t1: GpuMbGravityAndLuT1,
    gravity_and_lu_t8: GpuMbGravityAndLuT8,
    gravity_and_lu_t16: GpuMbGravityAndLuT16,
    gravity_and_lu_t32: GpuMbGravityAndLuT32,
    compute_dynamics_pre: GpuMbComputeDynamicsPre,
    compute_dynamics_without_coriolis_pre: GpuMbComputeDynamicsWithoutCoriolisPre,
    init_joint_with_bias: GpuMbInitJointConstraints,
    /// Explicit-coriolis fast path: per-substep refresh of the joint rhs /
    /// limit activity (the columns and `inv_lhs` are per-step constants
    /// there, so the full build + back-solves run once per step).
    refresh_joint_constraints: GpuMbRefreshJointConstraints,
    init_contact_constraints: GpuMbInitContactConstraints,
    finalize_contact_constraints: GpuMbFinalizeContactConstraints,
    /// Fused joint+contact PGS sweep (one workgroup per multibody, shared-
    /// memory dof velocities). `use_bias = 0` runs the stabilization form,
    /// replacing the former separate remove-bias dispatches.
    solve_constraints: GpuMbSolveConstraints,
    /// Joint-only half of the sweep, used with the Delassus contact path
    /// (one kernel binding both joint and Delassus buffers would exceed the
    /// 8-storage-buffer budget).
    solve_joints: GpuMbSolveJoints,
    /// Fills the per-multibody Delassus blocks (`D = J M⁻¹ Jᵀ` + free-body
    /// coupling) right after the contact columns are finalized.
    build_contact_delassus: GpuMbBuildContactDelassus,
    /// Constraint-space contact sweep: `a = J·u` tracked incrementally in
    /// shared memory via the Delassus rows, breaking the per-iteration
    /// dof-space latency chain.
    solve_contacts_delassus: GpuMbSolveContactsDelassus,
    /// Zero the accumulated contact impulses once per frame (warmstart reset).
    reset_contact_warmstart: GpuMbResetContactWarmstart,
    /// Copy `contacts_len[batch]` into each `MultibodyInfo` once per step so
    /// `init_contact_constraints` (at the 8-storage-buffer limit) can bound
    /// its manifold scan by the actual count instead of the capacity.
    stash_contacts_len: GpuMbStashContactsLen,
    /// Re-apply the accumulated contact impulse each substep (warmstart).
    warmstart_contact_constraints: GpuMbWarmstartContactConstraints,
    update_impulse_joint_constraints: GpuMbUpdateImpulseJointConstraints,
    /// Finalize pass for the impulse-joint build (LU back-solve + `inv_lhs`),
    /// split out so the build pass fits 8 storage buffers.
    finalize_impulse_joint_constraints: GpuMbFinalizeImpulseJointConstraints,
    solve_impulse_joint_constraints: GpuMbSolveImpulseJointConstraints,
    remove_impulse_joint_constraint_bias: GpuMbRemoveImpulseJointConstraintBias,
    integrate_velocities: GpuMbIntegrateVelocities,
    integrate: GpuMbIntegrate,
}

/// Arguments for one multibody dispatch. The poses buffer is shared with the rest
/// of the rigid-body pipeline (FK writes link poses there); mass properties are
/// now owned by the multibody itself.
pub struct MultibodySolverArgs<'a> {
    /// Body poses (written by FK; consumed by every per-body computation).
    pub poses: &'a mut Tensor<Pose>,
    /// Per-collider world poses, used by `init_contact_constraints` to
    /// recover world-space contact normals and points from manifold features
    /// expressed in collider-local space.
    pub collider_world_poses: &'a Tensor<Pose>,
    /// Free-body world mass properties (read by `init_contact_constraints`).
    pub mprops: &'a Tensor<WorldMassProperties>,
    /// Per-batch contact manifold list (filled by narrow-phase).
    pub contacts: &'a Tensor<GpuIndexedContact>,
    /// Per-batch contact count (parallel to `contacts`).
    pub contacts_len: &'a Tensor<u32>,
    /// Free-body solver velocities (updated in place by `solve_contact_constraints`).
    pub solver_vels: &'a mut Tensor<Velocity>,
    /// Shared `BatchIndices` uniform — per-batch caps and packed-section
    /// offsets read by every multibody kernel. Owned by `RbdState`.
    pub batch_indices: &'a Tensor<BatchIndices>,
    /// Per-color-index uniform tensors (`color_uniforms[c]` holds `c`),
    /// shared with the contact/joint solvers. Bound by each colored
    /// impulse-joint sweep instead of a GPU-incremented cursor.
    pub color_uniforms: &'a [Tensor<u32>],
    /// GPU-written workgroup grid for the per-multibody contact-constraint
    /// dispatches: `[multibodies_batch_capacity, num_batches, 1]`, collapsed
    /// to zero workgroups when no batch has any contact this step (see
    /// `gpu_narrow_phase_init_contacts_dispatch`). Only valid AFTER the
    /// narrow phase of the current step.
    pub mb_sweep_indirect: &'a Tensor<[u32; 3]>,
}

impl GpuMultibodySolver {
    /// Runs FK → jacobians → mass matrix → gravity → LU solve in sequence on one pass.
    ///
    /// After completion, `mb.gen_accelerations()` holds `ẍ = M⁻¹ τ_g` (one per DOF).
    pub fn solve_gravity(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        let mut args = args;
        if mb.is_empty() {
            return Ok(());
        }
        self.compute_dynamics(pass, mb, &mut args)
    }

    /// Once-per-visible-step setup. After this call, `gen_forces` holds the
    /// generalized acceleration `a = M⁻¹ τ` and `mass_matrices` holds the LU
    /// factors. The caller then runs the substep phases once per substep, with
    /// the last call carrying `is_last_substep = true`.
    pub fn init_step(
        &self,
        encoder: &mut khal::backend::GpuEncoder,
        mut timestamps: Option<&mut khal::backend::GpuTimestamps>,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        use khal::backend::Encoder;
        if mb.is_empty() {
            return Ok(());
        }
        // Zero the accumulated contact impulses so the first substep's warmstart
        // starts cold (within a frame they are then preserved across substeps).
        // Flat (slot, multibody, batch) grid — impulse-field-only stores.
        {
            let mut pass = encoder.begin_pass("[RBD] mbi/reset", timestamps.as_deref_mut());
            let total_slots = mb.num_active_multibodies
                * mb.num_batches
                * crate::shaders::dynamics::MAX_MB_CONTACT_CONSTRAINTS_PER_MB;
            self.reset_contact_warmstart.call(
                &mut pass,
                [total_slots, 1, 1],
                &mut mb.contact_constraints,
                args.batch_indices,
            )?;
        }
        let mut pass = encoder.begin_pass("[RBD] mbi/dynamics", timestamps.as_deref_mut());
        self.compute_dynamics(&mut pass, mb, args)
    }

    /// Stash `contacts_len[batch]` into each `MultibodyInfo` — must run after
    /// the narrow phase (which writes `contacts_len`) and before the first
    /// `substep_build_constraints` of the step.
    pub fn stash_contacts_len(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        self.stash_contacts_len.call(
            pass,
            mb.flat_mb_dispatch(),
            &mut mb.multibody_info,
            args.contacts_len,
            args.batch_indices,
        )?;
        Ok(())
    }

    // Per-substep work is split into five phases so the pipeline can interleave
    // them with the rigid-body substep: `substep_integrate_velocities` (P1),
    // `substep_build_constraints` (P2), `substep_solve_with_bias` (P3),
    // `substep_integrate_positions` (P4) and `substep_solve_no_bias` (P5).

    /// P1: `dof_velocities += a · dt'` (apply the velocity increment).
    pub fn substep_integrate_velocities(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        let dispatch = mb.flat_mb_dispatch();
        self.integrate_velocities.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.dof_state,
            &mb.gen_forces,
            &mb.dt,
            args.batch_indices,
        )
    }

    /// P2: build limit/motor constraints and (build + finalize) the contact
    /// constraints, then warmstart the contacts.
    ///
    /// Takes the encoder (not a pass) so each kernel gets its own labeled
    /// timestamp pass — this phase held the two dominant single-robot costs
    /// (the contact scan and the joint-constraint back-solves), so per-kernel
    /// visibility is worth the pass splits.
    pub fn substep_build_constraints(
        &self,
        encoder: &mut khal::backend::GpuEncoder,
        mut timestamps: Option<&mut khal::backend::GpuTimestamps>,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
        first_substep: bool,
    ) -> Result<(), GpuBackendError> {
        use khal::backend::Encoder;
        if mb.is_empty() {
            return Ok(());
        }

        // With implicit coriolis, the mass matrix / LU / body jacobians are
        // recomputed every substep, so the joint + contact constraints (whose
        // M⁻¹Jᵀ columns depend on them) must be rebuilt every substep too. In
        // the explicit mode every column-derived quantity is a per-step
        // constant: the full build runs ONCE per step (see
        // `build_contact_constraints`) and each substep only refreshes the
        // joint rhs / limit activity / accumulated impulse from the
        // integrated joint positions (a no-op on the first substep — the
        // once-per-step build just wrote those exact values).
        if mb.implicit_coriolis {
            self.build_contact_constraints(encoder, timestamps.as_deref_mut(), mb, args)?;
        } else if mb.has_joint_constraints && !first_substep {
            let mut pass =
                encoder.begin_pass("[RBD] mbb/refresh-joint", timestamps.as_deref_mut());
            let dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
            self.refresh_joint_constraints.call(
                &mut pass,
                dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mut mb.joint_constraints,
                &mb.constraint_softness,
                args.batch_indices,
            )?;
        }

        // Warmstart: re-apply the accumulated contact impulse to dof_state (and
        // the free-body solver velocities) so the contact starts "warm" each
        // substep — mirrors rapier's per-substep `contact_constraints.warmstart`
        // and matches what the rigid-body solver does for free contacts. On the
        // first substep the impulse was just reset to 0 (Phase 0's
        // `reset_contact_warmstart`), so it is provably a no-op — skipped.
        // One 64-lane workgroup per multibody (one DOF per lane).
        if !first_substep {
            let mut pass =
                encoder.begin_pass("[RBD] mbb/warmstart-contact", timestamps.as_deref_mut());
            // Contact-only work: indirect grid collapses to zero workgroups
            // when no batch has any contact this step.
            self.warmstart_contact_constraints.call(
                &mut pass,
                args.mb_sweep_indirect,
                &mb.multibody_info,
                &mb.contact_constraints,
                &mb.contact_constraint_columns,
                &mut mb.dof_state,
                args.solver_vels,
                args.batch_indices,
            )?;
        }

        Ok(())
    }

    /// Build + finalize the contact constraints (normal + friction slots,
    /// free-body × multibody and self-contact pairs). `init` PRESERVES the
    /// accumulated impulse across substeps (zeroed once per frame by
    /// `reset_contact_warmstart` in `init_step`); `finalize` computes
    /// `inv_lhs` and the M⁻¹Jᵀ columns.
    ///
    /// Inputs are the narrow-phase manifolds, the collider world poses, the
    /// body jacobians and the mass-matrix LU. The first two only change once
    /// per step; the last two change per substep ONLY with implicit coriolis.
    /// So this runs once per step (from `solve_tgs`'s init pass, after the
    /// narrow phase) in the explicit mode, and once per substep otherwise.
    pub fn build_contact_constraints(
        &self,
        encoder: &mut khal::backend::GpuEncoder,
        mut timestamps: Option<&mut khal::backend::GpuTimestamps>,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        use khal::backend::Encoder;
        if mb.is_empty() {
            return Ok(());
        }

        // Joint limit/motor constraints: one 64-lane workgroup per multibody
        // (lane 0 emits the metadata serially — cheap; the per-constraint
        // M⁻¹-column LU back-solves run one-per-lane).
        if mb.has_joint_constraints {
            let mut pass = encoder.begin_pass("[RBD] mbb/init-joint", timestamps.as_deref_mut());
            let init_joint_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
            self.init_joint_with_bias.call(
                &mut pass,
                init_joint_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mb.mass_matrices,
                &mb.lu_pivots,
                &mut mb.joint_constraints,
                &mut mb.joint_constraint_columns,
                &mb.constraint_softness,
                args.batch_indices,
            )?;
        }

        // One 64-lane workgroup per multibody: the uniform emission walk runs
        // redundantly on every lane, the per-DOF `Jᵀ`-row fills one-per-lane.
        {
            let mut pass =
                encoder.begin_pass("[RBD] mbb/init-contact", timestamps.as_deref_mut());
            let init_contact_dispatch =
                [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
            self.init_contact_constraints.call(
                &mut pass,
                init_contact_dispatch,
                &mut mb.multibody_info,
                &mb.body_jacobians,
                &mb.body_to_link,
                &mut mb.contact_constraints,
                &mut mb.contact_constraint_jacs,
                &mb.constraint_softness,
                args.batch_indices,
                args.mprops,
                args.collider_world_poses,
                args.contacts,
            )?;
        }

        // One 64-lane workgroup per multibody: the per-constraint LU
        // back-solves are independent, so they run one-per-lane instead of
        // sequentially on a single thread. Contact-only work: indirect grid
        // collapses to zero workgroups on contact-free steps.
        {
            let mut pass =
                encoder.begin_pass("[RBD] mbb/finalize-contact", timestamps.as_deref_mut());
            self.finalize_contact_constraints.call(
                &mut pass,
                args.mb_sweep_indirect,
                &mb.multibody_info,
                &mb.mass_matrices,
                &mb.lu_pivots,
                &mut mb.contact_constraints,
                &mb.contact_constraint_jacs,
                &mut mb.contact_constraint_columns,
                args.batch_indices,
            )?;
        }

        // Delassus blocks for the constraint-space contact sweep (consumes
        // the columns finalized just above).
        if let Some(delassus) = &mut mb.contact_delassus {
            let mut pass =
                encoder.begin_pass("[RBD] mbb/build-delassus", timestamps.as_deref_mut());
            self.build_contact_delassus.call(
                &mut pass,
                args.mb_sweep_indirect,
                &mb.multibody_info,
                &mb.contact_constraints,
                &mb.contact_constraint_jacs,
                &mb.contact_constraint_columns,
                delassus,
                args.batch_indices,
            )?;
        }

        Ok(())
    }

    /// One joint+contact PGS sweep — the dof-space fused kernel, or (when the
    /// Delassus blocks are allocated) the joint-only kernel followed by the
    /// constraint-space contact kernel. `use_bias_idx` indexes
    /// `color_uniforms` (0 or 1, holding those constants).
    fn dispatch_solve(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
        solve_dispatch: [u32; 3],
        use_bias_idx: usize,
    ) -> Result<(), GpuBackendError> {
        let use_bias = &args.color_uniforms[use_bias_idx];
        if let Some(delassus) = &mb.contact_delassus {
            if mb.has_joint_constraints {
                self.solve_joints.call(
                    pass,
                    solve_dispatch,
                    &mb.multibody_info,
                    &mut mb.joint_constraints,
                    &mb.joint_constraint_columns,
                    &mut mb.dof_state,
                    use_bias,
                    args.batch_indices,
                )?;
            }
            // Contact-only work: indirect grid collapses to zero workgroups
            // on contact-free steps.
            self.solve_contacts_delassus.call(
                pass,
                args.mb_sweep_indirect,
                &mb.multibody_info,
                &mut mb.contact_constraints,
                &mb.contact_constraint_jacs,
                &mb.contact_constraint_columns,
                delassus,
                use_bias,
                args.batch_indices,
                &mut mb.dof_state,
                args.solver_vels,
            )?;
        } else if mb.has_joint_constraints {
            self.solve_constraints.call(
                pass,
                solve_dispatch,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mb.joint_constraint_columns,
                &mut mb.contact_constraints,
                &mb.contact_constraint_jacs,
                &mb.contact_constraint_columns,
                use_bias,
                args.batch_indices,
                &mut mb.dof_state,
                args.solver_vels,
            )?;
        } else {
            // No joint limits/motors anywhere: the fused sweep is contact-only
            // work, so the indirect grid (zero workgroups on contact-free
            // steps) replaces the full per-(multibody, batch) launch.
            self.solve_constraints.call(
                pass,
                args.mb_sweep_indirect,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mb.joint_constraint_columns,
                &mut mb.contact_constraints,
                &mb.contact_constraint_jacs,
                &mb.contact_constraint_columns,
                use_bias,
                args.batch_indices,
                &mut mb.dof_state,
                args.solver_vels,
            )?;
        }
        Ok(())
    }

    /// P3: one PGS sweep WITH bias over the joint, contact, and multibody-
    /// touching impulse-joint constraints.
    pub fn substep_solve_with_bias(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }

        // One 64-lane workgroup per multibody with the generalized velocities
        // held in workgroup memory (`color_uniforms[1]` holds the constant
        // 1 = use_bias). With the Delassus blocks allocated, the contact half
        // runs in constraint space instead (joints first, same order).
        let solve_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
        self.dispatch_solve(pass, mb, args, solve_dispatch, 1)?;

        // Multibody-touching impulse joints — generic (rb-mb / mb-mb)
        // constraints. Mirrors rapier's `JointGenericExternalConstraintBuilder::update`
        // plus a PGS sweep WITH bias.
        if mb.mb_imp_joints_per_batch > 0 {
            let imp_dispatch = [mb.mb_imp_joints_per_batch, mb.num_batches, 1];
            self.update_impulse_joint_constraints.call(
                pass,
                imp_dispatch,
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mut mb.mb_imp_joint_jacobians,
                &mb.constraint_softness,
                args.batch_indices,
                &mb.multibody_info,
                &mb.links_workspace,
                &mb.body_jacobians,
                args.poses,
                args.mprops,
            )?;
            // Finalize pass: LU back-solve `M⁻¹·Jᵀ` for the multibody sides and
            // compute `inv_lhs` (split out so the build pass above fits 8
            // storage buffers).
            self.finalize_impulse_joint_constraints.call(
                pass,
                imp_dispatch,
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mut mb.mb_imp_joint_jacobians,
                args.batch_indices,
                &mb.multibody_info,
                &mb.mass_matrices,
                &mb.lu_pivots,
            )?;
            // Colored PGS sweep WITH bias: one dispatch per color, each
            // color's joints solved race-free in parallel (graph coloring
            // done at init in `set_impulse_joints`). The color index is a
            // pre-built uniform instead of a GPU-incremented cursor.
            for c in 0..mb.mb_imp_joint_num_colors as usize {
                self.solve_impulse_joint_constraints.call(
                    pass,
                    // One workgroup (MB_LU_LANES threads) per joint; thread
                    // count = joints-in-largest-color × workgroup size.
                    [
                        mb.mb_imp_joint_max_color_group_len * MB_LU_LANES,
                        mb.num_batches,
                        1,
                    ],
                    &mb.mb_imp_joint_builders,
                    &mut mb.mb_imp_joint_constraints,
                    &mb.mb_imp_joint_jacobians,
                    &mb.mb_imp_joint_color_groups,
                    args.batch_indices,
                    &args.color_uniforms[c],
                    &mb.multibody_info,
                    &mut mb.dof_state,
                    args.solver_vels,
                )?;
            }
        }

        Ok(())
    }

    /// P4: integrate the multibody positions with the corrected `v`, then (if
    /// not the last substep) recompute the dynamics (M, LU, `a`) for the next
    /// substep's velocity update.
    pub fn substep_integrate_positions(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
        is_last_substep: bool,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        let dispatch = mb.flat_mb_dispatch();

        self.integrate.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mut mb.dof_values,
            &mb.dof_state,
            &mb.dt,
            args.batch_indices,
        )?;

        // Recompute `a` for the next substep — orientations / positions just
        // changed so M and τ are stale. Skipped on the last substep (rapier
        // skips it too: `if !is_last_substep`).
        // NOTE: we also only update the mass matrix a single time if running without
        //       `implicit_coriolis`. This further improves performances as that’s the main
        //       purpose of disabling the implicit handling of coriolis forces (and makes it
        //       closer to Mujoco/Genesis).
        if !is_last_substep && mb.implicit_coriolis {
            self.compute_dynamics(pass, mb, args)?;
        }

        Ok(())
    }

    /// P5: stabilization — fused remove-bias + final PGS sweep WITHOUT bias for
    /// joint limits/motors, contacts, and multibody-touching impulse joints.
    /// Settles velocity along constrained DOFs to zero (no rebound from the
    /// positional bias).
    pub fn substep_solve_no_bias(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }

        // Stabilization sweep: `use_bias = 0` (`color_uniforms[0]`) makes the
        // kernels read `rhs_wo_bias` directly, which replaces the former
        // remove-bias read-modify-write dispatches (every constraint is
        // re-initialized next substep, so the persistent `rhs` rewrite was
        // never needed).
        let solve_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
        self.dispatch_solve(pass, mb, args, solve_dispatch, 0)?;
        if mb.mb_imp_joints_per_batch > 0 {
            let imp_dispatch = [mb.mb_imp_joints_per_batch, mb.num_batches, 1];
            self.remove_impulse_joint_constraint_bias.call(
                pass,
                imp_dispatch,
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mb.mb_imp_joint_count,
                args.batch_indices,
            )?;
            // Final stabilization sweep WITHOUT bias — colored, one
            // dispatch per color (see the with-bias sweep above).
            for c in 0..mb.mb_imp_joint_num_colors as usize {
                self.solve_impulse_joint_constraints.call(
                    pass,
                    // One workgroup (MB_LU_LANES threads) per joint; thread
                    // count = joints-in-largest-color × workgroup size.
                    [
                        mb.mb_imp_joint_max_color_group_len * MB_LU_LANES,
                        mb.num_batches,
                        1,
                    ],
                    &mb.mb_imp_joint_builders,
                    &mut mb.mb_imp_joint_constraints,
                    &mb.mb_imp_joint_jacobians,
                    &mb.mb_imp_joint_color_groups,
                    args.batch_indices,
                    &args.color_uniforms[c],
                    &mb.multibody_info,
                    &mut mb.dof_state,
                    args.solver_vels,
                )?;
            }
        }

        Ok(())
    }

    /// Recompute the dynamics (mass matrix, LU factors, generalized
    /// acceleration). After this call, `gen_forces` holds the generalized
    /// acceleration `a` for the *next* substep's velocity update.
    fn compute_dynamics(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        // Fused FK + body-jacobians + velocity propagation + Mass-matrix
        // assembly. Packed: `64 / mb_pack_lanes` multibodies per workgroup,
        // flattened (multibody, batch) grid.
        let pre_dispatch = mb.packed_wg_dispatch();
        if mb.implicit_coriolis {
            self.compute_dynamics_pre.call(
                pass,
                pre_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mut mb.links_workspace,
                args.poses,
                &mut mb.body_jacobians,
                &mut mb.mass_matrices,
                &mut mb.coriolis_packed,
                &mb.dof_state,
                &mb.dt,
                args.batch_indices,
            )?;
        } else {
            self.compute_dynamics_without_coriolis_pre.call(
                pass,
                pre_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mut mb.links_workspace,
                args.poses,
                &mut mb.body_jacobians,
                &mut mb.mass_matrices,
                &mb.dof_state,
                &mb.dt,
                args.batch_indices,
            )?;
        }

        // Fused gravity + LU factor + LU solve. Packed tiers put `64/T`
        // multibodies in each workgroup (T×T shared tile per slot, flattened
        // (multibody, batch) grid — the shared tile size forces compile-time
        // variants, unlike the runtime-tiered `pre` kernel); the 64×64-tile
        // fallback keeps the legacy one-workgroup-per-multibody 2D grid.
        macro_rules! grav_lu {
            ($kernel:ident) => {
                self.$kernel.call(
                    pass,
                    mb.packed_wg_dispatch(),
                    &mb.multibody_info,
                    &mb.links_static,
                    &mut mb.links_workspace,
                    &mb.body_jacobians,
                    &mut mb.gen_forces,
                    &mut mb.mass_matrices,
                    &mut mb.lu_pivots,
                    &mb.dof_state,
                    &mb.gravity,
                    args.batch_indices,
                )?
            };
        }
        match mb.pack_lanes() {
            1 => grav_lu!(gravity_and_lu_t1),
            8 => grav_lu!(gravity_and_lu_t8),
            16 => grav_lu!(gravity_and_lu_t16),
            32 => grav_lu!(gravity_and_lu_t32),
            _ => {
                let grav_lu_dispatch =
                    [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
                self.gravity_and_lu.call(
                    pass,
                    grav_lu_dispatch,
                    &mb.multibody_info,
                    &mb.links_static,
                    &mut mb.links_workspace,
                    &mb.body_jacobians,
                    &mut mb.gen_forces,
                    &mut mb.mass_matrices,
                    &mut mb.lu_pivots,
                    &mb.dof_state,
                    &mb.gravity,
                    args.batch_indices,
                )?;
            }
        }

        Ok(())
    }
}
