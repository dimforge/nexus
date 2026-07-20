//! The [`GpuMultibodySolver`] shader bundle and its per-substep dispatch phases.

use super::multibody_set::*;
use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::dynamics::{
    GpuMbComputeDynamicsPre,
    GpuMbComputeDynamicsWithoutCoriolisPre,
    GpuMbFinalizeContactConstraints, GpuMbGravityAndLu, GpuMbInitContactConstraints,
    GpuMbInitJointConstraints, GpuMbIntegrate, GpuMbIntegrateVelocities,
    GpuMbRemoveContactConstraintBias, GpuMbRemoveImpulseJointConstraintBias,
    GpuMbResetContactWarmstart, GpuMbWarmstartContactConstraints,
    GpuMbRemoveSolveJointNoBias, GpuMbSolveContactConstraints, GpuMbSolveImpulseJointConstraints,
    GpuMbFinalizeImpulseJointConstraints, GpuMbSolveJointConstraints,
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
    compute_dynamics_pre: GpuMbComputeDynamicsPre,
    compute_dynamics_without_coriolis_pre: GpuMbComputeDynamicsWithoutCoriolisPre,
    solve_joint_with_bias: GpuMbSolveJointConstraints,
    init_joint_with_bias: GpuMbInitJointConstraints,
    /// Fused remove-bias + solve-without-bias for the stabilization sweep.
    remove_solve_joint_no_bias: GpuMbRemoveSolveJointNoBias,
    init_contact_constraints: GpuMbInitContactConstraints,
    finalize_contact_constraints: GpuMbFinalizeContactConstraints,
    solve_contact_constraints: GpuMbSolveContactConstraints,
    /// Zero the accumulated contact impulses once per frame (warmstart reset).
    reset_contact_warmstart: GpuMbResetContactWarmstart,
    /// Re-apply the accumulated contact impulse each substep (warmstart).
    warmstart_contact_constraints: GpuMbWarmstartContactConstraints,
    remove_contact_constraint_bias: GpuMbRemoveContactConstraintBias,
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
        if mb.is_empty() {
            return Ok(());
        }
        // Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis
        // mass-matrix assembly (4 dispatches → 1) — see
        // `gpu_mb_compute_dynamics_pre`. Only the implicit-Coriolis path is
        // wired through the fused kernel; the explicit-Coriolis fallback keeps
        // the legacy split path.
        let pre_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
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

        // Fused: gravity / Coriolis force assembly + LU factor + LU solve in
        // a single dispatch. Replaces the previous 2-dispatch chain
        // (apply_gravity_with_coriolis → lu_factor_and_solve) — drops one
        // WebGPU dispatch per `compute_dynamics` call.
        let grav_lu_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
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

        Ok(())
    }

    /// Once-per-visible-step setup. After this call, `gen_forces` holds the
    /// generalized acceleration `a = M⁻¹ τ` and `mass_matrices` holds the LU
    /// factors. The caller then runs the substep phases once per substep, with
    /// the last call carrying `is_last_substep = true`.
    pub fn init_step(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        // Zero the accumulated contact impulses so the first substep's warmstart
        // starts cold (within a frame they are then preserved across substeps).
        self.reset_contact_warmstart.call(
            pass,
            [mb.multibodies_per_batch, mb.num_batches, 1],
            &mb.multibody_info,
            &mut mb.contact_constraints,
            args.batch_indices,
        )?;
        self.compute_dynamics(pass, mb, args)
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
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];
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
    pub fn substep_build_constraints(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

        if mb.has_joint_constraints {
            // TODO(PERF): joints init could parallelized. We either need to rework
            //             the flow of the kernel to that the LU parts are not in
            //             potentially diverging code paths, or we need to have
            //             each link have its limits/motors generated by a separate
            //             threadgroups (which might actually be better for lower
            //             divergence and allow us to theadgroup-parallelize the LU
            //             solve).
            self.init_joint_with_bias.call(
                pass,
                dispatch,
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

        // Build + finalize contact constraints (normal-only, free body ×
        // multibody pairs only). `init` PRESERVES the accumulated impulse across
        // substeps (zeroed once per frame by `reset_contact_warmstart` in
        // `init_step`); `finalize` recomputes `inv_lhs` and the M⁻¹Jᵀ columns.
        self.init_contact_constraints.call(
            pass,
            dispatch,
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

        self.finalize_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.mass_matrices,
            &mb.lu_pivots,
            &mut mb.contact_constraints,
            &mb.contact_constraint_jacs,
            &mut mb.contact_constraint_columns,
            args.batch_indices,
        )?;

        // Warmstart: re-apply the accumulated contact impulse to dof_state (and
        // the free-body solver velocities) so the contact starts "warm" each
        // substep — mirrors rapier's per-substep `contact_constraints.warmstart`
        // and matches what the rigid-body solver does for free contacts. On the
        // first substep the impulse was just reset to 0, so this is a no-op.
        self.warmstart_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.contact_constraints,
            &mb.contact_constraint_columns,
            &mut mb.dof_state,
            args.solver_vels,
            args.batch_indices,
        )?;

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
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

        if mb.has_joint_constraints {
            self.solve_joint_with_bias.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mut mb.joint_constraint_columns,
                &mut mb.dof_state,
                args.batch_indices,
            )?;
        }

        self.solve_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.contact_constraints,
            &mb.contact_constraint_jacs,
            &mb.contact_constraint_columns,
            &mut mb.dof_state,
            args.solver_vels,
            args.batch_indices,
        )?;

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
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

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
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

        if mb.has_joint_constraints {
            self.remove_solve_joint_no_bias.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mb.joint_constraint_columns,
                &mut mb.dof_state,
                args.batch_indices,
            )?;
        }
        self.remove_contact_constraint_bias.call(
            pass,
            dispatch,
            &mut mb.contact_constraints,
            &mb.multibody_info,
            args.batch_indices,
        )?;
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
        }

        // (joint sweep WITHOUT bias was fused into `remove_solve_joint_no_bias` above.)
        self.solve_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.contact_constraints,
            &mb.contact_constraint_jacs,
            &mb.contact_constraint_columns,
            &mut mb.dof_state,
            args.solver_vels,
            args.batch_indices,
        )?;
        if mb.mb_imp_joints_per_batch > 0 {
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
        // Fused FK + body-jacobians + velocity propagation + Mass-matrix assembly
        let pre_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
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

        // Fused gravity + LU factor + LU solve.
        let grav_lu_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
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

        Ok(())
    }
}
