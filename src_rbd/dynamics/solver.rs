//! GPU-parallel constraint solver using graph coloring.
//!
//! Constraint-based physics solver running entirely on the GPU, using graph
//! coloring to solve constraints in parallel without data races. Uses the
//! `Soft-TGS` algorithm (as in Rapier).
//!
//! The contact constraints are stored in color-ordered tiles (see the `contact_tiles` shader
//! module): every iteration walks a contiguous range of them. Their topology and identity stay
//! in contact order, in the contact links used for the matching, mass splitting and coloring.

use super::contact_kernels::{
    ContactConstraints, GpuGatherWarmstartVelocities, GpuPrepareConstraints,
    GpuScaleConstraintImpulses, GpuSolveConstraints, GpuSolveConstraintsBiased,
    GpuSolveConstraintsFused, GpuSolveConstraintsFusedCached, GpuSolveConstraintsTail,
    GpuSolveConstraintsTailBiased, GpuSolveConstraintsTailUnbiased,
    GpuSolveConstraintsTailUnbiasedCached, GpuSolveConstraintsUnbiased,
    GpuSolveConstraintsUnbiasedCached, GpuWarmstartConstraints, GpuWarmstartConstraintsFused,
};
use crate::dynamics::joint::{GpuJointSolver, JointSolverArgs};
use crate::dynamics::mass_splitting::{GpuMassSplitting, HubState, SplitArgs};
#[cfg(feature = "dim3")]
use crate::dynamics::multibody::{GpuMultibodySet, GpuMultibodySolver, MultibodySolverArgs};
use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::broad_phase::ContactPlan;
#[cfg(feature = "dim3")]
use crate::shaders::dynamics::MbContactIndexEntry;
use crate::shaders::dynamics::contact_tiles::TAIL_COLOR;
use crate::shaders::dynamics::{BIAS_MODE_BIAS, BIAS_MODE_BIAS_FRICTION};
use crate::shaders::dynamics::{
    ContactLink, GpuApplySolverVelsInc, GpuInitContactLinks, GpuInitSolverBodies,
    GpuInitSolverVelsInc, GpuIntegrateLinearized, GpuMatchContactLinks, GpuSolverCleanup,
    GpuSolverCountConstraints, GpuSolverFinalize, GpuSolverSortConstraints, LocalMassProperties,
    RbdSimParams, Velocity, WorldMassProperties,
};
use crate::utils::{GpuPrefixSum, PrefixSumWorkspace};
use khal::Shader;
use khal::backend::{
    DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuPass, GpuTimestamps,
};
use vortx::tensor::Tensor;

/// GPU shader bundle for the constraint solver.
#[derive(Shader)]
pub struct GpuSolver {
    /// Initializes the contact links from the contact manifolds.
    init_links: GpuInitContactLinks,
    /// Matches the contact links with the previous step's (warmstart and contact recycling).
    match_links: GpuMatchContactLinks,
    /// Counts the constraints of each body.
    count_constraints: GpuSolverCountConstraints,
    /// Lists the constraints of each body.
    sort_constraints: GpuSolverSortConstraints,
    /// Builds the contact constraints and gives them their previous impulses, or recycled
    /// contacts.
    prepare_constraints: GpuPrepareConstraints,
    /// Scales the warmstart impulses (warmstart coefficients other than 1).
    scale_constraint_impulses: GpuScaleConstraintImpulses,
    /// Sums the cached warmstart velocity changes of each body's constraints.
    gather_warmstart_velocities: GpuGatherWarmstartVelocities,
    /// Applies the warmstart of one color's constraints (scatter-style).
    warmstart_constraints: GpuWarmstartConstraints,
    /// Applies the warmstart of every color, with one workgroup per batch.
    warmstart_constraints_fused: GpuWarmstartConstraintsFused,
    /// Gauss-Seidel iteration over one color, in the mode given by a uniform.
    solve_constraints: GpuSolveConstraints,
    /// Fixed bias-only mode for the usual biased iteration.
    solve_constraints_biased: GpuSolveConstraintsBiased,
    /// Fixed friction/stabilization mode for the unbiased iteration.
    solve_constraints_unbiased: GpuSolveConstraintsUnbiased,
    /// Unbiased iteration that also caches the warmstart of the next substep.
    solve_constraints_unbiased_cached: GpuSolveConstraintsUnbiasedCached,
    /// The iterations over the sparse colors from [`TAIL_COLOR`], in one workgroup.
    solve_constraints_tail: GpuSolveConstraintsTail,
    solve_constraints_tail_biased: GpuSolveConstraintsTailBiased,
    solve_constraints_tail_unbiased: GpuSolveConstraintsTailUnbiased,
    solve_constraints_tail_unbiased_cached: GpuSolveConstraintsTailUnbiasedCached,
    /// The iterations over every color with one workgroup per batch, used when per-batch
    /// constraint counts are small.
    solve_constraints_fused: GpuSolveConstraintsFused,
    solve_constraints_fused_cached: GpuSolveConstraintsFusedCached,
    /// Clears solver velocities and constraint counts.
    cleanup: GpuSolverCleanup,
    /// Initializes solver velocity increments.
    init_solver_vels_inc: GpuInitSolverVelsInc,
    /// Seeds the COM-centered solver poses from the body world poses
    /// (rapier's `SolverBodies::copy_from`). Run once per step.
    init_solver_bodies: GpuInitSolverBodies,
    /// Applies accumulated solver velocity increments.
    apply_solver_vels_inc: GpuApplySolverVelsInc,
    /// Integrates positions from velocities.
    integrate_linearized: GpuIntegrateLinearized,
    /// Writes solver velocities and converts the COM-centered solver poses
    /// back to body-origin poses.
    finalize: GpuSolverFinalize,
}

/// Arguments for constraint preparation and TGS dispatch.
pub struct SolverArgs<'a> {
    /// Total number of colors from graph coloring.
    pub num_colors: u32,
    /// Number of simulation batches.
    pub num_batches: u32,
    /// Number of colliders.
    pub num_colliders: u32,
    /// Contact manifolds generated by narrow-phase.
    pub contacts: &'a Tensor<GpuIndexedContact>,
    /// Clamped per-frame list totals (see `gpu_contact_plan`); `[PLAN_BOUND]`
    /// bounds the flat contact dispatches.
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// Contact→multibody index storage, rebuilt each step by
    /// `GpuMultibodySolver::layout_contact_constraints`.
    #[cfg(feature = "dim3")]
    pub mb_contact_index: &'a mut Tensor<MbContactIndexEntry>,
    /// Deterministic mode only: output of the sort of the multibody contact-index segments.
    #[cfg(feature = "dim3")]
    pub stable_mb_contact_index: Option<&'a mut Tensor<MbContactIndexEntry>>,
    /// Flat dispatch grid over the whole contacts range.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// This step's contact constraints.
    pub constraints: &'a mut ContactConstraints,
    /// The previous step's contact constraints, with their final impulses.
    pub old_constraints: &'a ContactConstraints,
    /// Previous-step per-body adjacency ends.
    pub old_body_constraint_counts: &'a Tensor<u32>,
    /// Previous-step adjacency entries.
    pub old_body_constraint_ids: &'a Tensor<u32>,
    /// Previous and current contact validity metadata (updated during preparation).
    pub recycle_states: &'a mut super::ContactRecycleStates,
    /// Global simulation parameters.
    pub sim_params: &'a Tensor<RbdSimParams>,
    /// Rigid body world-origin poses. Mirrors rapier's `RigidBody::position`.
    /// Read at the start of each step to seed [`Self::solver_body_poses`] and
    /// written back at the end of the substep loop by `finalize`.
    pub body_poses: &'a mut Tensor<Pose>,
    /// Rigid-body poses centered at the rigid-body center-of-mass — rapier's
    /// `SolverPose`. This is the only pose buffer the solver substep loop
    /// touches. Seeded from `body_poses` at step start and written back to
    /// `body_poses` at step end.
    pub solver_body_poses: &'a mut Tensor<Pose>,
    /// Per-collider local pose relative to the rigid-body it is attached to.
    pub collider_local_poses: &'a Tensor<Pose>,
    /// Per-collider world poses (= `body_poses[i] * collider_local_poses[i]`),
    /// kept up-to-date once per step before broad/narrow-phase.
    pub collider_world_poses: &'a Tensor<Pose>,
    /// Rigid body velocities.
    pub vels: &'a mut Tensor<Velocity>,
    /// Solver working velocities.
    pub solver_vels: &'a mut Tensor<Velocity>,
    /// Accumulated velocity increments during substeps.
    pub solver_vels_inc: &'a mut Tensor<Velocity>,
    /// World-space mass properties.
    pub mprops: &'a Tensor<WorldMassProperties>,
    /// Local-space mass properties.
    pub local_mprops: &'a Tensor<LocalMassProperties>,
    /// Number of constraints per body.
    ///
    /// All constraints of all the bodies part of the same multibody are counted in a single
    /// entry at its root’s body index.
    pub body_constraint_counts: &'a mut Tensor<u32>,
    /// Constraint IDs associated with each body.
    ///
    /// All constraints of all the bodies part of the same multibody are in the same list associated
    /// to the multibody’s root.
    pub body_constraint_ids: &'a mut Tensor<u32>,
    /// The `(color, batch)` bucket buffer (color-major, post-scatter exclusive
    /// ends): bucket `k` of `sorted_links` spans `[buckets[k-1], buckets[k])`.
    pub color_buckets: &'a Tensor<u32>,
    /// The constraint links bucket-sorted by `(color, batch)`.
    pub sorted_links: &'a Tensor<ContactLink>,
    /// Previous-frame per-color thread counts. Zero means feedback is not available.
    pub color_dispatch_threads: &'a [u32; 64],
    /// Whether previous-frame statistics suggest a short late-color suffix, solved by one
    /// workgroup.
    pub fuse_tail_colors: bool,
    /// Current GPU grid, used before feedback arrives or when readback is disabled.
    pub color_dispatch_indirect: &'a Tensor<[u32; 3]>,
    /// Per-color-index uniform tensors: `color_uniforms[c] == c`.
    pub color_uniforms: &'a [Tensor<u32>],
    /// Prefix sum shader for building constraint ranges.
    pub prefix_sum: &'a GpuPrefixSum,
    /// Number of solver iterations (max across all environments).
    pub num_solver_iterations: u32,
    /// PGS iterations of the biased pass per substep, shared by the
    /// rigid-body and multibody iterations (`RbdSimParams::num_internal_pgs_iterations`).
    pub num_internal_pgs_iterations: u32,
    /// Per-body graph-coloring group id (multibody-aware).
    pub body_group: &'a Tensor<u32>,
    /// Per-body flag, 1 for multibody links, whose contacts the rigid-body
    /// pipeline leaves to the multibody solver.
    pub body_is_multibody: &'a Tensor<u32>,
    /// When `true` (no multibody in the scene), warmstart uses the single
    /// gather-per-body dispatch instead of one scatter dispatch per color.
    /// The gather variant looks bodies up by their own id, which is only
    /// correct when `body_group` is the identity (multibody constraints are
    /// counted on their root's slot with link-id constraint sides).
    pub colorless_warmstart: bool,
    /// Whether the fused colored kernels are used.
    ///
    /// This is generally used when the number of constraints is small wrt.
    /// the number of environments.
    pub fused_color_dispatches: bool,
    /// `true` when every rigid-body contact constraint is provably a no-op.
    pub rb_contacts_inert: bool,
    /// Solve friction rows during the biased pass too
    /// (`RbdSimParams::friction_in_bias_pass`).
    pub friction_in_bias_pass: bool,
    /// Shared per-batch indices.
    pub batch_indices: &'a Tensor<crate::shaders::utils::BatchIndices>,
    /// The one gravity uniform every rigid-body and multibody kernel reads.
    pub gravity: &'a Tensor<glamx::Vec4>,
    /// Mass-splitting state of the bodies with many contacts.
    pub hubs: &'a mut HubState,
    /// Mass-splitting kernels.
    pub mass_splitting: &'a GpuMassSplitting,
    /// Whether the warmstart impulses are scaled (a warmstart coefficient other than 1).
    pub scale_warmstart_impulses: bool,
    /// GPU-written workgroup grid for the per-multibody contact-constraint
    /// dispatches (zero workgroups on contact-free steps).
    pub mb_dispatch_indirect: &'a Tensor<[u32; 3]>,
}

impl<'a> SolverArgs<'a> {
    fn color_dispatch_grid(&self, color: u32) -> DispatchGrid<'a, GpuBackend> {
        match self.color_dispatch_threads.get(color as usize).copied() {
            Some(threads) if threads > 0 => threads.into(),
            _ => self.color_dispatch_indirect.into(),
        }
    }
}

impl GpuSolver {
    /// Prepares the constraints before their coloring: their links, per-body lists, match
    /// with the previous step and mass splitting.
    pub fn prepare<'a>(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        args: SolverArgs<'a>,
        prefix_sum_workspace: &'a mut PrefixSumWorkspace,
    ) -> Result<(), GpuBackendError> {
        // Cleanup zeroes body_constraint_counts, solver_vels, vels, mprops.
        self.cleanup.call(
            pass,
            args.num_colliders * args.num_batches,
            args.body_constraint_counts,
            args.solver_vels,
            args.vels,
            args.mprops,
            args.batch_indices,
            &mut args.hubs.first_slot,
            &mut args.hubs.counts,
        )?;

        // Seed `solver_body_poses` from `body_poses`: rapier's
        // `SolverBodies::copy_from`. After this, only the COM-centered solver
        // poses are touched until the final `finalize` writeback.
        self.init_solver_bodies.call(
            pass,
            args.num_colliders * args.num_batches,
            args.body_poses,
            args.local_mprops,
            args.solver_body_poses,
            args.batch_indices,
        )?;

        if args.rb_contacts_inert {
            return Ok(());
        }

        self.init_links.call(
            pass,
            args.contacts_len_indirect,
            args.contacts,
            &mut args.constraints.links,
            args.body_is_multibody,
            args.contact_plan,
        )?;

        self.count_constraints.call(
            pass,
            args.contacts_len_indirect,
            args.contacts,
            args.body_constraint_counts,
            args.body_is_multibody,
            args.mprops,
            args.contact_plan,
        )?;

        // One global cumulative scan (bodies of every batch): the constraint
        // ranges live in one flat `body_constraint_ids` list.
        args.prefix_sum.launch(
            backend,
            pass,
            prefix_sum_workspace,
            args.body_constraint_counts,
            1,
        )?;

        self.sort_constraints.call(
            pass,
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.mprops,
            args.contacts,
            args.contact_plan,
            args.body_constraint_ids,
            args.body_is_multibody,
        )?;

        let (recycle_states, recycle_offsets) = args.recycle_states.bindings();
        self.match_links.call(
            pass,
            args.contacts_len_indirect,
            args.contacts,
            &mut args.constraints.links,
            args.old_body_constraint_counts,
            args.old_body_constraint_ids,
            &args.old_constraints.links,
            &args.old_constraints.constraint_indices,
            recycle_states,
            args.collider_world_poses,
            args.contact_plan,
            args.sim_params,
            recycle_offsets,
        )?;

        args.mass_splitting.split(
            pass,
            args.hubs,
            SplitArgs {
                num_body_slots: args.num_colliders * args.num_batches,
                body_constraint_counts: args.body_constraint_counts,
                body_constraint_ids: args.body_constraint_ids,
                links: &mut args.constraints.links,
                mprops: args.mprops,
                body_group: args.body_group,
                batch_indices: args.batch_indices,
            },
        )?;

        Ok(())
    }

    /// Builds the contact constraints once their order is known (after the color bucket sort),
    /// and caches their warmstart for the first substep.
    pub fn build_constraints(
        &self,
        pass: &mut GpuPass,
        args: &mut SolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if args.rb_contacts_inert {
            return Ok(());
        }
        let (recycle_states, recycle_offsets) = args.recycle_states.bindings();
        self.prepare_constraints.call(
            pass,
            args.contacts_len_indirect,
            args.sorted_links,
            args.contacts,
            args.mprops,
            &*args.solver_body_poses,
            &*args.vels,
            &*recycle_states,
            &args.old_constraints.tiles,
            &mut args.constraints.tiles,
            args.contact_plan,
            recycle_offsets,
        )
    }

    /// Solves constraints using the TGS (Total Gauss-Seidel) algorithm.
    ///
    /// When `multibody` is `Some`, multibody substep work is interleaved with
    /// the rigid-body substep work.
    pub fn solve_tgs<'a>(
        &self,
        encoder: &mut GpuEncoder,
        mut timestamps: Option<&mut GpuTimestamps>,
        joint_solver: &GpuJointSolver,
        args: SolverArgs<'a>,
        mut joint_args: JointSolverArgs<'a>,
        #[cfg(feature = "dim3")] multibody: Option<(&GpuMultibodySolver, &mut GpuMultibodySet)>,
    ) -> Result<(), GpuBackendError> {
        let num_substeps = args.num_solver_iterations;
        #[cfg(feature = "dim3")]
        let (mb_solver, mut mb_state) = match multibody {
            Some((s, st)) => (Some(s), Some(st)),
            None => (None, None),
        };

        let skip_rb = args.rb_contacts_inert;
        let joints_empty = joint_args.joints.is_empty();
        #[cfg(feature = "dim3")]
        let mut args = args;
        #[cfg(feature = "dim3")]
        let stable_mb_contact_index = args.stable_mb_contact_index.take();

        /*
         * Init solver vel increments.
         */
        {
            let mut pass = encoder.begin_pass("[RBD] slv/init", timestamps.as_deref_mut());
            if !skip_rb {
                self.init_solver_vels_inc.call(
                    &mut pass,
                    args.num_colliders * args.num_batches,
                    args.solver_vels_inc,
                    args.mprops,
                    args.sim_params,
                    args.batch_indices,
                    args.gravity,
                )?;
            }

            joint_solver.init(&mut pass, &mut joint_args)?;

            // Lay out the multibody contact-constraint segments and the
            // contact→multibody index, after the narrow phase and before the
            // first substep build.
            #[cfg(feature = "dim3")]
            if let (Some(solver), Some(state)) = (mb_solver, mb_state.as_deref_mut()) {
                let mut mb_args = MultibodySolverArgs {
                    poses: &mut *args.solver_body_poses,
                    collider_world_poses: args.collider_world_poses,
                    mprops: args.mprops,
                    contacts: args.contacts,
                    contact_plan: args.contact_plan,
                    contacts_indirect: args.contacts_len_indirect,
                    mb_contact_index: &mut *args.mb_contact_index,
                    solver_vels: &mut *args.solver_vels,
                    batch_indices: args.batch_indices,
                    gravity: args.gravity,
                    color_uniforms: args.color_uniforms,
                    mb_dispatch_indirect: args.mb_dispatch_indirect,
                    friction_in_bias_pass: args.friction_in_bias_pass,
                };
                solver.layout_contact_constraints(
                    &mut pass,
                    state,
                    &mut mb_args,
                    stable_mb_contact_index,
                )?;
            }
        }

        // Per substep, the multibody work is split into five phases that are
        // interleaved with the matching rigid-body phases.
        // Each `mb_phase!($method $(, $extra)*)` invocation runs one multibody
        // phase.
        macro_rules! mb_phase {
            ($label:expr, $method:ident $(, $extra:expr)*) => {{
                #[cfg(feature = "dim3")]
                if let (Some(solver), Some(state)) = (mb_solver, mb_state.as_deref_mut()) {
                    let mut pass = encoder.begin_pass($label, timestamps.as_deref_mut());
                    let mut mb_args = MultibodySolverArgs {
                        poses: &mut *args.solver_body_poses,
                        collider_world_poses: args.collider_world_poses,
                        mprops: args.mprops,
                        contacts: args.contacts,
                        contact_plan: args.contact_plan,
                        contacts_indirect: args.contacts_len_indirect,
                        mb_contact_index: &mut *args.mb_contact_index,
                        solver_vels: &mut *args.solver_vels,
                        batch_indices: args.batch_indices,
                        gravity: args.gravity,
                        color_uniforms: args.color_uniforms,
                        mb_dispatch_indirect: args.mb_dispatch_indirect,
                        friction_in_bias_pass: args.friction_in_bias_pass,
                    };
                    solver.$method(&mut pass, state, &mut mb_args $(, $extra)*)?;
                }
            }};
        }

        // Bias-mode uniform of the biased pass (see `decode_bias_mode`).
        let bias_mode = if args.friction_in_bias_pass {
            BIAS_MODE_BIAS_FRICTION as usize
        } else {
            BIAS_MODE_BIAS as usize
        };

        for substep_id in 0..num_substeps {
            let is_last_substep = substep_id == num_substeps - 1;
            // Only consumed by the dim3-only multibody phases.
            #[cfg(not(feature = "dim3"))]
            let _ = is_last_substep;

            /*
             * Integrate velocities (apply `a · dt'` / gravity increment).
             */
            mb_phase!("[RBD] slv/mb-integrate-vels", substep_integrate_velocities);
            if !skip_rb {
                let mut pass =
                    encoder.begin_pass("[RBD] slv/rb-apply-inc", timestamps.as_deref_mut());
                self.apply_solver_vels_inc.call(
                    &mut pass,
                    args.num_colliders * args.num_batches,
                    args.solver_vels,
                    args.solver_vels_inc,
                    args.batch_indices,
                )?;
            }

            /*
             * Build + warmstart constraints.
             */
            {
                #[cfg(feature = "dim3")]
                if let (Some(solver), Some(state)) = (mb_solver, mb_state.as_deref_mut()) {
                    let mut mb_args = MultibodySolverArgs {
                        poses: &mut *args.solver_body_poses,
                        collider_world_poses: args.collider_world_poses,
                        mprops: args.mprops,
                        contacts: args.contacts,
                        contact_plan: args.contact_plan,
                        contacts_indirect: args.contacts_len_indirect,
                        mb_contact_index: &mut *args.mb_contact_index,
                        solver_vels: &mut *args.solver_vels,
                        batch_indices: args.batch_indices,
                        gravity: args.gravity,
                        color_uniforms: args.color_uniforms,
                        mb_dispatch_indirect: args.mb_dispatch_indirect,
                        friction_in_bias_pass: args.friction_in_bias_pass,
                    };
                    solver.substep_build_constraints(
                        encoder,
                        timestamps.as_deref_mut(),
                        state,
                        &mut mb_args,
                        substep_id == 0,
                    )?;
                }
            }
            if !skip_rb || !joints_empty {
                let mut pass =
                    encoder.begin_pass("[RBD] slv/rb-build-warmstart", timestamps.as_deref_mut());
                let pass = &mut pass;
                // The solve kernels compute the normal right-hand sides themselves: this pass only
                // rescales the warmstart impulses, for warmstart coefficients other than 1.
                if !skip_rb && args.scale_warmstart_impulses {
                    self.scale_constraint_impulses.call(
                        pass,
                        args.contacts_len_indirect,
                        args.color_buckets,
                        &mut args.constraints.tiles,
                        args.batch_indices,
                        args.sim_params,
                    )?;
                }
                joint_solver.update(pass, &mut joint_args, args.solver_body_poses)?;
                if skip_rb {
                    // Contact warmstart skipped: no rigid-body contact
                    // constraint can carry an impulse here.
                } else if args.colorless_warmstart {
                    self.gather_warmstart_velocities.call(
                        pass,
                        args.num_colliders * args.num_batches,
                        args.body_constraint_counts,
                        args.body_constraint_ids,
                        &args.constraints.constraint_indices,
                        &args.constraints.tiles,
                        args.solver_vels,
                        &args.hubs.first_slot,
                        args.batch_indices,
                    )?;
                    // Split bodies are warmstarted through their sub-bodies.
                    args.mass_splitting
                        .scatter(pass, args.hubs, args.solver_vels)?;
                    args.mass_splitting.warmstart(
                        pass,
                        args.hubs,
                        args.solver_vels,
                        &args.constraints.tiles,
                        &args.constraints.constraint_indices,
                    )?;
                    args.mass_splitting.average(
                        pass,
                        args.hubs,
                        args.solver_vels,
                        args.body_constraint_counts,
                    )?;
                } else if args.fused_color_dispatches {
                    args.mass_splitting
                        .scatter(pass, args.hubs, args.solver_vels)?;
                    // One dispatch, one workgroup per batch, colors looped
                    // internally. `color_uniforms[num_colors]` holds the
                    // constant `num_colors`.
                    self.warmstart_constraints_fused.call(
                        pass,
                        [64, args.num_batches, 1],
                        &args.constraints.tiles,
                        args.solver_vels,
                        args.color_buckets,
                        &args.color_uniforms[args.num_colors as usize],
                        args.batch_indices,
                    )?;
                } else {
                    args.mass_splitting
                        .scatter(pass, args.hubs, args.solver_vels)?;
                    // NOTE: contact colors start at 1 (0 = unassigned).
                    for c in 1..=args.num_colors {
                        self.warmstart_constraints.call(
                            pass,
                            args.color_dispatch_grid(c),
                            &args.constraints.tiles,
                            args.solver_vels,
                            args.color_buckets,
                            &args.color_uniforms[c as usize],
                            args.batch_indices,
                        )?;
                    }
                }
                if !skip_rb && !args.colorless_warmstart {
                    args.mass_splitting.average(
                        pass,
                        args.hubs,
                        args.solver_vels,
                        args.body_constraint_counts,
                    )?;
                }
            }

            /*
             * Solve all joints + contacts with bias. The multibody and
             * rigid-body iterations interleave, one at a time, so a body
             * squeezed between a multibody link and a rigid body sees both
             * sides converge at the same rate.
             */
            for iteration in 0..args.num_internal_pgs_iterations.max(1) {
                let first_iteration = iteration == 0;
                // Only consumed by the dim3-only multibody phase.
                #[cfg(not(feature = "dim3"))]
                let _ = first_iteration;
                mb_phase!(
                    "[RBD] slv/mb-solve-bias",
                    substep_solve_with_bias,
                    first_iteration
                );
                if !skip_rb || !joints_empty {
                    let mut pass =
                        encoder.begin_pass("[RBD] slv/rb-solve-bias", timestamps.as_deref_mut());
                    let pass = &mut pass;
                    joint_solver.solve(pass, &mut joint_args, args.solver_vels, true)?;
                    if !skip_rb {
                        args.mass_splitting
                            .scatter(pass, args.hubs, args.solver_vels)?;
                    }
                    if skip_rb {
                        // Contact iterations skipped (inert constraints).
                    } else if args.fused_color_dispatches {
                        self.solve_constraints_fused.call(
                            pass,
                            [64, args.num_batches, 1],
                            &mut args.constraints.tiles,
                            args.solver_vels,
                            args.color_buckets,
                            args.solver_body_poses,
                            &args.color_uniforms[args.num_colors as usize],
                            args.batch_indices,
                            // Biased pass: `color_uniforms[bias_mode]` holds `bias_mode`.
                            &args.color_uniforms[bias_mode],
                            args.sim_params,
                        )?;
                    } else {
                        for c in 1..=args.num_colors {
                            let tail = args.fuse_tail_colors && c >= TAIL_COLOR;
                            if tail && c > TAIL_COLOR {
                                continue;
                            }
                            // The tail kernel solves every color from `TAIL_COLOR` to its
                            // color uniform, in one workgroup.
                            let (grid, color): (DispatchGrid<GpuBackend>, u32) = if tail {
                                (64u32.into(), args.num_colors)
                            } else {
                                (args.color_dispatch_grid(c), c)
                            };
                            macro_rules! solve_color {
                                ($kernel:expr) => {
                                    $kernel.call(
                                        pass,
                                        grid,
                                        &mut args.constraints.tiles,
                                        args.solver_vels,
                                        args.color_buckets,
                                        args.solver_body_poses,
                                        &args.color_uniforms[color as usize],
                                        args.batch_indices,
                                        &args.color_uniforms[bias_mode],
                                        args.sim_params,
                                    )?
                                };
                            }
                            match (tail, args.friction_in_bias_pass) {
                                (true, true) => solve_color!(self.solve_constraints_tail),
                                (true, false) => solve_color!(self.solve_constraints_tail_biased),
                                (false, true) => solve_color!(self.solve_constraints),
                                (false, false) => solve_color!(self.solve_constraints_biased),
                            }
                        }
                    }
                    if !skip_rb {
                        args.mass_splitting.average(
                            pass,
                            args.hubs,
                            args.solver_vels,
                            args.body_constraint_counts,
                        )?;
                    }
                }
            }

            /*
             * Integrate all positions once.
             */
            mb_phase!(
                "[RBD] slv/mb-integrate-pos",
                substep_integrate_positions,
                is_last_substep
            );
            if !skip_rb {
                let mut pass =
                    encoder.begin_pass("[RBD] slv/rb-integrate", timestamps.as_deref_mut());
                self.integrate_linearized.call(
                    &mut pass,
                    args.num_colliders * args.num_batches,
                    args.solver_body_poses,
                    args.solver_vels,
                    args.sim_params,
                    args.batch_indices,
                )?;
            }

            /*
             * Solve all joints + contacts without bias (stabilization).
             */
            mb_phase!("[RBD] slv/mb-solve-nobias", substep_solve_no_bias);
            // The last iteration of a substep caches the warmstart of the next one, gathered per
            // body. Rescaled impulses recompute it anyway.
            let cache_warmstart =
                !is_last_substep && args.colorless_warmstart && !args.scale_warmstart_impulses;
            if !skip_rb || !joints_empty {
                let mut pass =
                    encoder.begin_pass("[RBD] slv/rb-solve-nobias", timestamps.as_deref_mut());
                let pass = &mut pass;
                joint_solver.solve(pass, &mut joint_args, args.solver_vels, false)?;
                if !skip_rb {
                    args.mass_splitting
                        .scatter(pass, args.hubs, args.solver_vels)?;
                }
                if skip_rb {
                    // Contact iterations skipped (inert constraints).
                } else if args.fused_color_dispatches {
                    // `color_uniforms[0]` holds 0: no bias, with friction.
                    macro_rules! solve_fused {
                        ($kernel:expr) => {
                            $kernel.call(
                                pass,
                                [64, args.num_batches, 1],
                                &mut args.constraints.tiles,
                                args.solver_vels,
                                args.color_buckets,
                                args.solver_body_poses,
                                &args.color_uniforms[args.num_colors as usize],
                                args.batch_indices,
                                &args.color_uniforms[0],
                                args.sim_params,
                            )?
                        };
                    }
                    if cache_warmstart {
                        solve_fused!(self.solve_constraints_fused_cached);
                    } else {
                        solve_fused!(self.solve_constraints_fused);
                    }
                } else {
                    for c in 1..=args.num_colors {
                        let tail = args.fuse_tail_colors && c >= TAIL_COLOR;
                        if tail && c > TAIL_COLOR {
                            continue;
                        }
                        let (grid, color): (DispatchGrid<GpuBackend>, u32) = if tail {
                            (64u32.into(), args.num_colors)
                        } else {
                            (args.color_dispatch_grid(c), c)
                        };
                        macro_rules! solve_color {
                            ($kernel:expr) => {
                                $kernel.call(
                                    pass,
                                    grid,
                                    &mut args.constraints.tiles,
                                    args.solver_vels,
                                    args.color_buckets,
                                    args.solver_body_poses,
                                    &args.color_uniforms[color as usize],
                                    args.batch_indices,
                                    &args.color_uniforms[0],
                                    args.sim_params,
                                )?
                            };
                        }
                        match (tail, cache_warmstart) {
                            (true, true) => {
                                solve_color!(self.solve_constraints_tail_unbiased_cached)
                            }
                            (true, false) => solve_color!(self.solve_constraints_tail_unbiased),
                            (false, true) => solve_color!(self.solve_constraints_unbiased_cached),
                            (false, false) => solve_color!(self.solve_constraints_unbiased),
                        }
                    }
                }
                if !skip_rb {
                    args.mass_splitting.average(
                        pass,
                        args.hubs,
                        args.solver_vels,
                        args.body_constraint_counts,
                    )?;
                }
            }
        }

        mb_phase!("[RBD] slv/mb-sense-contacts", sense_contact_impulses);
        mb_phase!("[RBD] slv/mb-restitution", apply_restitution);

        /*
         * Writeback body velocities and convert COM-centered solver poses
         * back to body-origin poses.
         */
        {
            let mut pass = encoder.begin_pass("[RBD] slv/finalize", timestamps);
            self.finalize.call(
                &mut pass,
                args.num_colliders * args.num_batches,
                args.vels,
                args.solver_vels,
                args.body_poses,
                args.solver_body_poses,
                args.local_mprops,
                args.batch_indices,
            )?;
        }

        Ok(())
    }
}
