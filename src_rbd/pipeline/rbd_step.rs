//! The [`RbdPipeline`] running one full simulation step on the GPU.

use crate::broad_phase::{GpuNarrowPhase, Lbvh};
#[cfg(feature = "dim3")]
use crate::dynamics::GpuMultibodySolver;
use crate::dynamics::{
    CanonicalContactsArgs, ColorStatsBuffer, ColoringArgs, GpuCanonicalOrder, GpuColoring,
    GpuJointSolver, GpuMassSplitting, GpuMpropsUpdate, GpuSolver, GpuWarmstart, JointSolverArgs,
    SolverArgs,
};
use crate::shaders::broad_phase::LbvhNode;
use crate::utils::GpuPrefixSum;
use khal::Shader;

use super::lbvh_validation::validate_lbvh_topology;
use super::rbd_state::*;
use bytemuck::Zeroable;
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuTimestamps};
use std::slice;
use vortx::tensor::Tensor;

/// The main GPU physics pipeline coordinating all simulation stages.
pub struct RbdPipeline {
    mprops_update: GpuMpropsUpdate,
    sync_collider_poses: crate::dynamics::GpuSyncColliderPosesShader,
    narrow_phase: GpuNarrowPhase,
    solver: GpuSolver,
    joint_solver: GpuJointSolver,
    #[cfg(feature = "dim3")]
    multibody_solver: GpuMultibodySolver,
    prefix_sum: GpuPrefixSum,
    lbvh: Lbvh,
    coloring: GpuColoring,
    warmstart: GpuWarmstart,
    mass_splitting: GpuMassSplitting,
    /// Optional (default `false`): merge each collider pair's manifolds
    /// (e.g. per-triangle trimesh contacts) into one before the solvers.
    pub contact_reduction: bool,
    /// Passes that put the per-step lists in a fixed order (deterministic mode only).
    canonical_order: GpuCanonicalOrder,
}

impl RbdPipeline {
    /// Creates a new physics pipeline from a GPU backend.
    ///
    /// This method loads all the compute shaders needed for the physics simulation.
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            mprops_update: GpuMpropsUpdate::from_backend(backend)?,
            sync_collider_poses: crate::dynamics::GpuSyncColliderPosesShader::from_backend(
                backend,
            )?,
            narrow_phase: GpuNarrowPhase::from_backend(backend)?,
            solver: GpuSolver::from_backend(backend)?,
            joint_solver: GpuJointSolver::from_backend(backend)?,
            #[cfg(feature = "dim3")]
            multibody_solver: GpuMultibodySolver::from_backend(backend)?,
            prefix_sum: GpuPrefixSum::from_backend(backend)?,
            lbvh: Lbvh::from_backend(backend),
            coloring: GpuColoring::from_backend(backend)?,
            warmstart: GpuWarmstart::from_backend(backend)?,
            mass_splitting: GpuMassSplitting::from_backend(backend)?,
            contact_reduction: false,
            canonical_order: GpuCanonicalOrder::from_backend(backend)?,
        })
    }

    /// Executes one physics simulation timestep on the GPU.
    ///
    /// Automatically resizes buffers (next power of two) if collision pair count exceeds capacity.
    pub fn step(
        &self,
        backend: &GpuBackend,
        state: &mut RbdState,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<RunStats, GpuBackendError> {
        let mut encoder = backend.begin_encoding();
        let stats = self.step_impl(backend, state, timestamps, &mut encoder, true)?;
        backend.submit(encoder)?;
        Ok(stats)
    }

    /// Records one timestep into a caller-owned `encoder` instead of managing
    /// (and submitting) its own. Nothing is submitted: the caller submits, so
    /// its own dispatches can share the command buffer with the physics step.
    ///
    /// The intra-step submits `step` uses to overlap CPU encoding with GPU work
    /// are skipped here; WebGPU guarantees dispatch-order visibility within a
    /// single encoder, so the step stays correct without them.
    pub fn step_encoded(
        &self,
        backend: &GpuBackend,
        state: &mut RbdState,
        timestamps: Option<&mut GpuTimestamps>,
        encoder: &mut <GpuBackend as Backend>::Encoder,
    ) -> Result<RunStats, GpuBackendError> {
        self.step_impl(backend, state, timestamps, encoder, false)
    }

    /// Whether the step uses the fused colored kernels (one workgroup per
    /// batch walking every color) instead of one dispatch per color.
    ///
    /// Chosen from the expected pair count: small pair counts with many
    /// environments benefit from the fused kernels. The fused path is correct
    /// at any size, just serialized past ~64 lanes.
    ///
    /// The estimate is per batch: the read-back counter (or the capacity when
    /// the readback is disabled or hasn't completed yet) is a total over the whole flat pair
    /// buffer.
    pub fn fused_color_dispatches(state: &RbdState) -> bool {
        let readback_enabled = state.capacities.solver_colors_resize_policy
            != RbdResizePolicy::Fixed
            || state.capacities.collisions_resize_policy != RbdResizePolicy::Fixed;
        let pairs = match state.collision_pairs_len_cpu {
            Some(len) if readback_enabled => len,
            _ => state.collision_pairs_capacity_cpu,
        };
        let est_pairs = pairs.div_ceil(state.num_batches);
        est_pairs <= 128
    }

    fn step_impl(
        &self,
        backend: &GpuBackend,
        state: &mut RbdState,
        mut timestamps: Option<&mut GpuTimestamps>,
        encoder: &mut <GpuBackend as Backend>::Encoder,
        allow_splits: bool,
    ) -> Result<RunStats, GpuBackendError> {
        // Submit what is recorded so far and start a fresh encoder, so CPU
        // encoding overlaps GPU work. A no-op in encoded mode, where everything
        // stays in the caller's encoder.
        let split = |enc: &mut <GpuBackend as Backend>::Encoder| -> Result<(), GpuBackendError> {
            if allow_splits {
                let done = std::mem::replace(enc, backend.begin_encoding());
                backend.submit(done)?;
            }
            Ok(())
        };
        let mut stats = RunStats::default();
        state.stepped_since_readback = true;

        // A multibody capacity edit (e.g. reserving the dry-friction
        // constraint slots on the first `set_dof_frictionloss`) leaves the
        // shared `BatchIndices` uniform describing the old buffer sizes, so
        // the kernels would index the resized buffers with stale per-batch
        // capacities. Re-upload it before anything reads it.
        #[cfg(feature = "dim3")]
        if state.multibodies.constraint_caps_dirty {
            state.rebuild_batch_indices(backend);
            state.multibodies.constraint_caps_dirty = false;
        }

        // Make sure the color index uniforms are up-to-date.
        // This is the maximum over the colors needed for contacts, joints, and multibodies.
        {
            // At least 3: the bias-mode constants 0..=2 (see `decode_bias_mode`).
            let mut needed = (state.max_colors + 2).max(3);
            needed = needed.max(state.joints.num_colors() + 1);
            #[cfg(feature = "dim3")]
            {
                needed = needed.max(state.multibodies.mb_imp_joint_num_colors() + 1);
            }
            state.ensure_color_uniforms(backend, needed);
        }

        // Phase 0: Multibody once-per-visible-step setup (3D only for now).
        #[cfg(feature = "dim3")]
        {
            if !state.multibodies.is_empty() {
                let mut args = crate::dynamics::MultibodySolverArgs {
                    poses: &mut state.body_poses,
                    collider_world_poses: &state.collider_world_poses,
                    mprops: &state.mprops,
                    contacts: &state.contacts,
                    contact_plan: &state.contact_plan,
                    contacts_indirect: &state.contacts_indirect,
                    mb_contact_index: &mut state.mb_contact_index,
                    solver_vels: &mut state.solver_vels,
                    batch_indices: &state.batch_indices,
                    color_uniforms: &state.color_uniforms,
                    mb_dispatch_indirect: &state.mb_dispatch_indirect,
                    gravity: &state.gravity,
                    friction_in_bias_pass: state.sim_params_cpu.friction_in_bias_pass != 0,
                };
                self.multibody_solver.init_step(
                    &mut *encoder,
                    timestamps.as_deref_mut(),
                    &mut state.multibodies,
                    &mut args,
                )?;
            }
        }

        // Phase 1: Update mass properties, build LBVH, and find collision pairs.
        {
            let mut pass = encoder.begin_pass("[RBD] update-mprops", timestamps.as_deref_mut());

            // Update mass properties — uses body world poses to compute the
            // world COM and inertia tensor.
            self.mprops_update.dispatch(
                &mut pass,
                &mut state.mprops,
                &state.local_mprops,
                &state.body_poses,
                &state.batch_indices,
                state.num_colliders_per_batch,
                state.num_batches,
            )?;

            // Update collider world-space poses from their parent rigid-body poses.
            self.sync_collider_poses.dispatch(
                &mut pass,
                &state.body_poses,
                &state.collider_local_poses,
                &mut state.collider_world_poses,
                &state.collider_parent,
                &state.batch_indices,
                state.num_colliders_per_batch,
                state.num_batches,
            )?;

            drop(pass);

            let use_bf = state.uses_brute_force_broad_phase();
            if use_bf {
                let mut pass = encoder.begin_pass("[RBD] bf-find-pairs", timestamps.as_deref_mut());
                self.lbvh.brute_force_pairs(
                    backend,
                    &mut pass,
                    &mut state.lbvh,
                    state.collider_local_poses.len() as u32,
                    state.num_active_colliders,
                    state.num_batches,
                    &state.collider_world_poses,
                    &state.vertex_buffers,
                    &state.shapes,
                    &state.batch_indices,
                    &mut state.collision_pairs,
                    &mut state.collision_pairs_len,
                    &mut state.collision_pairs_indirect,
                    &state.collision_groups,
                    &state.pair_filter,
                    &state.sim_params,
                )?;
                drop(pass);
                split(&mut *encoder)?;
            } else {
                // Build LBVH and find collision pairs.
                self.lbvh.update_tree(
                    backend,
                    &mut *encoder,
                    &mut state.lbvh,
                    state.collider_local_poses.len() as u32,
                    state.num_active_colliders,
                    state.num_batches,
                    &state.collider_world_poses,
                    &state.vertex_buffers,
                    &state.shapes,
                    &state.batch_indices,
                    timestamps.as_deref_mut(),
                )?;

                // Debug: validate LBVH topology after tree construction
                if crate::VALIDATE_LBVH_TOPOLOGY && allow_splits {
                    split(&mut *encoder)?;

                    let num_colliders = state.collider_world_poses.len() as u32;
                    let tree: Vec<LbvhNode> = futures::executor::block_on(
                        backend.slow_read_vec(state.lbvh.tree().buffer()),
                    )?;
                    let sorted_colliders: Vec<u32> = futures::executor::block_on(
                        backend.slow_read_vec(state.lbvh.sorted_colliders().buffer()),
                    )?;
                    validate_lbvh_topology(&tree, &sorted_colliders, num_colliders);

                    let _pass = encoder
                        .begin_pass("[RBD] broad-phase-find-pairs", timestamps.as_deref_mut());
                }

                let mut pass =
                    encoder.begin_pass("[RBD] lbvh-find-pairs", timestamps.as_deref_mut());
                self.lbvh.find_pairs(
                    &mut pass,
                    &mut state.lbvh,
                    state.num_active_colliders,
                    state.num_batches,
                    &state.batch_indices,
                    &mut state.collision_pairs,
                    &mut state.collision_pairs_len,
                    &mut state.collision_pairs_indirect,
                    &state.collision_groups,
                    &state.pair_filter,
                )?;

                drop(pass);
                split(&mut *encoder)?;
            }
        }

        let fused_color_dispatches = Self::fused_color_dispatches(state);

        // In small scenes, submit less frequently. In big scenes submit more
        // to overlap compute and encoding.
        let merge_submits = fused_color_dispatches && state.num_batches <= 64;

        // Phase 2a: Narrow phase. Split out from solver-prep + coloring
        // so its CPU encoding overlaps with Phase 1's GPU work and its
        // own GPU work overlaps with Phase 2b's CPU encoding.
        {
            let mut pass = encoder.begin_pass("[RBD] narrow-phase", timestamps.as_deref_mut());

            self.narrow_phase.dispatch(
                backend,
                &mut pass,
                &state.collider_world_poses,
                &state.shapes,
                &state.vertex_buffers,
                &state.index_buffers,
                &state.collision_pairs,
                &mut state.collision_pairs_len,
                &mut state.contacts,
                &mut state.contacts_indirect,
                &mut state.contact_plan,
                &mut state.mb_dispatch_indirect,
                &mut state.pfm_pairs,
                &mut state.pfm_pairs_len,
                &mut state.pfm_pairs_indirect,
                &mut state.pfm_sort,
                &state.batch_indices,
                &state.collider_parent,
                &state.collider_materials,
                &state.sim_params,
                self.contact_reduction,
                state.determinism.enabled && state.determinism.has_composite_shapes,
                &state.collision_pairs_indirect,
                state.collision_pairs_capacity_cpu,
            )?;

            // Deterministic mode: the contact order follows the pair order, which changes per run.
            // Sort the contacts by `(collider_a, collider_b, subshape)`.
            if state.determinism.enabled {
                self.canonical_order.canonicalize_contacts(
                    backend,
                    &mut pass,
                    CanonicalContactsArgs {
                        contacts: &state.contacts,
                        contacts_scratch: &mut state.determinism.contacts_scratch,
                        contact_plan: &state.contact_plan,
                        contacts_indirect: &state.contacts_indirect,
                        sort_n: &mut state.determinism.contact_sort_n,
                        sort_keys: &mut state.determinism.contact_sort_keys,
                        sort_ids: &mut state.determinism.contact_sort_ids,
                        sort_keys_out: &mut state.determinism.contact_sort_keys_out,
                        sort_ids_out: &mut state.determinism.contact_sort_ids_out,
                        sort_workspace: &mut state.determinism.contact_sort_workspace,
                        key_selector_uniforms: &state.determinism.contact_key_selectors,
                        batch_indices: &state.batch_indices,
                        num_colliders: state.num_active_colliders * state.num_batches,
                        has_composite_shapes: state.determinism.has_composite_shapes,
                    },
                )?;
                std::mem::swap(&mut state.contacts, &mut state.determinism.contacts_scratch);
            }

            drop(pass);
            if !merge_submits {
                split(&mut *encoder)?;
            }
        }

        // Phase 2b: solver-prep + warmstart + bounded coloring. Separate
        // submit from narrow-phase to enable CPU/GPU overlap with the
        // upcoming Phase 3 solver substep loop.
        {
            let mut pass = encoder.begin_pass("[RBD] solver-prep", timestamps.as_deref_mut());

            // Solver preparation - create args here to avoid borrow conflicts
            let prepare_args = SolverArgs {
                contacts: &state.contacts,
                contact_plan: &state.contact_plan,
                #[cfg(feature = "dim3")]
                mb_contact_index: &mut state.mb_contact_index,
                #[cfg(feature = "dim3")]
                stable_mb_contact_index: None,
                contacts_len_indirect: &state.contacts_indirect,
                constraints: &mut state.new_constraints,
                old_constraints: &state.old_constraints,
                old_body_constraint_counts: &state.old_constraints_counts,
                old_body_constraint_ids: &state.old_body_constraint_ids,
                recycle_states: &mut state.recycle_states,
                sim_params: &state.sim_params,
                body_poses: &mut state.body_poses,
                solver_body_poses: &mut state.solver_body_poses,
                collider_local_poses: &state.collider_local_poses,
                collider_world_poses: &state.collider_world_poses,
                vels: &mut state.vels,
                solver_vels: &mut state.solver_vels,
                solver_vels_inc: &mut state.solver_vels_inc,
                mprops: &state.mprops,
                local_mprops: &state.local_mprops,
                body_constraint_counts: &mut state.new_constraints_counts,
                body_constraint_ids: &mut state.new_body_constraint_ids,
                color_buckets: &state.color_buckets,
                sorted_links: &state.sorted_links,
                color_dispatch_indirect: &state.coloring_dispatch.dispatch_indirect,
                color_dispatch_threads: &state.coloring_dispatch.dispatch_threads,
                fuse_tail_colors: state.coloring_dispatch.fuse_tail_colors,
                color_uniforms: &state.color_uniforms,
                prefix_sum: &self.prefix_sum,
                num_colors: 0,
                num_batches: state.num_batches,
                num_colliders: state.num_colliders_per_batch,
                num_solver_iterations: state.num_solver_iterations,
                num_internal_pgs_iterations: state.sim_params_cpu.num_internal_pgs_iterations,
                body_group: &state.body_group,
                body_is_multibody: &state.body_is_multibody,
                batch_indices: &state.batch_indices,
                mb_dispatch_indirect: &state.mb_dispatch_indirect,
                colorless_warmstart: false,
                fused_color_dispatches,
                rb_contacts_inert: state.rb_contacts_inert,
                friction_in_bias_pass: state.sim_params_cpu.friction_in_bias_pass != 0,
                gravity: &state.gravity,
                hubs: &mut state.hubs,
                mass_splitting: &self.mass_splitting,
                scale_warmstart_impulses: state.sim_params_cpu.warmstart_coefficient != 1.0,
            };
            self.solver.prepare(
                backend,
                &mut pass,
                prepare_args,
                &mut state.prefix_sum_workspace,
            )?;

            // Deterministic mode: the body constraint lists are filled with atomics,
            // and the warmstart sums the impulses in list order.
            if state.determinism.enabled && !state.rb_contacts_inert {
                self.canonical_order.stabilize_body_constraint_ids(
                    &mut pass,
                    &state.new_constraints_counts,
                    &state.new_body_constraint_ids,
                    &mut state.determinism.stable_body_constraint_ids,
                    &state.batch_indices,
                    state.num_active_colliders * state.num_batches,
                )?;
                std::mem::swap(
                    &mut state.new_body_constraint_ids,
                    &mut state.determinism.stable_body_constraint_ids,
                );
            }

            if state.rb_contacts_inert {
                stats.num_colors = state.max_colors + 1;
                drop(pass);
            } else {
                drop(pass);
                let mut pass = encoder.begin_pass("[RBD] prep/coloring", timestamps.as_deref_mut());

                let coloring_args = ColoringArgs {
                    contacts_len_indirect: &state.contacts_indirect,
                    body_constraint_counts: &state.new_constraints_counts,
                    body_constraint_ids: &state.new_body_constraint_ids,
                    links: &state.new_constraints.links,
                    constraints_colors: &mut state.constraints_colors,
                    constraints_pending_colors: &mut state.determinism.pending_colors,
                    constraints_rands: &mut state.constraints_rands,
                    curr_color: &mut state.curr_color,
                    uncolored: &mut state.uncolored,
                    uncolored_staging: &state.uncolored_staging,
                    contact_plan: &state.contact_plan,
                    colored: &mut state.colored,
                    batch_indices: &state.batch_indices,
                    body_group: &state.body_group,
                    coloring_indirect: &mut state.coloring_dispatch.coloring_indirect,
                };
                self.coloring
                    .dispatch_topo_gc_reset(&mut pass, coloring_args)?;

                // Seed the coloring from the previous frame's colors (contacts
                // persist, so most constraints can reuse their old color and the
                // topo-gc iterations converge in 1-2 rounds instead of ~num_colors).
                let seed_args = crate::dynamics::warmstart::SeedColorsArgs {
                    contact_plan: &state.contact_plan,
                    links: &state.new_constraints.links,
                    old_constraints_colors: &state.old_constraints_colors,
                    constraints_colors: &mut state.constraints_colors,
                    colored: &mut state.colored,
                    contacts_len_indirect: &state.contacts_indirect,
                };
                self.warmstart
                    .seed_colors_from_warmstart(&mut pass, seed_args)?;

                let coloring_args = ColoringArgs {
                    contacts_len_indirect: &state.contacts_indirect,
                    body_constraint_counts: &state.new_constraints_counts,
                    body_constraint_ids: &state.new_body_constraint_ids,
                    links: &state.new_constraints.links,
                    constraints_colors: &mut state.constraints_colors,
                    constraints_pending_colors: &mut state.determinism.pending_colors,
                    constraints_rands: &mut state.constraints_rands,
                    curr_color: &mut state.curr_color,
                    uncolored: &mut state.uncolored,
                    uncolored_staging: &state.uncolored_staging,
                    contact_plan: &state.contact_plan,
                    colored: &mut state.colored,
                    batch_indices: &state.batch_indices,
                    body_group: &state.body_group,
                    coloring_indirect: &mut state.coloring_dispatch.coloring_indirect,
                };
                self.coloring.dispatch_topo_gc_iterations(
                    &mut pass,
                    coloring_args,
                    state.max_colors,
                )?;

                // Bucket-sort the constraint ids by color so each colored solver
                // iteration only touches its own constraints.
                drop(pass);
                let mut pass = encoder.begin_pass("[RBD] prep/buckets", timestamps.as_deref_mut());
                let bucket_args = crate::dynamics::ColorBucketsArgs {
                    contacts_len_indirect: &state.contacts_indirect,
                    constraints_colors: &state.constraints_colors,
                    links: &state.new_constraints.links,
                    contact_plan: &state.contact_plan,
                    color_buckets: &mut state.color_buckets,
                    sorted_links: &mut state.sorted_links,
                    constraint_indices: &mut state.new_constraints.constraint_indices,
                    batch_indices: &state.batch_indices,
                    color_stats: &mut state.coloring_dispatch.color_stats,
                    dispatch_indirect: &mut state.coloring_dispatch.dispatch_indirect,
                };
                self.coloring.dispatch_build_color_buckets(
                    backend,
                    &mut pass,
                    bucket_args,
                    &self.prefix_sum,
                    &mut state.bucket_prefix_workspace,
                )?;

                // `+1` because solver iterates 1..=max_colors (color 0 is unassigned).
                let num_colors = state.max_colors + 1;
                stats.num_colors = num_colors;

                drop(pass);
            }
            if !merge_submits {
                split(&mut *encoder)?;
            }
        }

        let num_colors = stats.num_colors;

        // Create solver_args for solve phase (after coloring is complete)
        let mut solver_args = SolverArgs {
            contacts: &state.contacts,
            contact_plan: &state.contact_plan,
            #[cfg(feature = "dim3")]
            mb_contact_index: &mut state.mb_contact_index,
            #[cfg(feature = "dim3")]
            stable_mb_contact_index: state
                .determinism
                .enabled
                .then_some(&mut state.determinism.stable_mb_contact_index),
            contacts_len_indirect: &state.contacts_indirect,
            constraints: &mut state.new_constraints,
            old_constraints: &state.old_constraints,
            old_body_constraint_counts: &state.old_constraints_counts,
            old_body_constraint_ids: &state.old_body_constraint_ids,
            recycle_states: &mut state.recycle_states,
            sim_params: &state.sim_params,
            body_poses: &mut state.body_poses,
            solver_body_poses: &mut state.solver_body_poses,
            collider_local_poses: &state.collider_local_poses,
            collider_world_poses: &state.collider_world_poses,
            vels: &mut state.vels,
            solver_vels: &mut state.solver_vels,
            solver_vels_inc: &mut state.solver_vels_inc,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            body_constraint_counts: &mut state.new_constraints_counts,
            body_constraint_ids: &mut state.new_body_constraint_ids,
            color_buckets: &state.color_buckets,
            sorted_links: &state.sorted_links,
            color_dispatch_indirect: &state.coloring_dispatch.dispatch_indirect,
            color_dispatch_threads: &state.coloring_dispatch.dispatch_threads,
            fuse_tail_colors: state.coloring_dispatch.fuse_tail_colors,
            color_uniforms: &state.color_uniforms,
            prefix_sum: &self.prefix_sum,
            num_colors,
            num_batches: state.num_batches,
            num_colliders: state.num_colliders_per_batch,
            num_solver_iterations: state.num_solver_iterations,
            num_internal_pgs_iterations: state.sim_params_cpu.num_internal_pgs_iterations,
            body_group: &state.body_group,
            body_is_multibody: &state.body_is_multibody,
            batch_indices: &state.batch_indices,
            mb_dispatch_indirect: &state.mb_dispatch_indirect,
            // The gather warmstart is only valid without multibody grouping;
            // see `SolverArgs::colorless_warmstart`.
            #[cfg(feature = "dim3")]
            colorless_warmstart: state.multibodies.is_empty(),
            #[cfg(not(feature = "dim3"))]
            colorless_warmstart: true,
            fused_color_dispatches,
            rb_contacts_inert: state.rb_contacts_inert,
            friction_in_bias_pass: state.sim_params_cpu.friction_in_bias_pass != 0,
            gravity: &state.gravity,
            hubs: &mut state.hubs,
            mass_splitting: &self.mass_splitting,
            scale_warmstart_impulses: state.sim_params_cpu.warmstart_coefficient != 1.0,
        };

        // The constraints follow the color order, known only now.
        {
            let mut pass = encoder.begin_pass("[RBD] prep/constraints", timestamps.as_deref_mut());
            self.solver.build_constraints(&mut pass, &mut solver_args)?;
        }

        // Phase 3: Solve constraints
        let joint_solver_args = JointSolverArgs {
            num_batches: state.num_batches,
            sim_params: &state.sim_params,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            joints: &mut state.joints,
            batch_indices: &state.batch_indices,
            color_uniforms: &state.color_uniforms,
        };

        {
            #[cfg(feature = "dim3")]
            let mb = if state.multibodies.is_empty() {
                None
            } else {
                Some((&self.multibody_solver, &mut state.multibodies))
            };
            self.solver.solve_tgs(
                &mut *encoder,
                timestamps.as_deref_mut(),
                &self.joint_solver,
                solver_args,
                joint_solver_args,
                #[cfg(feature = "dim3")]
                mb,
            )?;

            // Resolve all accumulated timestamps before the final submit.
            if let Some(ts) = &timestamps {
                ts.resolve(&mut *encoder);
            }
            split(&mut *encoder)?;
        }

        // Swap buffers for warm-starting next frame
        std::mem::swap(&mut state.old_constraints, &mut state.new_constraints);
        state.recycle_states.swap_frames();
        std::mem::swap(
            &mut state.old_body_constraint_ids,
            &mut state.new_body_constraint_ids,
        );
        std::mem::swap(
            &mut state.old_constraints_counts,
            &mut state.new_constraints_counts,
        );
        std::mem::swap(
            &mut state.old_constraints_colors,
            &mut state.constraints_colors,
        );

        Ok(stats)
    }

    /// Grows the collision-pair / contact / constraint buffers when the previous
    /// step overflowed (or is close to overflow) them.
    ///
    /// Note that the readback done by this function is asynchronous. Therefore, it might not
    /// apply any resizing at the current frame, and might read slightly stale data.
    pub fn auto_resize_buffers(
        &self,
        backend: &GpuBackend,
        state: &mut RbdState,
    ) -> Result<(), GpuBackendError> {
        // Disable auto-resize if all policies are fixed.
        let readback_enabled = state.capacities.solver_colors_resize_policy
            != RbdResizePolicy::Fixed
            || state.capacities.collisions_resize_policy != RbdResizePolicy::Fixed;

        let mut feedback = ResizeFeedback::zeroed();
        if state.resize_readback.try_take(
            backend,
            bytemuck::cast_slice_mut(slice::from_mut(&mut feedback)),
        ) {
            let stats = feedback.color_stats;
            for (threads, size) in state
                .coloring_dispatch
                .dispatch_threads
                .iter_mut()
                .zip(&stats.sizes)
            {
                // Always launch some work, even if the previous frame's bucket was empty.
                // A stale/small hint only changes how much the shader's for loop strides.
                *threads = size.div_ceil(64).clamp(8, 256) * 64;
            }
            // Empty/short late colors share one workgroup. An old hint never skips
            // contacts: if the graph changes, that workgroup simply does more work.
            // Color 64 has no individual readback slot, so require it to be unused.
            state.coloring_dispatch.fuse_tail_colors = stats.highest_color < 64
                && stats.sizes[crate::shaders::dynamics::contact_tiles::TAIL_COLOR as usize..]
                    .iter()
                    .map(|&n| n as u64)
                    .sum::<u64>()
                    <= 512;
            // TODO: make the coloring update optional (and pre-configurable) too?
            // The flat pair and PFM work-lists share one buffer capacity, so
            // whichever is larger drives that resize.
            let pairs_len = feedback.collision_pairs.max(feedback.pfm_pairs);
            let coloring_converged = feedback.coloring_converged;
            state.collision_pairs_len_cpu = Some(feedback.collision_pairs);
            let nb = state.num_batches;

            // TODO: Fit will act like Grow. To be able to auto-shrink the max color count, we need
            //       to readback the actual color count. This would also allow us to grow the color
            //       count earlier, before it gets a chance to fail.
            let min_colors = state
                .capacities
                .minimum_solver_colors(
                    state.num_active_colliders,
                    (coloring_converged != 0).then_some(stats.highest_color),
                )
                .max(1);
            let grow_colors = state.capacities.solver_colors_resize_policy
                != RbdResizePolicy::Fixed
                && (coloring_converged == 0 || state.max_colors < min_colors)
                && !state.rb_contacts_inert;
            // Every color costs a coloring iteration and a dispatch per iteration: once the
            // coloring converges, shrink the budget back to a few colors above those in use.
            let highest_color = stats.highest_color;
            let shrink_colors = state.capacities.solver_colors_resize_policy
                != RbdResizePolicy::Fixed
                && coloring_converged != 0
                && !state.rb_contacts_inert
                && highest_color + 8 < state.max_colors
                && state.max_colors > min_colors;

            // Decide every resize up front, then drain the GPU once before
            // applying any of them: `rebuild_batch_indices` rewrites the shared
            // uniform, and on Metal a `write_buffer` is observed by every
            // still-queued submission — an in-flight step would read the new
            // capacities while bound to the old (smaller) buffers and write out
            // of bounds. Resizes are rare, so the stall is negligible.
            let total_capacity = state.collision_pairs_capacity_cpu;
            let safe_total = pairs_len.saturating_add(pairs_len / 4);
            let new_total = pairs_len
                .saturating_add(pairs_len / 2)
                .max(state.capacities.collisions_capacity.saturating_mul(nb));
            let resize_pairs = match state.capacities.collisions_resize_policy {
                RbdResizePolicy::Fixed => false,
                RbdResizePolicy::Grow => safe_total >= total_capacity,
                RbdResizePolicy::Fit => safe_total >= total_capacity || total_capacity >= new_total,
            };

            #[cfg(feature = "dim3")]
            let (resize_mb, new_mb) = {
                let mb_demand = feedback.mb_cons_demand;
                state.mb_cons_demand_cpu = mb_demand;
                let mb_capacity = state.multibodies.contact_constraints_capacity();
                let safe_mb = mb_demand.saturating_add(mb_demand / 4);
                let new_mb = mb_demand
                    .saturating_add(mb_demand / 2)
                    .max(state.multibodies.min_contact_slab_capacity())
                    .max(
                        state
                            .capacities
                            .mb_contact_constraints_capacity
                            .saturating_mul(nb),
                    );
                let resize_mb = match state.capacities.collisions_resize_policy {
                    RbdResizePolicy::Fixed => false,
                    RbdResizePolicy::Grow => safe_mb >= mb_capacity,
                    RbdResizePolicy::Fit => safe_mb >= mb_capacity || mb_capacity >= new_mb * 2,
                };
                (resize_mb, new_mb)
            };
            #[cfg(not(feature = "dim3"))]
            let resize_mb = false;

            if grow_colors || shrink_colors || resize_pairs || resize_mb {
                backend.synchronize()?;
                state.graph_generation += 1;
            }

            if grow_colors || shrink_colors {
                if grow_colors {
                    state.max_colors = (state.max_colors + 5).max(min_colors);
                } else {
                    state.max_colors = (highest_color + 4).max(min_colors);
                }

                // The color-bucket buffer is strided by `max_colors + 3`:
                // regrow it and update the stride in `BatchIndices`.
                let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
                let stride = state.max_colors + 3;
                let nb = state.num_batches;
                state.color_buckets = Tensor::vector_uninit(backend, stride * nb, storage)?;
                state.rebuild_batch_indices(backend);
            }

            let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

            // Flat pair / PFM buffers: sized by the TOTAL demand across all
            // batches (the whole point of the flat layout: one hot batch no
            // longer multiplies every batch's capacity).
            //
            // Since the auto-resize always lags a bit behind, resizes trigger
            // with less than 25% padding available and grow with 50% slack;
            // never below the configured per-batch floor × num_batches.
            if resize_pairs {
                state.collision_pairs = Tensor::vector_uninit(backend, new_total, storage)?;
                state.pfm_pairs = Tensor::vector_uninit(backend, new_total, storage)?;
                state.pfm_sort.resize(backend, new_total);
                state.collision_pairs_capacity_cpu = new_total;

                // Contacts-keyed buffers follow the pair capacity: contact
                // slots are positional (pair `t` owns slot `t`, PFM entry `i`
                // owns slot `pairs_total + i`), so their capacity is always
                // `2 ×` the pair/PFM capacity.
                let new_contacts = new_total * 2;
                state.contacts = Tensor::vector_uninit(backend, new_contacts, storage)?;
                #[cfg(feature = "dim3")]
                {
                    state.mb_contact_index = Tensor::vector_uninit(backend, new_contacts, storage)?;
                }
                state.old_constraints = crate::dynamics::ContactConstraints::new(
                    backend,
                    new_contacts,
                    storage,
                    &state.sim_params_cpu,
                )?;
                state.old_body_constraint_ids =
                    Tensor::vector_uninit(backend, new_contacts * 2, storage)?;
                state.new_constraints = crate::dynamics::ContactConstraints::new(
                    backend,
                    new_contacts,
                    storage,
                    &state.sim_params_cpu,
                )?;
                state.recycle_states =
                    crate::dynamics::ContactRecycleStates::new(backend, new_contacts, storage)?;
                state.new_body_constraint_ids =
                    Tensor::vector_uninit(backend, new_contacts * 2, storage)?;
                state.constraints_colors = Tensor::vector_uninit(backend, new_contacts, storage)?;
                state.determinism.pending_colors =
                    Tensor::vector_uninit(backend, new_contacts, storage)?;
                // Zeroed (not uninit): 0 = "uncolored" disables color seeding
                // for the frame right after the resize.
                state.old_constraints_colors =
                    Tensor::vector(backend, vec![0u32; new_contacts as usize], storage)?;
                state.colored = Tensor::vector_uninit(backend, new_contacts, storage)?;
                state.constraints_rands = Tensor::vector_uninit(backend, new_contacts, storage)?;
                state.sorted_links = Tensor::vector_uninit(backend, new_contacts, storage)?;
                // The old counts index the old (now discarded) constraint list:
                // zero them so the next warmstart transfer sees empty ranges
                // instead of stale offsets into the fresh buffers.
                let counts_len = state.old_constraints_counts.len() as usize;
                state.old_constraints_counts =
                    Tensor::vector(backend, vec![0u32; counts_len], storage)?;

                state.contacts_capacity_cpu = new_contacts;
            }
            // Multibody contact-constraint slots: flat and demand-sized.
            #[cfg(feature = "dim3")]
            if resize_mb {
                state.multibodies.resize_contact_slabs(backend, new_mb);
            }
            if resize_pairs || resize_mb {
                state.rebuild_batch_indices(backend);
            }
        }

        if readback_enabled && state.stepped_since_readback && state.resize_readback.is_idle() {
            state.stepped_since_readback = false;
            // Gathered in the field order of `ResizeFeedback`.
            let sources = [
                (state.collision_pairs_len.buffer(), 0, 1),
                (state.pfm_pairs_len.buffer(), 0, 1),
                (state.uncolored.buffer(), 0, 1),
                #[cfg(feature = "dim3")]
                (state.multibodies.mb_cons_demand().buffer(), 0, 1),
                (
                    state.coloring_dispatch.color_stats.0.buffer(),
                    0,
                    ColorStatsBuffer::WORDS,
                ),
            ];
            debug_assert_eq!(
                sources.iter().map(|s| s.2).sum::<usize>(),
                ResizeFeedback::WORDS
            );
            state.resize_readback.request(backend, &sources)?;
        }

        Ok(())
    }
}
