//! The [`RbdPipeline`] running one full simulation step on the GPU.

use crate::broad_phase::{BRUTE_FORCE_MAX_COLLIDERS, GpuNarrowPhase, Lbvh};
#[cfg(feature = "dim3")]
use crate::dynamics::GpuMultibodySolver;
use crate::dynamics::{
    ColoringArgs, GpuColoring, GpuJointSolver, GpuMpropsUpdate, GpuSolver, GpuWarmstart,
    JointSolverArgs, SolverArgs, warmstart::WarmstartArgs,
};
use crate::shaders::broad_phase::LbvhNode;
use crate::utils::GpuPrefixSum;
use khal::Shader;

use super::lbvh_validation::validate_lbvh_topology;
use super::rbd_state::*;
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuTimestamps};
use vortx::Reduce;
use vortx::tensor::Tensor;

/// Forces the fused colored-sweep kernels regardless of the estimated pair
/// count: the programmatic twin of `NEXUS_FUSED_SWEEPS=1`, for targets without
/// environment variables (wasm).
pub static FORCE_FUSED_SWEEPS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

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
    reduce: Reduce,
    /// Optional (default `false`): merge each collider pair's manifolds
    /// (e.g. per-triangle trimesh contacts) into one before the solvers.
    pub contact_reduction: bool,
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
            reduce: Reduce::from_backend(backend)?,
            contact_reduction: false,
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
            let mut needed = state.max_colors + 2;
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
                    contacts_len: &state.contacts_len,
                    solver_vels: &mut state.solver_vels,
                    batch_indices: &state.batch_indices,
                    color_uniforms: &state.color_uniforms,
                    mb_sweep_indirect: &state.mb_sweep_indirect,
                    gravity: &state.gravity,
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

            let use_bf = state.num_active_colliders <= BRUTE_FORCE_MAX_COLLIDERS
                && std::env::var("NEXUS_DISABLE_BF").is_err();
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

        let readback_enabled = state.capacities.solver_colors_resize_policy
            != RbdResizePolicy::Fixed
            || state.capacities.collisions_resize_policy != RbdResizePolicy::Fixed;
        let est_pairs = if readback_enabled {
            state.collision_pairs_len_cpu
        } else {
            state.collision_pairs_per_batch_cpu
        };

        // Choose the kernel depending on the expected pairs count: small pair
        // counts with many environments benefit from the fused kernels. The
        // fused path can also be forced regardless of size (an A/B knob: an env
        // var natively, [`FORCE_FUSED_SWEEPS`] on wasm where env vars do not
        // exist). It is correct at any size, just serialized past ~64 lanes,
        // which may still win where per-dispatch latency rules, i.e. small
        // batch counts in the browser.
        let fused_color_sweeps = est_pairs <= 128
            || FORCE_FUSED_SWEEPS.load(core::sync::atomic::Ordering::Relaxed)
            || std::env::var("NEXUS_FUSED_SWEEPS").as_deref() == Ok("1");

        // In small scenes, submit less frequently. In big scenes submit more
        // to overlap compute and encoding.
        let merge_submits = fused_color_sweeps && state.num_batches <= 64;

        // Phase 2a: Narrow phase. Split out from solver-prep + coloring
        // so its CPU encoding overlaps with Phase 1's GPU work and its
        // own GPU work overlaps with Phase 2b's CPU encoding.
        {
            let mut pass = encoder.begin_pass("[RBD] narrow-phase", timestamps.as_deref_mut());

            self.narrow_phase.dispatch(
                &mut pass,
                state.body_poses.len() as u32,
                &state.collider_world_poses,
                &state.shapes,
                &state.vertex_buffers,
                &state.index_buffers,
                &state.collision_pairs,
                &state.collision_pairs_len,
                &state.collision_pairs_indirect,
                &mut state.contacts,
                &mut state.contacts_len,
                &mut state.contacts_indirect,
                &mut state.mb_sweep_indirect,
                &mut state.pfm_pairs,
                &mut state.pfm_pairs_len,
                &mut state.pfm_pairs_indirect,
                &state.batch_indices,
                &state.collider_parent,
                &state.collider_materials,
                &state.sim_params,
                self.contact_reduction,
            )?;

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
                contacts_len: &state.contacts_len,
                contacts_len_indirect: &state.contacts_indirect,
                constraints: &mut state.new_constraints,
                constraint_builders: &mut state.new_constraint_builders,
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
                color_bucket_starts: &state.color_bucket_starts,
                color_sorted_ids: &state.color_sorted_ids,
                color_uniforms: &state.color_uniforms,
                prefix_sum: &self.prefix_sum,
                num_colors: 0,
                num_batches: state.num_batches,
                num_colliders: state.num_colliders_per_batch,
                num_solver_iterations: state.num_solver_iterations,
                body_group: &state.body_group,
                batch_indices: &state.batch_indices,
                mb_sweep_indirect: &state.mb_sweep_indirect,
                colorless_warmstart: false,
                fused_color_sweeps,
                rb_contacts_inert: state.rb_contacts_inert,
                gravity: &state.gravity,
            };
            self.solver.prepare(
                backend,
                &mut pass,
                prepare_args,
                &mut state.prefix_sum_workspace,
            )?;

            if state.rb_contacts_inert {
                stats.num_colors = state.max_colors + 1;
                drop(pass);
            } else {
                // Warmstart
                let warmstart_args = WarmstartArgs {
                    contacts_len: &state.contacts_len,
                    old_body_constraint_counts: &state.old_constraints_counts,
                    old_constraint_builders: &state.old_constraint_builders,
                    old_body_constraint_ids: &state.old_body_constraint_ids,
                    old_constraints: &state.old_constraints,
                    new_constraints: &mut state.new_constraints,
                    new_constraint_builders: &state.new_constraint_builders,
                    contacts_len_indirect: &state.contacts_indirect,
                    batch_indices: &state.batch_indices,
                };

                self.warmstart
                    .transfer_warmstart_impulses(&mut pass, warmstart_args)?;

                let coloring_args = ColoringArgs {
                    contacts_len_indirect: &state.contacts_indirect,
                    body_constraint_counts: &state.new_constraints_counts,
                    body_constraint_ids: &state.new_body_constraint_ids,
                    constraints: &state.new_constraints,
                    constraints_colors: &mut state.constraints_colors,
                    constraints_rands: &mut state.constraints_rands,
                    curr_color: &mut state.curr_color,
                    uncolored: &mut state.uncolored,
                    uncolored_staging: &state.uncolored_staging,
                    contacts_len: &state.contacts_len,
                    colored: &mut state.colored,
                    batch_indices: &state.batch_indices,
                    body_group: &state.body_group,
                };
                self.coloring
                    .dispatch_topo_gc_reset(&mut pass, coloring_args)?;

                // Seed the coloring from the previous frame's colors (contacts
                // persist, so most constraints can reuse their old color and the
                // topo-gc iterations converge in 1-2 rounds instead of ~num_colors).
                let seed_args = crate::dynamics::warmstart::SeedColorsArgs {
                    contacts_len: &state.contacts_len,
                    old_body_constraint_counts: &state.old_constraints_counts,
                    old_body_constraint_ids: &state.old_body_constraint_ids,
                    old_constraints: &state.old_constraints,
                    new_constraints: &state.new_constraints,
                    old_constraints_colors: &state.old_constraints_colors,
                    constraints_colors: &mut state.constraints_colors,
                    colored: &mut state.colored,
                    contacts_len_indirect: &state.contacts_indirect,
                    batch_indices: &state.batch_indices,
                };
                self.warmstart
                    .seed_colors_from_warmstart(&mut pass, seed_args)?;

                let coloring_args = ColoringArgs {
                    contacts_len_indirect: &state.contacts_indirect,
                    body_constraint_counts: &state.new_constraints_counts,
                    body_constraint_ids: &state.new_body_constraint_ids,
                    constraints: &state.new_constraints,
                    constraints_colors: &mut state.constraints_colors,
                    constraints_rands: &mut state.constraints_rands,
                    curr_color: &mut state.curr_color,
                    uncolored: &mut state.uncolored,
                    uncolored_staging: &state.uncolored_staging,
                    contacts_len: &state.contacts_len,
                    colored: &mut state.colored,
                    batch_indices: &state.batch_indices,
                    body_group: &state.body_group,
                };
                self.coloring.dispatch_topo_gc_iterations(
                    &mut pass,
                    coloring_args,
                    state.max_colors,
                )?;

                // Bucket-sort the constraint ids by color so each colored solver
                // sweep only touches its own constraints.
                let bucket_args = crate::dynamics::ColorBucketsArgs {
                    contacts_len_indirect: &state.contacts_indirect,
                    constraints_colors: &state.constraints_colors,
                    contacts_len: &state.contacts_len,
                    color_bucket_counts: &mut state.color_bucket_counts,
                    color_bucket_starts: &mut state.color_bucket_starts,
                    color_bucket_cursors: &mut state.color_bucket_cursors,
                    color_sorted_ids: &mut state.color_sorted_ids,
                    batch_indices: &state.batch_indices,
                };
                self.coloring.dispatch_build_color_buckets(
                    &mut pass,
                    bucket_args,
                    state.max_colors + 3,
                    state.num_batches,
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
        let solver_args = SolverArgs {
            contacts: &state.contacts,
            contacts_len: &state.contacts_len,
            contacts_len_indirect: &state.contacts_indirect,
            constraints: &mut state.new_constraints,
            constraint_builders: &mut state.new_constraint_builders,
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
            color_bucket_starts: &state.color_bucket_starts,
            color_sorted_ids: &state.color_sorted_ids,
            color_uniforms: &state.color_uniforms,
            prefix_sum: &self.prefix_sum,
            num_colors,
            num_batches: state.num_batches,
            num_colliders: state.num_colliders_per_batch,
            num_solver_iterations: state.num_solver_iterations,
            body_group: &state.body_group,
            batch_indices: &state.batch_indices,
            mb_sweep_indirect: &state.mb_sweep_indirect,
            // The gather warmstart is only valid without multibody grouping;
            // see `SolverArgs::colorless_warmstart`.
            #[cfg(feature = "dim3")]
            colorless_warmstart: state.multibodies.is_empty(),
            #[cfg(not(feature = "dim3"))]
            colorless_warmstart: true,
            fused_color_sweeps,
            rb_contacts_inert: state.rb_contacts_inert,
            gravity: &state.gravity,
        };

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
        std::mem::swap(
            &mut state.old_constraint_builders,
            &mut state.new_constraint_builders,
        );
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

        // The readback holds `[max collision-pair count across batches, uncolored
        // count]`. The max is computed on the GPU.
        let mut counts = [0u32; 2];
        if state.resize_readback.try_take(backend, &mut counts) {
            // TODO: make the coloring update optional (and pre-configurable) too?
            let collision_pairs_len = counts[0];
            let coloring_converged = counts[1];
            state.collision_pairs_len_cpu = collision_pairs_len;

            // TODO: Fit will act like Grow. To be able to auto-shrink the max color count, we need
            //       to readback the actual color count. This would also allow us to grow the color
            //       count earlier, before it gets a chance to fail.
            if state.capacities.solver_colors_resize_policy != RbdResizePolicy::Fixed
                && coloring_converged == 0
                && !state.rb_contacts_inert
            {
                state.max_colors += 5;

                // The color-bucket buffers are strided by `max_colors + 3`:
                // regrow them and update the stride in `BatchIndices`.
                let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
                let stride = state.max_colors + 3;
                let nb = state.num_batches;
                state.color_bucket_counts = Tensor::vector_uninit(backend, stride * nb, storage)?;
                state.color_bucket_starts = Tensor::vector_uninit(backend, stride * nb, storage)?;
                state.color_bucket_cursors = Tensor::vector_uninit(backend, stride * nb, storage)?;
                state.rebuild_batch_indices(backend);
            }

            // Lazy resize based on the *previous* frame's max pair count.
            let per_batch_capacity =
                (state.collision_pairs.len() as u32).div_ceil(state.num_batches);

            // Since the auto-resize always lags a bit behind, consider resizing if we have less than 25%
            // padding available, reducing the risks of missing contacts.
            let safe_capacity = collision_pairs_len.saturating_add(collision_pairs_len / 4);
            // Add a 50% extra so we don’t need to reallocate immediately if the
            // collision count grows further. Can never be smaller than the `RbdCapacities::collisions_capacity`.
            let new_capacity = collision_pairs_len
                .saturating_add(collision_pairs_len / 2)
                .max(state.capacities.collisions_capacity);

            let resize = match state.capacities.collisions_resize_policy {
                RbdResizePolicy::Fixed => false,
                RbdResizePolicy::Grow => safe_capacity >= per_batch_capacity,
                RbdResizePolicy::Fit => {
                    safe_capacity >= per_batch_capacity || per_batch_capacity >= new_capacity
                }
            };

            if resize {
                let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
                let nb = state.num_batches;

                state.collision_pairs = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.contacts = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.pfm_pairs = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.old_constraints = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.old_constraint_builders =
                    Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.old_body_constraint_ids =
                    Tensor::vector_uninit(backend, new_capacity * 2 * nb, storage)?;
                state.new_constraints = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.new_constraint_builders =
                    Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.new_body_constraint_ids =
                    Tensor::vector_uninit(backend, new_capacity * 2 * nb, storage)?;
                state.constraints_colors =
                    Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                // Zeroed (not uninit): 0 = "uncolored" disables color seeding
                // for the frame right after the resize.
                state.old_constraints_colors =
                    Tensor::vector(backend, vec![0u32; (new_capacity * nb) as usize], storage)?;
                state.colored = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.constraints_rands =
                    Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.color_sorted_ids =
                    Tensor::vector_uninit(backend, new_capacity * nb, storage)?;

                state.collision_pairs_per_batch_cpu = new_capacity;
                state.contacts_per_batch_cpu = new_capacity;
                state.rebuild_batch_indices(backend);
            }
        }

        if readback_enabled && state.resize_readback.is_idle() {
            let pairs_source = if state.num_batches > 1 {
                let mut encoder = backend.begin_encoding();
                let mut pass = encoder.begin_pass("[RBD] calc-max-coll-len", None);
                self.reduce.reduce_max_u32.call(
                    &mut pass,
                    1,
                    &state.num_batches_uniform,
                    &state.collision_pairs_len,
                    &mut state.collision_pairs_len_max,
                )?;
                drop(pass);
                backend.submit(encoder)?;
                state.collision_pairs_len_max.buffer()
            } else {
                state.collision_pairs_len.buffer()
            };

            state.resize_readback.request(
                backend,
                &[(pairs_source, 0, 1), (state.uncolored.buffer(), 0, 1)],
            )?;
        }

        Ok(())
    }
}
