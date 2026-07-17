//! The [`RbdPipeline`] running one full simulation step on the GPU.

use crate::broad_phase::{GpuNarrowPhase, Lbvh};
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
        })
    }

    /// Executes one physics simulation timestep on the GPU.
    ///
    /// Automatically resizes buffers (next power of two) if collision pair count exceeds capacity.
    pub fn step(
        &self,
        backend: &GpuBackend,
        state: &mut RbdState,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<RunStats, GpuBackendError> {
        let mut stats = RunStats::default();

        // Phase 0: Multibody once-per-visible-step setup (3D only for now).
        #[cfg(feature = "dim3")]
        {
            if !state.multibodies.is_empty() {
                let mut encoder = backend.begin_encoding();
                let mut pass =
                    encoder.begin_pass("[RBD] multibody-init-step", timestamps.as_deref_mut());
                let mut args = crate::dynamics::MultibodySolverArgs {
                    poses: &mut state.body_poses,
                    collider_world_poses: &state.collider_world_poses,
                    mprops: &state.mprops,
                    contacts: &state.contacts,
                    contacts_len: &state.contacts_len,
                    solver_vels: &mut state.solver_vels,
                    batch_indices: &state.batch_indices,
                };
                self.multibody_solver
                    .init_step(&mut pass, &mut state.multibodies, &mut args)?;
                drop(pass);
                backend.submit(encoder)?;
            }
        }

        // Phase 1: Update mass properties, build LBVH, and find collision pairs.
        {
            let mut encoder = backend.begin_encoding();
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

            // Build LBVH and find collision pairs.
            self.lbvh.update_tree(
                backend,
                &mut encoder,
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
            if crate::VALIDATE_LBVH_TOPOLOGY {
                backend.submit(encoder)?;

                let num_colliders = state.collider_world_poses.len() as u32;
                let tree: Vec<LbvhNode> =
                    futures::executor::block_on(backend.slow_read_vec(state.lbvh.tree().buffer()))?;
                let sorted_colliders: Vec<u32> = futures::executor::block_on(
                    backend.slow_read_vec(state.lbvh.sorted_colliders().buffer()),
                )?;
                validate_lbvh_topology(&tree, &sorted_colliders, num_colliders);

                encoder = backend.begin_encoding();
                let _pass =
                    encoder.begin_pass("[RBD] broad-phase-find-pairs", timestamps.as_deref_mut());
            }

            let mut pass = encoder.begin_pass("[RBD] lbvh-find-pairs", timestamps.as_deref_mut());
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
            )?;

            drop(pass);
            backend.submit(encoder)?;
        }

        // Phase 2a: Narrow phase. Split out from solver-prep + coloring
        // so its CPU encoding overlaps with Phase 1's GPU work and its
        // own GPU work overlaps with Phase 2b's CPU encoding.
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("[RBD] narrow-phase", timestamps.as_deref_mut());

            self.narrow_phase.dispatch(
                &mut pass,
                state.body_poses.len() as u32,
                &state.collider_world_poses,
                &state.shapes,
                &state.vertex_buffers,
                &state.index_buffers,
                &state.collision_pairs,
                &mut state.collision_pairs_len,
                &mut state.collision_pairs_indirect,
                &mut state.contacts,
                &mut state.contacts_len,
                &mut state.contacts_indirect,
                &mut state.pfm_pairs,
                &mut state.pfm_pairs_len,
                &mut state.pfm_pairs_indirect,
                &state.batch_indices,
                &state.collider_parent,
                &state.collider_materials,
                &mut state.pairs_flat_offsets,
                &mut state.pfm_flat_offsets,
            )?;

            drop(pass);
            backend.submit(encoder)?;
        }

        // Phase 2b: solver-prep + warmstart + bounded coloring. Separate
        // submit from narrow-phase to enable CPU/GPU overlap with the
        // upcoming Phase 3 solver substep loop.
        {
            let mut encoder = backend.begin_encoding();
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
                solver_vels_out: &state.solver_vels_out,
                solver_vels_inc: &mut state.solver_vels_inc,
                mprops: &state.mprops,
                local_mprops: &state.local_mprops,
                body_constraint_counts: &mut state.new_constraints_counts,
                body_constraint_ids: &mut state.new_body_constraint_ids,
                constraints_colors: &state.constraints_colors,
                curr_color: &mut state.curr_color,
                prefix_sum: &self.prefix_sum,
                num_colors: 0,
                num_batches: state.num_batches,
                num_colliders: state.num_colliders_per_batch,
                num_solver_iterations: state.num_solver_iterations,
                body_group: &state.body_group,
                batch_indices: &state.batch_indices,
            };
            self.solver.prepare(
                backend,
                &mut pass,
                prepare_args,
                &mut state.prefix_sum_workspace,
            )?;

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
                .dispatch_topo_gc_bounded(&mut pass, coloring_args, state.max_colors)?;

            // `+1` because solver iterates 1..=max_colors (color 0 is unassigned).
            let num_colors = state.max_colors + 1;
            stats.num_colors = num_colors;

            drop(pass);
            backend.submit(encoder)?;
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
            solver_vels_out: &state.solver_vels_out,
            solver_vels_inc: &mut state.solver_vels_inc,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            body_constraint_counts: &mut state.new_constraints_counts,
            body_constraint_ids: &mut state.new_body_constraint_ids,
            constraints_colors: &state.constraints_colors,
            curr_color: &mut state.curr_color,
            prefix_sum: &self.prefix_sum,
            num_colors,
            num_batches: state.num_batches,
            num_colliders: state.num_colliders_per_batch,
            num_solver_iterations: state.num_solver_iterations,
            body_group: &state.body_group,
            batch_indices: &state.batch_indices,
        };

        // Phase 3: Solve constraints
        let joint_solver_args = JointSolverArgs {
            num_batches: state.num_batches,
            sim_params: &state.sim_params,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            joints: &mut state.joints,
            batch_indices: &state.batch_indices,
        };

        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("[RBD] solver", timestamps.as_deref_mut());
            #[cfg(feature = "dim3")]
            let mb = if state.multibodies.is_empty() {
                None
            } else {
                Some((&self.multibody_solver, &mut state.multibodies))
            };
            self.solver.solve_tgs(
                &mut pass,
                &self.joint_solver,
                solver_args,
                joint_solver_args,
                #[cfg(feature = "dim3")]
                mb,
            )?;
            drop(pass);

            // Resolve all accumulated timestamps before the final submit.
            if let Some(ts) = &timestamps {
                ts.resolve(&mut encoder);
            }
            backend.submit(encoder)?;
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
            {
                state.max_colors += 5;
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
                state.colored = Tensor::vector_uninit(backend, new_capacity * nb, storage)?;
                state.constraints_rands =
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
