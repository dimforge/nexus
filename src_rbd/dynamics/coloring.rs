//! Graph coloring algorithms for parallel constraint solving on the GPU.
//!
//! Two algorithms are available:
//! - **TOPO-GC** (Topological Graph Coloring): the primary algorithm, used by default.
//!   May fail to converge for highly complex constraint graphs; falls back to Luby when
//!   it doesn't converge within the iteration limit.
//! - **Luby's Algorithm**: a randomized fallback that always converges (probabilistically)
//!   and handles arbitrary constraint graphs.

use crate::pipeline::RunStats;
use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::{
    ContactLink, GpuClearCompletionFlagTopoGc, GpuColorBucketsCount, GpuColorBucketsReset,
    GpuColorBucketsScatter, GpuColorSweepGrid, GpuFixConflictsTopoGc, GpuResetCompletionFlagTopoGc,
    GpuResetLuby, GpuResetTopoGc, GpuStepGraphColoringLuby, GpuStepGraphColoringTopoGc,
};
use crate::utils::{GpuPrefixSum, PrefixSumWorkspace};
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shaders for constraint graph coloring.
///
/// Contains compute pipelines for both TOPO-GC and Luby's algorithm.
#[derive(Shader)]
pub struct GpuColoring {
    /// Initializes state for Luby's algorithm.
    reset_luby_kernel: GpuResetLuby,
    /// One iteration of Luby's coloring.
    step_graph_coloring_luby_kernel: GpuStepGraphColoringLuby,
    /// Initializes state for TOPO-GC algorithm.
    reset_topo_gc_kernel: GpuResetTopoGc,
    /// One iteration of TOPO-GC coloring.
    step_graph_coloring_topo_gc_kernel: GpuStepGraphColoringTopoGc,
    /// Detects and fixes conflicts in TOPO-GC coloring.
    fix_conflicts_topo_gc_kernel: GpuFixConflictsTopoGc,
    reset_completion_flag_topo_gc: GpuResetCompletionFlagTopoGc,
    clear_completion_flag_topo_gc: GpuClearCompletionFlagTopoGc,
    /// Sizes the colored sweeps from the color buckets.
    color_sweep_grid: GpuColorSweepGrid,
    // Workspace for bucket-sorting constraint ids by color so each color iteration
    // only touches their own constraint.
    color_buckets_reset: GpuColorBucketsReset,
    color_buckets_count: GpuColorBucketsCount,
    color_buckets_scatter: GpuColorBucketsScatter,
}

/// Buffers for the per-color constraint bucket sort.
pub struct ColorBucketsArgs<'a> {
    /// Flat dispatch grid over the whole contacts range.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Color assigned to each constraint by graph coloring.
    pub constraints_colors: &'a Tensor<u32>,
    /// The links of the colored constraints (batch recovered from their global body ids).
    pub links: &'a Tensor<ContactLink>,
    /// Clamped per-frame list totals (the flat contact sweep bound).
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// The single `(color, batch)` bucket buffer, color-major
    /// (`solver_color_buckets_stride * num_batches` entries): counts, then
    /// scanned exclusive starts, then post-scatter exclusive ends.
    pub color_buckets: &'a mut Tensor<u32>,
    /// Output: the constraint links bucket-sorted by `(color, batch)`, inactive past the last
    /// constraint.
    pub sorted_links: &'a mut Tensor<ContactLink>,
    /// Output: the index of each constraint in the color order.
    pub constraint_indices: &'a mut Tensor<u32>,
    /// Shared per-batch capacity / section-offset uniform.
    pub batch_indices: &'a Tensor<crate::shaders::utils::BatchIndices>,
    /// Output: `[largest bucket, highest color, size of colors 0..64]`.
    pub color_stats: &'a mut Tensor<u32>,
    /// Output: the dispatch grid of the colored sweeps.
    pub sweep_indirect: &'a mut Tensor<[u32; 3]>,
}

/// Arguments for graph coloring dispatch.
///
/// Contains all GPU buffers needed by the coloring algorithms.
pub struct ColoringArgs<'a> {
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Number of constraints per body.
    pub body_constraint_counts: &'a Tensor<u32>,
    /// Constraint IDs associated with each body.
    pub body_constraint_ids: &'a Tensor<u32>,
    /// The links of the constraints to be colored.
    pub links: &'a Tensor<ContactLink>,
    /// Output: color assigned to each constraint.
    pub constraints_colors: &'a mut Tensor<u32>,
    /// Color picked by each constraint in the current round, before the conflict pass.
    /// Only read in deterministic mode, but always bound.
    pub constraints_pending_colors: &'a mut Tensor<u32>,
    /// Random values for Luby's algorithm.
    pub constraints_rands: &'a mut Tensor<u32>,
    /// Current color being assigned.
    pub curr_color: &'a mut Tensor<u32>,
    /// Count of uncolored constraints (or changed flag for TOPO-GC).
    pub uncolored: &'a mut Tensor<u32>,
    /// Staging buffer for reading uncolored count on CPU.
    pub uncolored_staging: &'a Tensor<u32>,
    /// Clamped per-frame list totals (the flat contact sweep bound).
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// Buffer tracking which constraints are colored.
    pub colored: &'a mut Tensor<u32>,
    /// Shared per-batch capacity / section-offset uniform.
    pub batch_indices: &'a Tensor<crate::shaders::utils::BatchIndices>,
    /// Per-body graph-coloring group id (multibody-aware): contacts touching
    /// different bodies of the same multibody share a group and never share
    /// a color. For free bodies, `body_group[i] = i`.
    pub body_group: &'a Tensor<u32>,
    /// Dispatch grid of the current topo-gc iteration (empty once converged).
    pub coloring_indirect: &'a mut Tensor<[u32; 3]>,
}

impl GpuColoring {
    /// Dispatches the reset_luby kernel.
    fn dispatch_reset_luby(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.reset_luby_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.constraints_rands,
            args.links,
            args.contact_plan,
        )?;
        Ok(())
    }

    /// Dispatches the step_graph_coloring_luby kernel.
    fn dispatch_step_luby(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.step_graph_coloring_luby_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.links,
            args.constraints_colors,
            args.constraints_rands,
            args.uncolored,
            args.body_group,
            args.curr_color,
            args.contact_plan,
        )?;
        Ok(())
    }

    /// Dispatches the reset_topo_gc kernel.
    fn dispatch_reset_topo_gc(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.reset_topo_gc_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.colored,
            args.links,
            args.contact_plan,
            args.uncolored,
            args.constraints_pending_colors,
        )?;
        Ok(())
    }

    /// Dispatches the step_graph_coloring_topo_gc kernel.
    fn dispatch_step_topo_gc(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.step_graph_coloring_topo_gc_kernel.call(
            pass,
            &*args.coloring_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.links,
            args.constraints_colors,
            args.colored,
            args.uncolored,
            args.contact_plan,
            args.body_group,
            args.batch_indices,
            args.constraints_pending_colors,
        )?;
        Ok(())
    }

    /// Dispatches the fix_conflicts_topo_gc kernel.
    fn dispatch_fix_conflicts_topo_gc(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.fix_conflicts_topo_gc_kernel.call(
            pass,
            &*args.coloring_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.links,
            args.constraints_colors,
            args.colored,
            args.uncolored,
            args.contact_plan,
            args.body_group,
            args.batch_indices,
            args.constraints_pending_colors,
        )?;
        Ok(())
    }

    /// Executes Luby's randomized graph coloring algorithm.
    ///
    /// Returns the total number of colors used (1-indexed).
    pub async fn dispatch_luby<'a>(
        &self,
        backend: &GpuBackend,
        mut args: ColoringArgs<'a>,
        stats: &mut RunStats,
    ) -> u32 {
        // Initialize coloring state
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("luby-coloring-reset", None);
            self.dispatch_reset_luby(&mut pass, &mut args).unwrap();
            drop(pass);
            backend.submit(encoder).unwrap();
        }

        let mut num_colors = 0;
        for color in 1u32.. {
            backend
                .write_buffer(args.curr_color.buffer_mut(), 0, &[color])
                .unwrap();
            backend
                .write_buffer(args.uncolored.buffer_mut(), 0, &[0u32])
                .unwrap();

            {
                let mut encoder = backend.begin_encoding();
                let mut pass = encoder.begin_pass("luby-coloring-step", None);
                self.dispatch_step_luby(&mut pass, &mut args).unwrap();
                drop(pass);
                backend.submit(encoder).unwrap();
            }

            let uncolored = backend
                .slow_read_vec(args.uncolored.buffer())
                .await
                .unwrap()[0];

            if uncolored == 0 {
                num_colors = color + 1;
                break;
            }
        }

        stats.num_colors = num_colors;
        num_colors
    }

    /// Bucket-sorts the constraint ids by `(color, batch)`: zero the buckets,
    /// count, exclusive-prefix-scan them in place, then scatter (which turns
    /// the starts into exclusive ends, the form the sweeps read).
    pub fn dispatch_build_color_buckets(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        args: ColorBucketsArgs<'_>,
        prefix_sum: &GpuPrefixSum,
        prefix_workspace: &mut PrefixSumWorkspace,
    ) -> Result<(), GpuBackendError> {
        let num_buckets = args.color_buckets.len() as u32;
        self.color_buckets_reset
            .call(pass, [num_buckets, 1, 1], args.color_buckets)?;
        self.color_buckets_count.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.links,
            args.contact_plan,
            args.color_buckets,
            args.batch_indices,
            args.sorted_links,
        )?;
        prefix_sum.launch(backend, pass, prefix_workspace, args.color_buckets, 1)?;
        self.color_buckets_scatter.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.links,
            args.contact_plan,
            args.color_buckets,
            args.sorted_links,
            args.batch_indices,
            args.constraint_indices,
        )?;
        // A single workgroup reduces over the colors.
        self.color_sweep_grid.call(
            pass,
            64u32,
            &*args.color_buckets,
            args.color_stats,
            args.sweep_indirect,
            args.batch_indices,
        )?;
        Ok(())
    }

    /// Runs a fixed number of iterations of the topo-gc coloring.
    pub fn dispatch_topo_gc_bounded<'a>(
        &self,
        pass: &mut GpuPass,
        mut args: ColoringArgs<'a>,
        max_colors: u32,
    ) -> Result<(), GpuBackendError> {
        // Reset coloring state.
        self.dispatch_reset_topo_gc(pass, &mut args)?;
        self.dispatch_topo_gc_iterations(pass, args, max_colors)
    }

    /// Resets the topo-gc coloring state (all constraints uncolored). Public
    /// so a seeding pass (e.g. warmstart color transfer) can run between the
    /// reset and [`Self::dispatch_topo_gc_iterations`].
    pub fn dispatch_topo_gc_reset<'a>(
        &self,
        pass: &mut GpuPass,
        mut args: ColoringArgs<'a>,
    ) -> Result<(), GpuBackendError> {
        self.dispatch_reset_topo_gc(pass, &mut args)
    }

    /// Runs the bounded topo-gc step/fix-conflicts iterations, assuming the
    /// coloring state was already reset (and possibly seeded).
    pub fn dispatch_topo_gc_iterations<'a>(
        &self,
        pass: &mut GpuPass,
        mut args: ColoringArgs<'a>,
        max_colors: u32,
    ) -> Result<(), GpuBackendError> {
        // The first fix-conflicts pass must validate the seeded colors even when the first step
        // colors nothing, so it starts from a cleared ("not converged") flag and a full grid. At
        // least two rounds run so the last pass can still record the color count.
        self.clear_completion_flag_topo_gc.call(
            pass,
            1u32,
            &mut *args.uncolored,
            args.contacts_len_indirect,
            &mut *args.coloring_indirect,
        )?;
        for i in 0..max_colors.max(2) {
            if i > 0 {
                self.reset_completion_flag_topo_gc.call(
                    pass,
                    1u32,
                    &mut *args.uncolored,
                    args.contacts_len_indirect,
                    &mut *args.coloring_indirect,
                )?;
            }
            self.dispatch_step_topo_gc(pass, &mut args)?;
            self.dispatch_fix_conflicts_topo_gc(pass, &mut args)?;
        }

        Ok(())
    }
}
