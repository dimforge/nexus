//! Graph coloring algorithms for parallel constraint solving on the GPU.
//!
//! Two algorithms are available:
//! - **TOPO-GC** (Topological Graph Coloring): the primary algorithm, used by default.
//!   May fail to converge for highly complex constraint graphs; falls back to Luby when
//!   it doesn't converge within the iteration limit.
//! - **Luby's Algorithm**: a randomized fallback that always converges (probabilistically)
//!   and handles arbitrary constraint graphs.

use crate::pipeline::RunStats;
use crate::shaders::dynamics::TwoBodyConstraint;
use crate::shaders::dynamics::{
    GpuColorBucketsCount, GpuColorBucketsReset, GpuColorBucketsScan, GpuColorBucketsScatter,
    GpuFixConflictsTopoGc, GpuResetCompletionFlagTopoGc, GpuResetLuby, GpuResetTopoGc,
    GpuStepGraphColoringLuby, GpuStepGraphColoringTopoGc,
};
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuPass, GpuTimestamps};
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
    /// Bucket-sort of constraint ids by color (reset / count / scan / scatter),
    /// run once per step after coloring so each colored solver sweep touches
    /// only its own constraints.
    color_buckets_reset: GpuColorBucketsReset,
    color_buckets_count: GpuColorBucketsCount,
    color_buckets_scan: GpuColorBucketsScan,
    color_buckets_scatter: GpuColorBucketsScatter,
}

/// Buffers for the per-color constraint bucket sort — see the
/// `gpu_color_buckets_*` kernels.
pub struct ColorBucketsArgs<'a> {
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Color assigned to each constraint by graph coloring.
    pub constraints_colors: &'a Tensor<u32>,
    /// Number of contacts per batch.
    pub contacts_len: &'a Tensor<u32>,
    /// Per-batch per-color counts (stride `solver_color_buckets_stride`).
    pub color_bucket_counts: &'a mut Tensor<u32>,
    /// Per-batch per-color exclusive prefix sums.
    pub color_bucket_starts: &'a mut Tensor<u32>,
    /// Scatter cursors (seeded from the starts).
    pub color_bucket_cursors: &'a mut Tensor<u32>,
    /// Constraint ids bucket-sorted by color (contacts layout).
    pub color_sorted_ids: &'a mut Tensor<u32>,
    /// Shared per-batch capacity / section-offset uniform.
    pub batch_indices: &'a Tensor<crate::shaders::utils::BatchIndices>,
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
    /// The constraints to be colored.
    pub constraints: &'a Tensor<TwoBodyConstraint>,
    /// Output: color assigned to each constraint.
    pub constraints_colors: &'a mut Tensor<u32>,
    /// Random values for Luby's algorithm.
    pub constraints_rands: &'a mut Tensor<u32>,
    /// Current color being assigned.
    pub curr_color: &'a mut Tensor<u32>,
    /// Count of uncolored constraints (or changed flag for TOPO-GC).
    pub uncolored: &'a mut Tensor<u32>,
    /// Staging buffer for reading uncolored count on CPU.
    pub uncolored_staging: &'a Tensor<u32>,
    /// Total number of contacts.
    pub contacts_len: &'a Tensor<u32>,
    /// Buffer tracking which constraints are colored.
    pub colored: &'a mut Tensor<u32>,
    /// Shared per-batch capacity / section-offset uniform.
    pub batch_indices: &'a Tensor<crate::shaders::utils::BatchIndices>,
    /// Per-body graph-coloring group id (multibody-aware): contacts touching
    /// different bodies of the same multibody share a group and never share
    /// a color. For free bodies, `body_group[i] = i`.
    pub body_group: &'a Tensor<u32>,
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
            args.contacts_len,
            args.batch_indices,
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
            args.constraints,
            args.constraints_colors,
            args.constraints_rands,
            args.uncolored,
            args.body_group,
            args.curr_color,
            args.contacts_len,
            args.batch_indices,
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
            args.contacts_len,
            args.batch_indices,
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
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.constraints,
            args.constraints_colors,
            args.colored,
            args.uncolored,
            args.contacts_len,
            args.body_group,
            args.batch_indices,
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
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.constraints,
            args.constraints_colors,
            args.colored,
            args.uncolored,
            args.contacts_len,
            args.body_group,
            args.batch_indices,
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

    /// Bucket-sorts the constraint ids by color. Must run after the coloring
    /// converged; the colored solver sweeps consume `color_bucket_starts` +
    /// `color_sorted_ids`.
    pub fn dispatch_build_color_buckets(
        &self,
        pass: &mut GpuPass,
        args: ColorBucketsArgs<'_>,
        color_buckets_stride: u32,
        num_batches: u32,
    ) -> Result<(), GpuBackendError> {
        self.color_buckets_reset.call(
            pass,
            [color_buckets_stride, num_batches, 1],
            args.color_bucket_counts,
            args.batch_indices,
        )?;
        self.color_buckets_count.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.contacts_len,
            args.color_bucket_counts,
            args.batch_indices,
        )?;
        self.color_buckets_scan.call(
            pass,
            [1, num_batches, 1],
            args.color_bucket_counts,
            args.color_bucket_starts,
            args.color_bucket_cursors,
            args.batch_indices,
        )?;
        self.color_buckets_scatter.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.contacts_len,
            args.color_bucket_cursors,
            args.color_sorted_ids,
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
        for _ in 0..max_colors {
            self.reset_completion_flag_topo_gc
                .call(pass, 1u32, args.uncolored)?;
            self.dispatch_step_topo_gc(pass, &mut args)?;
            self.dispatch_fix_conflicts_topo_gc(pass, &mut args)?;
        }

        Ok(())
    }

    /// Executes the TOPO-GC (Topological Graph Coloring) algorithm.
    ///
    /// Returns `Some(num_colors)` (1-indexed) on success, or `None` if convergence
    /// fails (caller should fall back to [`dispatch_luby`](Self::dispatch_luby)).
    pub async fn dispatch_topo_gc<'a>(
        &self,
        backend: &GpuBackend,
        mut args: ColoringArgs<'a>,
        stats: &mut RunStats,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Option<u32> {
        // Initialize TOPO-GC state
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("topo-gc-coloring-reset", timestamps.as_deref_mut());
            self.dispatch_reset_topo_gc(&mut pass, &mut args).unwrap();
            drop(pass);
            backend.submit(encoder).unwrap();
        }

        let mut num_loops = 0;
        loop {
            num_loops += 1;
            if num_loops > 64 {
                return None;
            }

            // Batch multiple iterations to reduce CPU-GPU sync overhead
            {
                let mut encoder = backend.begin_encoding();
                let mut pass =
                    encoder.begin_pass("topo-gc-coloring-step", timestamps.as_deref_mut());
                for _ in 0..10 {
                    self.reset_completion_flag_topo_gc
                        .call(&mut pass, 1u32, args.uncolored)
                        .unwrap();
                    self.dispatch_step_topo_gc(&mut pass, &mut args).unwrap();
                    self.dispatch_fix_conflicts_topo_gc(&mut pass, &mut args)
                        .unwrap();
                }
                drop(pass);
                backend.submit(encoder).unwrap();
            }

            let max_color = backend
                .slow_read_vec(args.uncolored.buffer())
                .await
                .unwrap()[0];

            if max_color != 0 {
                stats.num_colors = max_color;
                stats.coloring_iterations = num_loops;
                return Some(max_color + 1); // NOTE: color indices are 1-based.
            }
        }
    }
}
