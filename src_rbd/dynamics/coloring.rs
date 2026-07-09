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


/// Capacity-based dispatch grid for the coloring kernels (used when
/// fixed-grid dispatch replaces the indirect buffer; see `crate::dispatch_grid`).
fn coloring_grid(args: &ColoringArgs) -> [u32; 3] {
    let nb = (args.contacts_len.len() as u32).max(1);
    [
        (args.constraints.len() as u32 / nb).max(1).div_ceil(64),
        nb,
        1,
    ]
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
            crate::dispatch_grid(args.contacts_len_indirect, coloring_grid(args)),
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
            crate::dispatch_grid(args.contacts_len_indirect, coloring_grid(args)),
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
            crate::dispatch_grid(args.contacts_len_indirect, coloring_grid(args)),
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
            crate::dispatch_grid(args.contacts_len_indirect, coloring_grid(args)),
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
            crate::dispatch_grid(args.contacts_len_indirect, coloring_grid(args)),
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
