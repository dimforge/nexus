//! Warmstarting: reuses previous-frame impulses for faster solver convergence.

use crate::shaders::dynamics::{
    GpuSeedColorsFromWarmstart, GpuTransferWarmstartImpulses, TwoBodyConstraint,
    TwoBodyConstraintBuilder,
};
use crate::shaders::utils::BatchIndices;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader for transferring warmstart impulses between frames.
///
/// This shader matches new contacts against old contacts and transfers impulse
/// accumulators when a match is found.
#[derive(Shader)]
pub struct GpuWarmstart {
    /// Compute pipeline that matches contacts and transfers impulses.
    transfer_warmstart_impulses_kernel: GpuTransferWarmstartImpulses,
    /// Seeds the topo-gc coloring from the previous frame's colors (same
    /// old/new body-pair matching as the impulse transfer).
    seed_colors_kernel: GpuSeedColorsFromWarmstart,
}

/// Arguments for warmstart dispatch.
///
/// Contains buffers for both old (previous frame) and new (current frame) constraint data.
pub struct WarmstartArgs<'a> {
    /// Number of contacts in current frame.
    pub contacts_len: &'a Tensor<u32>,
    /// Constraint counts per body from previous frame.
    pub old_body_constraint_counts: &'a Tensor<u32>,
    /// Constraint IDs per body from previous frame.
    pub old_body_constraint_ids: &'a Tensor<u32>,
    /// Solver constraints from previous frame.
    pub old_constraints: &'a Tensor<TwoBodyConstraint>,
    /// Constraint builders from previous frame.
    pub old_constraint_builders: &'a Tensor<TwoBodyConstraintBuilder>,
    /// Solver constraints for current frame (to be warmstarted).
    pub new_constraints: &'a mut Tensor<TwoBodyConstraint>,
    /// Constraint builders for current frame.
    pub new_constraint_builders: &'a Tensor<TwoBodyConstraintBuilder>,
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Shared per-batch index uniform.
    pub batch_indices: &'a Tensor<BatchIndices>,
}

/// Arguments for the coloring seed dispatch — see
/// [`GpuWarmstart::seed_colors_from_warmstart`].
pub struct SeedColorsArgs<'a> {
    /// Number of contacts in current frame.
    pub contacts_len: &'a Tensor<u32>,
    /// Constraint counts per body from previous frame.
    pub old_body_constraint_counts: &'a Tensor<u32>,
    /// Constraint IDs per body from previous frame.
    pub old_body_constraint_ids: &'a Tensor<u32>,
    /// Solver constraints from previous frame.
    pub old_constraints: &'a Tensor<TwoBodyConstraint>,
    /// Solver constraints for current frame.
    pub new_constraints: &'a Tensor<TwoBodyConstraint>,
    /// Colors assigned to the previous frame's constraints.
    pub old_constraints_colors: &'a Tensor<u32>,
    /// Output: colors for the current frame's constraints (seeded slots only).
    pub constraints_colors: &'a mut Tensor<u32>,
    /// Output: per-constraint colored flag consumed by topo-gc.
    pub colored: &'a mut Tensor<u32>,
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Shared per-batch index uniform.
    pub batch_indices: &'a Tensor<BatchIndices>,
}

impl GpuWarmstart {
    /// Transfers warmstart impulses from old constraints to new constraints.
    pub fn transfer_warmstart_impulses<'a>(
        &self,
        pass: &mut GpuPass,
        args: WarmstartArgs<'a>,
    ) -> Result<(), GpuBackendError> {
        self.transfer_warmstart_impulses_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.old_body_constraint_counts,
            args.old_body_constraint_ids,
            args.old_constraints,
            args.old_constraint_builders,
            args.new_constraints,
            args.new_constraint_builders,
            args.contacts_len,
            args.batch_indices,
        )
    }

    /// Seeds the topo-gc coloring from the previous frame's colors. Must run
    /// after the topo-gc reset and before its iterations.
    pub fn seed_colors_from_warmstart(
        &self,
        pass: &mut GpuPass,
        args: SeedColorsArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.seed_colors_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.old_body_constraint_counts,
            args.old_body_constraint_ids,
            args.old_constraints,
            args.new_constraints,
            args.old_constraints_colors,
            args.constraints_colors,
            args.colored,
            args.contacts_len,
            args.batch_indices,
        )
    }
}
