//! Warmstarting: reuses previous-frame impulses for faster solver convergence.

use crate::shaders::dynamics::{
    GpuTransferWarmstartImpulses, TwoBodyConstraint, TwoBodyConstraintBuilder,
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

impl GpuWarmstart {
    /// Transfers warmstart impulses from old constraints to new constraints.
    pub fn transfer_warmstart_impulses<'a>(
        &self,
        pass: &mut GpuPass,
        args: WarmstartArgs<'a>,
    ) -> Result<(), GpuBackendError> {
        let nb = (args.contacts_len.len() as u32).max(1);
        let ws_grid = [
            (args.new_constraints.len() as u32 / nb).max(1).div_ceil(64),
            nb,
            1,
        ];
        self.transfer_warmstart_impulses_kernel.call(
            pass,
            crate::dispatch_grid(args.contacts_len_indirect, ws_grid),
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
}
