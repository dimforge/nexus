//! Warmstarting: reuses previous-frame impulses for faster solver convergence.

use crate::math::Pose;
use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::{
    ContactRecycleState, GpuSeedColorsFromWarmstart, GpuTransferWarmstartImpulses,
    TwoBodyConstraint, Velocity,
};
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
    /// Clamped per-frame list totals (the flat contact sweep bound).
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// Constraint counts per body from previous frame.
    pub old_body_constraint_counts: &'a Tensor<u32>,
    /// Constraint IDs per body from previous frame.
    pub old_body_constraint_ids: &'a Tensor<u32>,
    /// Solver constraints from previous frame.
    pub old_constraints: &'a Tensor<TwoBodyConstraint>,
    /// Solver constraints for current frame (to be warmstarted).
    pub new_constraints: &'a mut Tensor<TwoBodyConstraint>,
    /// When the previous frame's contacts were computed.
    pub old_recycle_states: &'a Tensor<ContactRecycleState>,
    /// When the current frame's contacts were computed (the recycled ones are updated).
    pub recycle_states: &'a mut Tensor<ContactRecycleState>,
    /// Per-collider world poses.
    pub collider_world_poses: &'a Tensor<Pose>,
    /// Rigid body velocities.
    pub vels: &'a Tensor<Velocity>,
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
}

/// Arguments for the coloring seed dispatch.
pub struct SeedColorsArgs<'a> {
    /// Clamped per-frame list totals (the flat contact sweep bound).
    pub contact_plan: &'a Tensor<ContactPlan>,
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
}

impl GpuWarmstart {
    /// Transfers warmstart impulses from old constraints to new constraints, or recycles the old
    /// contacts of the pairs that barely moved.
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
            args.new_constraints,
            args.contact_plan,
            args.old_recycle_states,
            args.recycle_states,
            args.collider_world_poses,
            args.vels,
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
            args.contact_plan,
        )
    }
}
