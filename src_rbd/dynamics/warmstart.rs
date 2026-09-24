//! Warmstarting: reuses previous-frame impulses for faster solver convergence.
//!
//! The contacts are matched with the previous step's by `GpuSolver::prepare`, and the
//! constraints receive their previous impulses in `GpuSolver::build_constraints`. The coloring
//! also starts from the previous colors of the matched constraints.

use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::{ContactLink, GpuSeedColorsFromWarmstart};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader seeding the coloring with the previous step's colors.
#[derive(Shader)]
pub struct GpuWarmstart {
    /// Seeds the topo-gc coloring from the previous frame's colors of the matched constraints.
    seed_colors_kernel: GpuSeedColorsFromWarmstart,
}

/// Arguments for the coloring seed dispatch.
pub struct SeedColorsArgs<'a> {
    /// Clamped per-frame list totals (the flat contact dispatch bound).
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// The contact links of the current frame, matched with the previous frame's.
    pub links: &'a Tensor<ContactLink>,
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
            args.links,
            args.old_constraints_colors,
            args.constraints_colors,
            args.colored,
            args.contact_plan,
        )
    }
}
