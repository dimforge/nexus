//! Timestep bound estimation kernels.
//!
//! Computes a CFL-based upper bound on the simulation timestep to prevent
//! numerical instability. Uses material sound speed and particle velocities
//! to determine the maximum safe timestep.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::solver::timestep_bound::{
    GpuEstimateTimestepBound, GpuResetTimestepBound, GpuTimestepBounds,
};
use crate::solver::GpuParticles;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuPass, GpuTimestamps};
use vortx::tensor::Tensor;

/// GPU kernel for estimating the maximum stable timestep (best-effort, does not eliminate all divergence risk).
#[derive(Shader)]
pub struct WgTimestepBounds {
    reset_timestep_bound: GpuResetTimestepBound,
    estimate_timestep_bound: GpuEstimateTimestepBound,
}

impl WgTimestepBounds {
    /// Launches the timestep bounds estimation and returns the estimated maximum timestep length.
    pub async fn compute_bounds(
        &self,
        backend: &GpuBackend,
        timestamps: Option<&mut GpuTimestamps>,
        grid: &GpuGrid,
        particles: &GpuParticles,
        bounds: &mut Tensor<GpuTimestepBounds>,
        bounds_staging: &mut Tensor<GpuTimestepBounds>,
    ) -> Result<f32, GpuBackendError> {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("timestep-bounds", timestamps);
        self.launch(&mut pass, grid, particles, bounds)?;
        drop(pass);
        bounds_staging.copy_from_view(&mut encoder, &*bounds)?;
        backend.submit(encoder)?;

        let mut result = [GpuTimestepBounds::default()];
        backend
            .read_buffer(bounds_staging.buffer(), &mut result)
            .await?;
        Ok(result[0].computed_max_dt_as_uint as f32 / GpuTimestepBounds::FLOAT_TO_INT)
    }

    fn launch(
        &self,
        pass: &mut GpuPass,
        grid: &GpuGrid,
        particles: &GpuParticles,
        bounds: &mut Tensor<GpuTimestepBounds>,
    ) -> Result<(), GpuBackendError> {
        self.reset_timestep_bound.call(pass, 1u32, bounds)?;

        let len = particles.len() as u32;
        self.estimate_timestep_bound.call(
            pass,
            [len, 1, 1],
            &grid.meta,
            &particles.models,
            &particles.kinematics,
            &particles.def_grad,
            &particles.properties,
            &particles.gpu_len,
            bounds,
        )
    }
}
