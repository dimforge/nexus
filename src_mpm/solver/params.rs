pub use crate::mpm_shaders::solver::params::SimulationParams;
use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuBackendError};
use vortx::tensor::Tensor;

/// GPU-resident simulation parameters.
pub struct GpuSimulationParams {
    /// Uniform buffer containing simulation parameters.
    pub params: Tensor<SimulationParams>,
}

impl GpuSimulationParams {
    /// Uploads simulation parameters to GPU memory.
    pub fn new(backend: &GpuBackend, params: SimulationParams) -> Result<Self, GpuBackendError> {
        Ok(Self {
            params: Tensor::scalar(
                backend,
                params,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
        })
    }
}
