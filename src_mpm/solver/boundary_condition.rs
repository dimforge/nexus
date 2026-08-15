pub use crate::mpm_shaders::solver::boundary_condition::{
    BodyMaterials, BoundaryCondition, MAX_COLLISION_BODIES,
};
use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuBackendError};
use vortx::tensor::Tensor;

/// GPU buffer storing the per-rigid-body boundary conditions.
///
/// Held as a single **uniform** `BodyMaterials` (a fixed `MAX_COLLISION_BODIES`
/// array) rather than a storage buffer, so the MPM kernels that read it stay
/// within the 8-storage-buffer WebGPU limit.
pub struct GpuMaterials {
    pub materials: Tensor<BodyMaterials>,
}

impl GpuMaterials {
    /// Creates the boundary-condition uniform buffer.
    ///
    /// Allocates space for up to `MAX_COLLISION_BODIES` bodies (CPIC limitation).
    pub fn new(
        backend: &GpuBackend,
        materials: &[BoundaryCondition],
    ) -> Result<Self, GpuBackendError> {
        assert!(
            materials.len() <= MAX_COLLISION_BODIES,
            "CPIC only supports up to {MAX_COLLISION_BODIES} colliders"
        );
        let mut mats = [BoundaryCondition::default(); MAX_COLLISION_BODIES];
        mats[..materials.len()].copy_from_slice(materials);
        Ok(Self {
            materials: Tensor::scalar(
                backend,
                BodyMaterials { mats },
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
        })
    }
}
