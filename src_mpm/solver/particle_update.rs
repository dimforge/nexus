//! Particle state update kernel.
//!
//! Updates particle positions, deformation gradients, and material state after
//! grid velocities have been transferred back to particles.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::solver::particle_update::GpuParticleUpdate;
use crate::solver::{GpuParticles, GpuSimulationParams};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

/// GPU compute kernel for updating particle state.
///
/// Integrates particle positions using updated velocities, updates deformation
/// gradients, and applies constitutive models (elasticity, plasticity).
#[derive(Shader)]
pub struct WgParticleUpdate {
    /// Compiled particle update compute shader.
    particle_update: GpuParticleUpdate,
}

impl WgParticleUpdate {
    /// Launches the particle update kernel.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &mut GpuParticles,
    ) -> Result<(), GpuBackendError> {
        let len = particles.len() as u32;
        self.particle_update.call(
            pass,
            [len, 1, 1],
            &sim_params.params,
            &grid.meta,
            &mut particles.models,
            &mut particles.positions,
            &mut particles.kinematics,
            &mut particles.def_grad,
            &particles.properties,
            &particles.gpu_len,
        )
    }
}
