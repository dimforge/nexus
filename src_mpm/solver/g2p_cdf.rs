//! Grid-to-Particle transfer with Collision Detection Field updates.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::g2p_cdf::GpuG2pCdf;
use crate::solver::{GpuParticles, GpuSimulationParams};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

/// GPU kernel for G2P transfer with CDF updates for rigid body coupling.
///
/// Updates particle CDF (Collision Detection Field) data based on proximity
/// to rigid bodies during the G2P phase.
#[derive(Shader)]
pub struct WgG2PCdf {
    /// Compiled G2P-CDF compute shader.
    g2p_cdf: GpuG2pCdf,
}

impl WgG2PCdf {
    /// Launches G2P with CDF updates for MPM particles.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &mut GpuParticles,
    ) -> Result<(), GpuBackendError> {
        self.g2p_cdf.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &sim_params.params,
            &grid.meta,
            &grid.hmap_entries,
            &grid.active_blocks,
            &grid.nodes,
            &particles.sorted_ids,
            &particles.positions,
            &mut particles.kinematics,
        )
    }
}
