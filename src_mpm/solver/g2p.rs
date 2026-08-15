//! Grid-to-Particle (G2P) transfer kernel.
//!
//! Interpolates grid velocities back to particles and updates particle velocity
//! gradients. This happens after grid forces have been applied.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::g2p::{GpuG2p, GpuG2pCpic};
use crate::solver::{GpuMaterials, GpuParticles, GpuSimulationParams};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;

/// GPU compute kernel for Grid-to-Particle (G2P) velocity interpolation.
///
/// Samples grid velocities at particle positions using quadratic B-spline weights
/// and updates particle velocity gradients for deformation tracking (APIC method).
#[derive(Shader)]
pub struct WgG2P {
    /// Compiled G2P compute shader.
    g2p: GpuG2p,
    g2p_cpic: GpuG2pCpic,
}

impl WgG2P {
    /// Launches the G2P kernel to update particle velocities from grid.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        use_cpic: bool,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &mut GpuParticles,
        bodies: &GpuBodySet,
        body_materials: &GpuMaterials,
    ) -> Result<(), GpuBackendError> {
        if use_cpic {
            self.g2p_cpic.call(
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
                &bodies.vels,
                &bodies.mprops,
                &body_materials.materials,
            )
        } else {
            self.g2p.call(
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
                &bodies.vels,
                &bodies.mprops,
                &body_materials.materials,
            )
        }
    }
}
