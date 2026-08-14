//! Particle-to-Grid (P2G) transfer kernel.
//!
//! Transfers particle mass, momentum, and forces to nearby grid nodes using
//! interpolation weights. This is the first major step of each MPM timestep.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::p2g::{GpuP2g, GpuP2gCpic};
use crate::solver::{GpuImpulses, GpuMaterials, GpuParticles};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;

/// GPU compute kernel for Particle-to-Grid (P2G) momentum transfer.
///
/// Rasterizes particle mass and momentum onto the background grid using quadratic
/// B-spline interpolation. Also handles impulse accumulation for rigid body coupling.
#[derive(Shader)]
pub struct WgP2G {
    /// Compiled P2G compute shader.
    p2g: GpuP2g,
    /// Compiled P2G compute shader with CPIC enabled.
    p2g_cpic: GpuP2gCpic,
}

impl WgP2G {
    /// Launches the P2G kernel to transfer particle data to grid nodes.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        use_cpic: bool,
        grid: &mut GpuGrid,
        particles: &GpuParticles,
        impulses: &mut GpuImpulses,
        bodies: &GpuBodySet,
        body_materials: &GpuMaterials,
    ) -> Result<(), GpuBackendError> {
        // Scatter-style P2G: one workgroup per active block, one thread per grid node,
        // streaming the block's particles (primaries + extras) from `sorted_particle_ids`.
        if use_cpic {
            self.p2g_cpic.call(
                pass,
                indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
                &grid.meta,
                &grid.active_blocks,
                &particles.sorted_ids,
                particles.positions(),
                particles.kinematics(),
                &mut grid.nodes,
                &bodies.vels,
                &body_materials.materials,
                &mut impulses.incremental_impulses,
            )
        } else {
            self.p2g.call(
                pass,
                indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
                &grid.meta,
                &grid.active_blocks,
                &particles.sorted_ids,
                particles.positions(),
                particles.kinematics(),
                &mut grid.nodes,
            )
        }
    }
}
