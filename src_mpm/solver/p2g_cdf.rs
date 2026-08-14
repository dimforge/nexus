//! Particle-to-Grid transfer with Collision Detection Field for rigid bodies.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::p2g_cdf::GpuP2gCdf;
use crate::solver::GpuRigidParticles;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;

/// GPU kernel for P2G transfer from rigid body particles.
///
/// Transfers momentum from rigid body surface particles to grid nodes,
/// enabling two-way coupling between MPM and rigid bodies.
#[derive(Shader)]
pub struct WgP2GCdf {
    /// Compiled P2G-CDF compute shader.
    p2g_cdf: GpuP2gCdf,
}

impl WgP2GCdf {
    /// Launches P2G transfer from rigid body particles to grid.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        grid: &mut GpuGrid,
        rigid_particles: &GpuRigidParticles,
        bodies: &GpuBodySet,
    ) -> Result<(), GpuBackendError> {
        if rigid_particles.is_empty() {
            return Ok(());
        }

        self.p2g_cdf.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &grid.meta,
            &grid.active_blocks,
            &rigid_particles.sorted_ids,
            &rigid_particles.sample_points,
            &bodies.shapes_vertex_buffers,
            &rigid_particles.sample_ids,
            &mut grid.nodes,
        )
    }
}
