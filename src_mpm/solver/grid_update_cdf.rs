//! Grid CDF (Collision Detection Field) update for rigid body coupling.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::grid_update_cdf::GpuGridUpdateCdf;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;

/// GPU kernel for updating grid node CDF data from rigid bodies.
///
/// Computes signed distance fields and closest points on rigid body surfaces
/// for each active grid node.
#[derive(Shader)]
pub struct WgGridUpdateCdf {
    /// Compiled grid CDF update shader.
    grid_update: GpuGridUpdateCdf,
}

impl WgGridUpdateCdf {
    /// Launches grid CDF update from rigid body geometries.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        grid: &mut GpuGrid,
        bodies: &GpuBodySet,
    ) -> Result<(), GpuBackendError> {
        if bodies.is_empty() {
            return Ok(());
        }

        self.grid_update.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &grid.meta,
            &grid.active_blocks,
            &bodies.shapes,
            &bodies.poses,
            &mut grid.nodes,
        )
    }
}
