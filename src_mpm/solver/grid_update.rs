//! Grid node update kernel.
//!
//! Updates grid node velocities by applying forces (gravity, boundary conditions)
//! and solving momentum equations on the grid.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::grid_update::GpuGridUpdate;
use crate::mpm_shaders::solver::grid_update_collide::GpuGridUpdateCollide;
use crate::solver::{GpuMaterials, GpuSimulationParams};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;

/// GPU compute kernel for updating grid node velocities.
///
/// Applies external forces (gravity), boundary conditions (sticky/slip walls),
/// and solves momentum equations on grid nodes. Runs between P2G and G2P stages.
#[derive(Shader)]
pub struct WgGridUpdate {
    /// Compiled grid update compute shader.
    grid_update: GpuGridUpdate,
    grid_update_collide: GpuGridUpdateCollide,
}

impl WgGridUpdate {
    /// Launches the grid update kernel.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        use_cpic: bool,
        sim_params: &GpuSimulationParams,
        grid: &mut GpuGrid,
        bodies: &GpuBodySet,
        body_materials: &GpuMaterials,
    ) -> Result<(), GpuBackendError> {
        self.grid_update.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &sim_params.params,
            &grid.meta,
            &grid.active_blocks,
            &mut grid.nodes,
        )?;

        if !use_cpic {
            self.grid_update_collide.call(
                pass,
                indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
                &sim_params.params,
                &grid.meta,
                &grid.active_blocks,
                &bodies.shapes,
                &bodies.poses,
                // The trimesh path projects a shape-local point, so it needs the
                // local-space vertices (with a valid interleaved BVH), not the
                // world-space buffer maintained for `p2g_cdf`.
                &bodies.shapes_local_vertex_buffers,
                &bodies.shapes_index_buffers,
                &bodies.vels,
                &bodies.mprops,
                &body_materials.materials,
                &mut grid.nodes,
            )?;
        }

        Ok(())
    }
}
