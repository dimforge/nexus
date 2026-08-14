//! Particle sorting kernels for spatial acceleration.
//!
//! These kernels handle spatial hashing and sorting to group particles by grid block
//! for efficient neighbor queries during P2G/G2P.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::grid::sort::{
    GpuCopyParticlesLenToScanValue, GpuCopyRigidParticlesLenToScanValue,
    GpuCopyScanValuesToFirstParticles, GpuCopyScanValuesToFirstRigidParticles,
    GpuFinalizeParticlesSort, GpuFinalizeRigidParticlesSort, GpuMarkRigidParticlesNeedingBlock,
    GpuTouchNeighborBlocks, GpuTouchParticleBlocks, GpuTouchPrimaryBlocks,
    GpuTouchRigidParticleBlocks, GpuUpdateBlockParticleCount, GpuUpdateBlockRigidParticleCount,
    GpuUpdateNbhBlockIds,
};
use crate::solver::GpuRigidParticles;
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use nexus_rbd::utils::{GpuPrefixSum, PrefixSumWorkspace};

/// GPU compute kernels for sorting particles into grid cells.
///
/// Implements spatial hashing and sorting to group particles by grid block
/// for efficient neighbor queries during P2G/G2P.
#[derive(Shader)]
pub struct WgSort {
    /// Legacy single-pass block activation, kept for the `launch_touch_for_test`
    /// correctness check against the two-pass `touch_primary_blocks`/`touch_neighbor_blocks`.
    pub(crate) touch_particle_blocks: GpuTouchParticleBlocks,
    pub(crate) touch_primary_blocks: GpuTouchPrimaryBlocks,
    pub(crate) touch_neighbor_blocks: GpuTouchNeighborBlocks,
    pub(crate) touch_rigid_particle_blocks: GpuTouchRigidParticleBlocks,
    pub(crate) mark_rigid_particles_needing_block: GpuMarkRigidParticlesNeedingBlock,
    pub(crate) update_block_particle_count: GpuUpdateBlockParticleCount,
    pub(crate) update_nbh_block_ids: GpuUpdateNbhBlockIds,
    pub(crate) copy_particles_len_to_scan_value: GpuCopyParticlesLenToScanValue,
    pub(crate) copy_scan_values_to_first_particles: GpuCopyScanValuesToFirstParticles,
    pub(crate) finalize_particles_sort: GpuFinalizeParticlesSort,
    pub(crate) update_block_rigid_particle_count: GpuUpdateBlockRigidParticleCount,
    pub(crate) copy_rigid_particles_len_to_scan_value: GpuCopyRigidParticlesLenToScanValue,
    pub(crate) copy_scan_values_to_first_rigid_particles: GpuCopyScanValuesToFirstRigidParticles,
    pub(crate) finalize_rigid_particles_sort: GpuFinalizeRigidParticlesSort,
}

impl WgSort {
    /// Sorts rigid body particles by grid block.
    ///
    /// Runs the same count / prefix-sum / finalize sequence as the regular particle
    /// sort, reusing `grid.scan_values` (which the regular sort no longer needs once
    /// `launch_sort` returned). The result feeds the scatter-style P2G-CDF kernel.
    pub fn launch_sort_rigid_particles(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        rigid_particles: &mut GpuRigidParticles,
        grid: &mut GpuGrid,
        prefix_sum: &mut PrefixSumWorkspace,
        prefix_sum_module: &GpuPrefixSum,
    ) -> Result<(), GpuBackendError> {
        if rigid_particles.is_empty() {
            return Ok(());
        }

        let rigid_particles_len = rigid_particles.len() as u32;

        self.update_block_rigid_particle_count.call(
            pass,
            rigid_particles_len,
            &grid.meta,
            &grid.hmap_entries,
            &rigid_particles.sample_points,
            &mut grid.active_blocks,
        )?;

        self.copy_rigid_particles_len_to_scan_value.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
            &grid.meta,
            &grid.active_blocks,
            &mut grid.scan_values,
        )?;
        prefix_sum_module.launch(backend, pass, prefix_sum, &mut grid.scan_values, 1)?;

        self.copy_scan_values_to_first_rigid_particles.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
            &grid.meta,
            &grid.scan_values,
            &mut grid.active_blocks,
        )?;

        self.finalize_rigid_particles_sort.call(
            pass,
            rigid_particles_len,
            &grid.meta,
            &grid.hmap_entries,
            &rigid_particles.sample_points,
            &mut grid.active_blocks,
            &mut rigid_particles.sorted_ids,
        )?;

        Ok(())
    }
}
