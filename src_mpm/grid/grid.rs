//! Grid data structures and GPU kernels for sparse grid management.

use crate::grid::sort::WgSort;
use crate::mpm_shaders::grid::grid::{
    ActiveBlockHeader, GpuCaptureNumActiveBlocks, GpuInitIndirectWorkgroups, GpuResetHmap, Grid,
    GridHashMapEntry, Node,
};
use crate::solver::{GpuParticles, GpuRigidParticles};
use khal::backend::{Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuPass, GpuTimestamps};
use khal::{BufferUsages, Shader};
use nexus_rbd::utils::{GpuPrefixSum, PrefixSumWorkspace};
use vortx::tensor::Tensor;

/// GPU kernels for grid initialization and management.
///
/// Handles sparse grid allocation, reset, and indirect dispatch setup.
#[derive(Shader)]
pub struct WgGrid {
    reset_hmap: GpuResetHmap,
    capture_num_active_blocks: GpuCaptureNumActiveBlocks,
    init_indirect_workgroups: GpuInitIndirectWorkgroups,
}

impl WgGrid {
    /// Sorts particles into grid cells and allocates sparse grid blocks.
    pub fn launch_sort(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        particles: &mut GpuParticles,
        mut rigid_particles: Option<&mut GpuRigidParticles>,
        grid: &mut GpuGrid,
        prefix_sum: &mut PrefixSumWorkspace,
        sort_module: &WgSort,
        prefix_sum_module: &GpuPrefixSum,
    ) -> Result<(), GpuBackendError> {
        let particles_len = particles.len() as u32;
        let hmap_capacity = grid.cpu_meta.hmap_capacity;

        // Retry until we allocated enough room on the sparse grid for all the blocks.
        let mut sparse_grid_has_the_correct_size = false;
        while !sparse_grid_has_the_correct_size {
            // - Reset next grid's hashmap.
            // - Reset grid.num_active_blocks to 0.
            // - Run touch_particle_blocks on the next grid.
            // - Readback num_active_blocks.
            // - Update the hashmap & grid buffer sizes if its occupancy is too high.

            // NOTE: num_active_blocks := 0 is set in reset_hmap.
            self.reset_hmap
                .call(pass, hmap_capacity, &mut grid.meta, &mut grid.hmap_entries)?;

            // Block activation in two passes (cheaper than every particle inserting
            // all NUM_ASSOC_BLOCKS of its stencil):
            // 1. Each particle activates only its primary (base) block.
            // 2. Each base block activates its +1 neighbour blocks (once per block,
            //    not once per particle). `active_blocks_snapshot` records the base-block
            //    count so pass 2 doesn't reprocess the neighbours it appends.
            sort_module.touch_primary_blocks.call(
                pass,
                particles_len,
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &particles.positions,
                &particles.gpu_len,
            )?;

            self.capture_num_active_blocks.call(
                pass,
                1u32,
                &grid.meta,
                &mut grid.active_blocks_snapshot,
            )?;

            // Indirect dispatch sized for the base blocks (the neighbour pass runs one
            // thread per base block).
            self.init_indirect_workgroups.call(
                pass,
                1u32,
                &grid.meta,
                &mut grid.indirect_n_blocks_groups,
                &mut grid.indirect_n_g2p_p2g_groups,
            )?;

            sort_module.touch_neighbor_blocks.call(
                pass,
                indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &grid.active_blocks_snapshot,
            )?;

            // Ensure blocks exist wherever we have rigid particles that might affect
            // other blocks. This is done in two passes:
            // 1. Mark all rigid particles that need to ensure its associated block exists
            // 2. Touch the blocks with marked rigid particles.
            if let Some(rigid_particles) = rigid_particles.as_deref_mut()
                && !rigid_particles.is_empty()
            {
                let rigid_particles_len = rigid_particles.len() as u32;
                sort_module.mark_rigid_particles_needing_block.call(
                    pass,
                    rigid_particles_len,
                    &grid.meta,
                    &grid.hmap_entries,
                    &rigid_particles.sample_points,
                    &mut rigid_particles.rigid_particle_needs_block,
                )?;

                sort_module.touch_rigid_particle_blocks.call(
                    pass,
                    rigid_particles_len,
                    &mut grid.meta,
                    &mut grid.hmap_entries,
                    &mut grid.active_blocks,
                    &rigid_particles.sample_points,
                    &rigid_particles.rigid_particle_needs_block,
                )?;
            }

            // TODO: handle grid buffer resizing
            sparse_grid_has_the_correct_size = true;
        }

        // - Launch update_block_particle_count
        // - Launch copy_particle_len_to_scan_value
        // - Launch cumulated sum.
        // - Launch copy_scan_values_to_first_particles
        // - Launch finalize_particles_sort
        // - Launch write_blocks_multiplicity_to_scan_value
        // - Launch cumulated sum

        // Prepare workgroups for indirect dispatches based on the number of active blocks.
        self.init_indirect_workgroups.call(
            pass,
            1u32,
            &grid.meta,
            &mut grid.indirect_n_blocks_groups,
            &mut grid.indirect_n_g2p_p2g_groups,
        )?;

        sort_module.update_nbh_block_ids.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
            &grid.meta,
            &grid.hmap_entries,
            &mut grid.active_blocks,
        )?;

        sort_module.update_block_particle_count.call(
            pass,
            particles_len,
            &grid.meta,
            &grid.hmap_entries,
            &particles.positions,
            &particles.gpu_len,
            &mut grid.active_blocks,
        )?;

        sort_module.copy_particles_len_to_scan_value.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
            &grid.meta,
            &grid.active_blocks,
            &mut grid.scan_values,
        )?;
        prefix_sum_module.launch(backend, pass, prefix_sum, &mut grid.scan_values, 1)?;

        sort_module.copy_scan_values_to_first_particles.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
            &grid.meta,
            &grid.scan_values,
            &mut grid.active_blocks,
        )?;

        sort_module.finalize_particles_sort.call(
            pass,
            particles_len,
            &grid.meta,
            &grid.hmap_entries,
            &particles.positions,
            &particles.gpu_len,
            &mut grid.active_blocks,
            &mut particles.sorted_ids,
        )?;

        Ok(())
    }

    /// Test helper: resets the hashmap and activates blocks for `particles` using
    /// either the legacy single-pass touch (`two_pass = false`) or the new two-pass
    /// touch (`two_pass = true`), leaving `grid.meta.num_active_blocks` readable.
    ///
    /// Both paths must activate the identical set of blocks; a benchmark compares the
    /// resulting `num_active_blocks` to validate the two-pass touch without depending
    /// on CPU/GPU rounding agreement.
    #[doc(hidden)]
    pub fn launch_touch_for_test(
        &self,
        pass: &mut GpuPass,
        particles: &GpuParticles,
        grid: &mut GpuGrid,
        sort_module: &WgSort,
        two_pass: bool,
    ) -> Result<(), GpuBackendError> {
        let particles_len = particles.len() as u32;
        let hmap_capacity = grid.cpu_meta.hmap_capacity;

        self.reset_hmap
            .call(pass, hmap_capacity, &mut grid.meta, &mut grid.hmap_entries)?;

        if two_pass {
            sort_module.touch_primary_blocks.call(
                pass,
                particles_len,
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &particles.positions,
                &particles.gpu_len,
            )?;
            self.capture_num_active_blocks.call(
                pass,
                1u32,
                &grid.meta,
                &mut grid.active_blocks_snapshot,
            )?;
            self.init_indirect_workgroups.call(
                pass,
                1u32,
                &grid.meta,
                &mut grid.indirect_n_blocks_groups,
                &mut grid.indirect_n_g2p_p2g_groups,
            )?;
            sort_module.touch_neighbor_blocks.call(
                pass,
                indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &grid.active_blocks_snapshot,
            )?;
        } else {
            sort_module.touch_particle_blocks.call(
                pass,
                particles_len,
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &particles.positions,
                &particles.gpu_len,
            )?;
        }

        Ok(())
    }

    /// Per-kernel-profiled variant of [`launch_sort`](Self::launch_sort) for the
    /// regular-particle (CPIC-disabled) path.
    ///
    /// Runs the same kernel sequence as `launch_sort` with `rigid_particles = None`,
    /// but wraps each sub-kernel in its own timestamp scope so a benchmark can see
    /// where the sort spends its time. Diagnostics only: the production pipeline
    /// uses `launch_sort`.
    #[doc(hidden)]
    pub fn launch_sort_profiled(
        &self,
        backend: &GpuBackend,
        encoder: &mut GpuEncoder,
        timestamps: &mut GpuTimestamps,
        particles: &mut GpuParticles,
        grid: &mut GpuGrid,
        prefix_sum: &mut PrefixSumWorkspace,
        sort_module: &WgSort,
        prefix_sum_module: &GpuPrefixSum,
    ) -> Result<(), GpuBackendError> {
        let particles_len = particles.len() as u32;
        let hmap_capacity = grid.cpu_meta.hmap_capacity;

        {
            let mut pass = encoder.begin_pass("sort:reset_hmap", Some(timestamps));
            self.reset_hmap.call(
                &mut pass,
                hmap_capacity,
                &mut grid.meta,
                &mut grid.hmap_entries,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:touch_primary_blocks", Some(timestamps));
            sort_module.touch_primary_blocks.call(
                &mut pass,
                particles_len,
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &particles.positions,
                &particles.gpu_len,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:capture_num_active_blocks", Some(timestamps));
            self.capture_num_active_blocks.call(
                &mut pass,
                1u32,
                &grid.meta,
                &mut grid.active_blocks_snapshot,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:init_indirect_workgroups", Some(timestamps));
            self.init_indirect_workgroups.call(
                &mut pass,
                1u32,
                &grid.meta,
                &mut grid.indirect_n_blocks_groups,
                &mut grid.indirect_n_g2p_p2g_groups,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:touch_neighbor_blocks", Some(timestamps));
            sort_module.touch_neighbor_blocks.call(
                &mut pass,
                indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
                &mut grid.meta,
                &mut grid.hmap_entries,
                &mut grid.active_blocks,
                &grid.active_blocks_snapshot,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:init_indirect_workgroups2", Some(timestamps));
            self.init_indirect_workgroups.call(
                &mut pass,
                1u32,
                &grid.meta,
                &mut grid.indirect_n_blocks_groups,
                &mut grid.indirect_n_g2p_p2g_groups,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:update_nbh_block_ids", Some(timestamps));
            sort_module.update_nbh_block_ids.call(
                &mut pass,
                indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
                &grid.meta,
                &grid.hmap_entries,
                &mut grid.active_blocks,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:update_block_particle_count", Some(timestamps));
            sort_module.update_block_particle_count.call(
                &mut pass,
                particles_len,
                &grid.meta,
                &grid.hmap_entries,
                &particles.positions,
                &particles.gpu_len,
                &mut grid.active_blocks,
            )?;
        }
        {
            let mut pass =
                encoder.begin_pass("sort:copy_particles_len_to_scan_value", Some(timestamps));
            sort_module.copy_particles_len_to_scan_value.call(
                &mut pass,
                indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
                &grid.meta,
                &grid.active_blocks,
                &mut grid.scan_values,
            )?;
        }
        {
            let mut pass = encoder.begin_pass("sort:prefix_sum", Some(timestamps));
            prefix_sum_module.launch(backend, &mut pass, prefix_sum, &mut grid.scan_values, 1)?;
        }
        {
            let mut pass =
                encoder.begin_pass("sort:copy_scan_values_to_first_particles", Some(timestamps));
            sort_module.copy_scan_values_to_first_particles.call(
                &mut pass,
                indirect_dispatch_tensor(&grid.indirect_n_blocks_groups),
                &grid.meta,
                &grid.scan_values,
                &mut grid.active_blocks,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("sort:finalize_particles_sort", Some(timestamps));
            sort_module.finalize_particles_sort.call(
                &mut pass,
                particles_len,
                &grid.meta,
                &grid.hmap_entries,
                &particles.positions,
                &particles.gpu_len,
                &mut grid.active_blocks,
                &mut particles.sorted_ids,
            )?;
        }

        Ok(())
    }
}

/// Reinterprets a `Tensor<u32>` (with 3 elements) as a `Tensor<[u32; 3]>` for indirect dispatch.
///
/// # Safety
/// The underlying GPU buffer is just raw bytes; `Tensor<u32>` with 3 elements has identical
/// memory layout to `Tensor<[u32; 3]>` with 1 element. Both `u32` and `[u32; 3]` are `Pod`.
pub(crate) fn indirect_dispatch_tensor(tensor: &Tensor<u32>) -> &Tensor<[u32; 3]> {
    unsafe { &*(tensor as *const Tensor<u32> as *const Tensor<[u32; 3]>) }
}

/// GPU-resident sparse grid structure.
///
/// The MPM grid uses a sparse representation with a hashmap to efficiently
/// store only active blocks (blocks containing particles). This dramatically
/// reduces memory usage for spatially localized simulations.
pub struct GpuGrid {
    /// CPU copy of grid metadata for readback.
    pub cpu_meta: Grid,
    /// GPU buffer containing grid metadata.
    pub meta: Tensor<Grid>,
    /// Pong buffer for grid metadata.
    pub prev_meta: Tensor<Grid>,
    /// Hash map entries for virtual-to-physical block mapping.
    pub hmap_entries: Tensor<GridHashMapEntry>,
    /// Pong buffer for hmap entries.
    pub prev_hmap_entries: Tensor<GridHashMapEntry>,
    /// Grid node data (momentum, mass, CDF).
    pub nodes: Tensor<Node>,
    /// Active block headers tracking particle ranges.
    pub active_blocks: Tensor<ActiveBlockHeader>,
    /// Workspace for prefix sum operations.
    pub scan_values: Tensor<u32>,
    /// Single-element snapshot of `num_active_blocks` taken after the primary-block
    /// touch pass, so the neighbour-block touch pass only iterates over base blocks.
    pub active_blocks_snapshot: Tensor<u32>,
    /// Indirect dispatch arguments for block-parallel kernels.
    ///
    /// Stored as `Tensor<u32>` with 3 elements so it can be written by
    /// `init_indirect_workgroups` (which operates on `&mut [u32]`).
    /// Use [`indirect_n_blocks_dispatch`](Self::indirect_n_blocks_dispatch)
    /// to obtain a `DispatchGrid` for indirect dispatch.
    pub indirect_n_blocks_groups: Tensor<u32>,
    /// Indirect dispatch arguments for node-parallel kernels.
    ///
    /// Same layout as `indirect_n_blocks_groups`. Use
    /// [`indirect_n_g2p_p2g_dispatch`](Self::indirect_n_g2p_p2g_dispatch)
    /// for indirect dispatch.
    pub indirect_n_g2p_p2g_groups: Tensor<u32>,
    /// Debug buffer for GPU-side diagnostics.
    pub debug: Tensor<u32>,
}

impl GpuGrid {
    /// Returns indirect dispatch arguments for block-parallel kernels.
    ///
    /// This reinterprets a `Tensor<u32>` (with 3 elements) as a `Tensor<[u32; 3]>`
    /// (with 1 element). This is sound because the memory layout is identical and both
    /// types are `Pod`.
    pub fn indirect_n_blocks_dispatch(&self) -> &Tensor<[u32; 3]> {
        indirect_dispatch_tensor(&self.indirect_n_blocks_groups)
    }

    /// Returns indirect dispatch arguments for node-parallel (G2P/P2G) kernels.
    ///
    /// See [`indirect_n_blocks_dispatch`](Self::indirect_n_blocks_dispatch) for safety rationale.
    pub fn indirect_n_g2p_p2g_dispatch(&self) -> &Tensor<[u32; 3]> {
        indirect_dispatch_tensor(&self.indirect_n_g2p_p2g_groups)
    }

    /// Creates a new sparse grid with the specified capacity.
    pub fn with_capacity(
        backend: &GpuBackend,
        capacity: u32,
        cell_width: f32,
    ) -> Result<Self, GpuBackendError> {
        const NODES_PER_BLOCK: u32 = 64; // 8 * 8 in 2D and 4 * 4 * 4 in 3D.
        let capacity = capacity.next_power_of_two();
        let cpu_meta = Grid {
            num_active_blocks: 0,
            cell_width,
            hmap_capacity: capacity,
            capacity,
        };
        let meta = Tensor::scalar(
            backend,
            cpu_meta,
            BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let prev_meta = Tensor::scalar(
            backend,
            cpu_meta,
            BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let default_entry = GridHashMapEntry {
            state: 0xFFFFFFFF,
            key: Default::default(),
            value: Default::default(),
            ownership: 0,
            padding: [0; _],
        };
        let default_entries = vec![default_entry; capacity as usize];
        let prev_hmap_entries = Tensor::vector(backend, &default_entries, BufferUsages::STORAGE)?;
        let hmap_entries = Tensor::vector(backend, &default_entries, BufferUsages::STORAGE)?;
        let nodes =
            Tensor::vector_uninit(backend, capacity * NODES_PER_BLOCK, BufferUsages::STORAGE)?;
        let active_blocks = Tensor::vector_uninit(backend, capacity, BufferUsages::STORAGE)?;
        let scan_values = Tensor::vector_uninit(backend, capacity, BufferUsages::STORAGE)?;
        let active_blocks_snapshot = Tensor::vector(backend, [0u32], BufferUsages::STORAGE)?;
        let indirect_n_blocks_groups =
            Tensor::vector_uninit(backend, 3, BufferUsages::STORAGE | BufferUsages::INDIRECT)?;
        let indirect_n_g2p_p2g_groups = Tensor::vector_uninit(
            backend,
            3,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        )?;
        let debug = Tensor::vector(backend, [0u32, 0], BufferUsages::STORAGE)?;

        Ok(Self {
            cpu_meta,
            meta,
            prev_meta,
            hmap_entries,
            prev_hmap_entries,
            nodes,
            active_blocks,
            scan_values,
            active_blocks_snapshot,
            indirect_n_blocks_groups,
            indirect_n_g2p_p2g_groups,
            debug,
        })
    }

    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.meta, &mut self.prev_meta);
        std::mem::swap(&mut self.prev_hmap_entries, &mut self.hmap_entries);
    }
}
