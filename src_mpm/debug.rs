//! Debug readback of the MPM state, for visualization and tests.

use crate::mpm_shaders::IVector;
use crate::mpm_shaders::grid::grid::{ActiveBlockHeader, Grid, Node};
use crate::mpm_shaders::solver::particle::{Kinematics, ParticleProperties, Position};
use crate::pipeline::MpmState;
use khal::backend::{Backend, GpuBackend};
use nexus_rbd::math::Vector;
use vortx::tensor::Tensor;

/// Nodes along each axis of a grid block (8x8 in 2D, 4x4x4 in 3D).
#[cfg(feature = "dim2")]
pub const BLOCK_SIZE: u32 = 8;
/// Nodes along each axis of a grid block.
#[cfg(feature = "dim3")]
pub const BLOCK_SIZE: u32 = 4;

/// Nodes per block, i.e. the stride between two blocks in the node buffer.
const NUM_CELL_PER_BLOCK: u32 = BLOCK_SIZE.pow(crate::mpm_shaders::DIM);

/// One MPM particle read back by [`MpmState::debug_particles`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DebugParticle {
    /// World-space position.
    pub position: Vector,
    /// World-space velocity.
    pub velocity: Vector,
    /// The particle's mass.
    pub mass: f32,
    /// The particle's initial radius.
    pub radius: f32,
    /// Render group the particle belongs to (carries no physics).
    pub group_id: u32,
    /// Whether the particle takes part in the transfers at all.
    pub enabled: bool,
    /// Whether the particle is pinned in place.
    pub fixed: bool,
    /// Outward normal of the closest collider surface.
    /// Only set with CPIC, and zero when no collider is near.
    pub cdf_normal: Vector,
    /// Signed distance to that surface (negative inside the collider).
    pub cdf_distance: f32,
    /// CPIC affinity bits: one (affinity, sign) pair per nearby collider.
    pub cdf_affinity: u32,
}

/// One allocated block of the sparse grid, read back by [`MpmState::debug_grid`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DebugGridBlock {
    /// The block coordinate in the virtual grid.
    pub virtual_id: IVector,
    /// World-space corner of the first node of the block.
    pub mins: Vector,
    /// The opposite corner, one cell after the last node of the block.
    pub maxs: Vector,
    /// Number of particles whose main block is this one.
    pub num_particles: u32,
    /// Number of particles touching the block, including the ones from a neighbor block.
    /// If `num_particles` is zero, the block exists only because of these.
    pub num_particles_with_extras: u32,
}

/// One grid node with mass, read back by [`MpmState::debug_grid`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DebugGridNode {
    /// World-space position of the node.
    pub position: Vector,
    /// Velocity after the last grid update.
    pub velocity: Vector,
    /// Mass given to the node by the last P2G.
    pub mass: f32,
    /// Signed distance to the closest collider surface, or a huge value if none is near.
    pub cdf_distance: f32,
    /// CPIC affinity bits of the node.
    pub cdf_affinity: u32,
}

/// The sparse grid: its allocated blocks and the nodes with mass.
#[derive(Clone, Debug, Default)]
pub struct DebugGrid {
    /// Distance between two neighbor nodes.
    pub cell_width: f32,
    /// All the blocks allocated by the last sort.
    pub blocks: Vec<DebugGridBlock>,
    /// The nodes of these blocks that have mass. Empty nodes are skipped.
    pub nodes: Vec<DebugGridNode>,
}

impl MpmState {
    /// Debug: reads the particles back from the GPU, in world space.
    /// Slow: copies all the particle buffers to the CPU.
    pub async fn debug_particles(&self, backend: &GpuBackend) -> Vec<DebugParticle> {
        let len = self.particles.len();
        let positions = read_prefix::<Position>(backend, &self.particles.positions, len).await;
        let kinematics = read_prefix::<Kinematics>(backend, &self.particles.kinematics, len).await;
        let properties =
            read_prefix::<ParticleProperties>(backend, &self.particles.properties, len).await;

        if positions.len() < len || kinematics.len() < len || properties.len() < len {
            return Vec::new();
        }

        (0..len)
            .map(|i| {
                let kin = &kinematics[i];
                let props = &properties[i];
                DebugParticle {
                    position: positions[i].pt,
                    velocity: kin.velocity,
                    mass: kin.mass,
                    radius: props.init_radius,
                    group_id: props.group_id,
                    enabled: kin.enabled != 0,
                    fixed: props.fixed != 0,
                    cdf_normal: kin.cdf.normal,
                    cdf_distance: kin.cdf.signed_distance,
                    cdf_affinity: kin.cdf.affinity.0,
                }
            })
            .collect()
    }

    /// Debug: reads the sparse grid back from the GPU, in world space.
    /// Only the blocks allocated by the last sort are converted. Slow.
    pub async fn debug_grid(&self, backend: &GpuBackend) -> DebugGrid {
        // `MpmPipeline::step` swaps the buffers before sorting, so `grid.meta` is the last one.
        let Ok(meta) = backend.slow_read_vec::<Grid>(self.grid.meta.buffer()).await else {
            return DebugGrid::default();
        };
        let Some(meta) = meta.first().copied() else {
            return DebugGrid::default();
        };

        let cell_width = meta.cell_width;
        let num_blocks = meta.num_active_blocks.min(meta.capacity) as usize;
        let headers =
            read_prefix::<ActiveBlockHeader>(backend, &self.grid.active_blocks, num_blocks).await;
        let nodes = read_prefix::<Node>(
            backend,
            &self.grid.nodes,
            num_blocks * NUM_CELL_PER_BLOCK as usize,
        )
        .await;

        let block_extent = Vector::splat(BLOCK_SIZE as f32 * cell_width);
        let mut blocks = Vec::with_capacity(headers.len());
        let mut debug_nodes = Vec::new();

        for (bid, header) in headers.iter().enumerate() {
            let mins = block_origin(header.virtual_id.id, cell_width);
            blocks.push(DebugGridBlock {
                virtual_id: header.virtual_id.id,
                mins,
                maxs: mins + block_extent,
                num_particles: header.num_particles,
                num_particles_with_extras: header.num_particles_with_extras,
            });

            let first_node = bid * NUM_CELL_PER_BLOCK as usize;
            for shift in 0..NUM_CELL_PER_BLOCK as usize {
                let Some(node) = nodes.get(first_node + shift) else {
                    continue;
                };
                if node.mass <= 0.0 {
                    continue;
                }
                debug_nodes.push(DebugGridNode {
                    position: mins + node_shift(shift as u32) * cell_width,
                    velocity: node.momentum_velocity,
                    mass: node.mass,
                    cdf_distance: node.cdf.distance,
                    cdf_affinity: node.cdf.affinities.0,
                });
            }
        }

        DebugGrid {
            cell_width,
            blocks,
            nodes: debug_nodes,
        }
    }
}

/// World-space position of the first node of a block (like `gpu_grid_update`).
fn block_origin(virtual_id: IVector, cell_width: f32) -> Vector {
    #[cfg(feature = "dim2")]
    let origin = Vector::new(virtual_id.x as f32, virtual_id.y as f32);
    #[cfg(feature = "dim3")]
    let origin = Vector::new(
        virtual_id.x as f32,
        virtual_id.y as f32,
        virtual_id.z as f32,
    );
    origin * (BLOCK_SIZE as f32 * cell_width)
}

/// Offset, in cells, of a node from the first node of its block.
fn node_shift(shift_in_block: u32) -> Vector {
    #[cfg(feature = "dim2")]
    {
        Vector::new(
            (shift_in_block % BLOCK_SIZE) as f32,
            (shift_in_block / BLOCK_SIZE) as f32,
        )
    }
    #[cfg(feature = "dim3")]
    {
        Vector::new(
            (shift_in_block % BLOCK_SIZE) as f32,
            ((shift_in_block / BLOCK_SIZE) % BLOCK_SIZE) as f32,
            (shift_in_block / (BLOCK_SIZE * BLOCK_SIZE)) as f32,
        )
    }
}

/// Reads the first `len` elements of `tensor`, or an empty vector on failure.
async fn read_prefix<T: bytemuck::Pod + Default + Send + Sync>(
    backend: &GpuBackend,
    tensor: &Tensor<T>,
    len: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); len.min(tensor.len() as usize)];
    if out.is_empty() {
        return out;
    }
    match backend.slow_read_buffer(tensor.buffer(), &mut out).await {
        Ok(()) => out,
        Err(_) => Vec::new(),
    }
}
