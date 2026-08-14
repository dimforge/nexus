//! Grid-to-Particle (G2P) transfer kernel.
//!
//! This kernel transfers grid node velocities back to particles using APIC
//! (Affine Particle-In-Cell) interpolation. It handles CPIC compatibility
//! checks, computes velocity gradients for the affine matrix, and accumulates
//! rigid body velocities for particles near colliders.

use crate::PaddingExt;
use crate::grid::grid::*;
use crate::grid::kernel::*;
use crate::nexus_rbd_shaders::dynamics::{
    Velocity as BodyVelocity, WorldMassProperties as BodyMassProperties,
};
use crate::solver::boundary_condition::{BodyMaterials, BoundaryCondition};
use crate::solver::params::SimulationParams;
use crate::solver::particle::{Kinematics, Position};
use crate::{Matrix, PaddedMatrix, Vector};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::{
    macros::{spirv, spirv_bindgen},
    sync::workgroup_memory_barrier_with_group_sync,
};
use unroll::unroll_for_loops;
/*
 * Constants.
 */

#[cfg(feature = "dim2")]
const NUM_SHARED_CELLS: usize = 10 * 10;
#[cfg(feature = "dim3")]
const NUM_SHARED_CELLS: usize = 6 * 6 * 6;

const WORKGROUP_SIZE: u32 = 64;

/*
 * Global -> shared memory transfer.
 */

#[inline]
#[unroll_for_loops]
fn global_shared_memory_transfers<const USE_CPIC: bool>(
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    nodes: &[Node],
    tid: khal_std::glamx::UVec3,
    active_block_vid: BlockVirtualId,
    shared_nodes_vel: &mut [Vector; NUM_SHARED_CELLS],
    shared_nodes_vel_incompatible: &mut [Vector; NUM_SHARED_CELLS],
    shared_nodes_cdf: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    let base_block_pos_int = active_block_vid.id;

    #[cfg(feature = "dim2")]
    {
        for i_loop in 0..2 {
            for j_loop in 0..2 {
                if !((i_loop == 1 && tid.x > 1) || (j_loop == 1 && tid.y > 1)) {
                    let octant = UVec2::new(i_loop as u32, j_loop as u32);
                    let octant_hid = grid.find_block_header_id(
                        hmap_entries,
                        &BlockVirtualId {
                            id: base_block_pos_int + IVec2::new(octant.x as i32, octant.y as i32),
                        },
                    );
                    let shared_index = octant * 8 + UVec2::new(tid.x, tid.y);
                    let flat_shared_index =
                        flatten_shared_index(shared_index.x, shared_index.y) as usize;

                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        let tid_xy = UVec2::new(tid.x, tid.y);
                        let global_node_id = global_chunk_id.node_id(tid_xy);
                        let node = nodes.read(global_node_id.id as usize);
                        shared_nodes_vel.write(flat_shared_index, node.momentum_velocity);

                        if USE_CPIC {
                            shared_nodes_vel_incompatible
                                .write(flat_shared_index, node.momentum_velocity_incompatible);
                            shared_nodes_cdf.write(flat_shared_index, node.cdf);
                        }
                    } else {
                        shared_nodes_vel.write(flat_shared_index, Vector::ZERO);

                        if USE_CPIC {
                            shared_nodes_vel_incompatible.write(flat_shared_index, Vector::ZERO);
                            shared_nodes_cdf.write(flat_shared_index, NodeCdf::NONE);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "dim3")]
    {
        for i_loop in 0..2 {
            for j_loop in 0..2 {
                for k_loop in 0..2 {
                    if !((i_loop == 1 && tid.x > 1)
                        || (j_loop == 1 && tid.y > 1)
                        || (k_loop == 1 && tid.z > 1))
                    {
                        let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
                        let octant_hid = grid.find_block_header_id(
                            hmap_entries,
                            &BlockVirtualId::new(
                                base_block_pos_int
                                    + IVec3::new(octant.x as i32, octant.y as i32, octant.z as i32),
                            ),
                        );
                        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                        let shared_index = octant * 4 + tid_xyz;
                        let flat_shared_index =
                            flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)
                                as usize;

                        if octant_hid.id != NONE {
                            let global_chunk_id = octant_hid.physical_id();
                            let global_node_id = global_chunk_id.node_id(tid_xyz);
                            let node = nodes.read(global_node_id.id as usize);
                            shared_nodes_vel.write(flat_shared_index, node.momentum_velocity);
                            if USE_CPIC {
                                shared_nodes_vel_incompatible
                                    .write(flat_shared_index, node.momentum_velocity_incompatible);
                                shared_nodes_cdf.write(flat_shared_index, node.cdf);
                            }
                        } else {
                            shared_nodes_vel.write(flat_shared_index, Vector::ZERO);
                            if USE_CPIC {
                                shared_nodes_vel_incompatible
                                    .write(flat_shared_index, Vector::ZERO);
                                shared_nodes_cdf.write(flat_shared_index, NodeCdf::NONE);
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Per-particle G2P interpolation.
 */

#[inline]
#[allow(clippy::too_many_arguments)]
#[unroll_for_loops]
fn particle_g2p<const USE_CPIC: bool>(
    body_vels: &[BodyVelocity],
    body_mprops: &[BodyMassProperties],
    body_materials: &BodyMaterials,
    particles_pos: &[Position],
    particles_kin: &mut [Kinematics],
    particle_id: u32,
    cell_width: f32,
    _dt: f32,
    shared_nodes_vel: &[Vector; NUM_SHARED_CELLS],
    shared_nodes_vel_incompatible: &[Vector; NUM_SHARED_CELLS],
    shared_nodes_cdf: &[NodeCdf; NUM_SHARED_CELLS],
) {
    let mut rigid_vel = Vector::ZERO;
    let mut velocity = Vector::ZERO;
    let mut velocity_gradient = Matrix::ZERO;
    let mut vel_grad_det = 0.0f32;

    // G2P
    if particles_kin.at(particle_id as usize).enabled != 0 {
        let particle_pos = particles_pos.read(particle_id as usize);
        let particle_cdf = particles_kin.at(particle_id as usize).cdf;
        let boundary_friction = particles_kin.at(particle_id as usize).boundary_friction;

        let inv_d = QuadraticKernel::inv_d(cell_width);
        let ref_elt_pos_minus_particle_pos = particle_pos.dir_to_associated_grid_node(cell_width);
        let w = QuadraticKernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

        let assoc_cell_index_in_block =
            particle_pos.associated_cell_index_in_block_off_by_one(cell_width);

        #[cfg(feature = "dim2")]
        let packed_cell_index_in_block =
            flatten_shared_index(assoc_cell_index_in_block.x, assoc_cell_index_in_block.y);
        #[cfg(feature = "dim3")]
        let packed_cell_index_in_block = flatten_shared_index(
            assoc_cell_index_in_block.x,
            assoc_cell_index_in_block.y,
            assoc_cell_index_in_block.z,
        );

        for i in 0..27 {
            // For loop unrolling, use the fixed bound (the maximum one between 2D and 3D).
            if i < NBH_LEN {
                let shift = NBH_SHIFTS.read(i);
                let packed_shift = NBH_SHIFT_SHARED.read(i);
                let shared_id = (packed_cell_index_in_block + packed_shift) as usize;
                let mut cell_vel = shared_nodes_vel.read(shared_id);

                #[cfg(feature = "dim2")]
                let dpt = ref_elt_pos_minus_particle_pos
                    + Vec2::new(shift.x as f32, shift.y as f32) * cell_width;
                #[cfg(feature = "dim3")]
                let dpt = ref_elt_pos_minus_particle_pos
                    + Vec3::new(shift.x as f32, shift.y as f32, shift.z as f32) * cell_width;

                if USE_CPIC {
                    let cell_cdf = shared_nodes_cdf.read(shared_id);
                    let is_compatible = particle_cdf.affinity.is_compatible(cell_cdf.affinities);

                    if !is_compatible {
                        cell_vel = shared_nodes_vel_incompatible.read(shared_id);

                        if cell_cdf.closest_id != NONE {
                            let body_vel = body_vels.read(cell_cdf.closest_id as usize);
                            let body_com = body_mprops.at(cell_cdf.closest_id as usize).com;
                            let body_material = body_materials.mats[cell_cdf.closest_id as usize];
                            // Scale the collider's friction by this particle's own
                            // factor, so one surface can grip sand and let water
                            // slide. Applied here because CPIC resolves the
                            // boundary per particle, not per grid node.
                            let material = BoundaryCondition::new(
                                body_material.ty,
                                body_material.friction * boundary_friction,
                            );
                            let cell_center = dpt + particle_pos.pt;
                            let body_pt_vel = body_vel.velocity_at_point(body_com, cell_center);

                            cell_vel = body_pt_vel
                                + material
                                    .project_velocity(cell_vel - body_pt_vel, particle_cdf.normal);
                        }
                    }
                }

                #[cfg(feature = "dim2")]
                let weight = vec3_extract(w[0], shift.x) * vec3_extract(w[1], shift.y);
                #[cfg(feature = "dim3")]
                let weight = vec3_extract(w[0], shift.x)
                    * vec3_extract(w[1], shift.y)
                    * vec3_extract(w[2], shift.z);

                velocity += cell_vel * weight;
                velocity_gradient += outer_product(cell_vel, dpt) * (weight * inv_d);
                vel_grad_det += weight * inv_d * cell_vel.dot(dpt);
            }
        }

        if USE_CPIC {
            // Accumulate rigid body velocities for all affinity-linked colliders.
            for i_collider in 0..16 {
                if particle_cdf.affinity.bit(i_collider as u32) {
                    let body_vel = body_vels.read(i_collider);
                    let body_com = body_mprops.at(i_collider).com;
                    rigid_vel += body_vel.velocity_at_point(body_com, particle_pos.pt);
                }
            }
        }
    }

    if USE_CPIC {
        particles_kin.at_mut(particle_id as usize).cdf.rigid_vel = rigid_vel;
    }

    // Set the particle velocity, and store the velocity gradient into the affine matrix.
    // The rest will be dealt with in the particle update kernel(s).
    particles_kin.at_mut(particle_id as usize).affine =
        PaddedMatrix::add_padding(velocity_gradient);
    particles_kin.at_mut(particle_id as usize).vel_grad_det = vel_grad_det;
    particles_kin.at_mut(particle_id as usize).velocity = velocity;
}

/*
 * GPU entry points.
 */

/// GPU kernel: G2P transfer (2D).
///
/// Transfers grid node velocities back to particles using APIC interpolation.
/// Dispatched with one workgroup per active block.
#[unroll_for_loops]
pub fn gpu_g2p_generic<const USE_CPIC: bool>(
    block_id: khal_std::glamx::UVec3,
    tid: khal_std::glamx::UVec3,
    tid_flat: u32,
    params: &SimulationParams,
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    active_blocks: &[ActiveBlockHeader],
    nodes: &[Node],
    sorted_particle_ids: &[u32],
    particles_pos: &[Position],
    particles_kin: &mut [Kinematics],
    body_vels: &[BodyVelocity],
    body_mprops: &[BodyMassProperties],
    body_materials: &BodyMaterials,
    shared_nodes_vel: &mut [Vector; NUM_SHARED_CELLS],
    shared_nodes_vel_incompatible: &mut [Vector; NUM_SHARED_CELLS],
    shared_nodes_cdf: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    let bid = block_id.x;
    // Force copy of the virtual ID (naga bug workaround).
    let vid_ = active_blocks.at(bid as usize).virtual_id.id;
    let vid = BlockVirtualId::new(vid_);

    // Block -> shared memory transfer.
    global_shared_memory_transfers::<USE_CPIC>(
        grid,
        hmap_entries,
        nodes,
        tid,
        vid,
        shared_nodes_vel,
        shared_nodes_vel_incompatible,
        shared_nodes_cdf,
    );

    // Sync after shared memory initialization.
    workgroup_memory_barrier_with_group_sync();

    // Particle update. Runs g2p on shared memory only.
    let first_particle = active_blocks.at(bid as usize).first_particle;
    let max_particle_id = first_particle + active_blocks.at(bid as usize).num_particles;

    let num_block_particles = max_particle_id - first_particle;
    let max_iters = num_block_particles.div_ceil(WORKGROUP_SIZE);
    let mut sorted_particle_id = first_particle + tid_flat;
    for _ in 0..max_iters {
        if sorted_particle_id >= max_particle_id {
            break;
        }
        let particle_id = sorted_particle_ids.read(sorted_particle_id as usize);
        particle_g2p::<USE_CPIC>(
            body_vels,
            body_mprops,
            body_materials,
            particles_pos,
            particles_kin,
            particle_id,
            grid.cell_width,
            params.dt,
            shared_nodes_vel,
            shared_nodes_vel_incompatible,
            shared_nodes_cdf,
        );
        sorted_particle_id += WORKGROUP_SIZE;
    }
}

/*
 * Shared memory flatten helpers for G2P.
 * Note: different from P2G -- no shift subtraction since the truncated blocks
 * are in the higher-index quadrants.
 */

#[cfg(feature = "dim2")]
#[inline]
fn flatten_shared_index(x: u32, y: u32) -> u32 {
    x + y * 10
}

#[cfg(feature = "dim3")]
#[inline]
fn flatten_shared_index(x: u32, y: u32, z: u32) -> u32 {
    x + y * 6 + z * 6 * 6
}

/*
 * Outer product helper.
 */

#[cfg(feature = "dim2")]
#[inline]
fn outer_product(a: Vec2, b: Vec2) -> Mat2 {
    Mat2::from_cols(a * b.x, a * b.y)
}

#[cfg(feature = "dim3")]
#[inline]
fn outer_product(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/*
 * Specialized entry points.
 */
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
#[unroll_for_loops]
pub fn gpu_g2p(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimulationParams,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] nodes: &[Node],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sorted_particle_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] particles_kin: &mut [Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] body_vels: &[BodyVelocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] body_mprops: &[BodyMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] body_materials: &BodyMaterials,
    // Shared memory.
    #[spirv(workgroup)] shared_nodes_vel: &mut [Vector; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes_vel_incompatible: &mut [Vector; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes_cdf: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    gpu_g2p_generic::<false>(
        block_id,
        tid,
        tid_flat,
        params,
        grid,
        hmap_entries,
        active_blocks,
        nodes,
        sorted_particle_ids,
        particles_pos,
        particles_kin,
        body_vels,
        body_mprops,
        body_materials,
        shared_nodes_vel,
        shared_nodes_vel_incompatible,
        shared_nodes_cdf,
    )
}

#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
#[unroll_for_loops]
pub fn gpu_g2p_cpic(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimulationParams,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] nodes: &[Node],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sorted_particle_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] particles_kin: &mut [Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] body_vels: &[BodyVelocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] body_mprops: &[BodyMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] body_materials: &BodyMaterials,
    // Shared memory.
    #[spirv(workgroup)] shared_nodes_vel: &mut [Vector; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes_vel_incompatible: &mut [Vector; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes_cdf: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    gpu_g2p_generic::<true>(
        block_id,
        tid,
        tid_flat,
        params,
        grid,
        hmap_entries,
        active_blocks,
        nodes,
        sorted_particle_ids,
        particles_pos,
        particles_kin,
        body_vels,
        body_mprops,
        body_materials,
        shared_nodes_vel,
        shared_nodes_vel_incompatible,
        shared_nodes_cdf,
    )
}
