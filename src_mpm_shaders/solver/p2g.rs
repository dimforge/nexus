//! Particle-to-Grid (P2G) transfer kernel (scatter style).
//!
//! The core MPM kernel that transfers particle data (momentum, mass, affine matrix)
//! onto the grid nodes. Dispatched with one workgroup per active block.
//!
//! The CPIC (Compatible Particle-In-Cell) variant also handles affinity checks: particles
//! incompatible with a node (different side of a collider) contribute to the node's
//! `incompatible` momentum field instead, and impulses are accumulated for the rigid body
//! coupling.

use crate::grid::grid::*;
use crate::grid::kernel::QuadraticKernel;
use crate::nexus_rbd_shaders::dynamics::Velocity as BodyVelocity;
use crate::solver::boundary_condition::{BodyMaterials, BoundaryCondition};
use crate::solver::particle::{Kinematics, Position};
use crate::{AngVector, Matrix, PaddingExt, TWO_WAYS_COUPLING_ENABLED, Vector};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::{atomic_add_i32, workgroup_memory_barrier_with_group_sync};

/// Workgroup size: one thread per grid node of a block (8*8 in 2D, 4*4*4 in 3D).
const WORKGROUP_SIZE: usize = 64;

/// Integer impulse atomic struct for accumulating impulses across threads.
///
/// Uses integer atomics to avoid floating-point atomic limitations on GPU.
/// The COM (center of mass) is stored alongside to reduce binding count.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct IntegerImpulse {
    pub com: Vector,
    pub linear_x: i32,
    pub linear_y: i32,
    #[cfg(feature = "dim3")]
    pub linear_z: i32,
    #[cfg(feature = "dim3")]
    pub _padding_a: i32,
    #[cfg(feature = "dim2")]
    pub angular: i32,
    #[cfg(feature = "dim2")]
    pub _padding: i32,
    #[cfg(feature = "dim3")]
    pub angular_x: i32,
    #[cfg(feature = "dim3")]
    pub angular_y: i32,
    #[cfg(feature = "dim3")]
    pub angular_z: i32,
    #[cfg(feature = "dim3")]
    pub _padding_b: [i32; 2],
}

const FLOAT_TO_INT_FACTOR: f32 = 1e5;

/// Converts a float to an integer for atomic accumulation.
#[inline]
fn flt2int(flt: f32) -> i32 {
    (flt * FLOAT_TO_INT_FACTOR) as i32
}

/// Generic scatter-style P2G shared by the plain and CPIC entry points.
#[allow(clippy::too_many_arguments)]
pub fn gpu_p2g_generic<const USE_CPIC: bool>(
    block_id: khal_std::glamx::UVec3,
    tid: khal_std::glamx::UVec3,
    tid_flat: u32,
    grid: &Grid,
    active_blocks: &[ActiveBlockHeader],
    sorted_particle_ids: &[u32],
    particles_pos: &[Position],
    particles_kin: &[Kinematics],
    nodes: &mut [Node],
    body_vels: &[BodyVelocity],
    body_impulses: &mut [IntegerImpulse],
    body_materials: &BodyMaterials,
    // Shared memory: one chunk of particles, loaded cooperatively (one per thread).
    shared_pos: &mut [Position; WORKGROUP_SIZE],
    shared_vel_mass: &mut [(Vector, f32); WORKGROUP_SIZE],
    shared_affine: &mut [Matrix; WORKGROUP_SIZE],
    // NOTE: these are only read/written under CPIC, but rust-gpu can't coerce a workgroup
    // `&mut [T; N]` to `&mut [T]`, so both entry points pass fixed-size arrays.
    shared_affinities: &mut [AffinityBits; WORKGROUP_SIZE],
    shared_normals: &mut [Vector; WORKGROUP_SIZE],
    // Per-particle slab key: associated-cell coordinate along the slowest node
    // axis, relative to the block. The per-chunk culling bounds derive from it.
    shared_zkey: &mut [i32; WORKGROUP_SIZE],
) {
    let bid = block_id.x;
    let cell_width = grid.cell_width;
    let inv_cell_width = 1.0 / cell_width;

    // Force copy of the virtual ID (naga bug workaround, as in the original kernel).
    let vid = active_blocks.at(bid as usize).virtual_id.id;

    // This thread owns one grid node of the block.
    #[cfg(feature = "dim2")]
    let (local_cell, cell_pos) = {
        let lc = UVec2::new(tid.x, tid.y);
        let c = vid * 8 + IVec2::new(tid.x as i32, tid.y as i32);
        (lc, Vec2::new(c.x as f32, c.y as f32) * cell_width)
    };
    #[cfg(feature = "dim3")]
    let (local_cell, cell_pos) = {
        let lc = UVec3::new(tid.x, tid.y, tid.z);
        let c = vid * 4 + IVec3::new(tid.x as i32, tid.y as i32, tid.z as i32);
        (
            lc,
            Vec3::new(c.x as f32, c.y as f32, c.z as f32) * cell_width,
        )
    };

    let gid = BlockHeaderId { id: bid }
        .physical_id()
        .node_id(local_cell)
        .id as usize;

    // Per-node CDF data (computed by an earlier pass), needed only for CPIC.
    let node_affinity = nodes.at(gid).cdf.affinities;
    let collider_id = if USE_CPIC {
        nodes.at(gid).cdf.closest_id
    } else {
        NONE
    };

    let mut acc_mv = Vector::ZERO;
    let mut acc_mass = 0.0f32;
    let mut acc_mv_incompatible = Vector::ZERO;
    let mut acc_mass_incompatible = 0.0f32;
    let mut impulse = Vector::ZERO;
    #[cfg(feature = "dim2")]
    let mut ang_impulse: AngVector = 0.0;
    #[cfg(feature = "dim3")]
    let mut ang_impulse: AngVector = Vec3::ZERO;

    let first = active_blocks.at(bid as usize).first_particle;
    let num = active_blocks.at(bid as usize).num_particles_with_extras;
    let last = first + num;
    // End of the primaries segment: the sorted slab keys are ascending within the
    // primaries and within the extras, so chunk bounds need this boundary.
    let primaries_end = first + active_blocks.at(bid as usize).num_particles;

    // This thread's node slab along the sort axis (the slowest-varying node axis).
    #[cfg(feature = "dim2")]
    let node_slab = tid.y as i32;
    #[cfg(feature = "dim3")]
    let node_slab = tid.z as i32;

    // Number of workgroup-sized chunks. Capped on the web (bounded loop with a per-chunk
    // guard) so the workgroup barriers stay in uniform control flow; off the web, the
    // exact count is used and the guard is always true.
    // We set it to 128, which would be exceeded in quite degenerate situations (for example
    // if we end up with more than 28 particles per cell in the entire block and its neighborhood.
    // The typical particle count per cell is 8). We could make the limit bigger, but starting
    // at 256 we’ve seen it result in a measurable negative performance impact.
    #[cfg(feature = "web-compat")]
    let num_chunks = 128u32;
    #[cfg(not(feature = "web-compat"))]
    let num_chunks = num.div_ceil(WORKGROUP_SIZE as u32);

    for chunk in 0..num_chunks {
        let chunk_base = first + chunk * WORKGROUP_SIZE as u32;
        let active_chunk = chunk_base < last;

        // Wait for the previous chunk's readers before overwriting shared memory.
        workgroup_memory_barrier_with_group_sync();

        if active_chunk {
            let load_idx = chunk_base + tid_flat;
            let slot = tid_flat as usize;
            if load_idx < last {
                let pid = sorted_particle_ids.read(load_idx as usize);
                let pos = particles_pos.read(pid as usize);

                // Slab key along the sort axis, relative to the block. Must match
                // the sort's bucket key (same associated-cell rounding, same clamp
                // at -2) so the shared keys stay ascending within each segment.
                let assoc_cell = (pos.pt / cell_width).round() - Vector::ONE;
                #[cfg(feature = "dim2")]
                let zkey = (assoc_cell.y as i32 - vid.y * 8).max(-2);
                #[cfg(feature = "dim3")]
                let zkey = (assoc_cell.z as i32 - vid.z * 4).max(-2);
                shared_zkey.write(slot, zkey);

                let pkin = particles_kin.at(pid as usize);
                if pkin.enabled != 0 {
                    // The first component holds the raw velocity when CPIC is on (the
                    // impulse computation needs it) or the precomputed momentum
                    // (velocity * mass) otherwise, so the inner loop never recomputes it.
                    let vel_or_momentum = if USE_CPIC {
                        pkin.velocity
                    } else {
                        pkin.velocity * pkin.mass
                    };
                    shared_pos.write(slot, pos);
                    shared_vel_mass.write(slot, (vel_or_momentum, pkin.mass));
                    shared_affine.write(slot, pkin.affine.remove_padding());
                    if USE_CPIC {
                        shared_affinities.write(slot, pkin.cdf.affinity);
                        shared_normals.write(slot, pkin.cdf.normal);
                    }
                } else {
                    // Disabled particle: contribute nothing (mass = 0).
                    shared_pos.at_mut(slot).pt = Vector::ZERO;
                    shared_vel_mass.write(slot, (Vector::ZERO, 0.0));
                    shared_affine.write(slot, Matrix::ZERO);
                    if USE_CPIC {
                        shared_affinities.write(slot, AffinityBits::EMPTY);
                        shared_normals.write(slot, Vector::ZERO);
                    }
                }
            }
        }

        workgroup_memory_barrier_with_group_sync();

        if active_chunk {
            // `chunk_len` is uniform across the workgroup.
            let chunk_len = (last - chunk_base).min(WORKGROUP_SIZE as u32);

            // Per-chunk slab bounds, exact thanks to the within-block sort: the keys
            // are ascending within the primaries and within the extras, so the range
            // is given by the chunk's first/last key, plus the two values around the
            // primaries/extras boundary if the chunk straddles it.
            let mut zmin = shared_zkey.read(0);
            let mut zmax = shared_zkey.read((chunk_len - 1) as usize);
            if primaries_end > chunk_base && primaries_end < chunk_base + chunk_len {
                let b = (primaries_end - chunk_base) as usize;
                zmin = zmin.min(shared_zkey.read(b));
                zmax = zmax.max(shared_zkey.read(b - 1));
            }

            // A particle with slab key `a` only influences nodes in slabs [a, a + 2]:
            // skip the whole chunk if this thread's node slab is outside the chunk's
            // dilated slab range. When that holds for every thread of a warp (e.g. a
            // chunk of extras below the block vs. the upper-half warp), the warp skips
            // the chunk entirely.
            let in_range = node_slab >= zmin && node_slab <= zmax + 2;
            let culled_len = if in_range { chunk_len } else { 0 };
            for p in 0..culled_len {
                let p = p as usize;
                let pos = shared_pos.read(p);
                // `vel_or_momentum` is the precomputed momentum (non-CPIC) or the raw
                // velocity (CPIC); see the chunk load above.
                let (vel_or_momentum, mass) = shared_vel_mass.read(p);
                let dpt = cell_pos - pos.pt;

                #[cfg(feature = "dim2")]
                let weight = QuadraticKernel::eval(dpt.x * inv_cell_width)
                    * QuadraticKernel::eval(dpt.y * inv_cell_width);
                #[cfg(feature = "dim3")]
                let weight = QuadraticKernel::eval(dpt.x * inv_cell_width)
                    * QuadraticKernel::eval(dpt.y * inv_cell_width)
                    * QuadraticKernel::eval(dpt.z * inv_cell_width);

                // The quadratic kernel is exactly zero outside the 3-node support, the
                // common case for the dense node x particle cross product.
                if weight != 0.0 {
                    let affine = shared_affine.at(p);
                    let momentum = if USE_CPIC {
                        vel_or_momentum * mass
                    } else {
                        vel_or_momentum
                    };
                    let vel_contribution = (affine * dpt + momentum) * weight;
                    let mass_contribution = mass * weight;

                    if USE_CPIC {
                        let particle_affinity = shared_affinities.read(p);
                        if !particle_affinity.is_compatible(node_affinity) {
                            if TWO_WAYS_COUPLING_ENABLED && collider_id != NONE {
                                let particle_normal = shared_normals.read(p);
                                let body_vel = body_vels.read(collider_id as usize);
                                let body_com = body_impulses.at(collider_id as usize).com;
                                let body_material = body_materials.mats[collider_id as usize];
                                let cell_center = cell_pos;
                                let body_pt_vel = body_vel.velocity_at_point(body_com, cell_center);
                                let particle_ghost_vel = body_pt_vel
                                    + body_material.project_velocity(
                                        vel_or_momentum - body_pt_vel,
                                        particle_normal,
                                    );
                                let delta_impulse =
                                    (vel_or_momentum - particle_ghost_vel) * (weight * mass);
                                let lever_arm = body_com - cell_center;

                                #[cfg(feature = "dim2")]
                                {
                                    ang_impulse +=
                                        delta_impulse.dot(Vec2::new(lever_arm.y, -lever_arm.x));
                                }
                                #[cfg(feature = "dim3")]
                                {
                                    ang_impulse += delta_impulse.cross(lever_arm);
                                }

                                impulse += delta_impulse;
                            }

                            acc_mv_incompatible += vel_contribution;
                            acc_mass_incompatible += mass_contribution;
                        } else {
                            acc_mv += vel_contribution;
                            acc_mass += mass_contribution;
                        }
                    } else {
                        acc_mv += vel_contribution;
                        acc_mass += mass_contribution;
                    }
                }
            }
        }
    }

    // Write the node state to global memory (one write per node, no atomics).
    nodes.at_mut(gid).momentum_velocity = acc_mv;
    nodes.at_mut(gid).mass = acc_mass;
    nodes.at_mut(gid).momentum_velocity_incompatible = acc_mv_incompatible;
    nodes.at_mut(gid).mass_incompatible = acc_mass_incompatible;

    if USE_CPIC {
        // Apply the accumulated impulse to the closest body using integer atomics.
        if TWO_WAYS_COUPLING_ENABLED && collider_id != NONE {
            let ci = collider_id as usize;
            #[cfg(feature = "dim2")]
            {
                atomic_add_i32(&mut body_impulses.at_mut(ci).linear_x, flt2int(impulse.x));
                atomic_add_i32(&mut body_impulses.at_mut(ci).linear_y, flt2int(impulse.y));
                atomic_add_i32(&mut body_impulses.at_mut(ci).angular, flt2int(ang_impulse));
            }
            #[cfg(feature = "dim3")]
            {
                atomic_add_i32(&mut body_impulses.at_mut(ci).linear_x, flt2int(impulse.x));
                atomic_add_i32(&mut body_impulses.at_mut(ci).linear_y, flt2int(impulse.y));
                atomic_add_i32(&mut body_impulses.at_mut(ci).linear_z, flt2int(impulse.z));
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular_x,
                    flt2int(ang_impulse.x),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular_y,
                    flt2int(ang_impulse.y),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular_z,
                    flt2int(ang_impulse.z),
                );
            }
        }
    }
}

/*
 * GPU entry points.
 */

/// GPU kernel: scatter-style P2G transfer (no CPIC).
///
/// Dispatched with one workgroup per active block.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_p2g(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_particle_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] particles_kin: &[Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] nodes: &mut [Node],
    #[spirv(workgroup)] shared_pos: &mut [Position; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_vel_mass: &mut [(Vector, f32); WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_affine: &mut [Matrix; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_affinities: &mut [AffinityBits; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_normals: &mut [Vector; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_zkey: &mut [i32; WORKGROUP_SIZE],
) {
    gpu_p2g_generic::<false>(
        block_id,
        tid,
        tid_flat,
        grid,
        active_blocks,
        sorted_particle_ids,
        particles_pos,
        particles_kin,
        nodes,
        &[],
        &mut [],
        &BodyMaterials::EMPTY,
        shared_pos,
        shared_vel_mass,
        shared_affine,
        shared_affinities,
        shared_normals,
        shared_zkey,
    );
}

/// GPU kernel: scatter-style P2G transfer with CPIC rigid-body coupling.
///
/// Dispatched with one workgroup per active block.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_p2g_cpic(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_particle_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] particles_kin: &[Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] nodes: &mut [Node],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] body_vels: &[BodyVelocity],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] body_materials: &BodyMaterials,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)]
    body_impulses: &mut [IntegerImpulse],
    #[spirv(workgroup)] shared_pos: &mut [Position; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_vel_mass: &mut [(Vector, f32); WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_affine: &mut [Matrix; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_affinities: &mut [AffinityBits; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_normals: &mut [Vector; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_zkey: &mut [i32; WORKGROUP_SIZE],
) {
    gpu_p2g_generic::<true>(
        block_id,
        tid,
        tid_flat,
        grid,
        active_blocks,
        sorted_particle_ids,
        particles_pos,
        particles_kin,
        nodes,
        body_vels,
        body_impulses,
        body_materials,
        shared_pos,
        shared_vel_mass,
        shared_affine,
        shared_affinities,
        shared_normals,
        shared_zkey,
    );
}
