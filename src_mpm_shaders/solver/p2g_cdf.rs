//! Particle-to-Grid CDF (Contact Distance Field) transfer kernel (scatter style).
//!
//! Transfers collision primitives (segments in 2D, triangles in 3D) from rigid body
//! surface particles onto nearby grid nodes, computing the signed distance and the
//! CPIC affinity bits. Dispatched with one workgroup per active block.

use crate::grid::grid::*;
use crate::solver::particle::{Position, RigidParticleIndices};
use crate::{IVector, Vector, abs};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
use nexus_rbd_shaders::PaddedVector;

/// Workgroup size: one thread per grid node of a block (8*8 in 2D, 4*4*4 in 3D).
const WORKGROUP_SIZE: usize = 64;

/// A collision primitive stored in shared memory.
/// In 2D: segment (two endpoints). In 3D: triangle (three vertices).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct SharedPrimitive {
    a: Vector,
    b: Vector,
    #[cfg(feature = "dim3")]
    c: Vector,
}

/*
 * Segment projection helper (2D).
 */

#[cfg(feature = "dim2")]
#[inline]
fn project_local_point_on_segment(a: Vec2, b: Vec2, point: Vec2) -> Vec2 {
    let ab = b - a;
    let ap = point - a;
    let ab_sqnorm = ab.dot(ab);

    if ab_sqnorm < 1.0e-10 {
        return a;
    }

    let t = ap.dot(ab) / ab_sqnorm;
    let t = t.clamp(0.0, 1.0);
    a + ab * t
}

/*
 * GPU entry points.
 */

/// GPU kernel: P2G CDF transfer.
///
/// Dispatched with one workgroup per active block.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_p2g_cdf(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_rigid_particle_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] collider_vertices: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    rigid_particle_indices: &[RigidParticleIndices],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] nodes: &mut [Node],
    // Shared memory: one chunk of rigid particles, loaded cooperatively (one per thread).
    #[spirv(workgroup)] shared_primitives: &mut [SharedPrimitive; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_collider_ids: &mut [u32; WORKGROUP_SIZE],
    #[spirv(workgroup)] shared_assoc_cells: &mut [IVector; WORKGROUP_SIZE],
) {
    let bid = block_id.x;
    let cell_width = grid.cell_width;

    // Force copy of the virtual ID (naga bug workaround, as in the original kernel).
    let vid = active_blocks.at(bid as usize).virtual_id.id;

    // This thread owns one grid node of the block.
    #[cfg(feature = "dim2")]
    let (local_cell, cell_int) = {
        let lc = UVec2::new(tid.x, tid.y);
        (lc, vid * 8 + IVec2::new(tid.x as i32, tid.y as i32))
    };
    #[cfg(feature = "dim3")]
    let (local_cell, cell_int) = {
        let lc = UVec3::new(tid.x, tid.y, tid.z);
        (
            lc,
            vid * 4 + IVec3::new(tid.x as i32, tid.y as i32, tid.z as i32),
        )
    };
    #[cfg(feature = "dim2")]
    let cell_pos = Vec2::new(cell_int.x as f32, cell_int.y as f32) * cell_width;
    #[cfg(feature = "dim3")]
    let cell_pos = Vec3::new(cell_int.x as f32, cell_int.y as f32, cell_int.z as f32) * cell_width;

    let gid = BlockHeaderId { id: bid }
        .physical_id()
        .node_id(local_cell)
        .id as usize;

    // Merge into the CDF computed by the analytical-shapes pass (`grid_update_cdf`).
    let mut node_cdf = nodes.at(gid).cdf;

    let first = active_blocks.at(bid as usize).first_rigid_particle;
    let num = active_blocks
        .at(bid as usize)
        .num_rigid_particles_with_extras;
    let last = first + num;

    // Number of workgroup-sized chunks. Capped on the web (bounded loop with a per-chunk
    // guard) so the workgroup barriers stay in uniform control flow; off the web, the
    // exact count is used and the guard is always true. Rigid particles are surface
    // samples spaced roughly one cell apart, so 128 chunks (8192 particles per block)
    // is far beyond anything reachable in practice.
    #[cfg(feature = "web-compat")]
    let num_chunks = 16u32;
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
                let pid = sorted_rigid_particle_ids.read(load_idx as usize);
                let rigid_idx = rigid_particle_indices.read(pid as usize);
                shared_collider_ids.write(slot, rigid_idx.collider);

                #[cfg(feature = "dim2")]
                shared_primitives.write(
                    slot,
                    SharedPrimitive {
                        a: collider_vertices.read(rigid_idx.segment.x as usize).0,
                        b: collider_vertices.read(rigid_idx.segment.y as usize).0,
                    },
                );
                #[cfg(feature = "dim3")]
                shared_primitives.write(
                    slot,
                    SharedPrimitive {
                        a: collider_vertices.read(rigid_idx.triangle.x as usize).0,
                        b: collider_vertices.read(rigid_idx.triangle.y as usize).0,
                        c: collider_vertices.read(rigid_idx.triangle.z as usize).0,
                    },
                );

                // The cell the particle is associated with (off-by-one convention): the
                // primitive only influences nodes in the 3-cell range starting there.
                // NOTE: must divide (not multiply by the inverse) to round exactly like
                // the sort kernels' block association.
                let assoc =
                    (rigid_particles_pos.read(pid as usize).pt / cell_width).round() - Vector::ONE;
                #[cfg(feature = "dim2")]
                shared_assoc_cells.write(slot, IVec2::new(assoc.x as i32, assoc.y as i32));
                #[cfg(feature = "dim3")]
                shared_assoc_cells.write(
                    slot,
                    IVec3::new(assoc.x as i32, assoc.y as i32, assoc.z as i32),
                );
            }
        }

        workgroup_memory_barrier_with_group_sync();

        if active_chunk {
            // `chunk_len` is uniform across the workgroup.
            let chunk_len = (last - chunk_base).min(WORKGROUP_SIZE as u32);
            for p in 0..chunk_len {
                let p = p as usize;

                // Restrict each primitive's influence to the quadratic-stencil-shaped
                // 3-cell neighbourhood of its associated cell, matching the original
                // gather implementation.
                let shift = cell_int - shared_assoc_cells.read(p);
                #[cfg(feature = "dim2")]
                let in_range = shift.x >= 0 && shift.x <= 2 && shift.y >= 0 && shift.y <= 2;
                #[cfg(feature = "dim3")]
                let in_range = shift.x >= 0
                    && shift.x <= 2
                    && shift.y >= 0
                    && shift.y <= 2
                    && shift.z >= 0
                    && shift.z <= 2;

                if in_range {
                    let collider_id = shared_collider_ids.read(p);
                    let primitive = shared_primitives.read(p);

                    #[cfg(feature = "dim2")]
                    {
                        // Project on Segment.
                        let proj =
                            project_local_point_on_segment(primitive.a, primitive.b, cell_pos);
                        // Check if this is a valid projection (not clamped to an endpoint).
                        let not_at_a = proj.x != primitive.a.x || proj.y != primitive.a.y;
                        let not_at_b = proj.x != primitive.b.x || proj.y != primitive.b.y;
                        if not_at_a && not_at_b {
                            let dpt = cell_pos - proj;
                            let distance = dpt.length();
                            let ab = primitive.b - primitive.a;
                            let sign = dpt.dot(Vec2::new(-ab.y, ab.x)) < 0.0;
                            node_cdf.affinities.set_bit(collider_id, sign);

                            if distance < node_cdf.distance {
                                node_cdf.distance = distance;
                                node_cdf.closest_id = collider_id;
                            }
                        }
                    }

                    #[cfg(feature = "dim3")]
                    {
                        // Project on Triangle.
                        let ap = cell_pos - primitive.a;
                        let bp = cell_pos - primitive.b;
                        let cp = cell_pos - primitive.c;
                        let ab = primitive.b - primitive.a;
                        let ac = primitive.c - primitive.a;
                        let bc = primitive.c - primitive.b;
                        let n = ab.cross(ac);
                        let n_length = n.length();

                        if n_length != 0.0
                            && ab.cross(n).dot(ap) <= 0.0
                            && bc.cross(n).dot(bp) <= 0.0
                            && ac.cross(n).dot(cp) >= 0.0
                        // Positive sign due to `ac` instead of `ca`.
                        {
                            // Valid projection on the face interior.
                            let signed_dist = n.dot(ap) / n_length;
                            let distance = abs(signed_dist);
                            node_cdf.affinities.set_bit(collider_id, signed_dist < 0.0);

                            if distance < node_cdf.distance {
                                node_cdf.distance = distance;
                                node_cdf.closest_id = collider_id;
                            }
                        }
                    }
                }
            }
        }
    }

    // Write the node cdf to global memory.
    nodes.at_mut(gid).cdf = node_cdf;
}
