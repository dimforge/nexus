//! Grid update CDF kernel: runs collision detection on each grid node.
//!
//! This kernel detects collisions between the grid nodes and the collision shapes,
//! storing the resulting contact distance field (CDF) data in each node's `cdf` field.

use crate::grid::grid::*;
use crate::nexus_rbd_shaders::dynamics::Velocity as BodyVelocity;
use crate::nexus_rbd_shaders::shapes::Shape;
use crate::{Pose, Vector};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Performs collision detection for a single grid node against all collision shapes.
///
/// Returns a `NodeCdf` with the closest collider distance, affinity bits, and collider ID.
#[inline]
fn collide(
    collision_shapes: &[Shape],
    collision_shape_poses: &[Pose],
    cell_width: f32,
    point: Vector,
) -> NodeCdf {
    let mut cdf = NodeCdf::NONE;

    let dist_cap = Vector::splat(cell_width * 1.5);

    // NOTE: we iterate using a fixed upper bound to avoid dynamic buffer length queries
    //       that may not be available in all SPIR-V environments. The caller must ensure
    //       the shapes buffer length matches the actual number of shapes.
    for i in 0..collision_shapes.len() {
        let shape = collision_shapes.read(i);
        let shape_pose = collision_shape_poses.read(i);
        let shape_type = shape.shape_type();

        use crate::nexus_rbd_shaders::shapes::{SHAPE_TYPE_POLYLINE, SHAPE_TYPE_TRIMESH};
        if shape_type != SHAPE_TYPE_POLYLINE && shape_type != SHAPE_TYPE_TRIMESH {
            let proj = shape.project_point_on_boundary(shape_pose, point);
            let dpt = proj.point - point;

            let abs_dpt = dpt.abs();
            #[cfg(feature = "dim2")]
            let within_cap = abs_dpt.x <= dist_cap.x && abs_dpt.y <= dist_cap.y;
            #[cfg(feature = "dim3")]
            let within_cap =
                abs_dpt.x <= dist_cap.x && abs_dpt.y <= dist_cap.y && abs_dpt.z <= dist_cap.z;

            if proj.is_inside || within_cap {
                let dist = dpt.length();
                if dist < cdf.distance {
                    cdf.closest_id = i as u32;
                    cdf.distance = dist;
                }
                cdf.affinities.set_bit(i as u32, proj.is_inside);
            }
        }
    }

    cdf
}

/// GPU kernel: grid update CDF.
///
/// For each active grid node, runs collision detection against all collision shapes
/// and writes the resulting `NodeCdf` (distance, affinity bits, closest collider ID).
///
/// Dispatched with one workgroup per active block, one thread per node in the block.
#[cfg(feature = "dim2")]
#[spirv_bindgen]
#[spirv(compute(threads(8, 8)))]
pub fn gpu_grid_update_cdf(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collision_shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] collision_shape_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] nodes: &mut [Node],
) {
    let bid = block_id.x;
    let vid = active_blocks.at(bid as usize).virtual_id;

    let global_chunk_id = BlockHeaderId { id: bid }.physical_id();
    let tid_xy = UVec2::new(tid.x, tid.y);
    let global_node_id = global_chunk_id.node_id(tid_xy);
    let cell_pos = Vec2::new(
        (vid.id.x * 8 + tid.x as i32) as f32,
        (vid.id.y * 8 + tid.y as i32) as f32,
    ) * grid.cell_width;

    let global_id = global_node_id.id;
    nodes.at_mut(global_id as usize).cdf = collide(
        collision_shapes,
        collision_shape_poses,
        grid.cell_width,
        cell_pos,
    );
}

/// GPU kernel: grid update CDF (3D version).
#[cfg(feature = "dim3")]
#[spirv_bindgen]
#[spirv(compute(threads(4, 4, 4)))]
pub fn gpu_grid_update_cdf(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collision_shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] collision_shape_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] nodes: &mut [Node],
) {
    let bid = block_id.x;
    let vid = active_blocks.at(bid as usize).virtual_id;

    let global_chunk_id = BlockHeaderId { id: bid }.physical_id();
    let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
    let global_node_id = global_chunk_id.node_id(tid_xyz);
    let cell_pos = Vec3::new(
        (vid.id.x * 4 + tid.x as i32) as f32,
        (vid.id.y * 4 + tid.y as i32) as f32,
        (vid.id.z * 4 + tid.z as i32) as f32,
    ) * grid.cell_width;

    let global_id = global_node_id.id;
    nodes.at_mut(global_id as usize).cdf = collide(
        collision_shapes,
        collision_shape_poses,
        grid.cell_width,
        cell_pos,
    );
}
