//! Grid update CDF kernel: runs collision detection on each grid node.
//!
//! This kernel detects collisions between the grid nodes and the collision shapes,
//! storing the resulting contact distance field (CDF) data in each node's `cdf` field.

use crate::grid::grid::*;
use crate::nexus_rbd_shaders::dynamics::{
    Velocity as BodyVelocity, WorldMassProperties as BodyMassProperties,
};
use crate::nexus_rbd_shaders::shapes::Shape;
use crate::solver::boundary_condition::{BodyMaterials, BoundaryCondition};
use crate::solver::params::SimulationParams;
use crate::{Pose, Vector};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use nexus_rbd_shaders::{MAX_FLT, PaddedVector};

struct Collision {
    normal: Vector,
    distance: f32,
    closest_id: usize,
}

/// Performs collision detection for a single grid node against all collision shapes.
///
/// Returns a `NodeCdf` with the closest collider distance, affinity bits, and collider ID.
#[inline]
#[cfg_attr(not(feature = "dim3"), allow(unused_variables))]
fn collide(
    collision_shapes: &[Shape],
    collision_shape_poses: &[Pose],
    collision_shape_indices: &[u32],
    collision_shape_vertices: &[PaddedVector],
    cell_width: f32,
    point: Vector,
) -> Collision {
    let dist_cap = Vector::splat(cell_width * 1.5);
    let mut collision = Collision {
        normal: Vector::ZERO,
        distance: MAX_FLT,
        closest_id: 0,
    };

    // NOTE: we iterate using a fixed upper bound to avoid dynamic buffer length queries
    //       that may not be available in all SPIR-V environments. The caller must ensure
    //       the shapes buffer length matches the actual number of shapes.
    for i in 0..collision_shapes.len() {
        let shape = collision_shapes.read(i);
        let shape_pose = collision_shape_poses.read(i);
        let shape_type = shape.shape_type();

        use crate::nexus_rbd_shaders::shapes::{SHAPE_TYPE_POLYLINE, SHAPE_TYPE_TRIMESH};

        #[cfg(feature = "dim3")]
        let (proj, valid) = if shape_type == SHAPE_TYPE_TRIMESH {
            let mesh = shape.to_trimesh();
            let local_pt = shape_pose.inverse() * point;
            let (mut proj, valid) = mesh.project_local_point(
                collision_shape_indices,
                collision_shape_vertices,
                local_pt,
                dist_cap.x,
            );
            // Transform the projected point back to world space.
            proj.point = shape_pose * proj.point;
            (proj, valid)
        } else {
            (shape.project_point_on_boundary(shape_pose, point), true)
        };

        #[cfg(feature = "dim2")]
        let (proj, valid) = (shape.project_point_on_boundary(shape_pose, point), true);

        if valid {
            let dpt = proj.point - point;
            let abs_dpt = dpt.abs();
            #[cfg(feature = "dim2")]
            let within_cap = abs_dpt.x <= dist_cap.x && abs_dpt.y <= dist_cap.y;
            #[cfg(feature = "dim3")]
            let within_cap =
                abs_dpt.x <= dist_cap.x && abs_dpt.y <= dist_cap.y && abs_dpt.z <= dist_cap.z;

            if proj.is_inside || within_cap {
                let sign = if proj.is_inside { -1.0 } else { 1.0 };
                let distance = dpt.length();
                let normal = dpt / (distance * -sign);
                let signed_dist = sign * distance;
                if signed_dist < collision.distance {
                    collision.distance = signed_dist;
                    collision.normal = normal;
                    collision.closest_id = i;
                }
            }
        }
    }

    collision
}

// TODO(PERF): merge with gpu_grid_update.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_grid_update_collide(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimulationParams,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] collision_shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] collision_shape_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    collision_shape_vertices: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] collision_shape_indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] body_vels: &[BodyVelocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] body_mprops: &[BodyMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] body_materials: &BodyMaterials,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] nodes: &mut [Node],
) {
    let dt = params.dt;
    let bid = block_id.x;
    let vid = active_blocks.at(bid as usize).virtual_id;

    let global_chunk_id = BlockHeaderId { id: bid }.physical_id();

    let global_node_id;
    let cell_pt;

    #[cfg(feature = "dim2")]
    {
        let tid_xy = UVec2::new(tid.x, tid.y);
        global_node_id = global_chunk_id.node_id(tid_xy);
        cell_pt = Vec2::new(
            (vid.id.x * 8 + tid.x as i32) as f32,
            (vid.id.y * 8 + tid.y as i32) as f32,
        ) * grid.cell_width;
    }

    #[cfg(feature = "dim3")]
    {
        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
        global_node_id = global_chunk_id.node_id(tid_xyz);
        cell_pt = Vec3::new(
            (vid.id.x * 4 + tid.x as i32) as f32,
            (vid.id.y * 4 + tid.y as i32) as f32,
            (vid.id.z * 4 + tid.z as i32) as f32,
        ) * grid.cell_width;
    }

    let global_id = global_node_id.id;
    let cell_width = grid.cell_width;
    let collision = collide(
        collision_shapes,
        collision_shape_poses,
        collision_shape_indices,
        collision_shape_vertices,
        cell_width,
        cell_pt,
    );

    if collision.distance != MAX_FLT {
        // Found a collision, apply the boundary condition.
        let body_vel = body_vels.at(collision.closest_id);
        let body_com = body_mprops.at(collision.closest_id).com;
        let body_vel_at_grid_pos = body_vel.velocity_at_point(body_com, cell_pt);
        let node_vel = nodes.at(global_id as usize).momentum_velocity;
        let body_material = body_materials.mats[collision.closest_id];
        let delta_vel = node_vel - body_vel_at_grid_pos;
        let normal_vel = delta_vel.dot(collision.normal);
        let margin = cell_width;

        if collision.distance <= margin {
            let corrected_vel =
                body_vel_at_grid_pos + body_material.project_velocity(delta_vel, collision.normal);

            nodes.at_mut(global_id as usize).momentum_velocity = corrected_vel;
        } else if -normal_vel * dt > collision.distance - margin {
            let excess_vel = (normal_vel + (collision.distance - margin) / dt) * collision.normal;
            nodes.at_mut(global_id as usize).momentum_velocity -= excess_vel;
        }
    }
}
