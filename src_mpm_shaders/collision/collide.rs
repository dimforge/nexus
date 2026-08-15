//! Collision detection for MPM grid nodes against collider shapes.

use crate::grid::grid::{AffinityBits, NONE, NodeCdf};
use crate::nexus_rbd_shaders::shapes::{SHAPE_TYPE_POLYLINE, SHAPE_TYPE_TRIMESH, Shape};
use crate::{Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// Tests a point against all collision shapes and returns the nearest CDF data.
///
/// For each non-polyline, non-trimesh shape, projects the point onto
/// the shape boundary and tracks distance, affinity bits, and closest collider ID.
pub fn collide(
    collision_shapes: &[Shape],
    collision_shape_poses: &[Pose],
    cell_width: f32,
    point: Vector,
) -> NodeCdf {
    let mut cdf = NodeCdf::NONE;

    let dist_cap = Vector::splat(cell_width * 1.5);

    // TODO: don't rely on the array length, e.g., if the user wants to
    //       preallocate the array to add more dynamically.
    let num_shapes = collision_shapes.len();

    for i in 0..num_shapes as u32 {
        // FIXME: figure out a way to support more than 16 colliders.
        let shape = collision_shapes.read(i as usize);
        let shape_pose = collision_shape_poses.read(i as usize);
        let shape_type = shape.shape_type();

        if shape_type != SHAPE_TYPE_POLYLINE && shape_type != SHAPE_TYPE_TRIMESH {
            let proj = shape.project_point_on_boundary(shape_pose, point);
            let dpt = proj.point - point;

            // Check if the projection is inside or within the distance cap.
            // `all(abs(dpt) <= dist_cap)` means every component of abs(dpt)
            // is <= the corresponding component of dist_cap.
            let abs_dpt = dpt.abs();
            #[cfg(feature = "dim2")]
            let within_cap = abs_dpt.x <= dist_cap.x && abs_dpt.y <= dist_cap.y;
            #[cfg(feature = "dim3")]
            let within_cap =
                abs_dpt.x <= dist_cap.x && abs_dpt.y <= dist_cap.y && abs_dpt.z <= dist_cap.z;

            if proj.is_inside || within_cap {
                let dist = dpt.length();
                // TODO: take is_inside into account to select the deepest
                //       penetration as the closest collider?
                if dist < cdf.distance {
                    cdf.closest_id = i;
                }
                cdf.distance = cdf.distance.min(dist);
                cdf.affinities.set_bit(i, proj.is_inside);
            }
        }
    }

    cdf
}
