module cone;

import nexus.aliases;
import nexus.queries.ray;
import nexus.queries.projection;
import nexus.shapes.segment;


/// A cone, defined by its radius.
public struct Cone {
    /// The cone’s principal axis.
    public float half_height;
    /// The cone’s radius.
    public float radius;

#if DIM == 3
    /// Projects a point on a cone.
    ///
    /// If the point is inside the cone, the point itself is returned.
    public func project_local_point(pt: Vect) -> Vect {
        // Project on the basis.
        let planar_dist_from_basis_center = length(pt.xz);
        let dir_from_basis_center = select(
            planar_dist_from_basis_center > 0.0,
            pt.xz / planar_dist_from_basis_center,
            float2(1.0, 0.0),
        );

        let projection_on_basis = float3(pt.x, -this.half_height, pt.z);

        if (pt.y < -this.half_height && planar_dist_from_basis_center <= this.radius) {
            // The projection is on the basis.
            return projection_on_basis;
        }

        // Project on the basis circle.
        let proj2d = dir_from_basis_center * this.radius;
        let projection_on_basis_circle = float3(proj2d[0], -this.half_height, proj2d[1]);

        // Project on the conic side.
        // TODO: we could solve this in 2D using the plane passing through the cone axis and the conic_side_segment to save some computation.
        let apex_point = float3(0.0, this.half_height, 0.0);
        let conic_side_segment = Segment(apex_point, projection_on_basis_circle);
        let conic_side_segment_dir = conic_side_segment.b - conic_side_segment.a;
        let proj = conic_side_segment.project_local_point(pt);

        let apex_to_basis_center = float3(0.0, -2.0 * this.half_height, 0.0);

        // Now determine if the point is inside of the this.
        if (pt.y >= -this.half_height
            && pt.y <= this.half_height
            && dot(
                   cross(conic_side_segment_dir, pt - apex_point),
                   cross(conic_side_segment_dir, apex_to_basis_center)
               ) >= 0.0)
        {
            // We are inside of the cone.
            return pt;
        } else {
            // We are outside of the cone, return the computed segment projection.
            return proj;
        }
    }

    /// Projects a point on a transformed cone.
    ///
    /// If the point is inside the cone, the point itself is returned.
    public func project_point(pose: Pose, pt: Vect) -> Vect {
        let local_pt = pose.inv_mul_pt(pt);
        return pose.mul_pt(this.project_local_point(local_pt));
    }


    /// Projects a point on the boundary of a cone.
    public func project_local_point_on_boundary(pt: Vect) -> ProjectionResult {
        // Project on the basis.
        let planar_dist_from_basis_center = length(pt.xz);
        let dir_from_basis_center = select(
            planar_dist_from_basis_center > 0.0,
            pt.xz / planar_dist_from_basis_center,
            float2(1.0, 0.0),
        );

        let projection_on_basis = float3(pt.x, -this.half_height, pt.z);

        if (pt.y < -this.half_height && planar_dist_from_basis_center <= this.radius) {
            // The projection is on the basis.
            return ProjectionResult(projection_on_basis, false);
        }

        // Project on the basis circle.
        let proj2d = dir_from_basis_center * this.radius;
        let projection_on_basis_circle = float3(proj2d[0], -this.half_height, proj2d[1]);

        // Project on the conic side.
        // TODO: we could solve this in 2D using the plane passing through the cone axis and the conic_side_segment to save some computation.
        let apex_point = float3(0.0, this.half_height, 0.0);
        let conic_side_segment = Segment(apex_point, projection_on_basis_circle);
        let conic_side_segment_dir = conic_side_segment.b - conic_side_segment.a;
        let proj = conic_side_segment.project_local_point(pt);

        let apex_to_basis_center = float3(0.0, -2.0 * this.half_height, 0.0);

        // Now determine if the point is inside of the this.
        if (pt.y >= -this.half_height
            && pt.y <= this.half_height
            && dot(
                   cross(conic_side_segment_dir, pt - apex_point),
                   cross(conic_side_segment_dir, apex_to_basis_center)
               ) >= 0.0)
        {
            // We are inside of the cone, so the correct projection is
            // either on the basis of the cone, or on the conic side.
            let pt_to_proj = proj - pt;
            let pt_to_basis_proj = projection_on_basis - pt;
            if (dot(pt_to_proj, pt_to_proj) > dot(pt_to_basis_proj, pt_to_basis_proj)) {
                return ProjectionResult(projection_on_basis, true);
            } else {
                return ProjectionResult(proj, true);
            }
        } else {
            // We are outside of the cone, return the computed segment projection as-is.
            return ProjectionResult(proj, false);
        }
    }

    /// Project a point of a transformed cone’s boundary.
    ///
    /// If the point is inside of the box, it will be projected on its boundary but
    /// `ProjectionResult::is_inside` will be set to `true`.
    public func project_point_on_boundary(pose: Pose, pt: Vect) -> ProjectionResult {
        let local_pt = pose.inv_mul_pt(pt);
        var result = this.project_local_point_on_boundary(local_pt);
        result.point = pose.mul_pt(result.point);
        return result;
    }
#endif
}
