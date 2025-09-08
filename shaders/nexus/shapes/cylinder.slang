module cylinder;

import nexus.aliases;
import nexus.queries.ray;
import nexus.queries.projection;

/// A cylinder, defined by its radius.
public struct Cylinder {
    /// The cylinder’s principal axis.
    public float half_height;
    /// The cylinder’s radius.
    public float radius;

#if DIM == 3
    /// Projects a point on a cylinder.
    ///
    /// If the point is inside the cylinder, the point itself is returned.
    public func project_local_point(pt: Vect) -> Vect {
        // Project on the basis.
        let planar_dist_from_basis_center = length(pt.xz);
        let dir_from_basis_center = select(
            planar_dist_from_basis_center > 0.0,
            pt.xz / planar_dist_from_basis_center,
            float2(1.0, 0.0),
        );

        let proj2d = dir_from_basis_center * this.radius;

        // PERF: reduce branching
        if (pt.y >= -this.half_height
            && pt.y <= this.half_height
            && planar_dist_from_basis_center <= this.radius)
        {
            return pt;
        } else {
            // The point is outside of the this.
            if (pt.y > this.half_height) {
                if (planar_dist_from_basis_center <= this.radius) {
                    return float3(pt.x, this.half_height, pt.z);
                } else {
                    return float3(proj2d[0], this.half_height, proj2d[1]);
                }
            } else if (pt.y < -this.half_height) {
                // Project on the bottom plane or the bottom circle.
                if (planar_dist_from_basis_center <= this.radius) {
                    return float3(pt.x, -this.half_height, pt.z);
                } else {
                    return float3(proj2d[0], -this.half_height, proj2d[1]);
                }
            } else {
                // Project on the side.
                return float3(proj2d[0], pt.y, proj2d[1]);
            }
        }
    }

    /// Projects a point on a transformed cylinder.
    ///
    /// If the point is inside the cylinder, the point itself is returned.
    public func project_point(pose: Pose, pt: Vect) -> Vect {
        let localPt = pose.inv_mul_pt(pt);
        return pose.mul_pt(this.project_local_point(localPt));
    }


    /// Projects a point on the boundary of a cylinder.
    public func project_local_point_on_boundary(pt: Vect) -> ProjectionResult {
        // Project on the basis.
        let planar_dist_from_basis_center = length(pt.xz);
        let dir_from_basis_center = select(
            planar_dist_from_basis_center > 0.0,
            pt.xz / planar_dist_from_basis_center,
            float2(1.0, 0.0),
        );

        let proj2d = dir_from_basis_center * this.radius;

        // PERF: reduce branching
        if (pt.y >= -this.half_height
            && pt.y <= this.half_height
            && planar_dist_from_basis_center <= this.radius)
        {
            // The point is inside of the cylinder.
            let dist_to_top = this.half_height - pt.y;
            let dist_to_bottom = pt.y - (-this.half_height);
            let dist_to_side = this.radius - planar_dist_from_basis_center;

            if (dist_to_top < dist_to_bottom && dist_to_top < dist_to_side) {
                let projection_on_top = float3(pt.x, this.half_height, pt.z);
                return ProjectionResult(projection_on_top, true);
            } else if (dist_to_bottom < dist_to_top && dist_to_bottom < dist_to_side) {
                let projection_on_bottom =
                    float3(pt.x, -this.half_height, pt.z);
                return ProjectionResult(projection_on_bottom, true);
            } else {
                let projection_on_side = float3(proj2d[0], pt.y, proj2d[1]);
                return ProjectionResult(projection_on_side, true);
            }
        } else {
            // The point is outside of the cylinder.
            if (pt.y > this.half_height) {
                if (planar_dist_from_basis_center <= this.radius) {
                    let projection_on_top = float3(pt.x, this.half_height, pt.z);
                    return ProjectionResult(projection_on_top, false);
                } else {
                    let projection_on_top_circle =
                        float3(proj2d[0], this.half_height, proj2d[1]);
                    return ProjectionResult(projection_on_top_circle, false);
                }
            } else if (pt.y < -this.half_height) {
                // Project on the bottom plane or the bottom circle.
                if (planar_dist_from_basis_center <= this.radius) {
                    let projection_on_bottom =
                        float3(pt.x, -this.half_height, pt.z);
                    return ProjectionResult(projection_on_bottom, false);
                } else {
                    let projection_on_bottom_circle =
                        float3(proj2d[0], -this.half_height, proj2d[1]);
                    return ProjectionResult(projection_on_bottom_circle, false);
                }
            } else {
                // Project on the side.
                let projection_on_side = float3(proj2d[0], pt.y, proj2d[1]);
                return ProjectionResult(projection_on_side, false);
            }
        }
    }

    /// Project a point of a transformed cylinder’s boundary.
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
