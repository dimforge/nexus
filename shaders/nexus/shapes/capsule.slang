module capsule;

import nexus.aliases;
import nexus.queries.ray;
import nexus.queries.projection;
import nexus.shapes.segment;

/// A capsule, defined by its radius.
public struct Capsule {
    /// The capsule’s principal axis.
    public Segment segment;
    /// The capsule’s radius.
    public float radius;

    /// Projects a point on a capsule.
    ///
    /// If the point is inside the capsule, the point itself is returned.
    public func project_local_point(pt: Vect) -> Vect {
        let proj_on_axis = this.segment.project_local_point(pt);
        let dproj = pt - proj_on_axis;
        let dist_to_axis = length(dproj);

        // PERF: call `select` instead?
        if (dist_to_axis > this.radius) {
            return proj_on_axis + dproj * (this.radius / dist_to_axis);
        } else {
            return pt;
        }
    }

    /// Projects a point on a transformed capsule.
    ///
    /// If the point is inside the capsule, the point itself is returned.
    public func project_point(pose: Pose, pt: Vect) -> Vect {
        let localPt = pose.inv_mul_pt(pt);
        return pose.mul_pt(this.project_local_point(localPt));
    }


    /// Projects a point on the boundary of a capsule.
    public func project_local_point_on_boundary(pt: Vect) -> ProjectionResult {
        let proj_on_axis = this.segment.project_local_point(pt);
        let dproj = pt - proj_on_axis;
        let dist_to_axis = length(dproj);

        if (dist_to_axis > 0.0) {
            let is_inside = dist_to_axis <= this.radius;
            return ProjectionResult(proj_on_axis + dproj * (this.radius / dist_to_axis), is_inside);
        } else {
            // Very rare occurence: the point lies on the capsule’s axis exactly.
            // Pick an arbitrary projection direction along an axis orthogonal to the principal axis.
            let axis_seg = this.segment.b - this.segment.a;
            let axis_len = length(axis_seg);
            let proj_dir = any_orthogonal_vector(axis_seg / axis_len);
            return ProjectionResult(proj_on_axis + proj_dir * this.radius, true);
        }
    }

    /// Project a point of a transformed capsule’s boundary.
    ///
    /// If the point is inside of the box, it will be projected on its boundary but
    /// `ProjectionResult::is_inside` will be set to `true`.
    public func project_point_on_boundary(pose: Pose, pt: Vect) -> ProjectionResult {
        let local_pt = pose.inv_mul_pt(pt);
        var result = this.project_local_point_on_boundary(local_pt);
        result.point = pose.mul_pt(result.point);
        return result;
    }

}

// TODO: upstream too gla
public func orthonormal_basis3(v: float3) -> Array<float3, 2> {
    // NOTE: not using `sign` because we don’t want the 0.0 case to return 0.0.
    let sign = select(v.z >= 0.0, 1.0, -1.0);
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;

    return {
        float3(
            1.0 + sign * v.x * v.x * a,
            sign * b,
            -sign * v.x,
        ),
        float3(b, sign + v.y * v.y * a, -v.y),
    };
}

// TODO: upstream to gla
public func any_orthogonal_vector(v: Vect) -> Vect {
#if DIM == 2
    return float2(v.y, -v.x);
#else
    return orthonormal_basis3(v)[0];
#endif
}
