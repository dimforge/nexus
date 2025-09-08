module cuboid;

import nexus.aliases;
import nexus.queries.ray;
import nexus.queries.projection;

/// A box, defined by its half-extents (half-length alon geach dimension).
public struct Cuboid {
    public Vect halfExtents;

    /// Projects a point on a box.
    ///
    /// If the point is inside the box, the point itself is returned.
    public func project_local_point(pt: Vect) -> Vect {
        let mins = -this.halfExtents;
        let maxs = this.halfExtents;

        let mins_pt = mins - pt; // -hext - pt
        let pt_maxs = pt - maxs; // pt - hext
        let shift = max(mins_pt, Vect<float>(0.0)) - max(pt_maxs, Vect<float>(0.0));

        return pt + shift;
    }

    /// Projects a point on a transformed box.
    ///
    /// If the point is inside the box, the point itself is returned.
    public func project_point(pose: Pose, pt: Vect) -> Vect {
        let local_pt = pose.inv_mul_pt(pt);
        return pose.mul_pt(this.project_local_point(local_pt));
    }

    /// Projects a point on the boundary of a box.
    public func project_local_point_on_boundary(pt: Vect) -> ProjectionResult {
        let out_proj = this.project_local_point(pt);

        // Projection if the point is inside the box.
        let pt_sgn_with_zero = Vect<float>(sign(pt));
        // This the sign of pt, or -1 for components that were zero.
        // This bias is arbitrary (we could have picked +1), but we picked it so
        // it matches the bias that’s in parry.
        let pt_sgn = pt_sgn_with_zero + (abs(pt_sgn_with_zero) - Vect<float>(1.0));
        let diff = this.halfExtents - pt_sgn * pt;

    #if DIM == 2
        let pick_x = diff.x <= diff.y;
        let shift_x = Vect<float>(diff.x * pt_sgn.x, 0.0);
        let shift_y = Vect<float>(0.0, diff.y * pt_sgn.y);
        let pen_shift = select(pick_x, shift_x, shift_y);
    #else
        let pick_x = diff.x <= diff.y && diff.x <= diff.z;
        let pick_y = diff.y <= diff.x && diff.y <= diff.z;
        let shift_x = Vect<float>(diff.x * pt_sgn.x, 0.0, 0.0);
        let shift_y = Vect<float>(0.0, diff.y * pt_sgn.y, 0.0);
        let shift_z = Vect<float>(0.0, 0.0, diff.z * pt_sgn.z);
        let pen_shift = select(pick_x, shift_x, select(pick_y, shift_y, shift_z));
    #endif
        let in_proj = pt + pen_shift;

        // Select between in and out proj.
        let is_inside = all(pt == out_proj);
        return ProjectionResult(select(is_inside, in_proj, out_proj), is_inside);
    }

    /// Project a point of a transformed box’s boundary.
    ///
    /// If the point is inside of the box, it will be projected on its boundary but
    /// `ProjectionResult::is_inside` will be set to `true`.
    public func project_point_on_boundary(pose: Pose, pt: Vect) -> ProjectionResult {
        let local_pt = pose.inv_mul_pt(pt);
        var result = this.project_local_point_on_boundary(local_pt);
        result.point = pose.mul_pt(result.point);
        return result;
    }


    // FIXME: ray.wgsl needs to support 2d/3d for these implementations to be commented-out.
    ///*
    // * Ray casting.
    // */
    ///// Casts a ray on a box.
    /////
    ///// Returns a negative value if there is no hit.
    ///// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
    //fn castLocalRay(ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    //    let mins = -box.halfExtents;
    //    let maxs = box.halfExtents;
    //    let inter1 = (mins - ray.origin) / ray.dir;
    //    let inter2 = (maxs - ray.origin) / ray.dir;
    //
    //    let vtmin = min(inter1, inter2);
    //    let vtmax = max(inter1, inter2);
    //
    //#if DIM == 2
    //    let tmin = max(max(vtmin.x, vtmin.y), 0.0);
    //    let tmax = min(min(vtmax.x, vtmax.y), maxTimeOfImpact);
    //#else
    //    let tmin = max(max(max(vtmin.x, vtmin.y), vtmin.z), 0.0);
    //    let tmax = min(min(min(vtmax.x, vtmax.y), vtmax.z), maxTimeOfImpact);
    //#endif
    //
    //    if tmin > tmax || tmax < 0.0 {
    //        return -1.0;
    //    } else {
    //        return tmin;
    //    }
    //}
    //
    ///// Casts a ray on a transformed box.
    /////
    ///// Returns a negative value if there is no hit.
    ///// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
    //fn castRay(pose: Pose, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    //    let localRay = Ray::Ray(Pose::invMulPt(pose, ray.origin), Pose::invMulVec(pose, ray.dir));
    //    return castLocalRay(box, localRay, maxTimeOfImpact);
    //}

}
