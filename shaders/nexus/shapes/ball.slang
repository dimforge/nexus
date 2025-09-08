module ball;

import nexus.aliases;
import nexus.queries.ray;
import nexus.queries.projection;


/// A ball, defined by its radius.
public struct Ball {
    /// The ball’s radius.
    public float radius;

    /// Projects a point on a ball.
    ///
    /// If the point is inside the ball, the point itself is returned.
    public func project_local_point(pt: Vect) -> Vect {
        let dist = length(pt);

        if (dist >= this.radius) {
            // The point is outside the ball.
            return pt * (this.radius / dist);
        } else {
            // The point is inside the ball.
            return pt;
        }
    }

    /// Projects a point on a transformed ball.
    ///
    /// If the point is inside the ball, the point itself is returned.
    public func project_point(pose: Pose, pt: Vect) -> Vect {
        let local_pt = pose.inv_mul_pt(pt);
        return pose.mul_pt(this.project_local_point(local_pt));
    }


    /// Projects a point on the boundary of a ball.
    public func project_local_point_on_boundary(pt: Vect) -> ProjectionResult {
        let dist = length(pt);
    #if DIM == 2
        let fallback = float2(0.0, this.radius);
    #else
        let fallback = float3(0.0, this.radius, 0.0);
    #endif

        let projected_point =
            select(dist != 0.0, pt * (this.radius / dist), fallback);
        let is_inside = dist <= this.radius;

        return ProjectionResult(projected_point, is_inside);
    }

    /// Project a point of a transformed ball’s boundary.
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

/*
/// Casts a ray on a ball.
///
/// Returns a negative value if there is no hit.
/// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
fn castLocalRay(ball: Ball, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    // Ray origin relative to the ball’s center. It’s the origin itself since it’s in the ball’s local frame.
    let dcenter = ray.origin;
    let a = dot(ray.dir, ray.dir);
    let b = dot(dcenter, ray.dir);
    let c = dot(dcenter, dcenter) - ball.radius * ball.radius;
    let delta = b * b - a * c;
    let t = -b - sqrt(delta);

    if (c > 0.0 && (b > 0.0 || a == 0.0)) || delta < 0.0 || t > maxTimeOfImpact * a {
        // No hit.
        return -1.0;
    } else if a == 0.0 {
        // Dir is zero but the ray started inside the ball.
        return 0.0;
    } else {
        // Hit. If t <= 0, the origin is inside the ball.
        return max(t / a, 0.0);
    }
}

/// Casts a ray on a transformed ball.
///
/// Returns a negative value if there is no hit.
/// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
fn castRay(ball: Ball, pose: Pose, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    let localRay = Ray::Ray(Pose::invMulPt(pose, ray.origin), Pose::invMulVec(pose, ray.dir));
    return castLocalRay(ball, localRay, maxTimeOfImpact);
}
*/
