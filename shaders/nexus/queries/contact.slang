#import wgebra::sim3 as Pose
#import wgparry::ball as Ball

mod contact;

/// A pair of contact points between two shapes.
public struct Contact {
    /// The contact point on the first shape.
    public Vect point1;
    /// The contact point on the second shape.
    public Vect point2;
    /// The first shape’s normal at its contact point.
    public Vect normal1;
    /// The second shape’s normal at its contact point.
    public Vect normal2;
    /// The distance between the two contact points.
    public float dist;

    public __init(Vect point1, Vect point2, Vect normal1, Vect normal2, float dist) {
        this.point1 = point1;
        this.point2 = point2;
        this.normal1 = normal1;
        this.normal2 = normal2;
    }

#if DIM == 3
    /// Computes the contact between two balls.
    public static func ballBall(pose12: Pose::Sim3, ball1: Ball::Ball, ball2: Ball::Ball) -> Contact {
        let r1 = ball1.radius;
        let r2 = ball2.radius;
        let center2_1 = pose12.translation_scale.xyz;
        let distance = length(center2_1);
        let sum_radius = r1 + r2;

        var normal1 = vec3(1.0, 0.0, 0.0);

        if (distance != 0.0) {
            normal1 = center2_1 / distance;
        }

        let normal2 = -pose12.inv_mul_unit_vec(normal1);
        let point1 = normal1 * r1;
        let point2 = normal2 * r2;

        return Contact(
            point1,
            point2,
            normal1,
            normal2,
            distance - sum_radius,
        );
    }
#endif
}
