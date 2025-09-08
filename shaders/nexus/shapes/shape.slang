module shape;

import nexus.aliases;
import nexus.queries.projection;
import nexus.shapes.ball;
import nexus.shapes.capsule;
import nexus.shapes.cone;
import nexus.shapes.cuboid;
import nexus.shapes.cylinder;
import nexus.shapes.segment;
import nexus.shapes.triangle;


// A generic shape. This is an enum that might contain any shape information.
// PERF: if it wasn’t for the capsule, we could store all the shapes into a single vec4…
public struct Shape {
    public float4 a;
    public float4 b;

    public static const uint SHAPE_TYPE_BALL = 0;
    public static const uint SHAPE_TYPE_CUBOID = 1;
    public static const uint SHAPE_TYPE_CAPSULE = 2;
    public static const uint SHAPE_TYPE_CONE = 3;
    public static const uint SHAPE_TYPE_CYLINDER = 4;
    public static const uint SHAPE_TYPE_POLYLINE = 5;
    public static const uint SHAPE_TYPE_TRIMESH = 6;

    public func shape_type() -> uint {
        return bit_cast<uint, float>(this.a.w);
    }

    /*
     *
     * Shape conversions.
     *
     */
    public func to_ball() -> Ball {
        // Ball layout:
        //     vec4(radius, _, _, shape_type)
        //     vec4(_, _, _, _)
        return Ball(this.a.x);
    }

    public func to_capsule() -> Capsule {
        // Capsule layout:
        //     vec4(ax, ay, az, shape_type)
        //     vec4(bx, by, bz, radius)
    #if DIM == 2
        return Capsule(Segment(this.a.xy, this.b.xy), this.b.w);
    #else
        return Capsule(Segment(this.a.xyz, this.b.xyz), this.b.w);
    #endif
    }

    public func to_cuboid() -> Cuboid {
        // Cuboid layout:
        //     vec4(hx, hy, hz, shape_type)
        //     vec4(_, _, _, _)
    #if DIM == 2
        return Cuboid(this.a.xy);
    #else
        return Cuboid(this.a.xyz);
    #endif
    }

    #if DIM == 3
        public func to_cone() -> Cone {
            // Cone layout:
            //     vec4(half_height, radius, _, shape_type)
            //     vec4(_, _, _, _)
            return Cone(this.a.x, this.a.y);
        }

        public func to_cylinder() -> Cylinder {
            // Cylinder layout:
            //     vec4(half_height, radius, _, shape_type)
            //     vec4(_, _, _, _)
            return Cylinder(this.a.x, this.a.y);
        }
    #endif


    /*
     *
     * Geometric operations.
     *
     */
    /// Projects a point on this shape.
    ///
    /// If the point is inside the shape, the point itself is returned.
    public func project_local_point(pt: Vect) -> Vect {
        let ty = this.shape_type();
        if (ty == SHAPE_TYPE_BALL) {
            return this.to_ball().project_local_point(pt);
        }
        if (ty == SHAPE_TYPE_CUBOID) {
            return this.to_cuboid().project_local_point(pt);
        }
        if (ty == SHAPE_TYPE_CAPSULE) {
            return this.to_capsule().project_local_point(pt);
        }
    #if DIM == 3
        if (ty == SHAPE_TYPE_CONE) {
            return this.to_cone().project_local_point(pt);
        }
        if (ty == SHAPE_TYPE_CYLINDER) {
            return this.to_cylinder().project_local_point(pt);
        }
    #endif
        return pt;
    }

    /// Projects a point on a transformed shape.
    ///
    /// If the point is inside the shape, the point itself is returned.
    public func project_point(pose: Pose, pt: Vect) -> Vect {
        let ty = this.shape_type();
        if (ty == SHAPE_TYPE_BALL) {
            return this.to_ball().project_point(pose, pt);
        }
        if (ty == SHAPE_TYPE_CUBOID) {
            return this.to_cuboid().project_point(pose, pt);
        }
        if (ty == SHAPE_TYPE_CAPSULE) {
            return this.to_capsule().project_point(pose, pt);
        }
    #if DIM == 3
        if (ty == SHAPE_TYPE_CONE) {
            return this.to_cone().project_point(pose, pt);
        }
        if (ty == SHAPE_TYPE_CYLINDER) {
            return this.to_cylinder().project_point(pose, pt);
        }
    #endif
        return pt;
    }


    /// Projects a point on the boundary of a shape.
    public func project_local_point_on_boundary(pt: Vect) -> ProjectionResult {
        let ty = this.shape_type();
        if (ty == SHAPE_TYPE_BALL) {
            return this.to_ball().project_local_point_on_boundary(pt);
        }
        if (ty == SHAPE_TYPE_CUBOID) {
            return this.to_cuboid().project_local_point_on_boundary(pt);
        }
        if (ty == SHAPE_TYPE_CAPSULE) {
            return this.to_capsule().project_local_point_on_boundary(pt);
        }
    #if DIM == 3
        if (ty == SHAPE_TYPE_CONE) {
            return this.to_cone().project_local_point_on_boundary(pt);
        }
        if (ty == SHAPE_TYPE_CYLINDER) {
            return this.to_cylinder().project_local_point_on_boundary(pt);
        }
    #endif
        return ProjectionResult(pt, false);
    }

    /// Project a point of a transformed shape’s boundary.
    ///
    /// If the point is inside of the shape, it will be projected on its boundary but
    /// `ProjectionResult::is_inside` will be set to `true`.
    public func project_point_on_boundary(pose: Pose, pt: Vect) -> ProjectionResult {
        let ty = this.shape_type();

        if (ty == SHAPE_TYPE_BALL) {
            return this.to_ball().project_point_on_boundary(pose, pt);
        }
        if (ty == SHAPE_TYPE_CUBOID) {
            return this.to_cuboid().project_point_on_boundary(pose, pt);
        }
        if (ty == SHAPE_TYPE_CAPSULE) {
            return this.to_capsule().project_point_on_boundary(pose, pt);
        }
    #if DIM == 3
        if (ty == SHAPE_TYPE_CONE) {
            return this.to_cone().project_point_on_boundary(pose, pt);
        }
        if (ty == SHAPE_TYPE_CYLINDER) {
            return this.to_cylinder().project_point_on_boundary(pose, pt);
        }
    #endif
        return ProjectionResult(pt, false);
    }

}
