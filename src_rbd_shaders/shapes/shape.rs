//! Generic Shape Enumeration Module
//!
//! A unified interface for working with multiple shape types through runtime
//! polymorphism. Construct a [`Shape`] with the type-specific constructors
//! (`Shape::ball`, `Shape::cuboid`, …), query its kind with `shape_type()`, and
//! run geometric operations (projection, support point, AABB) that dispatch on
//! the type at runtime.

use crate::bounding_volumes::Aabb;
use crate::queries::{PolygonalFeature, ProjectionResult};
use crate::shapes::capsule::Capsule;
use crate::shapes::segment::Segment;
use crate::shapes::triangle::Triangle;
use crate::shapes::{Ball, Cuboid};
use crate::{PaddedVector, Pose, Vector};
use glamx::Vec4;
use parry::{query::PointQuery, shape::SupportMap};

#[cfg(feature = "dim3")]
use crate::shapes::cone::Cone;
#[cfg(feature = "dim3")]
use crate::shapes::cylinder::Cylinder;

use crate::shapes::convex_polyhedron::ConvexPolyhedron;
use crate::shapes::polyline::Polyline;
use crate::shapes::trimesh::TriMesh;

/// Shape type constants for runtime type identification
pub const SHAPE_TYPE_BALL: u32 = 0;
pub const SHAPE_TYPE_CUBOID: u32 = 1;
pub const SHAPE_TYPE_CAPSULE: u32 = 2;
pub const SHAPE_TYPE_CONE: u32 = 3;
pub const SHAPE_TYPE_CYLINDER: u32 = 4;
pub const SHAPE_TYPE_POLYLINE: u32 = 5;
pub const SHAPE_TYPE_TRIMESH: u32 = 6;
pub const SHAPE_TYPE_CONVEX_POLY: u32 = 7;
// TODO: since this shape type is only for trimesh, it doesn't implement all the
//       operations it could if it were a standalone self.
pub const SHAPE_TYPE_TRIANGLE: u32 = 8;

/// A generic shape that can represent any concrete shape type.
///
/// This is a tagged union encoded in two vec4 values. The shape type
/// is stored in the 'a.w' component as a bitcast u32.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Shape {
    /// First vec4 containing shape-specific data and type tag in 'w' component.
    pub a: Vec4,
    /// Second vec4 for additional shape data (primarily for capsule segment endpoint).
    pub b: Vec4,
    /// Third vec4, only used for triangles.
    pub c: Vec4,
}

/// A sub-shape for polygonal feature-based collision detection.
pub struct PfmSubShape {
    pub shape: Shape,
    pub thickness: f32,
    pub valid: bool,
}

impl Shape {
    /// Creates a new shape from three vec4 values.
    #[inline]
    pub fn new(a: Vec4, b: Vec4, c: Vec4) -> Self {
        Self { a, b, c }
    }

    /// Returns the shape type from the encoded shape.
    #[inline]
    pub fn shape_type(&self) -> u32 {
        f32::to_bits(self.a.w)
    }

    /*
     *
     * Shape conversions.
     *
     */

    /// Converts a Shape to a Ball.
    pub fn to_ball(&self) -> Ball {
        // Ball layout:
        //     vec4(radius, _, _, shape_type)
        //     vec4(_, _, _, _)
        Ball::new(self.a.x)
    }

    /// Converts a Shape to a Triangle.
    pub fn to_triangle(&self) -> Triangle {
        // Triangle layout:
        //     vec4(a.x, a.y, a.z, shape_type)
        //     vec4(b.x, b.y, b.z, _)
        //     vec4(c.x, c.y, c.z, _)
        #[cfg(feature = "dim2")]
        {
            Triangle::new(
                Vector::new(self.a.x, self.a.y),
                Vector::new(self.b.x, self.b.y),
                Vector::new(self.c.x, self.c.y),
            )
        }
        #[cfg(feature = "dim3")]
        {
            Triangle::new(
                Vector::new(self.a.x, self.a.y, self.a.z),
                Vector::new(self.b.x, self.b.y, self.b.z),
                Vector::new(self.c.x, self.c.y, self.c.z),
            )
        }
    }

    /// Converts a Triangle to a Shape.
    pub fn from_triangle(tri: &Triangle) -> Shape {
        let tag = f32::from_bits(SHAPE_TYPE_TRIANGLE);
        #[cfg(feature = "dim2")]
        {
            Shape::new(
                Vec4::new(tri.a.x, tri.a.y, 0.0, tag),
                Vec4::new(tri.b.x, tri.b.y, 0.0, 0.0),
                Vec4::new(tri.c.x, tri.c.y, 0.0, 0.0),
            )
        }
        #[cfg(feature = "dim3")]
        {
            Shape::new(
                Vec4::new(tri.a.x, tri.a.y, tri.a.z, tag),
                Vec4::new(tri.b.x, tri.b.y, tri.b.z, 0.0),
                Vec4::new(tri.c.x, tri.c.y, tri.c.z, 0.0),
            )
        }
    }

    /// Converts a Shape to a Capsule.
    pub fn to_capsule(&self) -> Capsule {
        // Capsule layout:
        //     vec4(ax, ay, az, shape_type)
        //     vec4(bx, by, bz, radius)
        #[cfg(feature = "dim2")]
        {
            Capsule::new(
                Segment::new(
                    Vector::new(self.a.x, self.a.y),
                    Vector::new(self.b.x, self.b.y),
                ),
                self.b.w,
            )
        }
        #[cfg(feature = "dim3")]
        {
            Capsule::new(
                Segment::new(
                    Vector::new(self.a.x, self.a.y, self.a.z),
                    Vector::new(self.b.x, self.b.y, self.b.z),
                ),
                self.b.w,
            )
        }
    }

    /// Converts a Capsule to a Shape.
    pub fn from_capsule(cap: &Capsule) -> Shape {
        #[cfg(feature = "dim2")]
        {
            let a = Vec4::new(
                cap.segment.a.x,
                cap.segment.a.y,
                0.0,
                f32::from_bits(SHAPE_TYPE_CAPSULE),
            );
            let b = Vec4::new(cap.segment.b.x, cap.segment.b.y, 0.0, cap.radius);
            Shape::new(a, b, Vec4::ZERO)
        }
        #[cfg(feature = "dim3")]
        {
            let a = Vec4::new(
                cap.segment.a.x,
                cap.segment.a.y,
                cap.segment.a.z,
                f32::from_bits(SHAPE_TYPE_CAPSULE),
            );
            let b = Vec4::new(
                cap.segment.b.x,
                cap.segment.b.y,
                cap.segment.b.z,
                cap.radius,
            );
            Shape::new(a, b, Vec4::ZERO)
        }
    }

    /// Converts a Shape to a Cuboid.
    pub fn to_cuboid(&self) -> Cuboid {
        // Cuboid layout:
        //     vec4(hx, hy, hz, shape_type)
        //     vec4(_, _, _, _)
        #[cfg(feature = "dim2")]
        {
            Cuboid::new(Vector::new(self.a.x, self.a.y))
        }
        #[cfg(feature = "dim3")]
        {
            Cuboid::new(Vector::new(self.a.x, self.a.y, self.a.z))
        }
    }

    #[cfg(feature = "dim3")]
    /// Converts a Shape to a Cone (3D only).
    pub fn to_cone(&self) -> Cone {
        // Cone layout:
        //     vec4(half_height, radius, _, shape_type)
        //     vec4(_, _, _, _)
        Cone::new(self.a.x, self.a.y)
    }

    #[cfg(feature = "dim3")]
    /// Converts a Shape to a Cylinder (3D only).
    pub fn to_cylinder(&self) -> Cylinder {
        // Cylinder layout:
        //     vec4(half_height, radius, _, shape_type)
        //     vec4(_, _, _, _)
        Cylinder::new(self.a.x, self.a.y)
    }

    /// Converts a Shape to a ConvexPolyhedron.
    pub fn to_convex_poly(&self) -> ConvexPolyhedron {
        // Convex polyhedron layout:
        //     vec4(first_vtx_id, end_vtx_id, _, shape_type)
        //     vec4(first_tri_id, end_tri_id, _, _)
        let first_vtx_id = f32::to_bits(self.a.x);
        let end_vtx_id = f32::to_bits(self.a.y);
        let first_tri_id = f32::to_bits(self.b.x);
        let end_tri_id = f32::to_bits(self.b.y);
        ConvexPolyhedron::new(first_vtx_id, end_vtx_id, first_tri_id, end_tri_id)
    }

    /// Converts a Shape to a TriMesh.
    pub fn to_trimesh(&self) -> TriMesh {
        // Trimesh layout:
        //     vec4(bvh_vtx_root_id, bvh_idx_root_id, bvh_node_len, shape_type)
        //     vec4(root_aabb.mins.xyz, num_triangles)
        //     vec4(root_aabb.maxs.xyz, num_vertices)
        let bvh_vtx_root_id = f32::to_bits(self.a.x);
        let bvh_idx_root_id = f32::to_bits(self.a.y);
        let bvh_node_len = f32::to_bits(self.a.z);
        let num_triangles = f32::to_bits(self.b.w);
        let num_vertices = f32::to_bits(self.c.w);
        #[cfg(feature = "dim2")]
        let root_aabb = Aabb::new(
            Vector::new(self.b.x, self.b.y),
            Vector::new(self.c.x, self.c.y),
        );
        #[cfg(feature = "dim3")]
        let root_aabb = Aabb::new(
            Vector::new(self.b.x, self.b.y, self.b.z),
            Vector::new(self.c.x, self.c.y, self.c.z),
        );
        TriMesh::new(
            bvh_vtx_root_id,
            bvh_idx_root_id,
            bvh_node_len,
            num_triangles,
            num_vertices,
            root_aabb,
        )
    }

    /// Converts a Shape to a Polyline.
    pub fn to_polyline(&self) -> Polyline {
        // Polyline layout:
        //     vec4(bvh_vtx_root_id, bvh_idx_root_id, bvh_node_len, shape_type)
        //     vec4(root_aabb.mins, _)
        //     vec4(root_aabb.maxs, _)
        let bvh_vtx_root_id = f32::to_bits(self.a.x);
        let bvh_idx_root_id = f32::to_bits(self.a.y);
        let bvh_node_len = f32::to_bits(self.a.z);
        #[cfg(feature = "dim2")]
        let root_aabb = Aabb::new(
            Vector::new(self.b.x, self.b.y),
            Vector::new(self.c.x, self.c.y),
        );
        #[cfg(feature = "dim3")]
        let root_aabb = Aabb::new(
            Vector::new(self.b.x, self.b.y, self.b.z),
            Vector::new(self.c.x, self.c.y, self.c.z),
        );
        Polyline::new(bvh_vtx_root_id, bvh_idx_root_id, bvh_node_len, root_aabb)
    }

    /*
     *
     * Geometric operations.
     *
     */

    /// Projects a point on this self.
    ///
    /// If the point is inside the shape, the point itself is returned.
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_BALL {
            return self.to_ball().project_local_point(pt, true).point;
        }
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().project_local_point(pt, true).point;
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().project_local_point(pt);
        }
        #[cfg(feature = "dim3")]
        {
            if ty == SHAPE_TYPE_CONE {
                return self.to_cone().project_local_point(pt);
            }
            if ty == SHAPE_TYPE_CYLINDER {
                return self.to_cylinder().project_local_point(pt);
            }
        }
        pt
    }

    /// Projects a point on a transformed self.
    ///
    /// If the point is inside the shape, the point itself is returned.
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_BALL {
            return self.to_ball().project_point(&pose, pt, true).point;
        }
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().project_point(&pose, pt, true).point;
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().project_point(pose, pt);
        }
        #[cfg(feature = "dim3")]
        {
            if ty == SHAPE_TYPE_CONE {
                return self.to_cone().project_point(pose, pt);
            }
            if ty == SHAPE_TYPE_CYLINDER {
                return self.to_cylinder().project_point(pose, pt);
            }
        }
        pt
    }

    /// Projects a point on the boundary of a self.
    pub fn project_local_point_on_boundary(&self, pt: Vector) -> ProjectionResult {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_BALL {
            return self.to_ball().project_local_point(pt, false).into();
        }
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().project_local_point(pt, false).into();
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().project_local_point_on_boundary(pt);
        }
        #[cfg(feature = "dim3")]
        {
            if ty == SHAPE_TYPE_CONE {
                return self.to_cone().project_local_point_on_boundary(pt);
            }
            if ty == SHAPE_TYPE_CYLINDER {
                return self.to_cylinder().project_local_point_on_boundary(pt);
            }
        }
        ProjectionResult::new(pt, false)
    }

    /// Project a point of a transformed shape's boundary.
    ///
    /// If the point is inside of the shape, it will be projected on its boundary but
    /// `ProjectionResult::is_inside` will be set to `true`.
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_BALL {
            return self.to_ball().project_point(&pose, pt, false).into();
        }
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().project_point(&pose, pt, false).into();
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().project_point_on_boundary(pose, pt);
        }
        #[cfg(feature = "dim3")]
        {
            if ty == SHAPE_TYPE_CONE {
                return self.to_cone().project_point_on_boundary(pose, pt);
            }
            if ty == SHAPE_TYPE_CYLINDER {
                return self.to_cylinder().project_point_on_boundary(pose, pt);
            }
        }
        ProjectionResult::new(pt, false)
    }

    /// Computes the support point of a transformed self.
    pub fn support_point(&self, pose: Pose, axis: Vector, vertices: &[PaddedVector]) -> Vector {
        let local_axis = pose.rotation.inverse() * axis;
        let local_pt = self.local_support_point(local_axis, vertices);
        pose * local_pt
    }

    /// Computes the local support point of a self.
    pub fn local_support_point(&self, dir: Vector, vertices: &[PaddedVector]) -> Vector {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_BALL {
            return self.to_ball().local_support_point(dir);
        }
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().local_support_point(dir);
        }
        if ty == SHAPE_TYPE_TRIANGLE {
            return self.to_triangle().local_support_point(dir);
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().local_support_point(dir);
        }
        #[cfg(feature = "dim3")]
        {
            if ty == SHAPE_TYPE_CONE {
                return self.to_cone().local_support_point(dir);
            }
            if ty == SHAPE_TYPE_CYLINDER {
                return self.to_cylinder().local_support_point(dir);
            }
        }

        if ty == SHAPE_TYPE_CONVEX_POLY {
            return self.to_convex_poly().local_support_point(vertices, dir);
        }

        Vector::ZERO
    }

    /// Computes the support face of a self.
    #[cfg(feature = "dim2")]
    pub fn support_face(&self, dir: Vector, vertices: &[PaddedVector]) -> PolygonalFeature {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().support_face(dir).into();
        }
        if ty == SHAPE_TYPE_TRIANGLE {
            return self.to_triangle().support_face(dir);
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().support_face(dir);
        }

        if ty == SHAPE_TYPE_CONVEX_POLY {
            return self.to_convex_poly().support_face(vertices, dir);
        }

        PolygonalFeature::default()
    }

    /// Computes the support face of a self.
    #[cfg(feature = "dim3")]
    pub fn support_face(
        &self,
        dir: Vector,
        vertices: &[PaddedVector],
        indices: &[u32],
    ) -> PolygonalFeature {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_CUBOID {
            return self.to_cuboid().support_face(dir).into();
        }
        if ty == SHAPE_TYPE_TRIANGLE {
            return self.to_triangle().support_face(dir);
        }
        if ty == SHAPE_TYPE_CAPSULE {
            return self.to_capsule().support_face(dir);
        }
        if ty == SHAPE_TYPE_CONE {
            return self.to_cone().support_face(dir);
        }
        if ty == SHAPE_TYPE_CYLINDER {
            return self.to_cylinder().support_face(dir);
        }

        if ty == SHAPE_TYPE_CONVEX_POLY {
            return self.to_convex_poly().support_face(vertices, indices, dir);
        }

        PolygonalFeature::default()
    }

    /// Returns the polygonal feature sub-shape for collision detection.
    pub fn pfm_subshape(&self) -> PfmSubShape {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_CUBOID
            || ty == SHAPE_TYPE_CONE
            || ty == SHAPE_TYPE_CYLINDER
            || ty == SHAPE_TYPE_CONVEX_POLY
            || ty == SHAPE_TYPE_TRIANGLE
        {
            // No subshape, return the original shape itself.
            return PfmSubShape {
                shape: *self,
                thickness: 0.0,
                valid: true,
            };
        }

        if ty == SHAPE_TYPE_BALL {
            let ball = self.to_ball();
            let segment = Capsule::default();
            return PfmSubShape {
                shape: Self::from_capsule(&segment),
                thickness: ball.radius,
                valid: true,
            };
        }

        if ty == SHAPE_TYPE_CAPSULE {
            let capsule = self.to_capsule();
            let without_radius = Capsule::new(capsule.segment, 0.0);
            return PfmSubShape {
                shape: Self::from_capsule(&without_radius),
                thickness: capsule.radius,
                valid: true,
            };
        }

        // Not a PFM.
        PfmSubShape {
            shape: *self,
            thickness: 0.0,
            valid: false,
        }
    }

    /// Creates an AABB from a transformed self.
    pub fn compute_aabb(&self, pose: Pose, vertices: &[PaddedVector]) -> Aabb {
        let ty = self.shape_type();
        if ty == SHAPE_TYPE_BALL {
            let ball = self.to_ball();
            #[cfg(feature = "dim2")]
            let (tra, rad) = {
                let tra = pose.translation;
                let rad = ball.radius; // No scale support
                (tra, rad)
            };
            #[cfg(feature = "dim3")]
            let (tra, rad) = {
                let tra = pose.translation;
                let rad = ball.radius; // No scale support
                (tra, rad)
            };

            #[cfg(feature = "dim2")]
            return Aabb::new(tra - Vector::splat(rad), tra + Vector::splat(rad));
            #[cfg(feature = "dim3")]
            return Aabb::new(tra - Vector::splat(rad), tra + Vector::splat(rad));
        }

        if ty == SHAPE_TYPE_CUBOID {
            let cuboid = self.to_cuboid();
            let local_aabb = Aabb::new(-cuboid.half_extents, cuboid.half_extents);
            return local_aabb.transform_by(pose);
        }

        if ty == SHAPE_TYPE_TRIANGLE {
            let tri = self.to_triangle();
            let local_aabb = tri.aabb();
            return local_aabb.transform_by(pose);
        }

        if ty == SHAPE_TYPE_CAPSULE {
            let capsule = self.to_capsule();
            let aa = pose * capsule.segment.a;
            let bb = pose * capsule.segment.b;
            #[cfg(feature = "dim2")]
            return Aabb::new(
                aa.min(bb) - Vector::splat(capsule.radius),
                aa.max(bb) + Vector::splat(capsule.radius),
            );
            #[cfg(feature = "dim3")]
            return Aabb::new(
                aa.min(bb) - Vector::splat(capsule.radius),
                aa.max(bb) + Vector::splat(capsule.radius),
            );
        }

        #[cfg(feature = "dim3")]
        {
            if ty == SHAPE_TYPE_CONE {
                let cone = self.to_cone();
                let local_aabb = Aabb::new(
                    -Vector::new(cone.radius, cone.half_height, cone.radius),
                    Vector::new(cone.radius, cone.half_height, cone.radius),
                );
                return local_aabb.transform_by(pose);
            }

            if ty == SHAPE_TYPE_CYLINDER {
                let cyl = self.to_cylinder();
                let local_aabb = Aabb::new(
                    -Vector::new(cyl.radius, cyl.half_height, cyl.radius),
                    Vector::new(cyl.radius, cyl.half_height, cyl.radius),
                );
                return local_aabb.transform_by(pose);
            }
        }

        if ty == SHAPE_TYPE_CONVEX_POLY {
            let poly = self.to_convex_poly();
            let local_aabb = poly.aabb(vertices);
            return local_aabb.transform_by(pose);
        }

        if ty == SHAPE_TYPE_TRIMESH {
            let mesh = self.to_trimesh();
            let local_aabb = mesh.aabb();
            return local_aabb.transform_by(pose);
        }

        if ty == SHAPE_TYPE_POLYLINE {
            let pline = self.to_polyline();
            let local_aabb = pline.aabb();
            return local_aabb.transform_by(pose);
        }

        Aabb::default()
    }

    /// Creates a ball (sphere/circle) shape.
    ///
    /// # Parameters
    ///
    /// - `radius`: The radius of the ball
    pub fn ball(radius: f32) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_BALL);
        Self {
            a: Vec4::new(radius, 0.0, 0.0, tag),
            b: Vec4::ZERO,
            c: Vec4::ZERO,
        }
    }

    /// Creates a cuboid (box/rectangle) shape.
    ///
    /// # Parameters
    ///
    /// - `half_extents`: Half-widths along each axis (vec2 for 2D, vec3 for 3D)
    pub fn cuboid(half_extents: Vector) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_CUBOID);
        Self {
            #[cfg(feature = "dim2")]
            a: Vec4::new(half_extents.x, half_extents.y, 0.0, tag),
            #[cfg(feature = "dim3")]
            a: Vec4::new(half_extents.x, half_extents.y, half_extents.z, tag),
            b: Vec4::ZERO,
            c: Vec4::ZERO,
        }
    }

    /// Creates a capsule shape.
    ///
    /// # Parameters
    ///
    /// - `a`: First endpoint of the capsule's central segment
    /// - `b`: Second endpoint of the capsule's central segment
    /// - `radius`: Radius of the capsule (distance from segment to surface)
    pub fn capsule(a: Vector, b: Vector, radius: f32) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_CAPSULE);
        #[cfg(feature = "dim2")]
        return Self {
            a: Vec4::new(a.x, a.y, 0.0, tag),
            b: Vec4::new(b.x, b.y, 0.0, radius),
            c: Vec4::ZERO,
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: Vec4::new(a.x, a.y, a.z, tag),
            b: Vec4::new(b.x, b.y, b.z, radius),
            c: Vec4::ZERO,
        };
    }

    /// Creates a polyline shape from BVH data.
    ///
    /// A polyline is a connected sequence of line segments defined by vertices.
    ///
    /// # Parameters
    ///
    /// - `bvh_vtx_root_id`: Start index for BVH vertex data
    /// - `bvh_idx_root_id`: Start index for BVH index data
    /// - `bvh_node_len`: Number of BVH nodes
    /// - `aabb_mins`: Minimum point of the bounding box
    /// - `aabb_maxs`: Maximum point of the bounding box
    pub fn polyline(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        aabb_mins: Vector,
        aabb_maxs: Vector,
    ) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_POLYLINE);
        let a0 = f32::from_bits(bvh_vtx_root_id);
        let a1 = f32::from_bits(bvh_idx_root_id);
        let a2 = f32::from_bits(bvh_node_len);
        #[cfg(feature = "dim2")]
        return Self {
            a: Vec4::new(a0, a1, a2, tag),
            b: Vec4::new(aabb_mins.x, aabb_mins.y, 0.0, 0.0),
            c: Vec4::new(aabb_maxs.x, aabb_maxs.y, 0.0, 0.0),
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: Vec4::new(a0, a1, a2, tag),
            b: Vec4::new(aabb_mins.x, aabb_mins.y, aabb_mins.z, 0.0),
            c: Vec4::new(aabb_maxs.x, aabb_maxs.y, aabb_maxs.z, 0.0),
        };
    }

    /// Creates a triangle mesh shape from BVH data.
    ///
    /// A trimesh is a collection of triangles sharing vertices.
    pub fn trimesh(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        num_triangles: u32,
        num_vertices: u32,
        aabb_mins: Vector,
        aabb_maxs: Vector,
    ) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_TRIMESH);
        let a0 = f32::from_bits(bvh_vtx_root_id);
        let a1 = f32::from_bits(bvh_idx_root_id);
        let a2 = f32::from_bits(bvh_node_len);
        let num_triangles = f32::from_bits(num_triangles);
        let num_vertices = f32::from_bits(num_vertices);
        #[cfg(feature = "dim2")]
        return Self {
            a: Vec4::new(a0, a1, a2, tag),
            b: Vec4::new(aabb_mins.x, aabb_mins.y, 0.0, num_triangles),
            c: Vec4::new(aabb_maxs.x, aabb_maxs.y, 0.0, num_vertices),
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: Vec4::new(a0, a1, a2, tag),
            b: Vec4::new(aabb_mins.x, aabb_mins.y, aabb_mins.z, num_triangles),
            c: Vec4::new(aabb_maxs.x, aabb_maxs.y, aabb_maxs.z, num_vertices),
        };
    }

    /// Creates a convex polyhedron from vertex and face buffer ranges.
    pub fn convex_poly(
        first_vtx_id: u32,
        end_vtx_id: u32,
        first_face_id: u32,
        end_face_id: u32,
    ) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_CONVEX_POLY);
        let a0 = f32::from_bits(first_vtx_id);
        let a1 = f32::from_bits(end_vtx_id);
        let b0 = f32::from_bits(first_face_id);
        let b1 = f32::from_bits(end_face_id);
        Self {
            a: Vec4::new(a0, a1, 0.0, tag),
            b: Vec4::new(b0, b1, 0.0, 0.0),
            c: Vec4::ZERO,
        }
    }

    /// Creates a cone shape (3D only).
    ///
    /// # Parameters
    ///
    /// - `half_height`: Half the height of the cone (from base to apex)
    /// - `radius`: Radius of the cone's base
    #[cfg(feature = "dim3")]
    pub fn cone(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_CONE);
        Self {
            a: Vec4::new(half_height, radius, 0.0, tag),
            b: Vec4::ZERO,
            c: Vec4::ZERO,
        }
    }

    /// Creates a cylinder shape (3D only).
    ///
    /// # Parameters
    ///
    /// - `half_height`: Half the height of the cylinder (distance from center to end caps)
    /// - `radius`: Radius of the cylinder
    #[cfg(feature = "dim3")]
    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(SHAPE_TYPE_CYLINDER);
        Self {
            a: Vec4::new(half_height, radius, 0.0, tag),
            b: Vec4::ZERO,
            c: Vec4::ZERO,
        }
    }

    /// Returns the raw shape type tag.
    pub fn shape_type_tag(&self) -> u32 {
        self.a.w.to_bits()
    }

    /// Returns the start index of the actual polyline vertices in the vertex buffer.
    ///
    /// # Panics
    ///
    /// Panics if this shape is not a polyline
    pub fn polyline_vertex_start(&self) -> u32 {
        assert!(self.shape_type_tag() == SHAPE_TYPE_POLYLINE);
        let bvh_vtx_root_id = self.a.x.to_bits();
        let bvh_node_len = self.a.z.to_bits();
        bvh_vtx_root_id + bvh_node_len * 2
    }

    /// Returns the start index of the actual trimesh vertices in the vertex buffer.
    ///
    /// # Panics
    ///
    /// Panics if this shape is not a triangle mesh
    pub fn trimesh_vertex_start(&self) -> u32 {
        assert!(self.shape_type_tag() == SHAPE_TYPE_TRIMESH);
        let bvh_vtx_root_id = self.a.x.to_bits();
        let bvh_node_len = self.a.z.to_bits();
        bvh_vtx_root_id + bvh_node_len * 2
    }
}
