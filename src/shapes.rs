//! GPU-compatible shape representations.
//!
//! This module provides structures for representing collision shapes in a format
//! optimized for GPU computation. It includes conversion utilities from Rapier/Parry
//! shapes to GPU-friendly formats with vertex buffers.

use na::{vector, Vector4};
use rapier::geometry::{Shape, ShapeType, TypedShape};

use crate::math::{Point, Vector};

/// GPU shape type identifiers.
///
/// These numeric values must match the type constants defined in `shape.wgsl`.
/// They are used to tag shape data on the GPU for runtime type identification.
// NOTE: this must match the type values in shape.wgsl
pub enum GpuShapeType {
    /// Spherical/circular shape (2D: circle, 3D: sphere)
    Ball = 0,
    /// Rectangular/box shape (2D: rectangle, 3D: cuboid)
    Cuboid = 1,
    /// Capsule shape (line segment with rounded ends)
    Capsule = 2,
    /// Conical shape (3D only)
    #[cfg(feature = "dim3")]
    Cone = 3,
    /// Cylindrical shape (3D only)
    #[cfg(feature = "dim3")]
    Cylinder = 4,
    /// Polyline shape (connected line segments)
    // TODO: not sure we want to keep the Polyline in the shape type.
    Polyline = 5,
    /// Triangle mesh shape
    TriMesh = 6,
}

/// Storage for shape vertex data.
///
/// Accumulates vertices from complex shapes (polylines, trimeshes) during
/// conversion from Rapier/Parry shapes. Shapes reference ranges within this buffer.
#[derive(Default, Clone, Debug)]
pub struct ShapeBuffers {
    /// Vertex positions for all complex shapes.
    ///
    /// Polyline and trimesh shapes store references to ranges within this buffer.
    pub vertices: Vec<Point<f32>>,
    // NOTE: a bit weird we don't have any index buffer here but
    //       we don't need it yet (slosh has its own indexing method).
}

/// GPU-compatible shape representation.
///
/// A compact, fixed-size representation of collision shapes suitable for GPU processing.
/// Shape data is encoded into two 4D vectors, with the shape type stored in the `w`
/// component of the first vector as a bit-cast integer.
///
/// # Memory Layout
/// - `a.xyz`: Primary shape parameters (radius, half-extents, vertex range start, etc.)
/// - `a.w`: Shape type identifier (bit-cast from [`GpuShapeType`])
/// - `b.xyz`: Secondary shape parameters (capsule endpoint, vertex range end, etc.)
/// - `b.w`: Additional parameter (e.g., capsule radius)
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuShape {
    a: Vector4<f32>,
    b: Vector4<f32>,
}

impl GpuShape {
    /// Create a ball/sphere shape.
    ///
    /// # Arguments
    /// * `radius` - The radius of the ball/sphere
    ///
    /// # Examples
    /// ```ignore
    /// # use nexus::shapes::GpuShape;
    /// let ball = GpuShape::ball(1.5);
    /// ```
    pub fn ball(radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Ball as u32);
        Self {
            a: vector![radius, 0.0, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a cuboid/rectangle shape.
    ///
    /// # Arguments
    /// * `half_extents` - The half-extents (half-width, half-height, half-depth) of the cuboid
    ///
    /// # Examples
    /// ```ignore
    /// # use nexus::shapes::GpuShape;
    /// # use nexus::math::vector;
    /// let cuboid = GpuShape::cuboid(vector![1.0, 2.0, 3.0]);
    /// ```
    pub fn cuboid(half_extents: Vector<f32>) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cuboid as u32);
        Self {
            #[cfg(feature = "dim2")]
            a: vector![half_extents.x, half_extents.y, 0.0, tag],
            #[cfg(feature = "dim3")]
            a: vector![half_extents.x, half_extents.y, half_extents.z, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a capsule shape.
    ///
    /// A capsule is a line segment with rounded ends of the specified radius.
    ///
    /// # Arguments
    /// * `a` - First endpoint of the capsule's central segment
    /// * `b` - Second endpoint of the capsule's central segment
    /// * `radius` - The radius of the capsule's rounded ends
    pub fn capsule(a: Point<f32>, b: Point<f32>, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Capsule as u32);
        #[cfg(feature = "dim2")]
        return Self {
            a: vector![a.x, a.y, 0.0, tag],
            b: vector![b.x, b.y, 0.0, radius],
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: vector![a.x, a.y, a.z, tag],
            b: vector![b.x, b.y, b.z, radius],
        };
    }

    /// Create a polyline shape from a vertex range.
    ///
    /// The vertices must already exist in a [`ShapeBuffers`] instance.
    ///
    /// # Arguments
    /// * `vertex_range` - `[start, end]` indices into the vertex buffer
    pub fn polyline(vertex_range: [u32; 2]) -> Self {
        let tag = f32::from_bits(GpuShapeType::Polyline as u32);
        let rng0 = f32::from_bits(vertex_range[0]);
        let rng1 = f32::from_bits(vertex_range[1]);
        Self {
            a: vector![rng0, rng1, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a triangle mesh shape from a vertex range.
    ///
    /// The vertices must already exist in a [`ShapeBuffers`] instance.
    ///
    /// # Arguments
    /// * `vertex_range` - `[start, end]` indices into the vertex buffer
    pub fn trimesh(vertex_range: [u32; 2]) -> Self {
        let tag = f32::from_bits(GpuShapeType::TriMesh as u32);
        let rng0 = f32::from_bits(vertex_range[0]);
        let rng1 = f32::from_bits(vertex_range[1]);
        Self {
            a: vector![rng0, rng1, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a cone shape (3D only).
    ///
    /// # Arguments
    /// * `half_height` - Half the height of the cone along its central axis
    /// * `radius` - The radius of the cone's base
    #[cfg(feature = "dim3")]
    pub fn cone(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cone as u32);
        Self {
            a: vector![half_height, radius, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a cylinder shape (3D only).
    ///
    /// # Arguments
    /// * `half_height` - Half the height of the cylinder along its central axis
    /// * `radius` - The radius of the cylinder
    #[cfg(feature = "dim3")]
    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cylinder as u32);
        Self {
            a: vector![half_height, radius, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Convert a Rapier/Parry shape to a GPU-compatible representation.
    ///
    /// For complex shapes (polylines, trimeshes, heightfields), vertex data is
    /// appended to the provided `buffers` and the shape stores references to those vertices.
    ///
    /// # Arguments
    /// * `shape` - The Rapier/Parry shape to convert
    /// * `buffers` - Storage for vertex data of complex shapes
    ///
    /// # Returns
    /// `Some(GpuShape)` if the shape type is supported, `None` otherwise
    pub fn from_parry(shape: &(impl Shape + ?Sized), buffers: &mut ShapeBuffers) -> Option<Self> {
        match shape.as_typed_shape() {
            TypedShape::Ball(shape) => Some(Self::ball(shape.radius)),
            TypedShape::Cuboid(shape) => Some(Self::cuboid(shape.half_extents)),
            TypedShape::Capsule(shape) => Some(Self::capsule(
                shape.segment.a,
                shape.segment.b,
                shape.radius,
            )),
            TypedShape::Polyline(shape) => {
                let base_id = buffers.vertices.len();
                buffers.vertices.extend_from_slice(shape.vertices());
                Some(Self::polyline([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            TypedShape::TriMesh(shape) => {
                let base_id = buffers.vertices.len();
                buffers.vertices.extend_from_slice(shape.vertices());
                Some(Self::trimesh([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            // HACK: we currently emulate heightfields as trimeshes or polylines
            #[cfg(feature = "dim2")]
            TypedShape::HeightField(shape) => {
                let base_id = buffers.vertices.len();
                let (vtx, _) = shape.to_polyline();
                buffers.vertices.extend_from_slice(&vtx);
                Some(Self::polyline([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            #[cfg(feature = "dim3")]
            TypedShape::HeightField(shape) => {
                let base_id = buffers.vertices.len();
                let (vtx, _) = shape.to_trimesh();
                buffers.vertices.extend_from_slice(&vtx);
                Some(Self::trimesh([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            #[cfg(feature = "dim3")]
            TypedShape::Cone(shape) => Some(Self::cone(shape.half_height, shape.radius)),
            #[cfg(feature = "dim3")]
            TypedShape::Cylinder(shape) => Some(Self::cylinder(shape.half_height, shape.radius)),
            _ => None,
        }
    }

    /// Get the shape type identifier.
    ///
    /// Extracts and decodes the shape type tag stored in the `w` component.
    ///
    /// # Returns
    /// The Rapier [`ShapeType`] enum variant corresponding to this shape
    ///
    /// # Panics
    /// Panics if the stored type tag is invalid
    pub fn shape_type(&self) -> ShapeType {
        let tag = self.a.w.to_bits();

        match tag {
            0 => ShapeType::Ball,
            1 => ShapeType::Cuboid,
            2 => ShapeType::Capsule,
            #[cfg(feature = "dim3")]
            3 => ShapeType::Cone,
            #[cfg(feature = "dim3")]
            4 => ShapeType::Cylinder,
            5 => ShapeType::Polyline,
            6 => ShapeType::TriMesh,
            _ => panic!("Unknown shape type: {}", tag),
        }
    }

    /// Get the vertex range for a polyline shape.
    ///
    /// # Returns
    /// `[start, end]` indices into the shape vertex buffer
    ///
    /// # Panics
    /// Panics if this shape is not a polyline
    pub fn polyline_rngs(&self) -> [u32; 2] {
        assert!(self.shape_type() == ShapeType::Polyline);
        [self.a.x.to_bits(), self.a.y.to_bits()]
    }

    /// Get the vertex range for a triangle mesh shape.
    ///
    /// # Returns
    /// `[start, end]` indices into the shape vertex buffer
    ///
    /// # Panics
    /// Panics if this shape is not a triangle mesh
    pub fn trimesh_rngs(&self) -> [u32; 2] {
        assert!(self.shape_type() == ShapeType::TriMesh);
        [self.a.x.to_bits(), self.a.y.to_bits()]
    }
}
