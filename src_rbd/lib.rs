//! GPU-accelerated rigid-body physics engine built on khal/vortx.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
pub use parry2d as parry;
#[cfg(feature = "dim3")]
pub use parry3d as parry;

#[cfg(feature = "dim2")]
pub use rapier2d as rapier;
#[cfg(feature = "dim3")]
pub use rapier3d as rapier;

// Re-export the shader crate
#[cfg(feature = "dim2")]
pub use nexus_rbd_shaders2d as shaders;
#[cfg(feature = "dim3")]
pub use nexus_rbd_shaders3d as shaders;

use khal::re_exports::include_dir::{Dir, include_dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

// Re-export commonly used types from shader crate
pub use shaders::bounding_volumes::Aabb;
pub use shaders::dynamics::{Force, Impulse, LocalMassProperties, Velocity, WorldMassProperties};
pub use shaders::shapes::Shape;
pub use shaders::{Pad, PaddedVector};

// Re-export glamx for users
pub use glamx;

pub mod dynamics;

/// Broad-phase collision detection (LBVH).
pub mod broad_phase;
/// Geometric queries (ray-casting, point projection, contact generation).
pub mod queries;
/// Shape definitions.
pub mod shapes;
/// Utilities (GPU radix sort, prefix sum, etc.).
pub mod utils;

#[cfg(feature = "dim3")]
pub mod math {
    //! Dimension-specific type aliases (3D).

    pub use glamx::{Mat3, Pose3, Quat, Vec3, Vec4};

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 3;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The vector type.
    pub type Vector = Vec3;

    /// The angular vector type.
    pub type AngVector = Vec3;

    /// The point type (same as Vector in glamx).
    pub type Point = Vec3;

    /// The matrix type.
    pub type Matrix = Mat3;

    /// The rotation matrix type.
    pub type Rotation = Quat;

    /// The transformation type.
    pub type Pose = Pose3;

    /// The angular inertia of a rigid body.
    pub type AngularInertia = Mat3;

    /// The principal angular inertia of a rigid body.
    pub type PrincipalAngularInertia = Vec3;
}

#[cfg(feature = "dim2")]
pub mod math {
    //! Dimension-specific type aliases (2D).

    pub use glamx::{Mat2, Pose2, Rot2, Vec2, Vec3};

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 2;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The vector type.
    pub type Vector = Vec2;

    /// The angular vector type (scalar in 2D).
    pub type AngVector = f32;

    /// The point type (same as Vector in glamx).
    pub type Point = Vec2;

    /// The matrix type.
    pub type Matrix = Mat2;

    /// The rotation type.
    pub type Rotation = Rot2;

    /// The transformation type.
    pub type Pose = Pose2;

    /// The angular inertia of a rigid body (scalar in 2D).
    pub type AngularInertia = f32;

    /// The principal angular inertia of a rigid body (scalar in 2D).
    pub type PrincipalAngularInertia = f32;
}
