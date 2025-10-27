#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

extern crate nalgebra as na;
/// Re-export of the Rapier 2D physics engine.
///
/// This is available when the `dim2` feature is enabled.
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
/// Re-export of the Rapier 3D physics engine.
///
/// This is available when the `dim3` feature is enabled.
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

use slang_hal::re_exports::include_dir;
use slang_hal::re_exports::minislang::SlangCompiler;

/// GPU-accelerated rigid body dynamics simulation.
///
/// This module provides structures and methods for managing physics bodies
/// on the GPU, including body state, integration, and coupling with colliders.
pub mod dynamics;
/// GPU-compatible shape representations.
///
/// This module defines shape types and utilities for converting Rapier/Parry shapes
/// to GPU-friendly formats with vertex buffers.
pub mod shapes;

/// Mathematical types and utilities for physics simulation.
///
/// Re-exports Rapier's math types and defines dimension-specific type aliases
/// for GPU simulation and angular inertia calculations.
pub mod math {
    /// Re-export all mathematical types from Rapier (vectors, matrices, etc.)
    pub use rapier::math::*;

    /// GPU similarity transformation for 2D simulations (translation + rotation).
    #[cfg(feature = "dim2")]
    pub type GpuSim = stensor::geometry::GpuSim2;
    /// GPU similarity transformation for 3D simulations (translation + rotation).
    #[cfg(feature = "dim3")]
    pub type GpuSim = stensor::geometry::GpuSim3;

    /// Angular inertia type for 2D simulations (scalar).
    #[cfg(feature = "dim2")]
    pub type AngularInertia<N> = N;
    /// Angular inertia type for 3D simulations (3x3 matrix).
    #[cfg(feature = "dim3")]
    pub type AngularInertia<N> = na::Matrix3<N>;
}

/// Directory containing the Slang shader source files.
///
/// This includes all shader code needed for GPU-accelerated physics simulation.
pub const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders");

/// Register all required shaders with a Slang compiler.
///
/// This function registers both stensor shaders and Nexus-specific shader sources
/// needed for GPU physics simulation.
///
/// # Arguments
/// * `compiler` - The Slang compiler instance to register shaders with
pub fn register_shaders(compiler: &mut SlangCompiler) {
    stensor::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}
