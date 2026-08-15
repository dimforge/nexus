//! GPU-accelerated Material Point Method simulation with rigid body coupling.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]
#![allow(missing_docs)]

#[cfg(feature = "dim2")]
pub use nexus_mpm_shaders2d as mpm_shaders;
#[cfg(feature = "dim3")]
pub use nexus_mpm_shaders3d as mpm_shaders;

#[cfg(feature = "dim2")]
pub extern crate nexus_rbd2d as nexus_rbd;
#[cfg(feature = "dim3")]
pub extern crate nexus_rbd3d as nexus_rbd;

#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

use khal::re_exports::include_dir::{Dir, include_dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

pub mod grid;
pub mod models;
pub mod pipeline;
pub use pipeline::MpmCapacities;
pub(crate) mod sampling;
pub mod solver;
#[cfg(feature = "dim3")]
pub mod trimesh;
