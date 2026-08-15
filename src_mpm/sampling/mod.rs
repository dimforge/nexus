//! Surface sampling for rigid body coupling.
//!
//! Samples particles on the surfaces of rigid body colliders for two-way
//! MPM-rigid body coupling. In 2D, samples polyline edges; in 3D, samples
//! triangle mesh surfaces.

#[cfg(feature = "dim2")]
pub use sample_polyline::*;
#[cfg(feature = "dim3")]
pub use sample_trimesh::*;

#[cfg(feature = "dim2")]
mod sample_polyline;
#[cfg(feature = "dim3")]
mod sample_trimesh;
