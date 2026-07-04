//! GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm) implementations.

mod cso_point;
#[cfg(feature = "dim2")]
mod epa2;
#[cfg(feature = "dim3")]
mod epa3;
mod gjk;
#[cfg(feature = "dim2")]
mod voronoi_simplex2;
#[cfg(feature = "dim3")]
mod voronoi_simplex3;

pub use cso_point::FLT_EPS;
#[cfg(feature = "dim2")]
pub use epa2::Epa2 as Epa;
#[cfg(feature = "dim3")]
pub use epa3::Epa3 as Epa;
pub use gjk::*;
#[cfg(feature = "dim2")]
pub use voronoi_simplex2::*;
#[cfg(feature = "dim3")]
pub use voronoi_simplex3::*;
