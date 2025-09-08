#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]

extern crate nalgebra as na;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

use slang_hal::re_exports::include_dir;
use slang_hal::re_exports::minislang::SlangCompiler;

pub mod dynamics;
pub mod shapes;

pub mod math {
    pub use rapier::math::*;

    #[cfg(feature = "dim2")]
    pub type GpuSim = gla::geometry::GpuSim2;
    #[cfg(feature = "dim3")]
    pub type GpuSim = gla::geometry::GpuSim3;

    #[cfg(feature = "dim2")]
    pub type AngularInertia<N> = N;
    #[cfg(feature = "dim3")]
    pub type AngularInertia<N> = na::Matrix3<N>;
}

pub const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders");
pub fn register_shaders(compiler: &mut SlangCompiler) {
    gla::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}
