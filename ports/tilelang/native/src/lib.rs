//! Native scene uploads and optional Metal execution for the TileLang port.
#[cfg(feature = "metal")]
mod metal_runtime;
#[cfg(feature = "metal")]
pub use metal_runtime::*;
#[cfg(feature = "metal")]
pub mod frame;
#[cfg(any(feature = "rust-reference", feature = "scene-upload"))]
pub mod upload;
#[cfg(any(feature = "rust-reference", feature = "scene-upload"))]
pub mod upload_cases;
