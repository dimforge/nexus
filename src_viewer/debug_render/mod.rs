//! Wireframe debug rendering of the physics scene.

mod backend;
mod mpm;
mod renderer;

pub use backend::{DebugLine, DebugPoint, LineCollector, hsla_to_rgba};
pub use mpm::MpmDebugRenderMode;
pub use renderer::{DebugRenderSettings, DebugRenderer, LbvhStatus};
