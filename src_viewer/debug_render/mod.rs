//! Wireframe debug rendering of the rigid-body scene.

mod backend;
mod renderer;

pub use backend::{DebugLine, DebugPoint, LineCollector, hsla_to_rgba};
pub use renderer::{DebugRenderSettings, DebugRenderer};
