#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
pub extern crate nexus2d as nexus;
#[cfg(feature = "dim3")]
pub extern crate nexus3d as nexus;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

mod backend;
pub mod debug_render;
mod graphics;
#[cfg(feature = "dim3")]
pub mod sensors;
mod ui;
pub mod viewer;

pub use backend::BackendType;
pub use debug_render::{DebugRenderSettings, DebugRenderer, LbvhStatus, MpmDebugRenderMode};
#[cfg(feature = "dim3")]
pub use graphics::{RenderMaterial, VisualTexture};
#[cfg(feature = "dim3")]
pub use sensors::{SensorAttachment, SensorCamera, SensorCamera3d};
pub use viewer::{NexusViewer, UiState};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
}

/// The tabs of the viewer panel. Only one is shown at a time.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UiSection {
    Performance,
    Settings,
    Examples,
    DebugRender,
}

/// The kind of solver a registered demo uses. Used only to group demos in the
/// picker UI.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DemoKind {
    Rbd,
    Mpm,
    /// Large rigid-body scenes for performance work.
    Stress,
}

/// A loop transition requested from the UI: stop entirely, or switch to another
/// registered demo. The target index is carried in [`UiState::selected_demo`];
/// this only signals the example-owned `while viewer.render()` loop to exit so
/// the browser can run the next demo.
pub(crate) enum Transition {
    Quit,
    Switch,
}
