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
mod graphics;
mod ui;
pub mod viewer;

pub use backend::BackendType;
#[cfg(feature = "dim3")]
pub use graphics::RenderMaterial;
pub use viewer::{NexusViewer, UiState};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
}

#[derive(Copy, Clone)]
pub struct UiSections {
    pub show_examples: bool,
    pub show_settings: bool,
    pub show_performance: bool,
}

/// The kind of solver a registered demo uses. Used only to group demos in the
/// picker UI.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DemoKind {
    Rbd,
}

/// A loop transition requested from the UI: stop entirely, or switch to another
/// registered demo. The target index is carried in [`UiState::selected_demo`];
/// this only signals the example-owned `while viewer.render()` loop to exit so
/// the browser can run the next demo.
pub(crate) enum Transition {
    Quit,
    Switch,
}
