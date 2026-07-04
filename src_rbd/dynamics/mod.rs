//! Rigid-body dynamics: forces, velocities, constraints, and solvers.

pub use crate::shaders::dynamics::RbdSimParams;
pub use coloring::{ColoringArgs, GpuColoring};
pub use mprops_update::{GpuMpropsUpdate, GpuSyncColliderPosesShader};
pub use prep_render::{RbdInstanceDesc, WgRbdPrepRender};
pub use warmstart::{GpuWarmstart, WarmstartArgs};

mod coloring;
mod mprops_update;
mod prep_render;
pub mod warmstart;
