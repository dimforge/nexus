//! Rigid-body dynamics: forces, velocities, constraints, and solvers.

pub use crate::shaders::dynamics::RbdSimParams;
pub use coloring::{ColorBucketsArgs, ColoringArgs, GpuColoring};
pub use joint::{GpuImpulseJointSet, GpuJointSolver, JointSolverArgs};
pub use mprops_update::{GpuMpropsUpdate, GpuSyncColliderPosesShader};
#[cfg(feature = "dim3")]
pub use multibody::{GpuMultibodySet, GpuMultibodySolver, MultibodySolverArgs};
pub use prep_render::{RbdInstanceDesc, WgRbdPrepRender};
pub use solver::{GpuSolver, SolverArgs};
pub use warmstart::{GpuWarmstart, WarmstartArgs};

mod coloring;
mod joint;
mod mprops_update;
#[cfg(feature = "dim3")]
pub(crate) mod multibody;
mod prep_render;
mod solver;
pub mod warmstart;
