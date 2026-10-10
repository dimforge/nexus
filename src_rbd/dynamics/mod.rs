//! Rigid-body dynamics: forces, velocities, constraints, and solvers.

#[cfg(feature = "dim3")]
pub use crate::shaders::dynamics::FrictionModel;
pub use crate::shaders::dynamics::RbdSimParams;
pub use body::{BodyCoupling, BodyCouplingEntry, BodyDesc, GpuBodySet};
pub use canonical_order::{CanonicalContactsArgs, GpuCanonicalOrder, contact_sort_collider_shift};
pub use coloring::{ColorBucketsArgs, ColorStatsBuffer, ColoringArgs, GpuColoring};
pub use contact_kernels::{ContactConstraints, ContactImpulseSnapshot, ContactTiles};
pub use contact_recycling::ContactRecycleStates;
pub use joint::{GpuImpulseJointSet, GpuJointSolver, JointSolverArgs, convert_joint_motor};
pub use mass_splitting::{GpuMassSplitting, HubState, SplitArgs};
pub use mprops_update::{GpuMpropsUpdate, GpuSyncColliderPosesShader};
#[cfg(feature = "dim3")]
pub use multibody::{
    GpuMultibodySet, GpuMultibodySnapshot, GpuMultibodySolver, MultibodySolverArgs,
};
pub use prep_render::{RbdInstanceDesc, WgRbdPrepRender};
pub use solver::{GpuSolver, SolverArgs};
pub use warmstart::{GpuWarmstart, SeedColorsArgs};

pub mod body;
mod canonical_order;
mod coloring;
pub(crate) mod contact_kernels;
mod contact_recycling;
mod joint;
mod mass_splitting;
mod mprops_update;
#[cfg(feature = "dim3")]
pub(crate) mod multibody;
mod prep_render;
mod solver;
pub mod warmstart;
