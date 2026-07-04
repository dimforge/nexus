//! Host-side GPU multibody: buffer packing and kernel dispatch.
//!
//! A single `GpuMultibodySet` packs N rapier `Multibody`s across multiple simulation
//! batches into flat GPU tensors. `GpuMultibodySolver::step` advances them one step,
//! dispatched one workgroup per multibody.
//!
//! Contacts and user-defined joint constraints are intentionally not handled.

#![cfg(feature = "dim3")]

mod loop_closing_joints;
mod multibody_from_rapier;
mod multibody_set;
mod multibody_solver;

pub use multibody_set::GpuMultibodySet;
pub use multibody_solver::{GpuMultibodySolver, MultibodySolverArgs};
