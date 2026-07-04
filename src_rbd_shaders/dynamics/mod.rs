//! Rigid body dynamics module.
//!
//! This module provides:
//! - Body state and mass properties
//! - Contact constraints
//! - Constraint solver (PGS/Sequential Impulse)
//! - Graph coloring for parallel solving

// Data structures and algorithms
mod body;
mod constraint;
mod sim_params;
mod solver_utils;
mod warmstart;

// GPU compute shader kernels
mod coloring;
mod mprops_update;
mod prep_render;
mod solver;

pub use body::*;
pub use constraint::*;
pub use sim_params::*;
pub use coloring::*;
pub use mprops_update::*;
pub use prep_render::*;
pub use solver::*;
pub use solver_utils::warmstart_body;
pub use warmstart::*;
