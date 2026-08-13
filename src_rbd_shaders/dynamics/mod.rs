//! Rigid body dynamics module.
//!
//! This module provides:
//! - Body state and mass properties
//! - Contact constraints
//! - Joint constraints
//! - Constraint solver (PGS/Sequential Impulse)
//! - Graph coloring for parallel solving

// Data structures and algorithms
mod body;
mod constraint;
mod joint;
mod joint_constraint;
mod joint_constraint_builder;
mod multibody;
mod sim_params;
mod solver_utils;
mod warmstart;

// GPU compute shader kernels
mod color_buckets;
mod coloring;
mod mprops_update;
mod prep_render;
mod solver;

pub use body::*;
pub use constraint::*;
pub use joint::{
    ACCELERATION_BASED, ANG_AXES_MASK, FORCE_BASED, GenericJoint, ImpulseJoint, JointLimits,
    JointMotor, LIN_AXES_MASK, MotorParameters, SPATIAL_DIM,
};
pub use joint_constraint::*;
pub use joint_constraint_builder::{JointConstraintBuilder, JointConstraintHelper, new_helper};
pub use multibody::*;
pub use sim_params::*;
// Re-export solver items; update_constraint comes from joint_constraint_builder for joints
pub use color_buckets::*;
pub use coloring::*;
pub use mprops_update::*;
pub use prep_render::*;
pub use solver::*;
pub use solver_utils::warmstart_body;
pub use warmstart::*;
