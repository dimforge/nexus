//! Physics simulation pipeline orchestrating broad-phase, narrow-phase, and constraint solving.
//!
//! This module provides the high-level physics pipeline that coordinates all stages of a physics
//! simulation step on the GPU. The pipeline manages collision detection, contact generation,
//! constraint solving, and integration.

mod insertion_removal;
mod lbvh_validation;
mod rbd_state;
mod rbd_state_from_rapier;
mod rbd_step;

pub use rbd_state::{RbdCapacities, RbdResizePolicy, RbdState, RunStats};
pub use rbd_step::RbdPipeline;
