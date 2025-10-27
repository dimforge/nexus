//! Rigid-body dynamics (forces, velocities, etc.)

pub use body::{
    BodyCoupling, BodyCouplingEntry, BodyDesc, GpuBodySet, GpuForce, GpuMassProperties, GpuVelocity,
};

/// Rigid body definitions and GPU body set management.
pub mod body;
/// Physics integration routines (position, velocity updates).
pub mod integrate;
