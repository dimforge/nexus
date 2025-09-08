//! Rigid-body dynamics (forces, velocities, etc.)

pub use body::{BodyDesc, GpuBodySet, GpuForce, GpuMassProperties, GpuVelocity};

pub mod body;
pub mod integrate;
