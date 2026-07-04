//! Tests for collision detection algorithms.
//!
//! These tests run on the CPU and call the shader functions directly.

mod epa;
mod gjk;
#[cfg(feature = "dim3")]
mod linalg;
mod pfm_pfm;
