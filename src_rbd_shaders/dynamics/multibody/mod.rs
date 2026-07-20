//! Reduced-coordinates articulated multibody (3D).
//!
//! GPU port of rapier's `Multibody`, restricted to:
//! - Forward kinematics (link world transforms from generalized coordinates).
//! - Body jacobians (one `6 × ndofs` per link).
//! - Augmented mass matrix via CRBA: `M = Σ Jᵢᵀ Mᵢ Jᵢ`.
//! - Gravity generalized force: `τ = Σ Jᵢᵀ (mᵢ g, 0)`.
//! - In-place LU solve of `M ẍ = τ`.
//!
//! No constraints, contacts, or Coriolis terms (not in scope).
//!
//! Links and multibodies are stored flat across all simulation batches. Each
//! batch has a capacity; unused slots are padded out and skipped via per-batch
//! length counts.

mod compute_dynamics_pre;
mod contact_constraints;
mod gravity_and_lu;
mod impulse_joint_constraints;
mod integrate;
mod jacobian;
mod joint_constraints;
mod lu;
mod mass_matrix;
mod solve_constraints;
mod types;
mod utils;

pub use compute_dynamics_pre::*;
pub use contact_constraints::*;
pub use gravity_and_lu::*;
pub use impulse_joint_constraints::*;
pub use integrate::*;
pub use joint_constraints::*;
pub use solve_constraints::*;
pub use types::*;
pub use utils::*;
