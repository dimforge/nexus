//! Generic impulse-joint constraints for joints touching one or two multibodies.
//!
//! Used for any impulse joint whose endpoints are not both free rigid bodies —
//! when at least one side is a multibody link the regular `JointConstraint`
//! solver path can't propagate impulses, so this dedicated path is used instead.
//!
//! Kernels must run in order per step:
//!
//!   1. `gpu_mb_init_impulse_joint_constraints` — once per step, after FK / LU.
//!   2. `gpu_mb_update_impulse_joint_constraints` — once per substep.
//!   3. `gpu_mb_solve_impulse_joint_constraints` — one PGS sweep, updates both
//!      sides' velocities.
//!   4. `gpu_mb_remove_impulse_joint_constraint_bias` — strips the positional
//!      bias from `rhs` before the stabilization sweep.

mod helper;
mod jacobians;
mod kernels;
mod types;
mod update;

pub use kernels::*;
pub use types::*;
