//! # nexus — GPU-resident physics
//!
//! ## Running on macOS (Metal) — known issue with naga ≤ 29
//!
//! On macOS the engine runs through wgpu's Metal backend, and **naga 29's MSL
//! writer miscompiles rust-gpu loops**: a loop's `break if` condition is
//! re-evaluated after the `continuing` block has advanced the loop variables,
//! so every `while` loop whose condition is computed in the loop body exits
//! one iteration early ([gfx-rs/wgpu#4558], fixed by [gfx-rs/wgpu#9815]).
//!
//! In this engine the multibody solver's per-lane `J·v` reductions have
//! exactly one iteration per lane, so on an unpatched naga they run **zero**
//! times: contact and joint sweeps produce **zero impulses**, bodies
//! free-fall through the ground, and raising solver iterations makes the
//! blow-up *worse* (gravity integrates per iteration with nothing to cancel
//! it). CUDA and Vulkan are unaffected — the same SPIR-V never goes through
//! naga there.
//!
//! **Workaround until the wgpu fix ships** — patch naga in the workspace that
//! builds your final binary (patches only apply from the workspace root) with
//! a naga checkout carrying the [gfx-rs/wgpu#9815] fix:
//!
//! ```toml
//! [patch.crates-io]
//! naga = { path = "../naga-fixed" } # e.g. github.com/haixuanTao/naga-fixed
//! ```
//!
//! then run `cargo update -p naga` — without it the lockfile keeps the
//! registry naga and the patch is silently ignored (`[[patch.unused]]`).
//!
//! Quick sanity check: a multibody resting on a ground collider must hold its
//! height; if it sinks ~`g·dt²` per step or launches upward, you are on an
//! unpatched naga.
//!
//! [gfx-rs/wgpu#4558]: https://github.com/gfx-rs/wgpu/issues/4558
//! [gfx-rs/wgpu#9815]: https://github.com/gfx-rs/wgpu/pull/9815

// The umbrella pipeline/state currently only drive the rbd subsystem, so they
// are gated on it: without `rbd` this crate must still compile (e.g. the
// `cargo publish` verification build runs with default features only).
#[cfg(feature = "rbd")]
pub mod pipeline;
#[cfg(feature = "rbd")]
pub mod state;

#[cfg(all(feature = "dim2", feature = "rbd"))]
pub use nexus_rbd2d as rbd;
#[cfg(all(feature = "dim3", feature = "rbd"))]
pub use nexus_rbd3d as rbd;

#[cfg(feature = "rbd")]
pub use rbd::{parry, rapier};

pub mod prelude {
    #[cfg(feature = "rbd")]
    pub use crate::pipeline::*;
    #[cfg(feature = "rbd")]
    pub use crate::state::*;
}
