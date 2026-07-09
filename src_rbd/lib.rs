//! GPU-accelerated rigid-body physics engine built on khal/vortx.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
pub use parry2d as parry;
#[cfg(feature = "dim3")]
pub use parry3d as parry;

#[cfg(feature = "dim2")]
pub use rapier2d as rapier;
#[cfg(feature = "dim3")]
pub use rapier3d as rapier;

// Re-export the shader crate
#[cfg(feature = "dim2")]
pub use nexus_rbd_shaders2d as shaders;
#[cfg(feature = "dim3")]
pub use nexus_rbd_shaders3d as shaders;

use khal::re_exports::include_dir::{Dir, include_dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$OUT_DIR/shaders-spirv");

// Re-export commonly used types from shader crate
pub use shaders::bounding_volumes::Aabb;
pub use shaders::dynamics::{Force, Impulse, LocalMassProperties, Velocity, WorldMassProperties};
pub use shaders::shapes::Shape;
pub use shaders::{Pad, PaddedVector};

// Re-export glamx for users
pub use glamx;

const VALIDATE_LBVH_TOPOLOGY: bool = false;

pub mod dynamics;
pub mod pipeline;
pub use pipeline::RbdCapacities;

/// Broad-phase collision detection (LBVH).
pub mod broad_phase;
/// Geometric queries (ray-casting, point projection, contact generation).
pub mod queries;
/// Shape definitions.
pub mod shapes;
/// Utilities (GPU radix sort, prefix sum, etc.).
pub mod utils;

#[cfg(feature = "dim3")]
pub mod math {
    //! Dimension-specific type aliases (3D).

    pub use glamx::{Mat3, Pose3, Quat, Vec3, Vec4};

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 3;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The vector type.
    pub type Vector = Vec3;

    /// The angular vector type.
    pub type AngVector = Vec3;

    /// The point type (same as Vector in glamx).
    pub type Point = Vec3;

    /// The matrix type.
    pub type Matrix = Mat3;

    /// The rotation matrix type.
    pub type Rotation = Quat;

    /// The transformation type.
    pub type Pose = Pose3;

    /// The angular inertia of a rigid body.
    pub type AngularInertia = Mat3;

    /// The principal angular inertia of a rigid body.
    pub type PrincipalAngularInertia = Vec3;
}

#[cfg(feature = "dim2")]
pub mod math {
    //! Dimension-specific type aliases (2D).

    pub use glamx::{Mat2, Pose2, Rot2, Vec2, Vec3};

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 2;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The vector type.
    pub type Vector = Vec2;

    /// The angular vector type (scalar in 2D).
    pub type AngVector = f32;

    /// The point type (same as Vector in glamx).
    pub type Point = Vec2;

    /// The matrix type.
    pub type Matrix = Mat2;

    /// The rotation type.
    pub type Rotation = Rot2;

    /// The transformation type.
    pub type Pose = Pose2;

    /// The angular inertia of a rigid body (scalar in 2D).
    pub type AngularInertia = f32;

    /// The principal angular inertia of a rigid body (scalar in 2D).
    pub type PrincipalAngularInertia = f32;
}

/// Fixed-grid dispatch state: `0` = off, `1` = on, `2` = not-yet-initialized.
///
/// Replaces per-dispatch CUDA INDIRECT grids — which each force a
/// `stream.synchronize()` + device→host count read (a full GPU drain) in khal's
/// CUDA backend — with a fixed capacity-based grid. The affected kernels are
/// dispatched at workgroup granularity and bounds-check each thread against the
/// true count buffer, so over-launch is always correct. Also a prerequisite for
/// CUDA-graph capture of the step (indirect reads are capture-unsafe).
///
/// Defaults ON for the CUDA backend, OFF for WebGPU/Metal (their native
/// indirect dispatch is free). `NEXUS_FIXED_GRID=1` / `=0` overrides.
static FIXED_GRID_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

fn fixed_grid_env_override() -> Option<bool> {
    match std::env::var("NEXUS_FIXED_GRID").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    }
}

/// Set the fixed-grid default from the active backend (see
/// [`FIXED_GRID_STATE`]). The `NEXUS_FIXED_GRID` env var, if set, always wins.
/// Called once when the physics state is built.
pub fn set_fixed_dispatch_grid_default(is_cuda: bool) {
    use std::sync::atomic::Ordering;
    let enabled = fixed_grid_env_override().unwrap_or(is_cuda);
    FIXED_GRID_STATE.store(enabled as u8, Ordering::Relaxed);
}

pub(crate) fn fixed_dispatch_grid_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match FIXED_GRID_STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => fixed_grid_env_override().unwrap_or(false),
    }
}

/// Pick the dispatch grid for an indirect-capable kernel: the fixed
/// capacity-based `fixed` grid when [`fixed_dispatch_grid_enabled`], else the
/// original `indirect` buffer. `fixed` MUST cover the kernel's max count (the
/// kernel bounds-checks against the true count internally, so over-launch is
/// safe).
pub(crate) fn dispatch_grid<'a>(
    indirect: &'a vortx::tensor::Tensor<[u32; 3]>,
    fixed: [u32; 3],
) -> khal::backend::DispatchGrid<'a, khal::backend::GpuBackend> {
    use khal::backend::DispatchGrid;
    if fixed_dispatch_grid_enabled() {
        DispatchGrid::Grid(fixed)
    } else {
        DispatchGrid::Indirect(indirect.buffer())
    }
}
