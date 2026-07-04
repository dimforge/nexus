//! Direct-to-renderer preparation shader.
//!
//! Computes per-instance render data (world position + deformation matrix + color)
//! straight into a renderer's instance buffers, so the result never has to leave
//! the GPU. The zero-readback counterpart of the CPU-side
//! `update_instances_from_poses` path in the viewer.
//!
//! The three outputs are flat `&mut [f32]` (never `&mut [Vec3]`) to match the
//! renderer's tight stride and to avoid the std430 `Vec3`-alignment/straddle
//! pitfall in storage buffers:
//! - 3D: `positions[i*3..]` (Vec3), `deformations[i*9..]` (Mat3, 3 columns),
//!   `colors[i*4..]` (RGBA).
//! - 2D: `positions[i*2..]` (Vec2), `deformations[i*4..]` (Mat2, 2 columns),
//!   `colors[i*4..]` (RGBA).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::{Pose, Vector};
use glamx::*;

/// Per-instance render descriptor, indexing into the body-pose buffer and
/// carrying the same collider-local offset / scale / color the CPU render path
/// stores in each `InstancedNodeEntry`. Built CPU-side and uploaded only when the
/// instance set changes.
///
/// Layout is std430-clean: every `Vec3` lands on a 16-byte boundary so it never
/// straddles, and the struct size is a multiple of 16.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct RbdInstanceDesc {
    /// RGBA color written verbatim to the renderer color buffer.
    pub color: Vec4, // 0..16
    /// Collider-local (or visual-mesh-local) offset, composed onto the body pose.
    pub local_pose: Pose, // 16..48 (Pose3 = 32 bytes)
    /// Per-axis render scale applied to the deformation columns.
    pub scale: Vector, // 48..60 (Vec3)
    /// Index of the source body pose in the `body_poses` buffer.
    pub pose_index: u32, // 60..64
}

/// Per-instance render descriptor (2D).
#[cfg(feature = "dim2")]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct RbdInstanceDesc {
    /// RGBA color written verbatim to the renderer color buffer.
    pub color: Vec4, // 0..16
    /// Collider-local (or visual-mesh-local) offset, composed onto the body pose.
    pub local_pose: Pose, // 16..32 (Pose2 = 16 bytes)
    /// Per-axis render scale applied to the deformation columns.
    pub scale: Vector, // 32..40 (Vec2)
    /// Index of the source body pose in the `body_poses` buffer.
    pub pose_index: u32, // 40..44
    /// Padding so the struct size is a multiple of 16.
    pub _pad: u32, // 44..48
}

/// GPU kernel: write per-instance render data straight into the renderer's
/// instance buffers. Dispatched with one thread per instance.
#[cfg(feature = "dim3")]
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_rbd_prep_render(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] positions: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] deformations: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] colors: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] descriptors: &[RbdInstanceDesc],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] count: &u32,
) {
    let i = invocation_id.x;
    if i >= *count {
        return;
    }
    let idx = i as usize;
    let desc = descriptors.at(idx);

    // Same composition as `update_instances_from_poses` / `pose_to_render_data`.
    let pose = *body_poses.at(desc.pose_index as usize) * desc.local_pose;
    let p = pose.translation;
    let rot = Mat3::from_quat(pose.rotation);
    let c0 = rot.x_axis * desc.scale.x;
    let c1 = rot.y_axis * desc.scale.y;
    let c2 = rot.z_axis * desc.scale.z;

    let pb = idx * 3;
    *positions.at_mut(pb) = p.x;
    *positions.at_mut(pb + 1) = p.y;
    *positions.at_mut(pb + 2) = p.z;

    let db = idx * 9;
    *deformations.at_mut(db) = c0.x;
    *deformations.at_mut(db + 1) = c0.y;
    *deformations.at_mut(db + 2) = c0.z;
    *deformations.at_mut(db + 3) = c1.x;
    *deformations.at_mut(db + 4) = c1.y;
    *deformations.at_mut(db + 5) = c1.z;
    *deformations.at_mut(db + 6) = c2.x;
    *deformations.at_mut(db + 7) = c2.y;
    *deformations.at_mut(db + 8) = c2.z;

    let cb = idx * 4;
    *colors.at_mut(cb) = desc.color.x;
    *colors.at_mut(cb + 1) = desc.color.y;
    *colors.at_mut(cb + 2) = desc.color.z;
    *colors.at_mut(cb + 3) = desc.color.w;
}

/// GPU kernel: write per-instance render data straight into the renderer's
/// instance buffers (2D). Dispatched with one thread per instance.
#[cfg(feature = "dim2")]
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_rbd_prep_render(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] positions: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] deformations: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] colors: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] descriptors: &[RbdInstanceDesc],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] count: &u32,
) {
    let i = invocation_id.x;
    if i >= *count {
        return;
    }
    let idx = i as usize;
    let desc = descriptors.at(idx);

    let pose = *body_poses.at(desc.pose_index as usize) * desc.local_pose;
    let p = pose.translation;
    // 2D rotation matrix columns: [cos, sin], [-sin, cos], scaled per axis.
    let cos = pose.rotation.re;
    let sin = pose.rotation.im;
    let c0 = Vec2::new(cos, sin) * desc.scale.x;
    let c1 = Vec2::new(-sin, cos) * desc.scale.y;

    let pb = idx * 2;
    *positions.at_mut(pb) = p.x;
    *positions.at_mut(pb + 1) = p.y;

    let db = idx * 4;
    *deformations.at_mut(db) = c0.x;
    *deformations.at_mut(db + 1) = c0.y;
    *deformations.at_mut(db + 2) = c1.x;
    *deformations.at_mut(db + 3) = c1.y;

    let cb = idx * 4;
    *colors.at_mut(cb) = desc.color.x;
    *colors.at_mut(cb + 1) = desc.color.y;
    *colors.at_mut(cb + 2) = desc.color.z;
    *colors.at_mut(cb + 3) = desc.color.w;
}
