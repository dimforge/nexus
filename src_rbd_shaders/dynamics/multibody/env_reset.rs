//! Per-environment reset scatter for RL teleport / reset primitives.
//!
//! Copies one environment's carry-over multibody state (SoA link workspace,
//! static link descriptors, generalized coordinates and velocities) from a
//! compact contiguous staging blob into the batch-interleaved live buffers.

use glamx::{UVec4, Vec4};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::types::MultibodyLinkStatic;
use super::ws_soa::{WS_COORDS, WS_LTP, WS_LTW, WS_QUADS};

/// Scatters one staged env state into the interleaved buffers. Dispatch
/// `[links_per_batch · WS_QUADS, 1, 1]` threads, the largest of the three
/// per-element loops (`links_per_batch · WS_QUADS >= links_per_batch`, and
/// `dofs_per_batch <= links_per_batch · WS_QUADS` for any real multibody).
///
/// `staging_dofs` holds `dofs_per_batch` generalized coordinates followed by
/// `dofs_per_batch` generalized velocities. Only the velocity section of
/// `dof_state` is written; the sections after it are static configuration
/// (damping, armature, springs), not per-episode state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_env_reset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] staging_ws: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    staging_links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] staging_dofs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_state: &mut [f32],
    // x = dst_env, y = num_batches, z = links_per_batch, w = dofs_per_batch.
    #[spirv(uniform, descriptor_set = 0, binding = 6)] params: &UVec4,
) {
    let i = invocation_id.x;
    let env = params.x;
    let nb = params.y;
    let lpb = params.z;
    let dpb = params.w;

    if i < lpb * WS_QUADS {
        let link = i / WS_QUADS;
        let q = i % WS_QUADS;
        links_workspace.write(
            ((link * nb + env) * WS_QUADS + q) as usize,
            staging_ws.read(i as usize),
        );
    }
    if i < lpb {
        links_static.write((i * nb + env) as usize, staging_links.read(i as usize));
    }
    if i < dpb {
        dof_state.write((i * nb + env) as usize, staging_dofs.read(i as usize));
    }
}

/// Batched, template-resident variant of [`gpu_mb_env_reset`]: N resets in one
/// dispatch, reading from template blobs that live on the GPU permanently
/// (uploaded once at build) instead of a per-reset staging upload. A terrain
/// teleport offset is applied in-kernel to the free root's world position
/// (local-to-world / local-to-parent translations plus coords c0..c2), so the
/// host never clones and translates a snapshot per reset.
///
/// This pass writes the link workspace only; [`gpu_mb_env_reset_batch_dofs`]
/// writes the static links and the DoF sections. They are split so each fits
/// the 8-storage-buffer WebGPU limit.
///
/// Dispatch `[lpb · WS_QUADS, num_resets, 1]` threads.
///
/// `link_flags` is constant per robot: bit 0 = a valid link of a free-root
/// multibody (translate `WS_LTW`), bit 1 = that link is the root (translate
/// `WS_LTP` and coords c0..c2).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_env_reset_batch(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] templates_ws: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] link_flags: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] resets: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] offsets: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] links_workspace: &mut [Vec4],
    // x = num_batches, y = links_per_batch, z = dofs_per_batch, w = num_resets.
    #[spirv(uniform, descriptor_set = 0, binding = 5)] params: &UVec4,
) {
    let i = invocation_id.x;
    let r = invocation_id.y;
    let nb = params.x;
    let lpb = params.y;
    if r >= params.w {
        return;
    }
    let meta = resets.read(r as usize);
    let env = meta.x;
    let t = meta.y;
    let off = offsets.read(r as usize);

    if i < lpb * WS_QUADS {
        let mut v = templates_ws.read((t * lpb * WS_QUADS + i) as usize);
        let link = i / WS_QUADS;
        let q = i % WS_QUADS;
        let f = link_flags.read(link as usize);
        // `WS_LTW` and `WS_LTP` are rot|trans quad pairs, so `+1` is the
        // translation quad. Coords c0..c3 share quad `WS_COORDS` (c3 is a
        // rotational DoF, never offset, so its `.w` lane stays untouched).
        if f & 1 != 0 && q == WS_LTW + 1 {
            v.x += off.x;
            v.y += off.y;
            v.z += off.z;
        }
        if f & 2 != 0 && (q == WS_LTP + 1 || q == WS_COORDS) {
            v.x += off.x;
            v.y += off.y;
            v.z += off.z;
        }
        links_workspace.write(((link * nb + env) * WS_QUADS + q) as usize, v);
    }
}

/// Static-link and DoF half of the batched reset, split from
/// [`gpu_mb_env_reset_batch`] so each pass fits 8 storage buffers.
///
/// The teleport offset is not needed here: static links carry no world
/// position, and generalized coords are translation-invariant (the free root's
/// world position lives in the workspace coords quad the other pass handles).
///
/// Dispatch `[max(lpb, dpb), num_resets, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_env_reset_batch_dofs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    templates_links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] resets: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dof_vels: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_state: &mut [f32],
    // x = num_batches, y = links_per_batch, z = dofs_per_batch, w = num_resets.
    #[spirv(uniform, descriptor_set = 0, binding = 5)] params: &UVec4,
) {
    let i = invocation_id.x;
    let r = invocation_id.y;
    let nb = params.x;
    let lpb = params.y;
    let dpb = params.z;
    if r >= params.w {
        return;
    }
    let meta = resets.read(r as usize);
    let env = meta.x;
    let t = meta.y;

    if i < lpb {
        links_static.write(
            (i * nb + env) as usize,
            templates_links.read((t * lpb + i) as usize),
        );
    }
    if i < dpb {
        dof_state.write(
            (i * nb + env) as usize,
            dof_vels.read((r * dpb + i) as usize),
        );
    }
}

/// Rigid-body half of the batched reset: copies each reset env's `body_poses`
/// and `vels` slices (env-major, unlike the interleaved multibody buffers)
/// from the resident templates, adding the teleport offset to the poses of the
/// bodies flagged in `body_mask` (free-multibody links; ground and terrain stay
/// put).
///
/// Dispatch `[max(bodies_per_env, vels_per_env), num_resets, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_env_reset_bodies(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] templates_poses: &[crate::Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    templates_vels: &[crate::dynamics::body::Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_mask: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] resets: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] offsets: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_poses: &mut [crate::Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    vels: &mut [crate::dynamics::body::Velocity],
    // x = bodies_per_env, y = vels_per_env, z = num_resets.
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &UVec4,
) {
    let i = invocation_id.x;
    let r = invocation_id.y;
    let bps = params.x;
    let vs = params.y;
    let nb = params.w;
    if r >= params.z {
        return;
    }
    let meta = resets.read(r as usize);
    let env = meta.x;
    let t = meta.y;
    let off = offsets.read(r as usize);

    if i < bps {
        let mut p = templates_poses.read((t * bps + i) as usize);
        if body_mask.read(i as usize) != 0 {
            p.translation.x += off.x;
            p.translation.y += off.y;
            p.translation.z += off.z;
        }
        body_poses.write((i * nb + env) as usize, p);
    }
    if i < vs {
        vels.write(
            (i * nb + env) as usize,
            templates_vels.read((t * vs + i) as usize),
        );
    }
}
