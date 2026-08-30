//! Per-environment reset scatter for RL teleport / reset primitives.
//!
//! Copies one environment's carry-over multibody state (SoA link workspace,
//! static link descriptors, generalized coordinates and velocities) from a
//! compact contiguous staging blob into the batch-interleaved live buffers.

use glamx::Vec4;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::types::MultibodyLinkStatic;
use super::ws_soa::{WS_COORDS, WS_LTP, WS_LTW, WS_QUADS};

/// One entry of the batched-reset list: restore environment `env` from the
/// GPU-resident template `template`.
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct EnvResetRecord {
    /// Destination environment (batch) index.
    pub env: u32,
    /// Index of the resident template to restore from.
    pub template: u32,
}

/// Parameters of the single-env staging reset ([`gpu_mb_env_reset`]).
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbEnvResetParams {
    /// Destination environment (batch) index.
    pub dst_env: u32,
    /// Total number of simulation batches (the interleave stride).
    pub num_batches: u32,
    /// Links per batch (with padding slots).
    pub links_per_batch: u32,
    /// Generalized-velocity entries per batch.
    pub dofs_per_batch: u32,
}

/// Parameters shared by the two batched multibody reset passes
/// ([`gpu_mb_env_reset_batch`] and [`gpu_mb_env_reset_batch_dofs`]).
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbEnvResetBatchParams {
    /// Total number of simulation batches (the interleave stride).
    pub num_batches: u32,
    /// Links per batch (with padding slots).
    pub links_per_batch: u32,
    /// Generalized-velocity entries per batch.
    pub dofs_per_batch: u32,
    /// Number of entries in the reset list.
    pub num_resets: u32,
}

/// Parameters of the rigid-body half of the batched reset
/// ([`gpu_env_reset_bodies`]).
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct EnvResetBodiesParams {
    /// Body slots per environment (template stride).
    pub bodies_per_env: u32,
    /// Velocity slots per environment (template stride).
    pub vels_per_env: u32,
    /// Number of entries in the reset list.
    pub num_resets: u32,
    /// Total number of simulation batches (the interleave stride).
    pub num_batches: u32,
}

/// Scatters one staged env state into the interleaved buffers. Dispatch
/// `[links_per_batch · WS_QUADS, 1, 1]` threads, the largest of the three
/// per-element loops (`links_per_batch · WS_QUADS >= links_per_batch`, and
/// `dofs_per_batch <= links_per_batch · WS_QUADS` for any real multibody).
///
/// `staging_dofs` holds `dofs_per_batch` generalized velocities. Only the
/// velocity section of `dof_state` is written; the sections after it are
/// static configuration (damping, armature, springs), not per-episode state.
/// The generalized coordinates live in the workspace quads copied above.
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
    #[spirv(uniform, descriptor_set = 0, binding = 6)] params: &MbEnvResetParams,
) {
    let i = invocation_id.x;
    let env = params.dst_env;
    let nb = params.num_batches;
    let lpb = params.links_per_batch;
    let dpb = params.dofs_per_batch;

    if i < lpb * WS_QUADS {
        // Staging is per-env dense; the live workspace is interleaved at
        // per-link-record granularity (record = `WS_QUADS` dense quads).
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] resets: &[EnvResetRecord],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] offsets: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] links_workspace: &mut [Vec4],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] params: &MbEnvResetBatchParams,
) {
    let i = invocation_id.x;
    let r = invocation_id.y;
    let nb = params.num_batches;
    let lpb = params.links_per_batch;
    if r >= params.num_resets {
        return;
    }
    let meta = resets.read(r as usize);
    let env = meta.env;
    let t = meta.template;
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
        // Live workspace is interleaved at per-link-record granularity.
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] resets: &[EnvResetRecord],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dof_vels: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] params: &MbEnvResetBatchParams,
) {
    let i = invocation_id.x;
    let r = invocation_id.y;
    let nb = params.num_batches;
    let lpb = params.links_per_batch;
    let dpb = params.dofs_per_batch;
    if r >= params.num_resets {
        return;
    }
    let meta = resets.read(r as usize);
    let env = meta.env;
    let t = meta.template;

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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] resets: &[EnvResetRecord],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] offsets: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_poses: &mut [crate::Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    vels: &mut [crate::dynamics::body::Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &EnvResetBodiesParams,
) {
    let i = invocation_id.x;
    let r = invocation_id.y;
    let bps = params.bodies_per_env;
    let vs = params.vels_per_env;
    let nb = params.num_batches;
    if r >= params.num_resets {
        return;
    }
    let meta = resets.read(r as usize);
    let env = meta.env;
    let t = meta.template;
    let off = offsets.read(r as usize);

    // Templates are dense per-env arrays; the live per-body buffers are
    // batch-interleaved (`local * num_batches + env`).
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
