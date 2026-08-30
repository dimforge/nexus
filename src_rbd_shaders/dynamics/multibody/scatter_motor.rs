//! GPU motor-target scatter: writes per-(env, actuated-joint) target positions
//! straight into `links_static` on the GPU, replacing a host-side
//! `set_motors` + whole-mirror upload every step. This is what lets an RL
//! policy drive the motors without a host round-trip, and therefore what makes
//! a rollout capturable into a CUDA graph (no per-step host writes).
//!
//! `links_static` is batch-interleaved: link `l` of env `e` lives at
//! `l · num_envs + e`. Targets are row-major `[num_actuated x num_envs]`,
//! element `(j, env)` at `j · num_envs + env`, matching the policy action
//! buffer layout.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::types::MultibodyLinkStatic;

/// Parameters of the on-device delay-state refresh
/// ([`gpu_mb_delay_state_update`]).
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbDelayStateParams {
    /// Number of actuated joints (rows of the target tensor).
    pub num_actuated: u32,
    /// Number of environments (columns of the target tensor).
    pub num_envs: u32,
    /// Per-env stride of the delay-state buffer (`2 + links_per_batch`).
    pub stride: u32,
}

/// Parameters of the per-step delay tick ([`gpu_mb_delay_tick`]).
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbDelayTickParams {
    /// Number of environments.
    pub num_envs: u32,
    /// Per-env stride of the delay-state buffer (`2 + links_per_batch`).
    pub stride: u32,
}

/// One thread per (actuated joint `x`, env `y`). Writes `target_pos` into the
/// matching motor and sets its `motor_axes` bit, like `set_motor` does on the
/// host, but without touching the CPU mirror.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_scatter_motor_targets(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] motor_targets: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] actuated_link_ids: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] num_actuated: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] num_envs: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] axis_id: &u32,
) {
    let j = invocation_id.x;
    let env = invocation_id.y;
    if j >= *num_actuated || env >= *num_envs {
        return;
    }
    let link_id = actuated_link_ids[j as usize];
    // Batch-interleaved links layout.
    let global_idx = (link_id * *num_envs + env) as usize;
    let target = motor_targets[(j * *num_envs + env) as usize];

    // The single-iteration loop matches `gpu_lbvh_reset_collision_pairs`:
    // rust-gpu sometimes prunes the SPIR-V for kernels it deems trivial, and
    // the loop shell keeps the entry point emitted.
    for _ in 0..1 {
        let link = &mut links_static[global_idx];
        link.data.motors[*axis_id as usize].target_pos = target;
        link.data.motor_axes |= 1u32 << *axis_id;
    }
}

/// Per-step actuator-delay state refresh, on device: `tick <- 0`,
/// `k <- k_eff[env]`, and the actuated links' `prev_target` lanes copied from
/// the previous step's motor-target tensor (row-major `[num_actuated x n]`, the
/// same buffer the target scatter consumed last step, read before this step's
/// scatter overwrites it).
///
/// This replaces a full `stride * n` host rebuild and upload every step with
/// one `[n]` upload (`k_eff`) plus this dispatch. Non-actuated `prev` lanes keep
/// their existing value, so held joints stay wherever the host last put them.
///
/// Dispatch `[num_actuated, num_envs, 1]` threads; lane `j == 0` also writes the
/// two scalar lanes.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_delay_state_update(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] prev_targets: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] k_eff: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] actuated_link_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] delay_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] params: &MbDelayStateParams,
) {
    let j = invocation_id.x;
    let env = invocation_id.y;
    if j >= params.num_actuated || env >= params.num_envs {
        return;
    }
    let base = (env * params.stride) as usize;
    if j == 0 {
        delay_state.write(base, 0.0);
        delay_state.write(base + 1, k_eff.read(env as usize));
    }
    let link = actuated_link_ids.read(j as usize);
    delay_state.write(
        base + 2 + link as usize,
        prev_targets.read((j * params.num_envs + env) as usize),
    );
}

/// Advances the actuator-delay step counter by one, for every batch.
///
/// Dispatched once per physics step, before the joint constraints are built, so
/// the tick is stable across all of that step's substeps regardless of the
/// constraint-refresh cadence. Dispatch `[num_envs, 1, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_delay_tick(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] delay_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] params: &MbDelayTickParams,
) {
    let env = invocation_id.x;
    if env >= params.num_envs {
        return;
    }
    let base = (env * params.stride) as usize;
    let tick = delay_state.read(base);
    delay_state.write(base, tick + 1.0);
}
