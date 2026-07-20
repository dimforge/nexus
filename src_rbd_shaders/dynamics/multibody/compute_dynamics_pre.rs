//! Fused FK + body-jacobians + velocity-propagation + CRBA mass-matrix
//! kernel.
//!
//! Replaces a 4-dispatch chain
//! (`gpu_mb_forward_kinematics` → `gpu_mb_body_jacobians` →
//! `gpu_mb_update_velocities` → `gpu_mb_mass_matrix_with_coriolis`) with a
//! single workgroup-parallel kernel (one workgroup per `(multibody, batch)`).
//! The follow-up `gpu_mb_gravity_and_lu` kernel finishes the dynamics pipeline
//! (gravity rhs + LU factor + LU solve).

use khal_std::glamx::UVec3;
use glamx::Vec4;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use super::types::{MultibodyInfo, MultibodyLinkStatic};
use super::ws_soa::{
    WS_JOINT_ROT, WS_JOINT_VEL, WS_LTP, WS_LTW, WS_RB_VELS, WS_SHIFT02, WS_SHIFT23, WsAddr,
    ws_coords, ws_pose, ws_rot, ws_set_pose, ws_set_vec, ws_set_vel, ws_vec, ws_vel, ws_vel_ang,
    ws_world_inertia,
};
use crate::dynamics::body::Velocity;
use crate::dynamics::joint::SPATIAL_DIM;
#[cfg(feature = "dim3")]
use crate::utils::linalg::gemm_skew_lhs_cross_buf_par;
use crate::utils::linalg::{
    copy_from_par, fill_par, gemm_inertia_lhs_par,
    gemm_omega_skew_tr_cross_buf_par, gemm_skew_tr_lhs_cross_buf_par, gemm_skew_tr_lhs_par,
    gemm_tr_par, quadform_spatial_par,
};
use crate::utils::{BatchIndices, ISlice, SliceMut};
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross_av};
use parry::math::VectorExt;

/// Workgroup barrier gated on the (uniform-sourced) lane tier: in the serial
/// tier (`t == 1`, one thread per multibody) every dependency is within a
/// single thread, so no synchronization is needed. `t` comes from a uniform
/// buffer, so the branch is uniform control flow (legal around barriers).
#[inline(always)]
fn sync_slots(t: u32) {
    if t > 1 {
        workgroup_memory_barrier_with_group_sync();
    }
}

/// Packed slot decode shared by the two `pre` kernels: `64 / mb_pack_lanes`
/// multibodies per 64-lane workgroup, `(multibody, batch)` flattened into the
/// workgroup X dimension. Returns `(t, lane, batch_id, mb_idx, active_slot)`;
/// inactive slots get clamped indices (their loops all no-op on the zeroed
/// dummy `MultibodyInfo` the caller substitutes). `mb_pack_lanes` is
/// uniform-sourced so the decode keeps uniform control flow for barriers.
#[inline(always)]
fn packed_decode(wg_id: UVec3, lid: UVec3, batch_ids: &BatchIndices) -> (u32, u32, u32, u32, bool) {
    let t = batch_ids.mb_pack_lanes;
    let slot = lid.x / t;
    let lane = lid.x % t;
    let slots = 64 / t;

    let num_mb = batch_ids.multibodies_len;
    let total_mb = num_mb * batch_ids.num_batches;
    let global_mb = wg_id.x * slots + slot;
    let active_slot = global_mb < total_mb;
    let clamped_mb = if active_slot { global_mb } else { total_mb - 1 };
    let batch_id = clamped_mb / num_mb;
    let mb_idx = clamped_mb % num_mb;
    (t, lane, batch_id, mb_idx, active_slot)
}

// TODO: refactor into multiple functions (but single kernel) to share between the coriolis and non-coriolis versions.
/// Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis.
#[spirv_bindgen(force_cpu_coroutines)]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_compute_dynamics_pre(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] coriolis_packed: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    // Packed layout — see `packed_decode`. No early-return for inactive
    // slots — WGSL's naga frontend can't prove a storage-loaded comparison is
    // uniform across the workgroup, so any subsequent `workgroupBarrier()`
    // would be flagged "called from non-uniform control flow"; inactive slots
    // instead run every loop with a zeroed dummy `MultibodyInfo` (zero links /
    // DOFs ⇒ zero iterations, no stores).
    let (t, lane, batch_id, mb_idx, active_slot) = packed_decode(wg_id, lid, batch_ids);

    let dt = *dt_uniform;

    let mb = if active_slot {
        batch_ids
            .ib(batch_id, multibody_info)
            .read(mb_idx as usize)
    } else {
        MultibodyInfo::default()
    };
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let mb_jac_base = mb.jacobian_offset as usize;
    let mb_mm_base = mb.mass_matrix_offset as usize;
    let mb_cor_base = mb.coriolis_offset as usize;
    let mb_cor_w_base = batch_ids.coriolis_batch_capacity as usize + mb_cor_base;
    let mb_icdt_base =
        2 * batch_ids.coriolis_batch_capacity as usize + mb.i_coriolis_dt_offset as usize;
    let vel_base = mb.first_dof as usize;

    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);
    let mut poses_slice = batch_ids.coll_batch_mut(batch_id, poses);
    let damping_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(batch_ids.dof_batch_capacity as usize + vel_base);
    // Armature (reflected rotor inertia) section sits right after damping, at
    // `2 · dof_damping_section_offset` (= 2·N). Added to the mass-matrix diagonal
    // alongside `damping·dt`, matching rapier's `update_mass_matrix`.
    let armature_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(2 * batch_ids.dof_batch_capacity as usize + vel_base);
    let vel_slice = batch_ids.ib(batch_id, dof_state).offset(vel_base);

    // 1) Forward Kinematics (single-threaded)
    if active_slot && num_links > 0 && lane == 0 {
        forward_kinematics(&mb, &stat_slice, &mut poses_slice, links_workspace, wa, num_links);
    }
    sync_slots(t);

    // 2) Update body jacobians
    update_body_jacobians(
        lane,
        t,
        mb_jac_base,
        ndofs,
        num_links,
        batch_ids.mb_max_links,
        &stat_slice,
        links_workspace,
        wa,
        body_jacobians,
        batch_ids,
        batch_id,
    );

    // 3) Propagate velocities (single-threaded)
    if active_slot && num_links > 0 && lane == 0 {
        propagate_velocities(num_links, &stat_slice, &vel_slice, links_workspace, wa);
    }
    sync_slots(t);

    // 3) Mass matrix (with semi-implicit coriolis handling).
    let acc_augmented_mass = batch_ids.imat(batch_id, mb_mm_base, ndofs, ndofs);
    fill_par(mass_matrices, acc_augmented_mass, 0.0, lane, t);

    let i_coriolis_dt_view = batch_ids.imat(batch_id, mb_icdt_base, SPATIAL_DIM as u32, ndofs);
    let i_coriolis_dt_v = i_coriolis_dt_view.fixed_rows(0, DIM);
    let i_coriolis_dt_w = i_coriolis_dt_view.fixed_rows(DIM, ANG_DIM);

    sync_slots(t);

    // NOTE: uniform trip count (from the `BatchIndices` uniform).
    for k in 0..batch_ids.mb_max_links {
        let loop_is_active = k < num_links;
        let mut inv_mass_x = 0.0;
        let mut mass = 0.0;

        if loop_is_active {
            let lmp = stat_slice[k as usize].local_mprops;

            inv_mass_x = lmp.inv_mass.x;

            if inv_mass_x == 0.0 {
                let coriolis_block = batch_ids.imat(batch_id, 
                    mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
                    DIM,
                    ndofs,
                );
                fill_par(coriolis_packed, coriolis_block, 0.0, lane, t);
                fill_par(
                    coriolis_packed,
                    batch_ids.imat(batch_id, 
                        mb_cor_w_base + (k as usize) * (DIM as usize) * (ndofs as usize),
                        DIM,
                        ndofs,
                    ),
                    0.0,
                    lane,
                    t,
                );
            }
        }
        // Uniform barrier so subsequent parent-coriolis reads see consistent
        // state — WebGPU forbids a barrier inside divergent control flow.
        sync_slots(t);

        let loop_is_active = k < num_links && inv_mass_x != 0.0;
        let coriolis_v_i = batch_ids.imat(batch_id, 
            mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
            DIM,
            ndofs,
        );
        let coriolis_w_i = batch_ids.imat(batch_id, 
            mb_cor_w_base + (k as usize) * (DIM as usize) * (ndofs as usize),
            ANG_DIM,
            ndofs,
        );
        let body_jacobian = batch_ids.imat(batch_id, 
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );
        let rb_j_w = body_jacobian.fixed_rows(DIM, ANG_DIM);
        let mut rb_inertia = Default::default();

        if loop_is_active {
            let lmp = stat_slice[k as usize].local_mprops;
            mass = 1.0 / inv_mass_x;
            rb_inertia = ws_world_inertia(links_workspace, wa, k, &lmp);

            #[cfg(feature = "dim3")]
            let augmented_inertia = {
                let angvel = ws_vel_ang(links_workspace, wa, k, WS_RB_VELS);
                let w_skew = crate::utils::linalg::skew(angvel);
                let i_omega = rb_inertia * angvel;
                let i_omega_skew = crate::utils::linalg::skew(i_omega);
                let gyro_mat = w_skew * rb_inertia - i_omega_skew;
                rb_inertia + gyro_mat * dt
            };
            #[cfg(feature = "dim2")]
            let augmented_inertia = rb_inertia;

            quadform_spatial_par(
                mass_matrices,
                acc_augmented_mass,
                1.0,
                mass,
                augmented_inertia,
                body_jacobians,
                body_jacobian,
                1.0,
                lane,
                t,
            );

            if k != 0 {
                let stat = stat_slice[k as usize];
                let parent_id = stat.parent_link_id;
                let parent_j = batch_ids.imat(batch_id, 
                    mb_jac_base + (parent_id as usize) * SPATIAL_DIM * (ndofs as usize),
                    SPATIAL_DIM as u32,
                    ndofs,
                );
                let parent_j_w = parent_j.fixed_rows(DIM, ANG_DIM);
                let parent_coriolis_v = batch_ids.imat(batch_id, 
                    mb_cor_base + (parent_id as usize) * (DIM as usize) * (ndofs as usize),
                    DIM,
                    ndofs,
                );
                let parent_coriolis_w = batch_ids.imat(batch_id, 
                    mb_cor_w_base + (parent_id as usize) * (DIM as usize) * (ndofs as usize),
                    ANG_DIM,
                    ndofs,
                );
                let parent_w = ws_vel_ang(links_workspace, wa, parent_id, WS_RB_VELS);
                let ws_shift02 = ws_vec(links_workspace, wa, k, WS_SHIFT02);
                let ws_joint_vel = ws_vel(links_workspace, wa, k, WS_JOINT_VEL);

                copy_from_par(
                    coriolis_packed,
                    coriolis_v_i,
                    parent_coriolis_v,
                    lane,
                    t,
                );
                copy_from_par(
                    coriolis_packed,
                    coriolis_w_i,
                    parent_coriolis_w,
                    lane,
                    t,
                );

                gemm_skew_tr_lhs_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    ws_shift02,
                    parent_coriolis_w,
                    1.0,
                    lane,
                    t,
                );

                let ws_rb_ang = ws_vel_ang(links_workspace, wa, k, WS_RB_VELS);
                let dvel = crate::gcross_av(ws_rb_ang, ws_shift02) + ws_joint_vel.linear * 2.0;
                gemm_skew_tr_lhs_cross_buf_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    dvel,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                    lane,
                    t,
                );

                gemm_skew_tr_lhs_cross_buf_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    ws_joint_vel.linear,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                    lane,
                    t,
                );

                gemm_omega_skew_tr_cross_buf_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    parent_w,
                    ws_shift02,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                    lane,
                    t,
                );

                #[cfg(feature = "dim3")]
                {
                    gemm_skew_lhs_cross_buf_par(
                        coriolis_packed,
                        coriolis_w_i,
                        -1.0,
                        ws_joint_vel.angular,
                        body_jacobians,
                        parent_j_w,
                        1.0,
                        lane,
                        t,
                    );
                }
            }
        }

        sync_slots(t);

        if loop_is_active {
            if k != 0 {
                let stat = stat_slice[k as usize];
                let parent_id = stat.parent_link_id;

                if stat.kinematic == 0 {
                    let transform_rot = ws_rot(links_workspace, wa, parent_id, WS_LTW)
                        * stat.data.local_frame_a.rotation;
                    let coriolis_v_part = coriolis_v_i.columns(stat.assembly_id, stat.ndofs);
                    let coriolis_w_part = coriolis_w_i.columns(stat.assembly_id, stat.ndofs);

                    #[cfg(feature = "dim3")]
                    {
                        let parent_w_skew = crate::utils::linalg::skew(
                            ws_vel_ang(links_workspace, wa, parent_id, WS_RB_VELS),
                        );
                        let c = lane;
                        if c < stat.ndofs {
                            let (jv, jw) = stat.joint_jacobian_column(transform_rot, c);
                            let pv = parent_w_skew * jv;
                            let pw = parent_w_skew * jw;
                            let iv0 = coriolis_v_part.idx(0, c);
                            let iv1 = coriolis_v_part.idx(1, c);
                            let iv2 = coriolis_v_part.idx(2, c);
                            coriolis_packed.write(iv0, coriolis_packed.read(iv0) + 2.0 * pv.x);
                            coriolis_packed.write(iv1, coriolis_packed.read(iv1) + 2.0 * pv.y);
                            coriolis_packed.write(iv2, coriolis_packed.read(iv2) + 2.0 * pv.z);
                            let iw0 = coriolis_w_part.idx(0, c);
                            let iw1 = coriolis_w_part.idx(1, c);
                            let iw2 = coriolis_w_part.idx(2, c);
                            coriolis_packed.write(iw0, coriolis_packed.read(iw0) + pw.x);
                            coriolis_packed.write(iw1, coriolis_packed.read(iw1) + pw.y);
                            coriolis_packed.write(iw2, coriolis_packed.read(iw2) + pw.z);
                        }
                    }
                    #[cfg(feature = "dim2")]
                    {
                        let parent_w = ws_vel_ang(links_workspace, wa, parent_id, WS_RB_VELS);
                        let c = lane;
                        if c < stat.ndofs {
                            let (jv, _) = stat.joint_jacobian_column(transform_rot, c);
                            let iv0 = coriolis_v_part.idx(0, c);
                            let iv1 = coriolis_v_part.idx(1, c);
                            coriolis_packed
                                .write(iv0, coriolis_packed.read(iv0) + 2.0 * (-parent_w * jv.y));
                            coriolis_packed
                                .write(iv1, coriolis_packed.read(iv1) + 2.0 * (parent_w * jv.x));
                        }
                        let _ = coriolis_w_part;
                    }
                }
            } else {
                fill_par(coriolis_packed, coriolis_v_i, 0.0, lane, t);
                fill_par(coriolis_packed, coriolis_w_i, 0.0, lane, t);
            }
        }

        sync_slots(t);

        if loop_is_active {
            let ws_shift23 = ws_vec(links_workspace, wa, k, WS_SHIFT23);
            let ws_rb_ang = ws_vel_ang(links_workspace, wa, k, WS_RB_VELS);
            gemm_skew_tr_lhs_par(
                coriolis_packed,
                coriolis_v_i,
                1.0,
                ws_shift23,
                coriolis_w_i,
                1.0,
                lane,
                t,
            );

            let dvel_23 = crate::gcross_av(ws_rb_ang, ws_shift23);
            gemm_skew_tr_lhs_cross_buf_par(
                coriolis_packed,
                coriolis_v_i,
                1.0,
                dvel_23,
                body_jacobians,
                rb_j_w,
                1.0,
                lane,
                t,
            );

            gemm_omega_skew_tr_cross_buf_par(
                coriolis_packed,
                coriolis_v_i,
                1.0,
                ws_rb_ang,
                ws_shift23,
                body_jacobians,
                rb_j_w,
                1.0,
                lane,
                t,
            );
        }

        sync_slots(t);

        if loop_is_active {
            // i_coriolis_dt assembly: dt · (mass·coriolis_v, I·coriolis_w).
            {
                let scale = mass * dt;
                let c = lane;
                if c < ndofs {
                    for r in 0..DIM {
                        let v = coriolis_packed.read(coriolis_v_i.idx(r, c));
                        coriolis_packed.write(i_coriolis_dt_v.idx(r, c), scale * v);
                    }
                }
            }
            gemm_inertia_lhs_par(
                coriolis_packed,
                i_coriolis_dt_w,
                dt,
                rb_inertia,
                coriolis_w_i,
                0.0,
                lane,
                t,
            );
        }

        sync_slots(t);

        if loop_is_active {
            gemm_tr_par(
                mass_matrices,
                acc_augmented_mass,
                1.0,
                body_jacobians,
                body_jacobian,
                coriolis_packed,
                i_coriolis_dt_view,
                1.0,
                lane,
                t,
            );
        }

        sync_slots(t);
    }

    // Diagonal: M[i, i] += damping[i] * dt + armature[i] — parallel.
    // Matches rapier's `update_mass_matrix`: `diag = damping·dt + armature`.
    let d = lane;
    if d < ndofs {
        let diag_idx = acc_augmented_mass.idx(d, d);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(
            diag_idx,
            cur + damping_slice[d as usize] * dt + armature_slice[d as usize],
        );
    }
}

/// Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis.
#[spirv_bindgen(force_cpu_coroutines)]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_compute_dynamics_without_coriolis_pre(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dof_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    // Packed layout — see `packed_decode` and the uniformity note on
    // `gpu_mb_compute_dynamics_pre`.
    let (t, lane, batch_id, mb_idx, active_slot) = packed_decode(wg_id, lid, batch_ids);

    let dt = *dt_uniform;

    let mb = if active_slot {
        batch_ids
            .ib(batch_id, multibody_info)
            .read(mb_idx as usize)
    } else {
        MultibodyInfo::default()
    };
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let mb_jac_base = mb.jacobian_offset as usize;
    let mb_mm_base = mb.mass_matrix_offset as usize;
    let vel_base = mb.first_dof as usize;

    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);
    let mut poses_slice = batch_ids.coll_batch_mut(batch_id, poses);
    let damping_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(batch_ids.dof_batch_capacity as usize + vel_base);
    // Armature (reflected rotor inertia) section sits right after damping, at
    // `2 · dof_damping_section_offset` (= 2·N). Added to the mass-matrix diagonal
    // alongside `damping·dt`, matching rapier's `update_mass_matrix`.
    let armature_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(2 * batch_ids.dof_batch_capacity as usize + vel_base);
    let vel_slice = batch_ids.ib(batch_id, dof_state).offset(vel_base);

    // 1) Forward Kinematics (single-threaded)
    if active_slot && num_links > 0 && lane == 0 {
        forward_kinematics(&mb, &stat_slice, &mut poses_slice, links_workspace, wa, num_links);
    }
    sync_slots(t);

    // 2) Update body jacobians
    update_body_jacobians(
        lane,
        t,
        mb_jac_base,
        ndofs,
        num_links,
        batch_ids.mb_max_links,
        &stat_slice,
        links_workspace,
        wa,
        body_jacobians,
        batch_ids,
        batch_id,
    );

    // 3) Velocities propagation (single-threaded)
    if active_slot && num_links > 0 && lane == 0 {
        propagate_velocities(num_links, &stat_slice, &vel_slice, links_workspace, wa);
    }
    sync_slots(t);

    // 4) Mass matrix (without coriolis).
    let acc_augmented_mass = batch_ids.imat(batch_id, mb_mm_base, ndofs, ndofs);
    fill_par(mass_matrices, acc_augmented_mass, 0.0, lane, t);
    sync_slots(t);

    // NOTE: uniform trip count (from the `BatchIndices` uniform).
    for k in 0..batch_ids.mb_max_links {
        let mut active = k < num_links;
        if active {
            let lmp = stat_slice[k as usize].local_mprops;
            if lmp.inv_mass.x == 0.0 {
                active = false;
            }
        }

        if active {
            let lmp = stat_slice[k as usize].local_mprops;
            let mass = 1.0 / lmp.inv_mass.x;
            let inertia = ws_world_inertia(links_workspace, wa, k, &lmp);

            let body_jacobian = batch_ids.imat(batch_id, 
                mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
                SPATIAL_DIM as u32,
                ndofs,
            );

            quadform_spatial_par(
                mass_matrices,
                acc_augmented_mass,
                1.0,
                mass,
                inertia,
                body_jacobians,
                body_jacobian,
                1.0,
                lane,
                t,
            );
        }

        sync_slots(t);
    }

    // Diagonal: M[i, i] += damping[i] * dt + armature[i] — parallel.
    // Matches rapier's `update_mass_matrix`: `diag = damping·dt + armature`.
    let d = lane;
    if d < ndofs {
        let diag_idx = acc_augmented_mass.idx(d, d);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(
            diag_idx,
            cur + damping_slice[d as usize] * dt + armature_slice[d as usize],
        );
    }
}

/// Body-local velocity contributed by a joint, reading dof velocities directly
/// from the slice. Mirrors `velocity::jacobian_mul_coordinates`.
#[inline]
fn jacobian_mul_coordinates(
    locked_axes: u32,
    assembly_id: u32,
    vel_slice: &ISlice<f32>,
) -> (Vector, AngVector) {
    let mut lin = Vector::ZERO;
    #[cfg(feature = "dim3")]
    let mut ang = AngVector::ZERO;
    #[cfg(feature = "dim2")]
    let mut ang: AngVector = 0.0;
    let mut curr = 0u32;

    for i in 0..DIM {
        if (locked_axes & (1 << i)) == 0 {
            let v = vel_slice[(assembly_id + curr) as usize];
            lin += Vector::ith(i as usize, v);
            curr += 1;
        }
    }

    let ang_locked = (locked_axes >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        #[cfg(feature = "dim3")]
        {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let v = vel_slice[(assembly_id + curr) as usize];
            ang += Vector::ith(dof_id as usize, v);
        }
        #[cfg(feature = "dim2")]
        {
            let v = vel_slice[(assembly_id + curr) as usize];
            ang += v;
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let vx = vel_slice[(assembly_id + curr) as usize];
            let vy = vel_slice[(assembly_id + curr + 1) as usize];
            let vz = vel_slice[(assembly_id + curr + 2) as usize];
            ang += AngVector::new(vx, vy, vz);
        }
    }
    (lin, ang)
}

// Forward-kinematics traversing all the links of a given mulitbody
// sequentially on a single thread.
fn forward_kinematics(
    mb: &MultibodyInfo,
    stat_slice: &ISlice<MultibodyLinkStatic>,
    poses_slice: &mut SliceMut<Pose>,
    ws: &mut [Vec4],
    wa: WsAddr,
    num_links: u32,
) {
    // Root pose.
    let root_config = stat_slice[0];
    let root_pose = if mb.root_is_dynamic == 0 {
        poses_slice[root_config.rb_id as usize]
    } else {
        let jr = ws_rot(ws, wa, 0, WS_JOINT_ROT);
        let coords = ws_coords(ws, wa, 0);
        let pose = root_config.body_to_parent(jr, &coords);
        poses_slice[root_config.rb_id as usize] = pose;
        pose
    };
    ws_set_pose(ws, wa, 0, WS_LTP, root_pose);
    ws_set_pose(ws, wa, 0, WS_LTW, root_pose);

    for k in 1..num_links {
        let k_usize = k as usize;
        let stat = &stat_slice[k_usize];
        let parent_to_world = ws_pose(ws, wa, stat.parent_link_id, WS_LTW);
        let jr = ws_rot(ws, wa, k, WS_JOINT_ROT);
        let coords = ws_coords(ws, wa, k);
        let local_to_parent = stat.body_to_parent(jr, &coords);
        let local_to_world = parent_to_world * local_to_parent;

        let parent_lmp = stat_slice[stat.parent_link_id as usize].local_mprops;
        let lmp = stat.local_mprops;
        let world_com = local_to_world * lmp.com;
        let parent_com_world = parent_to_world * parent_lmp.com;
        let child_anchor_world = local_to_world * stat.data.local_frame_b.translation;
        let shift02 = child_anchor_world - parent_com_world;
        let shift23 = world_com - child_anchor_world;

        ws_set_pose(ws, wa, k, WS_LTP, local_to_parent);
        ws_set_pose(ws, wa, k, WS_LTW, local_to_world);
        ws_set_vec(ws, wa, k, WS_SHIFT02, shift02);
        ws_set_vec(ws, wa, k, WS_SHIFT23, shift23);
        poses_slice[stat.rb_id as usize] = local_to_world;
    }
}

fn update_body_jacobians(
    lane: u32,
    // Lanes owned by this multibody's slot (`BatchIndices::mb_pack_lanes`).
    lanes: u32,
    mb_jac_base: usize,
    ndofs: u32,
    num_links: u32,
    // Uniform-sourced upper bound for `num_links` (`BatchIndices::mb_max_links`).
    max_links: u32,
    stat_slice: &ISlice<MultibodyLinkStatic>,
    ws: &[Vec4],
    wa: WsAddr,
    body_jacobians: &mut [f32],
    batch_ids: &BatchIndices,
    batch_id: u32,
) {
    // TODO(PERF): instead of copying the body jacobian over and over for each body, we should
    //             precompute a bit set that indicates which dofs are part of the kinematic tree
    //             of each node. For a max number of DOFs set to 32, this means a single addition 32-bits
    //             value per node.
    for k in 0..max_links {
        let mut parent_to_world = Pose::default();
        let link_j = batch_ids.imat(batch_id, 
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        if k < num_links {
            let link_infos = &stat_slice[k as usize];

            if k != 0 {
                let parent_j = batch_ids.imat(batch_id, 
                    mb_jac_base
                        + (link_infos.parent_link_id as usize) * SPATIAL_DIM * (ndofs as usize),
                    SPATIAL_DIM as u32,
                    ndofs,
                );
                parent_to_world = ws_pose(ws, wa, link_infos.parent_link_id, WS_LTW);

                copy_from_par(body_jacobians, link_j, parent_j, lane, lanes);
                let link_j_v = link_j.fixed_rows(0, DIM);
                let parent_j_w = parent_j.fixed_rows(DIM, ANG_DIM);
                gemm_skew_tr_lhs_par(
                    body_jacobians,
                    link_j_v,
                    1.0,
                    ws_vec(ws, wa, k, WS_SHIFT02),
                    parent_j_w,
                    1.0,
                    lane,
                    lanes,
                );
            } else {
                fill_par(body_jacobians, link_j, 0.0, lane, lanes);
            }
        }

        sync_slots(lanes);

        if k < num_links {
            let link_infos = &stat_slice[k as usize];
            let link_j_part = link_j.columns(link_infos.assembly_id, link_infos.ndofs);
            link_infos.joint_jacobian_accumulate_par(
                parent_to_world.rotation * link_infos.data.local_frame_a.rotation,
                body_jacobians,
                link_j_part,
                lane,
                lanes,
            );
        }

        sync_slots(lanes);

        if k < num_links {
            let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
            gemm_skew_tr_lhs_par(
                body_jacobians,
                link_j_v,
                1.0,
                ws_vec(ws, wa, k, WS_SHIFT23),
                link_j_w,
                1.0,
                lane,
                lanes,
            );
        }

        sync_slots(lanes);
    }
}

fn propagate_velocities(
    num_links: u32,
    stat_slice: &ISlice<MultibodyLinkStatic>,
    vel_slice: &ISlice<f32>,
    ws: &mut [Vec4],
    wa: WsAddr,
) {
    for k in 0..num_links {
        let k_usize = k as usize;
        let stat = stat_slice[k_usize];

        let (jv_local_lin, jv_local_ang) =
            jacobian_mul_coordinates(stat.data.locked_axes, stat.assembly_id, vel_slice);

        let (joint_velocity, rb_vels) = if k == 0 {
            let jv = Velocity::new(jv_local_lin, jv_local_ang);
            (jv, jv)
        } else {
            let parent_id = stat.parent_link_id;
            let parent_world_com_pose = ws_pose(ws, wa, parent_id, WS_LTW);
            let parent_to_world_rot = parent_world_com_pose.rotation;
            let parent_rb = ws_vel(ws, wa, parent_id, WS_RB_VELS);
            let parent_rb_lin = parent_rb.linear;
            let parent_rb_ang = parent_rb.angular;

            let parent_lmp = stat_slice[parent_id as usize].local_mprops;
            let transform_rot = parent_to_world_rot * stat.data.local_frame_a.rotation;

            #[cfg(feature = "dim3")]
            let joint_velocity =
                Velocity::new(transform_rot * jv_local_lin, transform_rot * jv_local_ang);
            #[cfg(feature = "dim2")]
            let joint_velocity = Velocity::new(transform_rot * jv_local_lin, jv_local_ang);

            let self_local_to_world = ws_pose(ws, wa, k, WS_LTW);
            let self_shift23 = ws_vec(ws, wa, k, WS_SHIFT23);

            let lmp = stat.local_mprops;
            let world_com = self_local_to_world * lmp.com;
            let parent_world_com = parent_world_com_pose * parent_lmp.com;
            let shift = world_com - parent_world_com;

            let mut new_lin = parent_rb_lin + joint_velocity.linear;
            let new_ang = parent_rb_ang + joint_velocity.angular;
            new_lin += gcross_av(parent_rb_ang, shift);
            new_lin += gcross_av(joint_velocity.angular, self_shift23);

            (joint_velocity, Velocity::new(new_lin, new_ang))
        };

        ws_set_vel(ws, wa, k, WS_JOINT_VEL, joint_velocity);
        ws_set_vel(ws, wa, k, WS_RB_VELS, rb_vels);
    }
}
