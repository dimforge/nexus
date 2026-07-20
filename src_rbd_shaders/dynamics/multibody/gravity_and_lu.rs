//! Fused `apply_gravity_with_coriolis` + LU factor + LU solve kernel.
//!
//! Replaces a 3-dispatch chain (`gpu_mb_apply_gravity_with_coriolis` →
//! `gpu_mb_lu_factor_and_solve`) with a single workgroup-parallel kernel,
//! removing two WebGPU dispatch round-trips per `compute_dynamics` call.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::Vec3;
use glamx::Vec4;

use crate::dynamics::body::Velocity;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, fill_par, gemv_tr_spatial_split_par, lu_decompose, lu_solve_in_place,
};
use crate::utils::{BatchIndices, Slice};
use crate::{AngVector, Vector, gcross_av};

use super::lu::{
    LANES, lu_apply_pivots, lu_apply_pivots_packed, lu_factor_in_shared,
    lu_factor_in_shared_packed, lu_triangular_solve_in_place,
    lu_triangular_solve_in_place_packed, sm_idx, sm_idx_packed,
};
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Fused gravity / Coriolis-force assembly + LU factor + LU solve.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_gravity_and_lu(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
    // Mass-matrix tile in shared memory.
    #[spirv(workgroup)] mat: &mut [f32; MAX_MB_DOFS * MAX_MB_DOFS],
    // RHS / solution vector.
    #[spirv(workgroup)] x: &mut [f32; MAX_MB_DOFS],
    // Per-lane partial sums for the triangular-solve tree reduction.
    #[spirv(workgroup)] partial: &mut [f32; LANES as usize],
    // Lane-0 → all-lanes broadcast slots used by the LU pivot step.
    #[spirv(workgroup)] pivot_row_shared: &mut u32,
    #[spirv(workgroup)] inv_akk_shared: &mut f32,
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // Uniform-sourced loop bounds (see `BatchIndices::mb_max_ndofs`).
    let max_ndofs = batch_ids.mb_max_ndofs;
    let max_links = batch_ids.mb_max_links;

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = gen_base;
    let rhs_offset = gen_base;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let vel_slice = Slice(dof_state, gen_base);
    let damping_slice = Slice(
        dof_state,
        batch_ids.dof_damping_section_offset as usize + gen_base,
    );

    // ---- Phase 1: zero the generalized-force vector (parallel across DOFs). ----
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    // TODO(perf): up to a certain number of degrees of freedom, we could actually run all the
    //             calculations in shared memory and only write the result in the end.
    //             Currently, the max number of dofs is 32 but we still accumulate forces/accelerations
    //             in global memory whereas it could be done in shared memory instead.
    fill_par(gen_forces, accelerations, 0.0, lane, LANES);
    workgroup_memory_barrier_with_group_sync();

    let _ = stat_slice;

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity.x, gravity.y, gravity.z);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity.x, gravity.y);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly. ----
    // NOTE: uniform trip count (from the `BatchIndices` uniform).
    for k in 0..max_links {
        let active = k < num_links;
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        if active {
            let (
                self_joint_vel_lin,
                self_joint_vel_ang,
                self_shift02,
                self_shift23,
                _self_local_to_world,
                self_rb_ang,
            ) = {
                let ws = &ws_slice[k as usize];
                (
                    ws.joint_velocity.linear,
                    ws.joint_velocity.angular,
                    ws.shift02,
                    ws.shift23,
                    ws.local_to_world,
                    ws.rb_vels.angular,
                )
            };

            if k != 0 {
                let stat = stat_slice[k as usize];
                let parent_ws = &ws_slice[stat.parent_link_id as usize];
                let parent_acc_lin = parent_ws.kinematic_acc.linear;
                let parent_acc_ang = parent_ws.kinematic_acc.angular;
                let parent_ang = parent_ws.rb_vels.angular;

                acc_lin = parent_acc_lin;
                acc_ang = parent_acc_ang;

                acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
                #[cfg(feature = "dim3")]
                {
                    acc_ang += parent_ang.cross(self_joint_vel_ang);
                }
                #[cfg(feature = "dim2")]
                {
                    let _ = self_joint_vel_ang;
                }
                acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
                acc_lin += gcross_av(parent_acc_ang, self_shift02);
            } else {
                let _ = self_joint_vel_ang;
                let _ = self_shift02;
            }
            let rb_ang = self_rb_ang;
            acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
            acc_lin += gcross_av(acc_ang, self_shift23);

            if lane == 0 {
                ws_slice[k as usize].kinematic_acc = Velocity::new(acc_lin, acc_ang);
            }
        }

        // Top-level barrier: reached uniformly by every lane on every outer
        // iteration so children see the just-published `kinematic_acc`.
        workgroup_memory_barrier_with_group_sync();

        if active {
            #[cfg(feature = "dim3")]
            let rb_ang = ws_slice[k as usize].rb_vels.angular;
            let lmp = stat_slice[k as usize].local_mprops;
            let inv_mass_x = lmp.inv_mass.x;
            if inv_mass_x != 0.0 {
                let mass = 1.0 / inv_mass_x;
                let rb_inertia = ws_slice[k as usize].link_world_inertia(&lmp);

                #[cfg(feature = "dim3")]
                let gyroscopic = {
                    let i_omega = rb_inertia * rb_ang;
                    rb_ang.cross(i_omega)
                };
                #[cfg(feature = "dim2")]
                let gyroscopic: AngVector = 0.0;

                let i_acc_ang = rb_inertia * acc_ang;

                #[cfg(feature = "dim3")]
                let f_lin = (g - acc_lin) * mass;
                #[cfg(feature = "dim2")]
                let f_lin = (g - acc_lin) * mass;
                let f_ang = -gyroscopic - i_acc_ang;

                let body_jacobian = MatSlice::dense(
                    mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
                    SPATIAL_DIM as u32,
                    ndofs,
                );

                gemv_tr_spatial_split_par(
                    gen_forces,
                    gen_base,
                    1.0,
                    body_jacobians,
                    body_jacobian,
                    f_lin,
                    f_ang,
                    1.0,
                    lane,
                    LANES,
                );
            }
        }
    }

    // Damping subtraction is handled at solve time via the LU rhs (we still
    // need the read-modify-write done by lane d): we don't have `dt` here, so
    // damping is applied identically to the original kernel: rhs -=
    // damping[d] * v[d] regardless of `dt` (matches rapier's `cmpy(-1.0,
    // damping, velocities, 1.0)` form, which does not include `dt`).
    workgroup_memory_barrier_with_group_sync();
    let i = lane;
    if i < ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        gen_forces.write(idx, cur - damping_slice[i as usize] * vel_slice[i as usize]);
    }
    workgroup_memory_barrier_with_group_sync();

    // ---- Phase 3: load M into shared memory, factor in place. ----
    let m_view = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    if lane < ndofs {
        for r in 0..ndofs {
            mat.write(sm_idx(r, lane), mass_matrices.read(m_view.idx(r, lane)));
        }
        x.write(lane as usize, gen_forces.read(rhs_offset + lane as usize));
    }
    workgroup_memory_barrier_with_group_sync();

    lu_factor_in_shared(
        ndofs,
        max_ndofs,
        lane,
        mat,
        lu_pivots,
        piv_offset,
        pivot_row_shared,
        inv_akk_shared,
    );

    // Persist LU factors to global memory (joint / contact constraint init
    // reuses them for unit-RHS solves).
    if lane < ndofs {
        for r in 0..ndofs {
            mass_matrices.write(m_view.idx(r, lane), mat.read(sm_idx(r, lane)));
        }
    }

    // ---- Phase 4: solve M·x = τ for the gravity rhs. ----
    lu_apply_pivots(ndofs, lane, lu_pivots, piv_offset, x);
    lu_triangular_solve_in_place(ndofs, max_ndofs, lane, mat, x, partial);

    if lane < ndofs {
        gen_forces.write(rhs_offset + lane as usize, x.read(lane as usize));
    }
}

/// Packed body of [`gpu_mb_gravity_and_lu`]: `SLOTS = 64 / T` multibodies per
/// 64-lane workgroup, each owning `T` lanes and a `T×T` shared tile
/// (`MATN = 64·T` floats total instead of `MAX_MB_DOFS² = 16 KB`, which
/// crippled shared-memory occupancy in the one-robot-per-environment regime).
/// The `(multibody, batch)` pair is flattened into the workgroup X dimension.
/// Inactive slots read a zeroed dummy `MultibodyInfo` (`num_links == ndofs ==
/// 0`) so every loop below no-ops for them while still reaching all barriers.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gravity_and_lu_packed_impl<const T: u32, const MATN: usize, const SLOTS: usize>(
    wg_id: UVec3,
    lid: UVec3,
    multibody_info: &[MultibodyInfo],
    links_static: &[MultibodyLinkStatic],
    links_workspace: &mut [MultibodyLinkWorkspace],
    body_jacobians: &[f32],
    gen_forces: &mut [f32],
    mass_matrices: &mut [f32],
    lu_pivots: &mut [u32],
    dof_state: &[f32],
    gravity: &Vec4,
    batch_ids: &BatchIndices,
    mat: &mut [f32; MATN],
    x: &mut [f32; 64],
    partial: &mut [f32; 64],
    pivot_row_shared: &mut [u32; SLOTS],
    inv_akk_shared: &mut [f32; SLOTS],
) {
    let slot = lid.x / T;
    let lane = lid.x % T;
    let seg = (slot * T) as usize;

    let num_mb = batch_ids.multibodies_len;
    let total_mb = num_mb * batch_ids.num_batches;
    let global_mb = wg_id.x * SLOTS as u32 + slot;
    let active_slot = global_mb < total_mb;
    // Clamped so index math stays in-bounds for inactive slots; their loops
    // all no-op (dummy `mb`) and every store is guarded.
    let clamped_mb = if active_slot { global_mb } else { total_mb - 1 };
    let batch_id = clamped_mb / num_mb;
    let mb_idx = clamped_mb % num_mb;

    // Uniform-sourced loop bounds (see `BatchIndices::mb_max_ndofs`).
    let max_ndofs = batch_ids.mb_max_ndofs;
    let max_links = batch_ids.mb_max_links;

    let mb = if active_slot {
        batch_ids
            .mb_batch(batch_id, multibody_info)
            .read(mb_idx as usize)
    } else {
        MultibodyInfo::default()
    };
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = gen_base;
    let rhs_offset = gen_base;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let vel_slice = Slice(dof_state, gen_base);
    let damping_slice = Slice(
        dof_state,
        batch_ids.dof_damping_section_offset as usize + gen_base,
    );

    // ---- Phase 1: zero the generalized-force vector (parallel across DOFs). ----
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    fill_par(gen_forces, accelerations, 0.0, lane, T);
    workgroup_memory_barrier_with_group_sync();

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity.x, gravity.y, gravity.z);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity.x, gravity.y);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly. ----
    // NOTE: uniform trip count (from the `BatchIndices` uniform).
    for k in 0..max_links {
        let active = k < num_links;
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        if active {
            let (
                self_joint_vel_lin,
                self_joint_vel_ang,
                self_shift02,
                self_shift23,
                _self_local_to_world,
                self_rb_ang,
            ) = {
                let ws = &ws_slice[k as usize];
                (
                    ws.joint_velocity.linear,
                    ws.joint_velocity.angular,
                    ws.shift02,
                    ws.shift23,
                    ws.local_to_world,
                    ws.rb_vels.angular,
                )
            };

            if k != 0 {
                let stat = stat_slice[k as usize];
                let parent_ws = &ws_slice[stat.parent_link_id as usize];
                let parent_acc_lin = parent_ws.kinematic_acc.linear;
                let parent_acc_ang = parent_ws.kinematic_acc.angular;
                let parent_ang = parent_ws.rb_vels.angular;

                acc_lin = parent_acc_lin;
                acc_ang = parent_acc_ang;

                acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
                #[cfg(feature = "dim3")]
                {
                    acc_ang += parent_ang.cross(self_joint_vel_ang);
                }
                #[cfg(feature = "dim2")]
                {
                    let _ = self_joint_vel_ang;
                }
                acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
                acc_lin += gcross_av(parent_acc_ang, self_shift02);
            } else {
                let _ = self_joint_vel_ang;
                let _ = self_shift02;
            }
            let rb_ang = self_rb_ang;
            acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
            acc_lin += gcross_av(acc_ang, self_shift23);

            if lane == 0 {
                ws_slice[k as usize].kinematic_acc = Velocity::new(acc_lin, acc_ang);
            }
        }

        // Top-level barrier: reached uniformly by every lane on every outer
        // iteration so children see the just-published `kinematic_acc`.
        workgroup_memory_barrier_with_group_sync();

        if active {
            #[cfg(feature = "dim3")]
            let rb_ang = ws_slice[k as usize].rb_vels.angular;
            let lmp = stat_slice[k as usize].local_mprops;
            let inv_mass_x = lmp.inv_mass.x;
            if inv_mass_x != 0.0 {
                let mass = 1.0 / inv_mass_x;
                let rb_inertia = ws_slice[k as usize].link_world_inertia(&lmp);

                #[cfg(feature = "dim3")]
                let gyroscopic = {
                    let i_omega = rb_inertia * rb_ang;
                    rb_ang.cross(i_omega)
                };
                #[cfg(feature = "dim2")]
                let gyroscopic: AngVector = 0.0;

                let i_acc_ang = rb_inertia * acc_ang;

                let f_lin = (g - acc_lin) * mass;
                let f_ang = -gyroscopic - i_acc_ang;

                let body_jacobian = MatSlice::dense(
                    mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
                    SPATIAL_DIM as u32,
                    ndofs,
                );

                gemv_tr_spatial_split_par(
                    gen_forces,
                    gen_base,
                    1.0,
                    body_jacobians,
                    body_jacobian,
                    f_lin,
                    f_ang,
                    1.0,
                    lane,
                    T,
                );
            }
        }
    }

    // Damping subtraction — see `gpu_mb_gravity_and_lu`.
    workgroup_memory_barrier_with_group_sync();
    let i = lane;
    if i < ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        gen_forces.write(idx, cur - damping_slice[i as usize] * vel_slice[i as usize]);
    }
    workgroup_memory_barrier_with_group_sync();

    // ---- Phase 3: load M into this slot's shared tile, factor in place. ----
    let m_view = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    if lane < ndofs {
        for r in 0..ndofs {
            mat.write(
                sm_idx_packed::<T>(slot, r, lane),
                mass_matrices.read(m_view.idx(r, lane)),
            );
        }
        x.write(seg + lane as usize, gen_forces.read(rhs_offset + lane as usize));
    }
    workgroup_memory_barrier_with_group_sync();

    lu_factor_in_shared_packed::<T, MATN, SLOTS>(
        ndofs,
        max_ndofs,
        slot,
        lane,
        active_slot,
        mat,
        lu_pivots,
        piv_offset,
        pivot_row_shared,
        inv_akk_shared,
    );

    // Persist LU factors to global memory (joint / contact constraint init
    // reuses them for unit-RHS solves).
    if lane < ndofs {
        for r in 0..ndofs {
            mass_matrices.write(m_view.idx(r, lane), mat.read(sm_idx_packed::<T>(slot, r, lane)));
        }
    }

    // ---- Phase 4: solve M·x = τ for the gravity rhs. ----
    lu_apply_pivots_packed::<T>(ndofs, slot, lane, active_slot, lu_pivots, piv_offset, x);
    lu_triangular_solve_in_place_packed::<T, MATN>(
        ndofs,
        max_ndofs,
        slot,
        lane,
        active_slot,
        mat,
        x,
        partial,
    );

    if lane < ndofs {
        gen_forces.write(rhs_offset + lane as usize, x.read(seg + lane as usize));
    }
}

/// SERIAL tier of the fused gravity + LU kernel (Genesis-style): one thread
/// per `(multibody, batch)`, 64 multibodies per workgroup with every lane
/// busy and NO barriers at all. Selected by the host when `max_ndofs ≤ 8`:
/// at that size the whole gravity assembly + 8×8 LU factor + solve is a
/// short serial chain per thread, and dropping the lane-parallel tiers'
/// ~60-barrier dependency chain wins at every batch count (dramatically so
/// when the batch count is too small to hide barrier latency). The factor
/// runs in place in global `mass_matrices` (the whole matrix is 2 cache
/// lines), the solve in place on `gen_forces`.
///
/// With no barriers, non-uniform control flow is fine: the kernel early-outs
/// on out-of-range threads and loops only over the real `num_links`.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_gravity_and_lu_t1(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / num_mb;
    let mb_idx = invocation_id.x % num_mb;

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = gen_base;
    let rhs_offset = gen_base;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let vel_slice = Slice(dof_state, gen_base);
    let damping_slice = Slice(
        dof_state,
        batch_ids.dof_damping_section_offset as usize + gen_base,
    );

    // ---- Phase 1: zero the generalized-force vector. ----
    for d in 0..ndofs {
        gen_forces.write(gen_base + d as usize, 0.0);
    }

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity.x, gravity.y, gravity.z);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity.x, gravity.y);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly (serial:
    // parents precede children in link order, so `kinematic_acc` reads see
    // the parent's write in program order). ----
    for k in 0..num_links {
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        let (self_joint_vel_lin, self_joint_vel_ang, self_shift02, self_shift23, self_rb_ang) = {
            let ws = &ws_slice[k as usize];
            (
                ws.joint_velocity.linear,
                ws.joint_velocity.angular,
                ws.shift02,
                ws.shift23,
                ws.rb_vels.angular,
            )
        };

        if k != 0 {
            let stat = stat_slice[k as usize];
            let parent_ws = &ws_slice[stat.parent_link_id as usize];
            let parent_acc_lin = parent_ws.kinematic_acc.linear;
            let parent_acc_ang = parent_ws.kinematic_acc.angular;
            let parent_ang = parent_ws.rb_vels.angular;

            acc_lin = parent_acc_lin;
            acc_ang = parent_acc_ang;

            acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
            #[cfg(feature = "dim3")]
            {
                acc_ang += parent_ang.cross(self_joint_vel_ang);
            }
            #[cfg(feature = "dim2")]
            {
                let _ = self_joint_vel_ang;
            }
            acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
            acc_lin += gcross_av(parent_acc_ang, self_shift02);
        } else {
            let _ = self_joint_vel_ang;
            let _ = self_shift02;
        }
        let rb_ang = self_rb_ang;
        acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
        acc_lin += gcross_av(acc_ang, self_shift23);

        ws_slice[k as usize].kinematic_acc = Velocity::new(acc_lin, acc_ang);

        let lmp = stat_slice[k as usize].local_mprops;
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x != 0.0 {
            let mass = 1.0 / inv_mass_x;
            let rb_inertia = ws_slice[k as usize].link_world_inertia(&lmp);

            #[cfg(feature = "dim3")]
            let gyroscopic = {
                let i_omega = rb_inertia * rb_ang;
                rb_ang.cross(i_omega)
            };
            #[cfg(feature = "dim2")]
            let gyroscopic: AngVector = 0.0;

            let i_acc_ang = rb_inertia * acc_ang;

            let f_lin = (g - acc_lin) * mass;
            let f_ang = -gyroscopic - i_acc_ang;

            let body_jacobian = MatSlice::dense(
                mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
                SPATIAL_DIM as u32,
                ndofs,
            );

            // Single lane owns the whole gemv (lane = 0, lanes = 1).
            gemv_tr_spatial_split_par(
                gen_forces,
                gen_base,
                1.0,
                body_jacobians,
                body_jacobian,
                f_lin,
                f_ang,
                1.0,
                0,
                1,
            );
        }
    }

    // Damping subtraction — see `gpu_mb_gravity_and_lu`.
    for i in 0..ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        gen_forces.write(idx, cur - damping_slice[i as usize] * vel_slice[i as usize]);
    }

    // ---- Phase 3 + 4: factor M in place in global memory, then solve
    // M·x = τ in place on the gravity rhs. ----
    let m_view = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    lu_decompose(mass_matrices, m_view, lu_pivots, piv_offset);
    lu_solve_in_place(
        mass_matrices,
        m_view,
        lu_pivots,
        piv_offset,
        gen_forces,
        rhs_offset,
    );
}

/// Stamps one packed-tier entry point of the fused gravity + LU kernel.
/// `MATN = 64·T`, `SLOTS = 64/T`.
macro_rules! gravity_and_lu_packed_entry {
    ($(#[$doc:meta])* $name:ident, $t:literal, $matn:literal, $slots:literal) => {
        $(#[$doc])*
        #[spirv_bindgen]
        #[spirv(compute(threads(64, 1, 1)))]
        pub fn $name(
            #[spirv(workgroup_id)] wg_id: UVec3,
            #[spirv(local_invocation_id)] lid: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
            multibody_info: &[MultibodyInfo],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
            links_static: &[MultibodyLinkStatic],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
            links_workspace: &mut [MultibodyLinkWorkspace],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &[f32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] gen_forces: &mut [f32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] lu_pivots: &mut [u32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
            #[spirv(uniform, descriptor_set = 0, binding = 8)] gravity: &Vec4,
            #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
            #[spirv(workgroup)] mat: &mut [f32; $matn],
            #[spirv(workgroup)] x: &mut [f32; 64],
            #[spirv(workgroup)] partial: &mut [f32; 64],
            #[spirv(workgroup)] pivot_row_shared: &mut [u32; $slots],
            #[spirv(workgroup)] inv_akk_shared: &mut [f32; $slots],
        ) {
            gravity_and_lu_packed_impl::<$t, $matn, $slots>(
                wg_id,
                lid,
                multibody_info,
                links_static,
                links_workspace,
                body_jacobians,
                gen_forces,
                mass_matrices,
                lu_pivots,
                dof_state,
                gravity,
                batch_ids,
                mat,
                x,
                partial,
                pivot_row_shared,
                inv_akk_shared,
            );
        }
    };
}

gravity_and_lu_packed_entry!(
    /// Packed tier for `max_ndofs ≤ 8`: 8 multibodies per workgroup.
    gpu_mb_gravity_and_lu_t8, 8u32, 512, 8
);
gravity_and_lu_packed_entry!(
    /// Packed tier for `max_ndofs ≤ 16`: 4 multibodies per workgroup.
    gpu_mb_gravity_and_lu_t16, 16u32, 1024, 4
);
gravity_and_lu_packed_entry!(
    /// Packed tier for `max_ndofs ≤ 32`: 2 multibodies per workgroup.
    gpu_mb_gravity_and_lu_t32, 32u32, 2048, 2
);
