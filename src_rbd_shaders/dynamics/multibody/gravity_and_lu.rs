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
use crate::utils::linalg::{MAX_MB_DOFS, MatSlice, fill_par, gemv_tr_spatial_split_par};
use crate::utils::{BatchIndices, Slice};
use crate::{AngVector, Vector, gcross_av};

use super::lu::{LANES, NO_PARENT, ltdl_factor_in_shared, ltdl_solve_in_shared, sm_idx};
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
    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `num_links` as the upper bound.
    for k in 0..MAX_MB_DOFS as u32 {
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

    // Per-DOF parent array (tree metadata for the sparse LᵀDL factor/solves),
    // written into the old pivots buffer so downstream solves read the tree
    // from the binding they already have. A joint's DOFs chain internally;
    // its first DOF hangs off the last DOF of the nearest ancestor link that
    // has any (welded links contribute none).
    if lane == 0 {
        let mut k = 0u32;
        while k < num_links {
            let stat = stat_slice[k as usize];
            if stat.ndofs > 0 {
                let mut p = NO_PARENT;
                if k != 0 {
                    let mut a = stat.parent_link_id;
                    for _ in 0..MAX_MB_DOFS as u32 {
                        let s = stat_slice[a as usize];
                        if s.ndofs > 0 {
                            p = s.assembly_id + s.ndofs - 1;
                            break;
                        }
                        if a == 0 {
                            break;
                        }
                        a = s.parent_link_id;
                    }
                }
                lu_pivots.write(piv_offset + stat.assembly_id as usize, p);
                for t in 1..stat.ndofs {
                    lu_pivots.write(
                        piv_offset + (stat.assembly_id + t) as usize,
                        stat.assembly_id + t - 1,
                    );
                }
            }
            k += 1;
        }
    }
    workgroup_memory_barrier_with_group_sync();

    ltdl_factor_in_shared(ndofs, lane, mat, lu_pivots, piv_offset);

    // Persist the LᵀDL factors to global memory (joint / contact constraint
    // init reuses them for unit-RHS solves).
    if lane < ndofs {
        for r in 0..ndofs {
            mass_matrices.write(m_view.idx(r, lane), mat.read(sm_idx(r, lane)));
        }
    }

    // ---- Phase 4: solve M·x = τ for the gravity rhs. ----
    ltdl_solve_in_shared(ndofs, lane, mat, lu_pivots, piv_offset, x);

    let _ = partial;
    let _ = pivot_row_shared;
    let _ = inv_akk_shared;
    if lane < ndofs {
        gen_forces.write(rhs_offset + lane as usize, x.read(lane as usize));
    }
}
