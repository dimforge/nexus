//! Recursive Newton-Euler generalized forces, reusing the unused Coriolis scratch.
use super::gravity_and_lu::apply_spring_forces;
use super::types::{MultibodyInfo, MultibodyLinkStatic};
use super::ws_soa::*;
use crate::dynamics::{Velocity, joint::SPATIAL_DIM};
use crate::utils::{BatchIndices, linalg::MatSlice};
use crate::{AngVector, Vector, gcross_av};
#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::Vec3;
use glamx::Vec4;
use khal_std::{
    glamx::UVec3,
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

#[inline]
fn read_wrench(buf: &[f32], lin0: usize, ang0: usize, k: u32) -> Velocity {
    let l = lin0 + k as usize * crate::DIM as usize;
    let a = ang0 + k as usize * crate::DIM as usize;
    #[cfg(feature = "dim3")]
    {
        Velocity::new(
            Vec3::new(buf.read(l), buf.read(l + 1), buf.read(l + 2)),
            Vec3::new(buf.read(a), buf.read(a + 1), buf.read(a + 2)),
        )
    }
    #[cfg(feature = "dim2")]
    {
        Velocity::new(Vec2::new(buf.read(l), buf.read(l + 1)), buf.read(a))
    }
}
#[inline]
fn write_wrench(buf: &mut [f32], lin0: usize, ang0: usize, k: u32, f: Velocity) {
    let l = lin0 + k as usize * crate::DIM as usize;
    let a = ang0 + k as usize * crate::DIM as usize;
    buf.write(l, f.linear.x);
    buf.write(l + 1, f.linear.y);
    #[cfg(feature = "dim3")]
    {
        buf.write(l + 2, f.linear.z);
        buf.write(a, f.angular.x);
        buf.write(a + 1, f.angular.y);
        buf.write(a + 2, f.angular.z);
    }
    #[cfg(feature = "dim2")]
    {
        buf.write(a, f.angular);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_recursive_forces(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] scratch: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] dt_uniform: &f32,
) {
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / num_mb;
    let mb_idx = invocation_id.x % num_mb;

    let mb = batch_ids.ib(batch_id, multibody_info).read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let jac0 = batch_ids.mb_region(
        batch_id,
        mb.jacobian_offset,
        num_links * SPATIAL_DIM as u32 * ndofs,
    );
    let gen0 = batch_ids.mb_region(batch_id, mb.first_dof, ndofs);
    let gen_base = mb.first_dof as usize;
    let region_len = num_links * crate::DIM * ndofs;
    let lin0 = batch_ids.mb_region(batch_id, mb.coriolis_offset, region_len);
    let ang0 = batch_ids.mb_region(
        batch_id,
        batch_ids.coriolis_batch_capacity + mb.coriolis_offset,
        region_len,
    );

    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);
    let vel_slice = batch_ids.ib(batch_id, dof_state).offset(gen_base);
    let damping_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(batch_ids.dof_batch_capacity as usize + gen_base);
    let stiffness_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(3 * batch_ids.dof_batch_capacity as usize + gen_base);
    let spring_ref_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(4 * batch_ids.dof_batch_capacity as usize + gen_base);
    let kin_mask_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(5 * batch_ids.dof_batch_capacity as usize + gen_base);

    // ---- Phase 1: zero the generalized-force vector. ----
    for d in 0..ndofs {
        gen_forces.write(gen0 + d as usize, 0.0);
    }

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity.x, gravity.y, gravity.z);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity.x, gravity.y);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly (serial:
    // parents precede children in link order, so `kinematic_acc` reads see
    // the parent's write in program order). ----
    for k in 0..num_links {
        write_wrench(scratch, lin0, ang0, k, Velocity::default());
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        let (self_joint_vel_lin, self_joint_vel_ang, self_shift02, self_shift23, self_rb_ang) = {
            let jv = ws_vel(links_workspace, wa, k, WS_JOINT_VEL);
            (
                jv.linear,
                jv.angular,
                ws_vec(links_workspace, wa, k, WS_SHIFT02),
                ws_vec(links_workspace, wa, k, WS_SHIFT23),
                ws_vel_ang(links_workspace, wa, k, WS_RB_VELS),
            )
        };

        if k != 0 {
            let stat = stat_slice[k as usize];
            let pid = stat.parent_link_id;
            let parent_acc = ws_vel(links_workspace, wa, pid, WS_KIN_ACC);
            let parent_acc_lin = parent_acc.linear;
            let parent_acc_ang = parent_acc.angular;
            let parent_ang = ws_vel_ang(links_workspace, wa, pid, WS_RB_VELS);

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

        ws_set_vel(
            links_workspace,
            wa,
            k,
            WS_KIN_ACC,
            Velocity::new(acc_lin, acc_ang),
        );

        let lmp = stat_slice[k as usize].local_mprops;
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x != 0.0 {
            let mass = 1.0 / inv_mass_x;
            let rb_inertia = ws_world_inertia(links_workspace, wa, k, &lmp);

            #[cfg(feature = "dim3")]
            let gyroscopic = {
                let i_omega = rb_inertia * rb_ang;
                rb_ang.cross(i_omega)
            };
            #[cfg(feature = "dim2")]
            let gyroscopic: AngVector = 0.0;

            let i_acc_ang = rb_inertia * acc_ang;
            let (ext_force, ext_torque, gravity_scale) = ws_ext_wrench(links_workspace, wa, k);

            let f_lin = g * (mass * gravity_scale) + ext_force - acc_lin * mass;
            let f_ang = ext_torque - gyroscopic - i_acc_ang;

            write_wrench(scratch, lin0, ang0, k, Velocity::new(f_lin, f_ang));
        }
    }

    for reverse in 0..num_links {
        let k = num_links - 1 - reverse;
        let stat = stat_slice[k as usize];
        let force = read_wrench(scratch, lin0, ang0, k);
        let jac = MatSlice::dense(
            jac0 + k as usize * SPATIAL_DIM * ndofs as usize,
            SPATIAL_DIM as u32,
            ndofs,
        );
        for dof in stat.assembly_id..stat.assembly_id + stat.ndofs {
            #[cfg(feature = "dim3")]
            let jlin = Vec3::new(
                body_jacobians.read(jac.idx(0, dof)),
                body_jacobians.read(jac.idx(1, dof)),
                body_jacobians.read(jac.idx(2, dof)),
            );
            #[cfg(feature = "dim3")]
            let jang = Vec3::new(
                body_jacobians.read(jac.idx(3, dof)),
                body_jacobians.read(jac.idx(4, dof)),
                body_jacobians.read(jac.idx(5, dof)),
            );
            #[cfg(feature = "dim2")]
            let jlin = Vec2::new(
                body_jacobians.read(jac.idx(0, dof)),
                body_jacobians.read(jac.idx(1, dof)),
            );
            #[cfg(feature = "dim2")]
            let jang = body_jacobians.read(jac.idx(2, dof));
            gen_forces.write(
                gen0 + dof as usize,
                jlin.dot(force.linear) + crate::gdot(jang, force.angular),
            );
        }
        if k > 0 {
            let parent = stat.parent_link_id;
            let old = read_wrench(scratch, lin0, ang0, parent);
            let r = ws_vec(links_workspace, wa, k, super::ws_soa::WS_WORLD_COM)
                - ws_vec(links_workspace, wa, parent, super::ws_soa::WS_WORLD_COM);
            write_wrench(
                scratch,
                lin0,
                ang0,
                parent,
                Velocity::new(
                    old.linear + force.linear,
                    old.angular + force.angular + crate::gcross(r, force.linear),
                ),
            );
        }
    }

    // Damping subtraction.
    for i in 0..ndofs {
        let idx = gen0 + i as usize;
        let cur = gen_forces.read(idx);
        let v = vel_slice[i as usize];
        gen_forces.write(idx, cur - damping_slice[i as usize] * v);
    }

    // Per-DoF joint springs.
    apply_spring_forces(
        gen_forces,
        gen0,
        &stat_slice,
        links_workspace,
        wa,
        num_links,
        &stiffness_slice,
        &spring_ref_slice,
        &vel_slice,
        *dt_uniform,
    );

    // Kinematic DOFs get zero acceleration.
    for i in 0..ndofs {
        if kin_mask_slice[i as usize] != 0.0 {
            gen_forces.write(gen0 + i as usize, 0.0);
        }
    }
}
