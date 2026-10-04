//! Composite inertia mass assembly in world coordinates.
use super::link_static_soa::LinkStatics;
use super::types::MultibodyLinkStatic;
use super::ws_soa::{WS_LTW, WS_WORLD_COM, WsAddr, world_inertia, ws_rot, ws_vec};
use crate::utils::linalg::MatSlice;
use glamx::{Mat3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;

/// One lane owns one generalized coordinate. Accumulate its subtree inertia,
/// project it against the ancestor Jacobians, and write the symmetric triangle.
#[inline]
pub(super) fn mass_column(
    lane: u32,
    ndofs: u32,
    num_links: u32,
    stat: &LinkStatics,
    workspace: &[Vec4],
    wa: WsAddr,
    jacobians: &[f32],
    jac0: usize,
    matrices: &mut [f32],
    matrix: MatSlice,
) {
    if lane >= ndofs {
        return;
    }
    let owner = super::compute_dynamics_pre::dof_owner(stat, num_links, lane);
    let origin = ws_vec(workspace, wa, owner, WS_WORLD_COM);
    let mut mass = 0.0;
    let mut moment = Vec3::ZERO;
    let mut inertia = Mat3::ZERO;
    // Links in pairs, both loaded before either is accumulated: the loads do
    // not depend on the sums, so their latency overlaps (same summation order).
    for pair in 0..num_links.div_ceil(2) {
        let k0 = 2 * pair;
        let k1 = k0 + 1;
        let has1 = k1 < num_links;
        let link0 = stat.get(k0 as usize);
        let link1 = stat.get(if has1 { k1 } else { k0 } as usize);
        let com0 = ws_vec(workspace, wa, k0, WS_WORLD_COM);
        let com1 = ws_vec(workspace, wa, if has1 { k1 } else { k0 }, WS_WORLD_COM);
        let rot0 = ws_rot(workspace, wa, k0, WS_LTW);
        let rot1 = ws_rot(workspace, wa, if has1 { k1 } else { k0 }, WS_LTW);
        accumulate_subtree(
            lane,
            &link0,
            com0,
            rot0,
            origin,
            &mut mass,
            &mut moment,
            &mut inertia,
        );
        if has1 {
            accumulate_subtree(
                lane,
                &link1,
                com1,
                rot1,
                origin,
                &mut mass,
                &mut moment,
                &mut inertia,
            );
        }
    }
    let j = MatSlice::dense(jac0 + owner as usize * 6 * ndofs as usize, 6, ndofs);
    let v = Vec3::new(
        jacobians.read(j.idx(0, lane)),
        jacobians.read(j.idx(1, lane)),
        jacobians.read(j.idx(2, lane)),
    );
    let w = Vec3::new(
        jacobians.read(j.idx(3, lane)),
        jacobians.read(j.idx(4, lane)),
        jacobians.read(j.idx(5, lane)),
    );
    let force = v * mass + w.cross(moment);
    let torque = moment.cross(v) + inertia * w;
    // Columns in pairs, loads first (see above).
    for pair in 0..(lane + 1).div_ceil(2) {
        let o0 = 2 * pair;
        let o1 = o0 + 1;
        let has1 = o1 <= lane;
        let (vj0, wj0) = jacobian_column(jacobians, j, o0);
        let (vj1, wj1) = jacobian_column(jacobians, j, if has1 { o1 } else { o0 });
        let value0 = vj0.dot(force) + wj0.dot(torque);
        matrices.write(matrix.idx(lane, o0), value0);
        if o0 != lane {
            matrices.write(matrix.idx(o0, lane), value0);
        }
        if has1 {
            let value1 = vj1.dot(force) + wj1.dot(torque);
            matrices.write(matrix.idx(lane, o1), value1);
            if o1 != lane {
                matrices.write(matrix.idx(o1, lane), value1);
            }
        }
    }
}

/// Column `c` of a body jacobian (linear, angular).
#[inline(always)]
fn jacobian_column(jacobians: &[f32], j: MatSlice, c: u32) -> (Vec3, Vec3) {
    (
        Vec3::new(
            jacobians.read(j.idx(0, c)),
            jacobians.read(j.idx(1, c)),
            jacobians.read(j.idx(2, c)),
        ),
        Vec3::new(
            jacobians.read(j.idx(3, c)),
            jacobians.read(j.idx(4, c)),
            jacobians.read(j.idx(5, c)),
        ),
    )
}

/// Adds link `link` (world COM `com`, local-to-world rotation `rot`) to the
/// subtree inertia of DOF `lane` about `origin` when the DOF is one of its
/// ancestors.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn accumulate_subtree(
    lane: u32,
    link: &MultibodyLinkStatic,
    com: Vec3,
    rot: crate::Rotation,
    origin: Vec3,
    mass: &mut f32,
    moment: &mut Vec3,
    inertia: &mut Mat3,
) {
    if link.ancestor_dofs[(lane / 32) as usize] & (1 << (lane % 32)) == 0 {
        return;
    }
    let im = link.local_mprops.inv_mass.x;
    if im == 0.0 {
        return;
    }
    let m = 1.0 / im;
    let r = com - origin;
    *mass += m;
    *moment += r * m;
    let parallel_axis =
        Mat3::from_diagonal(Vec3::splat(r.dot(r))) - Mat3::from_cols(r * r.x, r * r.y, r * r.z);
    *inertia += world_inertia(rot, &link.local_mprops) + parallel_axis * m;
}
