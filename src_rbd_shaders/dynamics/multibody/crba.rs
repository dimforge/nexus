//! Composite inertia mass assembly in world coordinates.
use super::types::MultibodyLinkStatic;
use super::ws_soa::{WS_WORLD_COM, WsAddr, ws_vec, ws_world_inertia};
use crate::utils::{ISlice, linalg::MatSlice};
use glamx::{Mat3, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;

/// One lane owns one generalized coordinate. Accumulate its subtree inertia,
/// project it against the ancestor Jacobians, and write the symmetric triangle.
#[inline]
pub(super) fn mass_column(
    lane: u32,
    ndofs: u32,
    num_links: u32,
    stat: &ISlice<MultibodyLinkStatic>,
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
    let mut owner = 0;
    for k in 0..num_links {
        let link = stat[k as usize];
        if lane >= link.assembly_id && lane < link.assembly_id + link.ndofs {
            owner = k;
        }
    }
    let origin = ws_vec(workspace, wa, owner, WS_WORLD_COM);
    let mut mass = 0.0;
    let mut moment = Vec3::ZERO;
    let mut inertia = Mat3::ZERO;
    for k in 0..num_links {
        let link = stat[k as usize];
        if link.ancestor_dofs[(lane / 32) as usize] & (1 << (lane % 32)) == 0 {
            continue;
        }
        let im = link.local_mprops.inv_mass.x;
        if im == 0.0 {
            continue;
        }
        let m = 1.0 / im;
        let r = ws_vec(workspace, wa, k, WS_WORLD_COM) - origin;
        mass += m;
        moment += r * m;
        let parallel_axis =
            Mat3::from_diagonal(Vec3::splat(r.dot(r))) - Mat3::from_cols(r * r.x, r * r.y, r * r.z);
        inertia += ws_world_inertia(workspace, wa, k, &link.local_mprops) + parallel_axis * m;
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
    for other in 0..=lane {
        let vj = Vec3::new(
            jacobians.read(j.idx(0, other)),
            jacobians.read(j.idx(1, other)),
            jacobians.read(j.idx(2, other)),
        );
        let wj = Vec3::new(
            jacobians.read(j.idx(3, other)),
            jacobians.read(j.idx(4, other)),
            jacobians.read(j.idx(5, other)),
        );
        let value = vj.dot(force) + wj.dot(torque);
        matrices.write(matrix.idx(lane, other), value);
        if other != lane {
            matrices.write(matrix.idx(other, lane), value);
        }
    }
}
