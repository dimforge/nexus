//! SoA layout for the per-link multibody workspace.
//!
//! The former `MultibodyLinkWorkspace` struct (~240 B in 3D) stored each
//! link's fields contiguously, so under the batch-interleaved layout every
//! lane still touched its own pair of cache lines — struct-granularity
//! elements cannot coalesce under ANY layout. This module splits the
//! workspace into `Vec4` QUADS inside ONE flat buffer (no extra bindings),
//! addressed as
//!
//! ```text
//! flat = ((link_intra · WS_QUADS + field_quad) · num_batches + batch_id)
//! ```
//!
//! so each 16-byte quad of each field is batch-interleaved individually:
//! adjacent lanes (adjacent batches) load adjacent `Vec4`s. Quad (rather
//! than scalar) granularity keeps the loads vectorized — a pose is 2 load
//! instructions instead of 7 scalar loads with per-component index math.
//!
//! The layout constants and addressing live here, compiled for BOTH the GPU
//! kernels and the host (which uses them to explode the initial
//! `MultibodyLinkWorkspace` structs into the SoA buffer) — one source of
//! truth for the layout.

use glamx::Vec4;
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::Vec3;

use super::types::MultibodyLinkWorkspace;
use crate::dynamics::body::Velocity;
use crate::{Pose, Rotation, Vector};

/*
 * Per-link QUAD offsets of each field (dim3): 15 quads / 240 B per link.
 */
#[cfg(feature = "dim3")]
mod layout {
    /// Joint rotation quat (xyzw).
    pub const WS_JOINT_ROT: u32 = 0;
    /// Generalized coordinates c0..c3 | c4, c5, pad, pad.
    pub const WS_COORDS: u32 = 1;
    /// Local-to-parent: rot quat | trans xyz, pad.
    pub const WS_LTP: u32 = 3;
    /// Local-to-world: rot quat | trans xyz, pad.
    pub const WS_LTW: u32 = 5;
    /// shift02 xyz, pad.
    pub const WS_SHIFT02: u32 = 7;
    /// shift23 xyz, pad.
    pub const WS_SHIFT23: u32 = 8;
    /// Joint velocity: lin xyz, pad | ang xyz, pad.
    pub const WS_JOINT_VEL: u32 = 9;
    /// Rigid-body velocity: lin | ang.
    pub const WS_RB_VELS: u32 = 11;
    /// Kinematic acceleration: lin | ang.
    pub const WS_KIN_ACC: u32 = 13;
    /// Total quads per link (per-link stride, in quad units).
    pub const WS_QUADS: u32 = 15;
}

/*
 * Per-link QUAD offsets of each field (dim2): 9 quads / 144 B per link.
 */
#[cfg(feature = "dim2")]
mod layout {
    /// Joint rotation (re, im, pad, pad).
    pub const WS_JOINT_ROT: u32 = 0;
    /// Generalized coordinates c0..c2, pad.
    pub const WS_COORDS: u32 = 1;
    /// Local-to-parent: (rot.re, rot.im, trans.x, trans.y) — one quad.
    pub const WS_LTP: u32 = 2;
    /// Local-to-world: (rot.re, rot.im, trans.x, trans.y) — one quad.
    pub const WS_LTW: u32 = 3;
    /// shift02 x, y, pad, pad.
    pub const WS_SHIFT02: u32 = 4;
    /// shift23 x, y, pad, pad.
    pub const WS_SHIFT23: u32 = 5;
    /// Joint velocity: (lin.x, lin.y, ang, pad).
    pub const WS_JOINT_VEL: u32 = 6;
    /// Rigid-body velocity: (lin.x, lin.y, ang, pad).
    pub const WS_RB_VELS: u32 = 7;
    /// Kinematic acceleration: (lin.x, lin.y, ang, pad).
    pub const WS_KIN_ACC: u32 = 8;
    /// Total quads per link (per-link stride, in quad units).
    pub const WS_QUADS: u32 = 9;
}

pub use layout::*;

/// Addressing view over the SoA workspace buffer for one multibody of one
/// batch: `base` is the multibody's first link (intra-batch), `stride` /
/// `shift` the batch interleave (`num_batches` / `batch_id`).
#[derive(Copy, Clone)]
pub struct WsAddr {
    pub base: usize,
    pub stride: u32,
    pub shift: u32,
}

impl WsAddr {
    /// View of batch `shift`'s workspace, based at link `base`.
    #[inline]
    pub fn new(base: usize, stride: u32, shift: u32) -> Self {
        Self {
            base,
            stride,
            shift,
        }
    }

    /// Re-based view (like `Slice::offset`).
    #[inline]
    pub fn offset(self, links: usize) -> Self {
        Self {
            base: self.base + links,
            ..self
        }
    }

    /// Flat index of quad `quad` (a `WS_*` field offset + quad index) of
    /// link `k` (relative to `base`).
    #[inline]
    pub fn at(&self, k: u32, quad: u32) -> usize {
        ((self.base + k as usize) * WS_QUADS as usize + quad as usize) * self.stride as usize
            + self.shift as usize
    }
}

/*
 * Field accessors. Each 16-byte quad is one `Vec4` load/store,
 * batch-interleaved (coalesced across lanes).
 */

#[cfg(feature = "dim3")]
#[inline]
pub fn ws_rot(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Rotation {
    let q = buf.read(a.at(k, f));
    Rotation::from_xyzw(q.x, q.y, q.z, q.w)
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_rot(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Rotation {
    let q = buf.read(a.at(k, f));
    Rotation::from_cos_sin_unchecked(q.x, q.y)
}

#[cfg(feature = "dim3")]
#[inline]
pub fn ws_set_rot(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, r: Rotation) {
    buf.write(a.at(k, f), Vec4::new(r.x, r.y, r.z, r.w));
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_set_rot(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, r: Rotation) {
    buf.write(a.at(k, f), Vec4::new(r.re, r.im, 0.0, 0.0));
}

#[cfg(feature = "dim3")]
#[inline]
pub fn ws_vec(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Vector {
    let q = buf.read(a.at(k, f));
    Vec3::new(q.x, q.y, q.z)
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_vec(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Vector {
    let q = buf.read(a.at(k, f));
    Vec2::new(q.x, q.y)
}

#[cfg(feature = "dim3")]
#[inline]
pub fn ws_set_vec(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, v: Vector) {
    buf.write(a.at(k, f), Vec4::new(v.x, v.y, v.z, 0.0));
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_set_vec(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, v: Vector) {
    buf.write(a.at(k, f), Vec4::new(v.x, v.y, 0.0, 0.0));
}

/// Pose accessors. 3D: rotation quad + translation quad. 2D: one quad
/// `(rot.re, rot.im, trans.x, trans.y)`.
#[cfg(feature = "dim3")]
#[inline]
pub fn ws_pose(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Pose {
    let rot = ws_rot(buf, a, k, f);
    let tr = ws_vec(buf, a, k, f + 1);
    Pose::from_parts(tr, rot)
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_pose(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Pose {
    let q = buf.read(a.at(k, f));
    Pose::from_parts(
        Vec2::new(q.z, q.w),
        Rotation::from_cos_sin_unchecked(q.x, q.y),
    )
}

#[cfg(feature = "dim3")]
#[inline]
pub fn ws_set_pose(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, p: Pose) {
    ws_set_rot(buf, a, k, f, p.rotation);
    ws_set_vec(buf, a, k, f + 1, p.translation);
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_set_pose(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, p: Pose) {
    buf.write(
        a.at(k, f),
        Vec4::new(
            p.rotation.re,
            p.rotation.im,
            p.translation.x,
            p.translation.y,
        ),
    );
}

/// Velocity accessors. 3D: linear quad + angular quad. 2D: one quad
/// `(lin.x, lin.y, ang, pad)`.
#[cfg(feature = "dim3")]
#[inline]
pub fn ws_vel(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Velocity {
    Velocity::new(ws_vec(buf, a, k, f), ws_vec(buf, a, k, f + 1))
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_vel(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> Velocity {
    let q = buf.read(a.at(k, f));
    Velocity::new(Vec2::new(q.x, q.y), q.z)
}

/// Angular part of a velocity field (3D loads only the angular quad instead
/// of the whole velocity).
#[cfg(feature = "dim3")]
#[inline]
pub fn ws_vel_ang(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> crate::AngVector {
    ws_vec(buf, a, k, f + 1)
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_vel_ang(buf: &[Vec4], a: WsAddr, k: u32, f: u32) -> crate::AngVector {
    buf.read(a.at(k, f)).z
}

#[cfg(feature = "dim3")]
#[inline]
pub fn ws_set_vel(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, v: Velocity) {
    ws_set_vec(buf, a, k, f, v.linear);
    ws_set_vec(buf, a, k, f + 1, v.angular);
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_set_vel(buf: &mut [Vec4], a: WsAddr, k: u32, f: u32, v: Velocity) {
    buf.write(a.at(k, f), Vec4::new(v.linear.x, v.linear.y, v.angular, 0.0));
}

/// Extract component `i` (0..4) of a `Vec4` by value (no reference indexing,
/// which would create SPIR-V pointer phis).
#[inline]
fn vec4_get(v: Vec4, i: u32) -> f32 {
    if i == 0 {
        v.x
    } else if i == 1 {
        v.y
    } else if i == 2 {
        v.z
    } else {
        v.w
    }
}

#[inline]
fn vec4_set(v: Vec4, i: u32, val: f32) -> Vec4 {
    let mut out = v;
    if i == 0 {
        out.x = val;
    } else if i == 1 {
        out.y = val;
    } else if i == 2 {
        out.z = val;
    } else {
        out.w = val;
    }
    out
}

/// Single generalized-coordinate accessors (`i < MAX_JOINT_DOFS`).
#[inline]
pub fn ws_coord(buf: &[Vec4], a: WsAddr, k: u32, i: u32) -> f32 {
    vec4_get(buf.read(a.at(k, WS_COORDS + i / 4)), i % 4)
}

/// Read-modify-write of one coordinate's quad.
#[inline]
pub fn ws_set_coord(buf: &mut [Vec4], a: WsAddr, k: u32, i: u32, v: f32) {
    let idx = a.at(k, WS_COORDS + i / 4);
    let q = buf.read(idx);
    buf.write(idx, vec4_set(q, i % 4, v));
}

/// Load the whole coords array (for `body_to_parent`).
#[cfg(feature = "dim3")]
#[inline]
pub fn ws_coords(buf: &[Vec4], a: WsAddr, k: u32) -> [f32; 6] {
    let q0 = buf.read(a.at(k, WS_COORDS));
    let q1 = buf.read(a.at(k, WS_COORDS + 1));
    [q0.x, q0.y, q0.z, q0.w, q1.x, q1.y]
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_coords(buf: &[Vec4], a: WsAddr, k: u32) -> [f32; 3] {
    let q0 = buf.read(a.at(k, WS_COORDS));
    [q0.x, q0.y, q0.z]
}

/// World-space inertia of link `k` — the SoA counterpart of the former
/// `MultibodyLinkWorkspace::link_world_inertia` (reads only the
/// local-to-world rotation).
#[cfg(feature = "dim3")]
#[inline]
pub fn ws_world_inertia(
    buf: &[Vec4],
    a: WsAddr,
    k: u32,
    lmp: &crate::dynamics::body::LocalMassProperties,
) -> glamx::Mat3 {
    use crate::rotation_to_matrix;
    let ipi = lmp.inv_principal_inertia;
    let px = if ipi.x != 0.0 { 1.0 / ipi.x } else { 0.0 };
    let py = if ipi.y != 0.0 { 1.0 / ipi.y } else { 0.0 };
    let pz = if ipi.z != 0.0 { 1.0 / ipi.z } else { 0.0 };
    let rot = ws_rot(buf, a, k, WS_LTW);
    let r = rotation_to_matrix(rot * lmp.inertia_ref_frame);
    // M = r · diag(px, py, pz) (column-scale); I = M · rᵀ.
    let m = glamx::Mat3::from_cols(r.x_axis * px, r.y_axis * py, r.z_axis * pz);
    m * r.transpose()
}

#[cfg(feature = "dim2")]
#[inline]
pub fn ws_world_inertia(
    _buf: &[Vec4],
    _a: WsAddr,
    _k: u32,
    lmp: &crate::dynamics::body::LocalMassProperties,
) -> f32 {
    if lmp.inv_inertia != 0.0 {
        1.0 / lmp.inv_inertia
    } else {
        0.0
    }
}

/*
 * Host-side explode of the AoS structs into the SoA buffer. `data` is
 * batch-major (`batch · links_cap + link`), as built by `from_rapier`
 * BEFORE interleaving; the output is the GPU-ready SoA buffer.
 */
#[cfg(not(target_arch_is_gpu))]
pub fn ws_soa_from_structs(
    data: &[MultibodyLinkWorkspace],
    links_cap: u32,
    num_batches: u32,
) -> std::vec::Vec<Vec4> {
    let mut out =
        std::vec![Vec4::ZERO; links_cap as usize * WS_QUADS as usize * num_batches as usize];
    for b in 0..num_batches {
        let a = WsAddr::new(0, num_batches, b);
        for k in 0..links_cap {
            let ws = &data[(b * links_cap + k) as usize];
            ws_set_rot(&mut out, a, k, WS_JOINT_ROT, ws.joint_rot);
            for (i, &c) in ws.coords.iter().enumerate() {
                ws_set_coord(&mut out, a, k, i as u32, c);
            }
            ws_set_pose(&mut out, a, k, WS_LTP, ws.local_to_parent);
            ws_set_pose(&mut out, a, k, WS_LTW, ws.local_to_world);
            ws_set_vec(&mut out, a, k, WS_SHIFT02, ws.shift02);
            ws_set_vec(&mut out, a, k, WS_SHIFT23, ws.shift23);
            ws_set_vel(&mut out, a, k, WS_JOINT_VEL, ws.joint_velocity);
            ws_set_vel(&mut out, a, k, WS_RB_VELS, ws.rb_vels);
            ws_set_vel(&mut out, a, k, WS_KIN_ACC, ws.kinematic_acc);
        }
    }
    out
}
