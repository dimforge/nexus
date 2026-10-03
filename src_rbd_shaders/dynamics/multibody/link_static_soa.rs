//! Quad-interleaved storage of the per-link static data.
//!
//! [`MultibodyLinkStatic`] records are stored as `LS_QUADS` quads interleaved
//! across batches at quad granularity: quad `q` of link `L` of batch `b` is at
//! `(L * LS_QUADS + q) * num_batches + b`. Kernels running one thread per
//! environment then read neighbouring quads from neighbouring lanes instead
//! of one cache line per lane and field. [`LinkStatics::get`] reassembles a
//! record; the loads of the fields a kernel never uses are dead code.

use super::types::MultibodyLinkStatic;
use crate::dynamics::body::LocalMassProperties;
use crate::dynamics::joint::{GenericJoint, JointLimits, JointMotor};
use crate::{Pose, Rotation, Vector};
use core::mem::{offset_of, size_of};
use glamx::UVec4;
use khal_std::index::MaybeIndexUnchecked;

/// Quads per link record (the last one zero-padded when the record size is not
/// a multiple of 16 bytes, as in 2D).
pub const LS_QUADS: u32 = size_of::<MultibodyLinkStatic>().div_ceil(16) as u32;

/// 32-bit words per link record.
pub const LS_WORDS: usize = size_of::<MultibodyLinkStatic>() / 4;

/// Flat quad index of quad `q` of link `link` (relative to the batch's link
/// slice start) of batch `shift` among `stride` batches.
#[inline(always)]
pub fn ls_quad_index(link: usize, q: usize, stride: u32, shift: u32) -> usize {
    (link * LS_QUADS as usize + q) * stride as usize + shift as usize
}

/// Word `w` (32-bit) of a record, as a quad index and lane.
#[inline(always)]
pub fn ls_word_position(w: usize) -> (usize, usize) {
    (w / 4, w % 4)
}

#[inline(always)]
fn quad_lane(q: UVec4, lane: usize) -> u32 {
    match lane {
        0 => q.x,
        1 => q.y,
        2 => q.z,
        _ => q.w,
    }
}

/// Interleaved view of one batch's link records (see the module docs).
#[derive(Copy, Clone)]
pub struct LinkStatics<'a> {
    pub buf: &'a [UVec4],
    pub base: usize,
    pub stride: u32,
    pub shift: u32,
}

impl<'a> LinkStatics<'a> {
    #[inline(always)]
    pub fn new(buf: &'a [UVec4], base: usize, stride: u32, shift: u32) -> Self {
        Self {
            buf,
            base,
            stride,
            shift,
        }
    }

    /// Re-based view (like `ISlice::offset`).
    #[inline(always)]
    pub fn offset(self, links: usize) -> Self {
        Self {
            base: self.base + links,
            ..self
        }
    }

    /// Word `w` of link `k`'s record.
    #[inline(always)]
    pub fn word(&self, k: usize, w: usize) -> u32 {
        let (q, lane) = ls_word_position(w);
        quad_lane(
            self.buf
                .read(ls_quad_index(self.base + k, q, self.stride, self.shift)),
            lane,
        )
    }

    /// Link `k`'s record.
    #[inline(always)]
    pub fn get(&self, k: usize) -> MultibodyLinkStatic {
        MultibodyLinkStatic::from_words(|w| self.word(k, w), 0)
    }
}

/// One link of a [`LinkStatics`] view, read field by field (only the fields a
/// kernel touches are loaded, even under divergent control flow).
#[derive(Copy, Clone)]
pub struct LinkStaticRef<'a> {
    pub statics: LinkStatics<'a>,
    pub k: usize,
}

impl<'a> LinkStatics<'a> {
    /// Lazy view of link `k`.
    #[inline(always)]
    pub fn at(&self, k: usize) -> LinkStaticRef<'a> {
        LinkStaticRef { statics: *self, k }
    }
}

impl LinkStaticRef<'_> {
    #[inline(always)]
    fn word(&self, w: usize) -> u32 {
        self.statics.word(self.k, w)
    }

    #[inline(always)]
    pub fn assembly_id(&self) -> u32 {
        self.word(offset_of!(MultibodyLinkStatic, assembly_id) / 4)
    }

    #[inline(always)]
    pub fn ndofs(&self) -> u32 {
        self.word(offset_of!(MultibodyLinkStatic, ndofs) / 4)
    }

    /// Word `i` of `ancestor_dofs` (3D only).
    #[cfg(feature = "dim3")]
    #[inline(always)]
    pub fn ancestor_dofs(&self, i: usize) -> u32 {
        self.word(offset_of!(MultibodyLinkStatic, ancestor_dofs) / 4 + i)
    }

    #[inline(always)]
    pub fn kinematic(&self) -> u32 {
        self.word(offset_of!(MultibodyLinkStatic, kinematic) / 4)
    }

    #[inline(always)]
    fn joint_word(&self, offset: usize) -> u32 {
        self.word((offset_of!(MultibodyLinkStatic, data) + offset) / 4)
    }

    #[inline(always)]
    pub fn locked_axes(&self) -> u32 {
        self.joint_word(offset_of!(GenericJoint, locked_axes))
    }

    #[inline(always)]
    pub fn limit_axes(&self) -> u32 {
        self.joint_word(offset_of!(GenericJoint, limit_axes))
    }

    #[inline(always)]
    pub fn motor_axes(&self) -> u32 {
        self.joint_word(offset_of!(GenericJoint, motor_axes))
    }

    /// The limits of `axis`.
    #[inline(always)]
    pub fn limit(&self, axis: usize) -> JointLimits {
        let base = (offset_of!(MultibodyLinkStatic, data)
            + offset_of!(GenericJoint, limits)
            + axis * size_of::<JointLimits>())
            / 4;
        JointLimits::from_words(|w| self.word(w), base)
    }

    /// The motor of `axis`.
    #[inline(always)]
    pub fn motor(&self, axis: usize) -> JointMotor {
        let base = (offset_of!(MultibodyLinkStatic, data)
            + offset_of!(GenericJoint, motors)
            + axis * size_of::<JointMotor>())
            / 4;
        JointMotor::from_words(|w| self.word(w), base)
    }
}

/// Reassembly of a record type from its 32-bit words (`w(i)` is word `i` of
/// the enclosing record; the value starts at word `base`).
pub trait FromWords: Sized {
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self;
}

impl FromWords for u32 {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        w(base)
    }
}

impl FromWords for f32 {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        f32::from_bits(w(base))
    }
}

impl FromWords for glamx::Vec2 {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        glamx::Vec2::new(f32::from_bits(w(base)), f32::from_bits(w(base + 1)))
    }
}

impl FromWords for glamx::Vec3 {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        glamx::Vec3::new(
            f32::from_bits(w(base)),
            f32::from_bits(w(base + 1)),
            f32::from_bits(w(base + 2)),
        )
    }
}

#[cfg(feature = "dim3")]
impl FromWords for glamx::Quat {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        glamx::Quat::from_xyzw(
            f32::from_bits(w(base)),
            f32::from_bits(w(base + 1)),
            f32::from_bits(w(base + 2)),
            f32::from_bits(w(base + 3)),
        )
    }
}

#[cfg(feature = "dim2")]
impl FromWords for glamx::Rot2 {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        glamx::Rot2::from_cos_sin_unchecked(f32::from_bits(w(base)), f32::from_bits(w(base + 1)))
    }
}

impl FromWords for Pose {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        let rotation = Rotation::from_words(w, base + offset_of!(Pose, rotation) / 4);
        let translation = Vector::from_words(w, base + offset_of!(Pose, translation) / 4);
        #[cfg(feature = "dim3")]
        {
            Pose {
                rotation,
                translation,
                padding: w(base + offset_of!(Pose, padding) / 4),
            }
        }
        #[cfg(feature = "dim2")]
        {
            Pose {
                rotation,
                translation,
            }
        }
    }
}

impl FromWords for JointLimits {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        JointLimits {
            min: f32::from_words(w, base + offset_of!(JointLimits, min) / 4),
            max: f32::from_words(w, base + offset_of!(JointLimits, max) / 4),
            impulse: f32::from_words(w, base + offset_of!(JointLimits, impulse) / 4),
        }
    }
}

impl FromWords for JointMotor {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        JointMotor {
            target_vel: f32::from_words(w, base + offset_of!(JointMotor, target_vel) / 4),
            target_pos: f32::from_words(w, base + offset_of!(JointMotor, target_pos) / 4),
            stiffness: f32::from_words(w, base + offset_of!(JointMotor, stiffness) / 4),
            damping: f32::from_words(w, base + offset_of!(JointMotor, damping) / 4),
            max_force: f32::from_words(w, base + offset_of!(JointMotor, max_force) / 4),
            impulse: f32::from_words(w, base + offset_of!(JointMotor, impulse) / 4),
            model: u32::from_words(w, base + offset_of!(JointMotor, model) / 4),
        }
    }
}

impl FromWords for GenericJoint {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        let limits = base + offset_of!(GenericJoint, limits) / 4;
        let limit =
            |i: usize| JointLimits::from_words(w, limits + i * size_of::<JointLimits>() / 4);
        let motors = base + offset_of!(GenericJoint, motors) / 4;
        let motor = |i: usize| JointMotor::from_words(w, motors + i * size_of::<JointMotor>() / 4);
        GenericJoint {
            local_frame_a: Pose::from_words(w, base + offset_of!(GenericJoint, local_frame_a) / 4),
            local_frame_b: Pose::from_words(w, base + offset_of!(GenericJoint, local_frame_b) / 4),
            locked_axes: u32::from_words(w, base + offset_of!(GenericJoint, locked_axes) / 4),
            limit_axes: u32::from_words(w, base + offset_of!(GenericJoint, limit_axes) / 4),
            motor_axes: u32::from_words(w, base + offset_of!(GenericJoint, motor_axes) / 4),
            coupled_axes: u32::from_words(w, base + offset_of!(GenericJoint, coupled_axes) / 4),
            #[cfg(feature = "dim3")]
            limits: [limit(0), limit(1), limit(2), limit(3), limit(4), limit(5)],
            #[cfg(feature = "dim2")]
            limits: [limit(0), limit(1), limit(2)],
            #[cfg(feature = "dim3")]
            motors: [motor(0), motor(1), motor(2), motor(3), motor(4), motor(5)],
            #[cfg(feature = "dim2")]
            motors: [motor(0), motor(1), motor(2)],
        }
    }
}

impl FromWords for LocalMassProperties {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        LocalMassProperties {
            #[cfg(feature = "dim3")]
            inertia_ref_frame: Rotation::from_words(
                w,
                base + offset_of!(LocalMassProperties, inertia_ref_frame) / 4,
            ),
            #[cfg(feature = "dim3")]
            inv_principal_inertia: glamx::Vec3::from_words(
                w,
                base + offset_of!(LocalMassProperties, inv_principal_inertia) / 4,
            ),
            #[cfg(feature = "dim3")]
            padding0: u32::from_words(w, base + offset_of!(LocalMassProperties, padding0) / 4),
            inv_mass: Vector::from_words(w, base + offset_of!(LocalMassProperties, inv_mass) / 4),
            #[cfg(feature = "dim3")]
            padding1: u32::from_words(w, base + offset_of!(LocalMassProperties, padding1) / 4),
            com: Vector::from_words(w, base + offset_of!(LocalMassProperties, com) / 4),
            padding2: u32::from_words(w, base + offset_of!(LocalMassProperties, padding2) / 4),
            #[cfg(feature = "dim2")]
            inv_inertia: f32::from_words(
                w,
                base + offset_of!(LocalMassProperties, inv_inertia) / 4,
            ),
        }
    }
}

impl FromWords for MultibodyLinkStatic {
    #[inline(always)]
    fn from_words<W: Fn(usize) -> u32 + Copy>(w: W, base: usize) -> Self {
        MultibodyLinkStatic {
            rb_id: u32::from_words(w, base + offset_of!(MultibodyLinkStatic, rb_id) / 4),
            parent_link_id: u32::from_words(
                w,
                base + offset_of!(MultibodyLinkStatic, parent_link_id) / 4,
            ),
            multibody_id: u32::from_words(
                w,
                base + offset_of!(MultibodyLinkStatic, multibody_id) / 4,
            ),
            assembly_id: u32::from_words(
                w,
                base + offset_of!(MultibodyLinkStatic, assembly_id) / 4,
            ),
            ndofs: u32::from_words(w, base + offset_of!(MultibodyLinkStatic, ndofs) / 4),
            kinematic: u32::from_words(w, base + offset_of!(MultibodyLinkStatic, kinematic) / 4),
            #[cfg(feature = "dim3")]
            ancestor_dofs: [
                w(base + offset_of!(MultibodyLinkStatic, ancestor_dofs) / 4),
                w(base + offset_of!(MultibodyLinkStatic, ancestor_dofs) / 4 + 1),
            ],
            data: GenericJoint::from_words(w, base + offset_of!(MultibodyLinkStatic, data) / 4),
            local_mprops: LocalMassProperties::from_words(
                w,
                base + offset_of!(MultibodyLinkStatic, local_mprops) / 4,
            ),
        }
    }
}

/// Word offset of the `target_pos` of motor `axis` in a record.
#[inline(always)]
pub fn ls_motor_target_word(axis: usize) -> usize {
    (offset_of!(MultibodyLinkStatic, data)
        + offset_of!(GenericJoint, motors)
        + axis * size_of::<JointMotor>()
        + offset_of!(JointMotor, target_pos))
        / 4
}

/// Word offset of `data.motor_axes` in a record.
#[inline(always)]
pub fn ls_motor_axes_word() -> usize {
    (offset_of!(MultibodyLinkStatic, data) + offset_of!(GenericJoint, motor_axes)) / 4
}

/*
 * Host-side conversions between batch-interleaved records (`link * num_batches
 * + batch`, the former layout) and the quad-interleaved buffer.
 */
#[cfg(not(target_arch_is_gpu))]
pub fn ls_soa_from_structs(data: &[MultibodyLinkStatic], num_batches: u32) -> std::vec::Vec<UVec4> {
    let nb = num_batches.max(1) as usize;
    let links = data.len() / nb;
    let mut out = std::vec![UVec4::ZERO; data.len() * LS_QUADS as usize];
    for link in 0..links {
        for b in 0..nb {
            let words: &[u32] = bytemuck::cast_slice(core::slice::from_ref(&data[link * nb + b]));
            for (w, word) in words.iter().enumerate() {
                let (q, lane) = ls_word_position(w);
                let quad = &mut out[ls_quad_index(link, q, nb as u32, b as u32)];
                match lane {
                    0 => quad.x = *word,
                    1 => quad.y = *word,
                    2 => quad.z = *word,
                    _ => quad.w = *word,
                }
            }
        }
    }
    out
}

#[cfg(not(target_arch_is_gpu))]
pub fn ls_soa_to_structs(buf: &[UVec4], num_batches: u32) -> std::vec::Vec<MultibodyLinkStatic> {
    let nb = num_batches.max(1) as usize;
    let records = buf.len() / LS_QUADS as usize;
    let links = records / nb;
    let mut out: std::vec::Vec<MultibodyLinkStatic> = bytemuck::zeroed_vec(records);
    for link in 0..links {
        for b in 0..nb {
            let words: &mut [u32] =
                bytemuck::cast_slice_mut(core::slice::from_mut(&mut out[link * nb + b]));
            for (w, word) in words.iter_mut().enumerate() {
                let (q, lane) = ls_word_position(w);
                *word = quad_lane(buf[ls_quad_index(link, q, nb as u32, b as u32)], lane);
            }
        }
    }
    out
}

/// The quads of one record, for a dense one-batch upload.
#[cfg(not(target_arch_is_gpu))]
pub fn ls_record_quads(record: &MultibodyLinkStatic) -> std::vec::Vec<UVec4> {
    ls_soa_from_structs(core::slice::from_ref(record), 1)
}

impl crate::utils::BatchIndices {
    /// Batch `batch_id`'s view of the quad-interleaved link records (use
    /// `.offset(...)` for the intra-batch link offset).
    #[inline(always)]
    pub fn ls<'s>(&self, batch_id: u32, buf: &'s [UVec4]) -> LinkStatics<'s> {
        LinkStatics::new(buf, 0, self.num_batches, batch_id)
    }
}
