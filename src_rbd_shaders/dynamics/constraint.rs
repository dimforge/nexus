//! Contact constraint data structures for the iterative solver.
//!
//! A [`TwoBodyConstraint`] is one contact manifold between two bodies. It is kept compact
//! since every solver sweep streams it: each contact point only stores its lever arms,
//! anchors, impulses and effective masses, and the constraint stores both bodies' inverse
//! mass and (symmetric) world inverse inertia, from which the angular Jacobians are
//! recomputed on the fly.

use crate::{AngVector, Pose, Vector, gcross, gdot};
use glamx::UVec2;
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim3")]
use glamx::{Vec2, Vec3};

#[cfg(feature = "dim2")]
/// Number of tangent constraint directions (2D: one tangent perpendicular to normal).
pub const SUB_LEN: usize = 1;

#[cfg(feature = "dim3")]
/// Number of tangent constraint directions (3D: two tangents in contact plane).
pub const SUB_LEN: usize = 2;

#[cfg(feature = "dim2")]
/// Maximum number of contact points per contact manifold (2D: typically 2).
pub const MAX_CONSTRAINTS_PER_MANIFOLD: usize = 2;

#[cfg(feature = "dim3")]
/// Maximum number of contact points per contact manifold (3D: up to 4).
pub const MAX_CONSTRAINTS_PER_MANIFOLD: usize = 4;

/// A world-space inverse angular inertia tensor (symmetric).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")]
pub struct SymInertia {
    /// The diagonal `(xx, yy, zz)`.
    pub diag: Vec3,
    pub _padding0: u32,
    /// The off-diagonal terms `(xy, xz, yz)`.
    pub off: Vec3,
    pub _padding1: u32,
}

#[cfg(feature = "dim3")]
impl SymInertia {
    /// The symmetric part of `m`'s upper-left 3x3 block.
    #[inline(always)]
    pub fn from_mat4(m: glamx::Mat4) -> Self {
        Self {
            diag: Vec3::new(m.x_axis.x, m.y_axis.y, m.z_axis.z),
            _padding0: 0,
            off: Vec3::new(m.y_axis.x, m.z_axis.x, m.z_axis.y),
            _padding1: 0,
        }
    }

    /// Multiplies `v` by this tensor.
    #[inline(always)]
    pub fn mul(&self, v: Vec3) -> Vec3 {
        let d = self.diag;
        let o = self.off;
        Vec3::new(
            d.x * v.x + o.x * v.y + o.y * v.z,
            o.x * v.x + d.y * v.y + o.z * v.z,
            o.y * v.x + o.z * v.y + d.z * v.z,
        )
    }

    /// Scales every term of this tensor.
    #[inline(always)]
    pub fn scale(&mut self, s: f32) {
        self.diag *= s;
        self.off *= s;
    }
}

/// A contact manifold between two rigid bodies, as the solver sees it.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")]
pub struct TwoBodyConstraint {
    /// Contact normal from body A's perspective (points away from A).
    pub dir_a: Vector,
    /// Number of active contact points in this manifold.
    pub len: u32,
    /// First friction direction (orthogonal to the normal); the second is `dir_a × tangent_a`.
    pub tangent_a: Vector,
    /// Friction coefficient.
    pub limit: f32,
    /// Inverse mass of body A along each axis.
    pub im_a: Vector,
    /// Index of body A (poses, mass properties).
    pub solver_body_a: u32,
    /// Inverse mass of body B along each axis.
    pub im_b: Vector,
    /// Index of body B.
    pub solver_body_b: u32,
    /// World-space inverse inertia of body A.
    pub ii_a: SymInertia,
    /// World-space inverse inertia of body B.
    pub ii_b: SymInertia,
    /// Index of body A's velocity in the solver velocity buffer: `solver_body_a`, or one of
    /// the sub-body slots when body A is split (see `mass_splitting`).
    pub vel_slot_a: u32,
    /// Index of body B's velocity in the solver velocity buffer.
    pub vel_slot_b: u32,
    /// Collider A of the source manifold, which identifies it across steps with
    /// `warmstart_collider_b` and `warmstart_subshape` (for the warmstart and color seeding).
    pub warmstart_collider_a: u32,
    /// Collider B of the source manifold.
    pub warmstart_collider_b: u32,
    /// [`IndexedManifold::subshape`] of the source manifold.
    ///
    /// [`IndexedManifold::subshape`]: crate::queries::IndexedManifold::subshape
    pub warmstart_subshape: u32,
    /// Combined restitution coefficient.
    pub restitution: f32,
    pub _padding: [u32; 2],
    /// The contact points (the first `len` are active).
    pub points: [ContactPoint; MAX_CONSTRAINTS_PER_MANIFOLD],
}

/// A contact manifold between two rigid bodies, as the solver sees it.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")]
pub struct TwoBodyConstraint {
    /// Contact normal from body A's perspective (points away from A).
    pub dir_a: Vector,
    /// Number of active contact points in this manifold.
    pub len: u32,
    /// Friction coefficient.
    pub limit: f32,
    /// Inverse mass of body A along each axis.
    pub im_a: Vector,
    /// Inverse mass of body B along each axis.
    pub im_b: Vector,
    /// World-space inverse inertia of body A.
    pub ii_a: f32,
    /// World-space inverse inertia of body B.
    pub ii_b: f32,
    /// Index of body A (poses, mass properties).
    pub solver_body_a: u32,
    /// Index of body B.
    pub solver_body_b: u32,
    /// Index of body A's velocity in the solver velocity buffer: `solver_body_a`, or one of
    /// the sub-body slots when body A is split (see `mass_splitting`).
    pub vel_slot_a: u32,
    /// Index of body B's velocity in the solver velocity buffer.
    pub vel_slot_b: u32,
    /// Collider A of the source manifold, which identifies it across steps with
    /// `warmstart_collider_b` and `warmstart_subshape` (for the warmstart and color seeding).
    pub warmstart_collider_a: u32,
    /// Collider B of the source manifold.
    pub warmstart_collider_b: u32,
    /// [`IndexedManifold::subshape`] of the source manifold.
    ///
    /// [`IndexedManifold::subshape`]: crate::queries::IndexedManifold::subshape
    pub warmstart_subshape: u32,
    /// Combined restitution coefficient.
    pub restitution: f32,
    pub _padding: [u32; 2],
    /// The contact points (the first `len` are active).
    pub points: [ContactPoint; MAX_CONSTRAINTS_PER_MANIFOLD],
}

/// When the contacts of a [`TwoBodyConstraint`] were last computed, for contact recycling (see
/// `RbdSimParams::normalized_contact_recycle_distance`).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ContactRecycleState {
    /// World pose of the first collider.
    pub pose_a: Pose,
    /// World pose of the second collider.
    pub pose_b: Pose,
    /// The collider pair.
    pub colliders: UVec2,
    /// See `IndexedManifold::recycle_extent`.
    pub max_extent: f32,
    /// The drift allowed before the contacts are computed again (0 if they can't be recycled).
    pub max_drift: f32,
}

/// One contact point of a [`TwoBodyConstraint`].
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")]
pub struct ContactPoint {
    /// The contact point relative to body A's center of mass, in world space (frozen at the
    /// start of the step, like the Jacobians).
    pub r_a: Vector,
    /// Accumulated normal impulse.
    pub normal_impulse: f32,
    /// The contact point relative to body B's center of mass, in world space.
    pub r_b: Vector,
    /// Inverse of the effective mass along the normal.
    pub normal_mass: f32,
    /// The contact point in body A's (center-of-mass) frame, to track its current position.
    pub local_pt_a: Vector,
    /// Signed distance at the start of the step (negative = penetration).
    pub dist: f32,
    /// The contact point in body B's frame.
    pub local_pt_b: Vector,
    /// Target normal relative velocity (restitution).
    pub normal_vel: f32,
    /// Accumulated friction impulses (along `tangent_a` and `dir_a × tangent_a`).
    pub tangent_impulse: Vec2,
    /// Effective mass of the friction directions: `[k00, k11, 2 k01]` (not inverted).
    pub tangent_k: [f32; 3],
    pub _padding: [u32; 3],
}

/// One contact point of a [`TwoBodyConstraint`].
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")]
pub struct ContactPoint {
    /// The contact point relative to body A's center of mass, in world space.
    pub r_a: Vector,
    /// The contact point relative to body B's center of mass, in world space.
    pub r_b: Vector,
    /// The contact point in body A's (center-of-mass) frame.
    pub local_pt_a: Vector,
    /// The contact point in body B's frame.
    pub local_pt_b: Vector,
    /// Accumulated normal impulse.
    pub normal_impulse: f32,
    /// Inverse of the effective mass along the normal.
    pub normal_mass: f32,
    /// Signed distance at the start of the step (negative = penetration).
    pub dist: f32,
    /// Target normal relative velocity (restitution).
    pub normal_vel: f32,
    /// Accumulated friction impulse.
    pub tangent_impulse: [f32; 1],
    /// Inverse of the effective mass along the tangent.
    pub tangent_mass: f32,
    pub _padding: [u32; 2],
}

#[inline(always)]
fn inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

impl TwoBodyConstraint {
    /// Body A's inverse inertia times `v`.
    #[inline(always)]
    pub fn ii_a_mul(&self, v: AngVector) -> AngVector {
        #[cfg(feature = "dim2")]
        return self.ii_a * v;
        #[cfg(feature = "dim3")]
        return self.ii_a.mul(v);
    }

    /// Body B's inverse inertia times `v`.
    #[inline(always)]
    pub fn ii_b_mul(&self, v: AngVector) -> AngVector {
        #[cfg(feature = "dim2")]
        return self.ii_b * v;
        #[cfg(feature = "dim3")]
        return self.ii_b.mul(v);
    }

    /// The friction directions.
    #[inline(always)]
    pub fn tangents(&self) -> [Vector; SUB_LEN] {
        #[cfg(feature = "dim2")]
        return [Vector::new(-self.dir_a.y, self.dir_a.x)];
        #[cfg(feature = "dim3")]
        return [self.tangent_a, self.dir_a.cross(self.tangent_a)];
    }

    /// Computes the effective masses of every contact point from the lever arms, inverse
    /// masses and inertias.
    #[inline(always)]
    pub fn compute_effective_masses(&mut self) {
        let dir = self.dir_a;
        let imsum = self.im_a + self.im_b;
        let tangents = self.tangents();

        for k in 0..self.len as usize {
            let r_a = self.points.at(k).r_a;
            let r_b = self.points.at(k).r_b;

            let ta = gcross(r_a, dir);
            let tb = gcross(r_b, -dir);
            let k_n =
                dir.dot(imsum * dir) + gdot(self.ii_a_mul(ta), ta) + gdot(self.ii_b_mul(tb), tb);
            self.points.at_mut(k).normal_mass = inv(k_n);

            #[cfg(feature = "dim2")]
            {
                let t = tangents.read(0);
                let ta = gcross(r_a, t);
                let tb = gcross(r_b, -t);
                let k_t =
                    t.dot(imsum * t) + gdot(self.ii_a_mul(ta), ta) + gdot(self.ii_b_mul(tb), tb);
                self.points.at_mut(k).tangent_mass = inv(k_t);
            }

            #[cfg(feature = "dim3")]
            {
                let t0 = tangents.read(0);
                let t1 = tangents.read(1);
                let ta0 = gcross(r_a, t0);
                let tb0 = gcross(r_b, -t0);
                let ta1 = gcross(r_a, t1);
                let tb1 = gcross(r_b, -t1);
                let iia1 = self.ii_a_mul(ta1);
                let iib1 = self.ii_b_mul(tb1);
                let k00 =
                    t0.dot(imsum * t0) + self.ii_a_mul(ta0).dot(ta0) + self.ii_b_mul(tb0).dot(tb0);
                let k11 = t1.dot(imsum * t1) + iia1.dot(ta1) + iib1.dot(tb1);
                let k01 = 2.0 * (ta0.dot(iia1) + tb0.dot(iib1));
                self.points.at_mut(k).tangent_k = [k00, k11, k01];
            }
        }
    }
}
