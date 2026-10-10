//! Manifold twist friction with dedicated compact storage.
//!
//! Normal rows store one frozen lever arm. The other differs by the same
//! frozen COM offset for every point, including recycled contacts. Sliding and
//! twist impulses/masses are stored once per manifold, never per point.
use super::solver_utils::{
    ContactPointAccess, ContactSolverData, cap_magnitude, compute_tangent_contact_directions,
    maybe_inv,
};
use super::{ContactLink, ContactPoint, RbdSimParams, SymInertia, Velocity, WorldMassProperties};
use crate::queries::IndexedManifold;
use crate::utils::Slice;
use crate::{Pose, Vector};
use crunchy::unroll;
use glamx::{Quat, Vec2, Vec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;

/// A normal-only contact row (32 bytes), in the frozen world frame.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwistContactPoint {
    pub r_a: Vec3,
    pub normal_impulse: f32,
    pub normal_mass: f32,
    pub dist: f32,
    pub normal_vel: f32,
    pub radius: f32,
}

/// One sliding row pair and one normal-axis angular row for the manifold.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwistFriction {
    pub center_a: Vec3,
    pub twist_mass: f32,
    /// k00, k11, 2*k01 of the central sliding rows.
    pub tangent_k: Vec3,
    pub twist_impulse: f32,
    pub tangent_impulse: Vec2,
    pub _padding: [u32; 2],
}

/// Compact twist-friction manifold (384 bytes versus Coulomb's 544).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwistConstraint {
    /// Number of active contact points in this manifold.
    pub len: u32,
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
    /// Contact normal from body A's perspective (points away from A).
    pub dir_a: Vector,
    /// Friction coefficient.
    pub limit: f32,
    /// First friction direction (orthogonal to the normal); the second is `dir_a × tangent_a`.
    pub tangent_a: Vector,
    /// Combined restitution coefficient.
    pub restitution: f32,
    /// Inverse mass of body A along each axis.
    pub im_a: Vector,
    pub _padding_a: u32,
    /// Inverse mass of body B along each axis.
    pub im_b: Vector,
    pub _padding_b: u32,
    /// World-space inverse inertia of body A.
    pub ii_a: SymInertia,
    /// World-space inverse inertia of body B.
    pub ii_b: SymInertia,
    pub points: [TwistContactPoint; 4],
    /// Frozen r_b - r_a, shared by all contact points.
    pub offset_b: Vec3,
    pub _padding_offset: u32,
    /// Body rotations when the contact geometry was frozen. Local anchors are
    /// recovered from these and the frozen lever arms, including after recycling.
    pub frame_a: Quat,
    pub frame_b: Quat,
    pub friction: TwistFriction,
}

impl TwistConstraint {
    #[inline(always)]
    pub fn init_header(&mut self, manifold: &IndexedManifold, mprops: &Slice<WorldMassProperties>) {
        let a = mprops[manifold.bodies.x as usize];
        let b = mprops[manifold.bodies.y as usize];
        self.im_a = a.inv_mass;
        self.im_b = b.inv_mass;
        self.ii_a = SymInertia::from_mat4(a.inv_inertia);
        self.ii_b = SymInertia::from_mat4(b.inv_inertia);
        self.solver_body_a = manifold.bodies.x;
        self.solver_body_b = manifold.bodies.y;
        self.vel_slot_a = self.solver_body_a;
        self.vel_slot_b = self.solver_body_b;
        self.limit = manifold.friction;
        self.restitution = manifold.restitution;
        self.warmstart_collider_a = manifold.colliders.x;
        self.warmstart_collider_b = manifold.colliders.y;
        self.warmstart_subshape = manifold.subshape;
        self.len = manifold.contact.len;
    }
    #[inline(always)]
    pub fn init_contact(
        &mut self,
        manifold: &IndexedManifold,
        mprops: &Slice<WorldMassProperties>,
        collider_poses: &Slice<Pose>,
        solver_poses: &Slice<Pose>,
        vels: &Slice<Velocity>,
    ) {
        self.init_contact_with_poses(
            manifold,
            mprops,
            collider_poses[manifold.colliders.x as usize],
            solver_poses[manifold.bodies.x as usize],
            solver_poses[manifold.bodies.y as usize],
            vels,
        );
    }
    #[inline(always)]
    pub fn init_contact_with_poses(
        &mut self,
        manifold: &IndexedManifold,
        mprops: &Slice<WorldMassProperties>,
        ca: Pose,
        a: Pose,
        b: Pose,
        vels: &Slice<Velocity>,
    ) {
        self.init_header(manifold, mprops);
        self.init_points_with_poses(manifold, ca, a, b, vels);
    }
    /// Initializes the contact points of a constraint whose header was initialized from
    /// `manifold`, and its effective masses.
    #[inline(always)]
    pub fn init_points_with_poses(
        &mut self,
        manifold: &IndexedManifold,
        ca: Pose,
        a: Pose,
        b: Pose,
        vels: &Slice<Velocity>,
    ) {
        let va = vels[self.solver_body_a as usize];
        let vb = vels[self.solver_body_b as usize];
        self.dir_a = -(ca.rotation * manifold.contact.normal_a);
        self.tangent_a = compute_tangent_contact_directions(self.dir_a).read(0);
        self.offset_b = a.translation - b.translation;
        self.friction = TwistFriction::default();
        self.frame_a = a.rotation;
        self.frame_b = b.rotation;
        for k in 0..self.len as usize {
            let contact = manifold.contact.points_a.at(k);
            let pt = ca * (contact.pt + manifold.contact.normal_a * contact.dist / 2.0);
            let ra = pt - a.translation;
            let rb = ra + self.offset_b;
            self.points.write(
                k,
                TwistContactPoint {
                    r_a: ra,
                    normal_impulse: 0.0,
                    normal_mass: 0.0,
                    dist: contact.dist,
                    normal_vel: self.restitution
                        * (va.linear + va.angular.cross(ra) - vb.linear - vb.angular.cross(rb))
                            .dot(self.dir_a),
                    radius: 0.0,
                },
            );
        }
        self.compute_effective_masses();
    }
    /// Applies the velocity slots and mass scales chosen by mass splitting.
    #[inline(always)]
    pub fn apply_link(&mut self, link: &ContactLink) {
        self.vel_slot_a = link.vel_slot_a;
        self.vel_slot_b = link.vel_slot_b;
        self.restitution = link.restitution;
        self.im_a *= link.mass_scale_a;
        self.im_b *= link.mass_scale_b;
        self.ii_a.scale(link.mass_scale_a);
        self.ii_b.scale(link.mass_scale_b);
    }
    #[inline(always)]
    pub fn compute_effective_masses(&mut self) {
        if self.len == 0 {
            return;
        }
        let mut center = Vec3::ZERO;
        let im = self.im_a + self.im_b;
        for k in 0..self.len as usize {
            let p = self.points.at_mut(k);
            let a = p.r_a.cross(self.dir_a);
            let b = (p.r_a + self.offset_b).cross(-self.dir_a);
            let kn =
                self.dir_a.dot(im * self.dir_a) + self.ii_a.mul(a).dot(a) + self.ii_b.mul(b).dot(b);
            p.normal_mass = if kn == 0.0 { 0.0 } else { 1.0 / kn };
            center += p.r_a;
        }
        center /= self.len as f32;
        self.friction.center_a = center;
        for k in 0..self.len as usize {
            let p = self.points.at_mut(k);
            p.radius = (p.r_a - center).length();
        }
        let t0 = self.tangent_a;
        let t1 = self.dir_a.cross(t0);
        let ra = center;
        let rb = center + self.offset_b;
        let a0 = ra.cross(t0);
        let a1 = ra.cross(t1);
        let b0 = rb.cross(-t0);
        let b1 = rb.cross(-t1);
        self.friction.tangent_k = Vec3::new(
            t0.dot(im * t0) + a0.dot(self.ii_a.mul(a0)) + b0.dot(self.ii_b.mul(b0)),
            t1.dot(im * t1) + a1.dot(self.ii_a.mul(a1)) + b1.dot(self.ii_b.mul(b1)),
            2.0 * (self.ii_a.mul(a0).dot(a1) + self.ii_b.mul(b0).dot(b1)),
        );
        self.friction.twist_mass = maybe_inv(
            self.dir_a.dot(self.ii_a.mul(self.dir_a)) + self.dir_a.dot(self.ii_b.mul(self.dir_a)),
        );
        if self.len == 1 {
            self.friction.twist_impulse = 0.0;
        }
    }
    #[inline(always)]
    pub fn recycle_from(&mut self, old: &Self, a: &Velocity, b: &Velocity) {
        self.dir_a = old.dir_a;
        self.tangent_a = old.tangent_a;
        self.len = old.len;
        self.offset_b = old.offset_b;
        self.friction = old.friction;
        self.frame_a = old.frame_a;
        self.frame_b = old.frame_b;
        for k in 0..self.len as usize {
            let mut p = old.points.read(k);
            p.normal_vel = self.restitution
                * (a.linear + a.angular.cross(p.r_a)
                    - b.linear
                    - b.angular.cross(p.r_a + self.offset_b))
                .dot(self.dir_a);
            self.points.write(k, p);
        }
        self.compute_effective_masses();
    }
    #[inline(always)]
    pub fn local_anchors(&self, k: usize) -> (Vec3, Vec3) {
        let r = self.points.at(k).r_a;
        (
            self.frame_a.inverse() * r,
            self.frame_b.inverse() * (r + self.offset_b),
        )
    }
    #[inline(always)]
    pub fn friction_warmstart(&self, _k: usize) -> Vec4 {
        let t = self.friction.tangent_impulse;
        (self.tangent_a * t.x + self.dir_a.cross(self.tangent_a) * t.y)
            .extend(self.friction.twist_impulse)
    }
    #[inline(always)]
    pub fn transfer_friction_point(&mut self, impulse: Vec4, _k: usize) {
        let world = impulse.truncate();
        self.friction.tangent_impulse += Vec2::new(
            world.dot(self.tangent_a),
            world.dot(self.dir_a.cross(self.tangent_a)),
        );
        self.friction.twist_impulse += impulse.w;
    }
    #[inline(always)]
    pub fn finish_friction_transfer(&mut self) {
        if self.len != 0 {
            self.friction.tangent_impulse /= self.len as f32;
            self.friction.twist_impulse = if self.len > 1 {
                self.friction.twist_impulse / self.len as f32
            } else {
                0.0
            };
        }
    }
    #[inline(always)]
    pub fn scale_impulses(&mut self, scale: f32) {
        for k in 0..self.len as usize {
            self.points.at_mut(k).normal_impulse *= scale;
        }
        self.friction.tangent_impulse *= scale;
        self.friction.twist_impulse *= scale;
    }
    #[inline(always)]
    pub(crate) fn solver_data(&self) -> ContactSolverData {
        ContactSolverData {
            dir_a: self.dir_a,
            len: self.len,
            tangent_a: self.tangent_a,
            limit: self.limit,
            im_a: self.im_a,
            im_b: self.im_b,
            ii_a: self.ii_a,
            ii_b: self.ii_b,
            solver_body_a: self.solver_body_a,
            solver_body_b: self.solver_body_b,
            vel_slot_a: self.vel_slot_a,
            vel_slot_b: self.vel_slot_b,
        }
    }
    #[inline(always)]
    pub fn warmstart_constraint(&self, a: &mut Velocity, b: &mut Velocity) {
        warmstart(&self.solver_data(), self, a, b);
    }
    #[inline(always)]
    pub fn solve_constraint_gauss_seidel(
        &mut self,
        poses: &Slice<Pose>,
        params: &RbdSimParams,
        a: &mut Velocity,
        b: &mut Velocity,
        use_bias: bool,
        friction: bool,
    ) {
        solve(
            &self.solver_data(),
            self,
            poses,
            params,
            a,
            b,
            use_bias,
            friction,
        );
    }
}

/// Access to a twist constraint, in registers or in a tile, for the shared twist arithmetic.
/// This is never used by the Coulomb kernels and contains no runtime friction-model branch.
pub(crate) trait TwistAccess: ContactPointAccess {
    fn friction(&self) -> TwistFriction;
    fn offset_b(&self) -> Vec3;
    fn frames(&self) -> (Quat, Quat);
    fn radius(&self, k: usize) -> f32;
    fn write_friction(&mut self, tangent: Vec2, twist: f32);
}
impl ContactPointAccess for TwistConstraint {
    #[inline(always)]
    fn read_point(&self, k: usize) -> ContactPoint {
        let p = self.points.at(k);
        ContactPoint {
            r_a: p.r_a,
            r_b: p.r_a + self.offset_b,
            local_pt_a: self.frame_a.inverse() * p.r_a,
            local_pt_b: self.frame_b.inverse() * (p.r_a + self.offset_b),
            normal_mass: p.normal_mass,
            normal_impulse: p.normal_impulse,
            normal_vel: p.normal_vel,
            dist: p.dist,
            ..ContactPoint::default()
        }
    }
    #[inline(always)]
    fn write_normal_impulse(&mut self, k: usize, value: f32) {
        self.points.at_mut(k).normal_impulse = value;
    }
    #[inline(always)]
    fn write_tangent_impulse(&mut self, _k: usize, _value: Vec2) {}
}
impl TwistAccess for TwistConstraint {
    #[inline(always)]
    fn friction(&self) -> TwistFriction {
        self.friction
    }
    #[inline(always)]
    fn offset_b(&self) -> Vec3 {
        self.offset_b
    }
    #[inline(always)]
    fn frames(&self) -> (Quat, Quat) {
        (self.frame_a, self.frame_b)
    }
    #[inline(always)]
    fn radius(&self, k: usize) -> f32 {
        self.points.at(k).radius
    }
    #[inline(always)]
    fn write_friction(&mut self, tangent: Vec2, twist: f32) {
        self.friction.tangent_impulse = tangent;
        self.friction.twist_impulse = twist;
    }
}
#[inline(always)]
pub(crate) fn warmstart<P: TwistAccess>(
    h: &ContactSolverData,
    points: &P,
    a: &mut Velocity,
    b: &mut Velocity,
) {
    if h.len == 0 {
        return;
    }
    let f = points.friction();
    let tangent =
        h.tangent_a * f.tangent_impulse.x + h.dir_a.cross(h.tangent_a) * f.tangent_impulse.y;
    let twist = h.dir_a * f.twist_impulse;
    let mut force = Vec3::ZERO;
    let (mut ta, mut tb) = (Vec3::ZERO, Vec3::ZERO);
    crunchy::unroll! { for k in 0..4 { if k < h.len as usize {
        let p = points.read_point(k);
        let normal = h.dir_a * p.normal_impulse;
        force += normal;
        ta += p.r_a.cross(normal);
        tb += p.r_b.cross(normal);
    } }}
    force += tangent;
    ta += f.center_a.cross(tangent) + twist;
    tb += (f.center_a + points.offset_b()).cross(tangent) + twist;
    a.linear += h.im_a * force;
    a.angular += h.ii_a.mul(ta);
    b.linear -= h.im_b * force;
    b.angular -= h.ii_b.mul(tb);
}
/// Transform the normal back into the frozen world frame once per manifold.
/// The two local anchors derive from r_a and r_a + offset_b, so their projected
/// separation needs just one dot product per point. No per-point anchors are stored.
/// The frozen Jacobians and Gauss-Seidel point order stay unchanged.
#[inline(always)]
fn solve_normals<P: TwistAccess>(
    h: &ContactSolverData,
    points: &mut P,
    poses: &Slice<Pose>,
    params: &RbdSimParams,
    a: &mut Velocity,
    b: &mut Velocity,
    use_bias: bool,
) {
    if h.len == 0 {
        return;
    }
    let pa = poses[h.solver_body_a as usize];
    let pb = poses[h.solver_body_b as usize];
    let (fa, fb) = points.frames();
    let na = fa * pa.inverse_transform_vector(h.dir_a);
    let nb = fb * pb.inverse_transform_vector(h.dir_a);
    let delta_n = na - nb;
    let separation = (pa.translation - pb.translation).dot(h.dir_a) - points.offset_b().dot(nb);
    let is_static = h.im_a == Vec3::ZERO || h.im_b == Vec3::ZERO;
    let (cfm, erp) = if is_static {
        (
            params.static_contact_cfm_factor(),
            params.static_contact_erp_inv_dt(),
        )
    } else {
        (params.contact_cfm_factor(), params.contact_erp_inv_dt())
    };
    let inv_dt = params.inv_dt();
    let max_corr = params.max_corrective_velocity();
    crunchy::unroll! { for k in 0..4 { if k < h.len as usize {
        let p = points.read_point(k);
        let dist = p.dist + (separation + p.r_a.dot(delta_n));
        let rhs = p.normal_vel + dist.max(0.0) * inv_dt;
        let (rhs, softness) = if use_bias {
            (rhs + (dist * erp).max(-max_corr).min(0.0), if dist <= 0.0 { cfm } else { 1.0 })
        } else {
            (rhs, 1.0)
        };
        let ta = p.r_a.cross(h.dir_a);
        let tb = p.r_b.cross(-h.dir_a);
        let dvel = h.dir_a.dot(a.linear) + ta.dot(a.angular)
            - h.dir_a.dot(b.linear) + tb.dot(b.angular) + rhs;
        let impulse = softness * (p.normal_impulse - p.normal_mass * dvel).max(0.0);
        let delta = impulse - p.normal_impulse;
        points.write_normal_impulse(k, impulse);
        a.linear += h.dir_a * h.im_a * delta;
        a.angular += h.ii_a.mul(ta) * delta;
        b.linear += h.dir_a * h.im_b * -delta;
        b.angular += h.ii_b.mul(tb) * delta;
    } }}
}
#[inline(always)]
pub(crate) fn solve<P: TwistAccess>(
    h: &ContactSolverData,
    points: &mut P,
    poses: &Slice<Pose>,
    params: &RbdSimParams,
    a: &mut Velocity,
    b: &mut Velocity,
    use_bias: bool,
    solve_friction: bool,
) {
    solve_normals(h, points, poses, params, a, b, use_bias);
    if !solve_friction || h.len == 0 {
        return;
    }
    let mut rhs = Vec2::ZERO;
    if use_bias {
        let pa = poses[h.solver_body_a as usize];
        let pb = poses[h.solver_body_b as usize];
        let (fa, fb) = points.frames();
        let center = points.friction().center_a;
        let drift = (pa * (fa.inverse() * center)
            - pb * (fb.inverse() * (center + points.offset_b())))
            * params.inv_dt();
        rhs = Vec2::new(
            drift.dot(h.tangent_a),
            drift.dot(h.dir_a.cross(h.tangent_a)),
        );
    }
    solve_friction_rows(h, points, a, b, rhs);
}
#[inline(always)]
pub(crate) fn solve_friction_rows<P: TwistAccess>(
    h: &ContactSolverData,
    points: &mut P,
    a: &mut Velocity,
    b: &mut Velocity,
    rhs: Vec2,
) {
    if h.len == 0 {
        return;
    }
    let f = points.friction();
    let (ra, rb) = (f.center_a, f.center_a + points.offset_b());
    let (mut tangent_limit, mut twist_limit) = (0.0, 0.0);
    crunchy::unroll! { for k in 0..4 { if k < h.len as usize {
        let p = points.read_point(k);
        tangent_limit += p.normal_impulse;
        twist_limit += p.normal_impulse * points.radius(k);
    } }}
    tangent_limit *= h.limit;
    twist_limit *= h.limit;
    let mut twist = 0.0;
    if h.len > 1 {
        let dvel = h.dir_a.dot(a.angular - b.angular);
        twist = (f.twist_impulse - f.twist_mass * dvel)
            .max(-twist_limit)
            .min(twist_limit);
        let delta = twist - f.twist_impulse;
        a.angular += h.ii_a.mul(h.dir_a) * delta;
        b.angular -= h.ii_b.mul(h.dir_a) * delta;
    }
    let t0 = h.tangent_a;
    let t1 = h.dir_a.cross(t0);
    let a0 = ra.cross(t0);
    let a1 = ra.cross(t1);
    let b0 = rb.cross(-t0);
    let b1 = rb.cross(-t1);
    let d0 = t0.dot(a.linear) + a0.dot(a.angular) - t0.dot(b.linear) + b0.dot(b.angular) + rhs.x;
    let d1 = t1.dot(a.linear) + a1.dot(a.angular) - t1.dot(b.linear) + b1.dot(b.angular) + rhs.y;
    let k00 = f.tangent_k.x;
    let k11 = f.tangent_k.y;
    let k01 = f.tangent_k.z * 0.5;
    let inv_det = maybe_inv(k00 * k11 - k01 * k01);
    let delta = Vec2::new(
        (k11 * d0 - k01 * d1) * inv_det,
        (k00 * d1 - k01 * d0) * inv_det,
    );
    let tangent = cap_magnitude(f.tangent_impulse - delta, tangent_limit);
    let delta = tangent - f.tangent_impulse;
    let force = t0 * delta.x + t1 * delta.y;
    a.linear += h.im_a * force;
    a.angular += h.ii_a.mul(a0 * delta.x + a1 * delta.y);
    b.linear -= h.im_b * force;
    b.angular += h.ii_b.mul(b0 * delta.x + b1 * delta.y);
    points.write_friction(tangent, twist);
}

#[cfg(all(test, not(target_arch_is_gpu)))]
#[path = "../tests/twist_friction.rs"]
mod tests;
