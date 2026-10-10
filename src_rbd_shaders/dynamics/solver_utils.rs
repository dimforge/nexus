//! Main physics solver functions (PGS/Sequential Impulse)
//!
//! This module contains the core physics solver implementation using iterative
//! constraint-based methods, using the `Soft-TGS` approach (as in Rapier).

use super::body::{Velocity, WorldMassProperties};
use super::constraint::{
    ContactLink, ContactPoint, MAX_CONSTRAINTS_PER_MANIFOLD, SUB_LEN, TwoBodyConstraint,
};
use super::sim_params::RbdSimParams;
use crate::{Pose, Vector, gcross, gcross_av, gdot};
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim3")]
use glamx::{Vec2, Vec3};

use crate::queries::IndexedManifold;
use crate::utils::Slice;

/// Helper function for maybe inverse with threshold.
#[cfg(feature = "dim3")]
pub(crate) fn maybe_inv(a: f32) -> f32 {
    const INV_EPSILON: f32 = 1.0e-20;
    if a < -INV_EPSILON || a > INV_EPSILON {
        1.0 / a
    } else {
        0.0
    }
}

/// Cap the magnitude of a 2D vector.
#[cfg(feature = "dim3")]
pub(crate) fn cap_magnitude(v: Vec2, limit: f32) -> Vec2 {
    let n = v.length();
    if n > limit { v * (limit / n) } else { v }
}

#[cfg(feature = "dim3")]
/// Computes an orthonormal vector perpendicular to the input (3D).
fn orthonormal_vector(vec: Vector) -> Vector {
    let sign = if vec.z == 0.0 {
        1.0
    } else if vec.z < 0.0 {
        -1.0
    } else {
        1.0
    };
    let a = -1.0 / (sign + vec.z);
    let b = vec.x * vec.y * a;
    Vec3::new(b, sign + vec.y * vec.y * a, -vec.y)
}

#[cfg(feature = "dim3")]
/// Computes the tangent contact directions for friction (3D version).
///
/// A deterministic basis derived from the contact normal.
pub(crate) fn compute_tangent_contact_directions(force_dir1: Vector) -> [Vector; SUB_LEN] {
    let tangent1 = orthonormal_vector(force_dir1);
    let bitangent1 = force_dir1.cross(tangent1);
    [tangent1, bitangent1]
}

// Constant indices let the shader compiler scalarize each manifold point.
// Preserve Gauss-Seidel point order; only the loop control changes.
macro_rules! for_contact_point {
    ($k:ident, $len:expr, $body:block) => {{
        let len = $len as usize;
        {
            let $k = 0usize;
            if $k < len {
                $body
            }
        }
        {
            let $k = 1usize;
            if $k < len {
                $body
            }
        }
        #[cfg(feature = "dim3")]
        {
            let $k = 2usize;
            if $k < len {
                $body
            }
        }
        #[cfg(feature = "dim3")]
        {
            let $k = 3usize;
            if $k < len {
                $body
            }
        }
    }};
}

pub(crate) use for_contact_point;

impl IndexedManifold {
    /// Initializes the manifold identity and current mass/material data without rebuilding points.
    #[inline(always)]
    pub fn contact_to_constraint_header(
        &self,
        mprops: &Slice<WorldMassProperties>,
        constraint: &mut TwoBodyConstraint,
    ) {
        let bid1 = self.bodies.x;
        let bid2 = self.bodies.y;
        let mprops1 = &mprops[bid1 as usize];
        let mprops2 = &mprops[bid2 as usize];
        #[cfg(feature = "dim3")]
        {
            constraint.ii_a = super::constraint::SymInertia::from_mat4(mprops1.inv_inertia);
            constraint.ii_b = super::constraint::SymInertia::from_mat4(mprops2.inv_inertia);
        }
        #[cfg(feature = "dim2")]
        {
            constraint.ii_a = mprops1.inv_inertia;
            constraint.ii_b = mprops2.inv_inertia;
        }
        constraint.im_a = mprops1.inv_mass;
        constraint.im_b = mprops2.inv_mass;
        constraint.limit = self.friction;
        constraint.restitution = self.restitution;
        constraint.solver_body_a = bid1;
        constraint.solver_body_b = bid2;
        constraint.vel_slot_a = bid1;
        constraint.vel_slot_b = bid2;
        constraint.warmstart_collider_a = self.colliders.x;
        constraint.warmstart_collider_b = self.colliders.y;
        constraint.warmstart_subshape = self.subshape;
        constraint.len = self.contact.len;
    }

    /// Converts a contact manifold to a solver constraint.
    ///
    /// `collider_world_poses` are used to recover the world-space contact normal
    /// and contact point (the manifold expresses both in collider-local space).
    /// `solver_body_poses` (rapier's COM-centered solver pose) are used for the
    /// world center of mass and to express the contact anchors in COM-local space.
    #[inline(always)]
    pub fn contact_to_constraint(
        &self,
        mprops: &Slice<WorldMassProperties>,
        collider_world_poses: &Slice<Pose>,
        solver_body_poses: &Slice<Pose>,
        vels: &Slice<Velocity>,
        constraint: &mut TwoBodyConstraint,
    ) {
        self.contact_to_constraint_header(mprops, constraint);
        self.contact_to_constraint_points(
            collider_world_poses[self.colliders.x as usize],
            solver_body_poses[self.bodies.x as usize],
            solver_body_poses[self.bodies.y as usize],
            vels,
            constraint,
        );
    }

    /// Initializes the contact points of `constraint`, whose header was initialized from this
    /// manifold, and their effective masses.
    #[inline(always)]
    pub fn contact_to_constraint_points(
        &self,
        cpose1: Pose,
        spose1: Pose,
        spose2: Pose,
        vels: &Slice<Velocity>,
        constraint: &mut TwoBodyConstraint,
    ) {
        let contact = &self.contact;
        let bid1 = self.bodies.x;
        let bid2 = self.bodies.y;
        let vel1 = &vels[bid1 as usize];
        let vel2 = &vels[bid2 as usize];

        let force_dir1 = -(cpose1.rotation * contact.normal_a);

        // Combined friction / restitution of the two colliders, resolved at
        // narrow-phase time (rapier's `CoefficientCombineRule`) and stored in the
        // manifold.
        let restitution = self.restitution;

        constraint.dir_a = force_dir1;
        #[cfg(feature = "dim3")]
        {
            constraint.tangent_a = compute_tangent_contact_directions(force_dir1).read(0);
        }

        for k in 0..(contact.len as usize) {
            let pt = cpose1
                * (contact.points_a.at(k).pt
                    + contact.normal_a * contact.points_a.at(k).dist / 2.0);
            // `mprops.com` and `solver_body_pose.translation` are equal (both are
            // the world COM); use the latter to mirror rapier's solver convention.
            let dp1 = pt - spose1.translation;
            let dp2 = pt - spose2.translation;
            let contact_vel1 = vel1.linear + gcross_av(vel1.angular, dp1);
            let contact_vel2 = vel2.linear + gcross_av(vel2.angular, dp2);

            // TODO: handle is_bouncy?
            let point = constraint.points.at_mut(k);
            point.r_a = dp1;
            point.r_b = dp2;
            // Anchors are stored in the solver-body (COM-centered) frame, matching rapier:
            // the iterations recover the world point as `solver_body_pose * local_pt`.
            point.local_pt_a = spose1.inverse_transform_point(pt);
            point.local_pt_b = spose2.inverse_transform_point(pt);
            point.dist = contact.points_a.at(k).dist;
            point.normal_vel = restitution * (contact_vel1 - contact_vel2).dot(force_dir1);
            point.normal_impulse = 0.0;
            #[cfg(feature = "dim2")]
            {
                point.tangent_impulse = [0.0];
            }
            #[cfg(feature = "dim3")]
            {
                point.tangent_impulse = Vec2::ZERO;
            }
        }

        constraint.len = contact.len;
        constraint.compute_effective_masses();
    }
}

impl TwoBodyConstraint {
    /// Scales the accumulated impulses (the warmstart coefficient).
    #[inline(always)]
    pub fn scale_impulses(&mut self, coeff: f32) {
        for k in 0..self.len as usize {
            let point = self.points.at_mut(k);
            point.normal_impulse *= coeff;
            #[cfg(feature = "dim2")]
            {
                point.tangent_impulse = [point.tangent_impulse[0] * coeff];
            }
            #[cfg(feature = "dim3")]
            {
                point.tangent_impulse *= coeff;
            }
        }
    }

    /// The total accumulated impulse of the contact point `k`, in world space, as applied to
    /// body A (body B receives its opposite).
    #[inline(always)]
    fn point_impulse(&self, k: usize, tangents: &[Vector; SUB_LEN]) -> Vector {
        let point = self.points.at(k);
        #[cfg(feature = "dim2")]
        return self.dir_a * point.normal_impulse + tangents.read(0) * point.tangent_impulse[0];
        #[cfg(feature = "dim3")]
        return self.dir_a * point.normal_impulse
            + tangents.read(0) * point.tangent_impulse.x
            + tangents.read(1) * point.tangent_impulse.y;
    }
}

impl TwoBodyConstraint {
    /// Applies warmstart impulses to a constraint (scatter-style, requires graph coloring).
    #[inline(always)]
    pub fn warmstart_constraint(&self, solver_vel1: &mut Velocity, solver_vel2: &mut Velocity) {
        let tangents = self.tangents();
        let mut force = Vector::ZERO;
        #[cfg(feature = "dim2")]
        let (mut torque_a, mut torque_b) = (0.0, 0.0);
        #[cfg(feature = "dim3")]
        let (mut torque_a, mut torque_b) = (Vec3::ZERO, Vec3::ZERO);

        for_contact_point!(k, self.len, {
            let f = self.point_impulse(k, &tangents);
            force += f;
            torque_a += gcross(self.points.at(k).r_a, f);
            torque_b += gcross(self.points.at(k).r_b, f);
        });

        solver_vel1.linear += self.im_a * force;
        solver_vel1.angular += self.ii_a_mul(torque_a);
        solver_vel2.linear -= self.im_b * force;
        solver_vel2.angular -= self.ii_b_mul(torque_b);
    }

    /// Main constraint solver iteration (Projected Gauss-Seidel). `solve_friction`
    /// gates the tangent rows: the stabilization iteration always solves them, the
    /// biased pass only when `RbdSimParams::friction_in_bias_pass` is set.
    ///
    /// The normal right-hand sides are computed here from the current solver poses (rather
    /// than stored by a separate pass): with bias for the biased iteration, without for the
    /// relaxation iteration, which runs after the positions are integrated.
    #[inline(always)]
    pub fn solve_constraint_gauss_seidel(
        &mut self,
        poses: &Slice<Pose>,
        params: &RbdSimParams,
        solver_vel1: &mut Velocity,
        solver_vel2: &mut Velocity,
        use_bias: bool,
        solve_friction: bool,
    ) {
        let data = ContactSolverData::from_constraint(self);
        data.solve_points(
            &mut self.points,
            poses,
            params,
            solver_vel1,
            solver_vel2,
            use_bias,
            solve_friction,
        );
    }
}

#[cfg(feature = "dim3")]
type SolverInertia = super::constraint::SymInertia;
#[cfg(feature = "dim2")]
type SolverInertia = f32;

/// Contact geometry shared by all point rows, cached independently of mutable impulses.
#[derive(Clone, Copy)]
pub(crate) struct ContactSolverData {
    pub dir_a: Vector,
    pub len: u32,
    #[cfg(feature = "dim3")]
    pub tangent_a: Vector,
    pub limit: f32,
    pub im_a: Vector,
    pub solver_body_a: u32,
    pub im_b: Vector,
    pub solver_body_b: u32,
    pub ii_a: SolverInertia,
    pub ii_b: SolverInertia,
    pub vel_slot_a: u32,
    pub vel_slot_b: u32,
}

#[cfg(feature = "dim3")]
type TangentImpulse = Vec2;
#[cfg(feature = "dim2")]
type TangentImpulse = [f32; 1];
pub(crate) trait ContactPointAccess {
    fn read_point(&self, k: usize) -> ContactPoint;
    fn write_normal_impulse(&mut self, k: usize, value: f32);
    fn write_tangent_impulse(&mut self, k: usize, value: TangentImpulse);
}
impl ContactPointAccess for [ContactPoint; MAX_CONSTRAINTS_PER_MANIFOLD] {
    #[inline(always)]
    fn read_point(&self, k: usize) -> ContactPoint {
        *self.at(k)
    }
    #[inline(always)]
    fn write_normal_impulse(&mut self, k: usize, value: f32) {
        self.at_mut(k).normal_impulse = value;
    }
    #[inline(always)]
    fn write_tangent_impulse(&mut self, k: usize, value: TangentImpulse) {
        self.at_mut(k).tangent_impulse = value;
    }
}

impl ContactSolverData {
    #[inline(always)]
    pub(crate) fn from_constraint(c: &TwoBodyConstraint) -> Self {
        Self {
            dir_a: c.dir_a,
            len: c.len,
            #[cfg(feature = "dim3")]
            tangent_a: c.tangent_a,
            limit: c.limit,
            im_a: c.im_a,
            solver_body_a: c.solver_body_a,
            im_b: c.im_b,
            solver_body_b: c.solver_body_b,
            ii_a: c.ii_a,
            ii_b: c.ii_b,
            vel_slot_a: c.vel_slot_a,
            vel_slot_b: c.vel_slot_b,
        }
    }
    #[inline(always)]
    pub(crate) fn ii_a_mul(&self, v: crate::AngVector) -> crate::AngVector {
        #[cfg(feature = "dim2")]
        return self.ii_a * v;
        #[cfg(feature = "dim3")]
        return self.ii_a.mul(v);
    }
    #[inline(always)]
    pub(crate) fn ii_b_mul(&self, v: crate::AngVector) -> crate::AngVector {
        #[cfg(feature = "dim2")]
        return self.ii_b * v;
        #[cfg(feature = "dim3")]
        return self.ii_b.mul(v);
    }
    #[inline(always)]
    pub(crate) fn tangents(&self) -> [Vector; SUB_LEN] {
        #[cfg(feature = "dim2")]
        return [Vector::new(-self.dir_a.y, self.dir_a.x)];
        #[cfg(feature = "dim3")]
        return [self.tangent_a, self.dir_a.cross(self.tangent_a)];
    }

    #[inline(always)]
    pub fn solve_points<P: ContactPointAccess>(
        &self,
        points: &mut P,
        poses: &Slice<Pose>,
        params: &RbdSimParams,
        solver_vel1: &mut Velocity,
        solver_vel2: &mut Velocity,
        use_bias: bool,
        solve_friction: bool,
    ) {
        let dir_a = self.dir_a;
        let im_a = self.im_a;
        let im_b = self.im_b;

        let is_static = im_a == Vector::ZERO || im_b == Vector::ZERO;
        let (cfm_factor, erp_inv_dt) = if is_static {
            (
                params.static_contact_cfm_factor(),
                params.static_contact_erp_inv_dt(),
            )
        } else {
            (params.contact_cfm_factor(), params.contact_erp_inv_dt())
        };
        let inv_dt = params.inv_dt();
        let max_corr_velocity = params.max_corrective_velocity();
        let pose1 = poses[self.solver_body_a as usize];
        let pose2 = poses[self.solver_body_b as usize];

        // Solve the normal parts of the constraint.
        for_contact_point!(k, self.len, {
            let point = points.read_point(k);
            let p1 = pose1 * point.local_pt_a;
            let p2 = pose2 * point.local_pt_b;
            let dist = point.dist + (p1 - p2).dot(dir_a);
            let rhs_wo_bias = point.normal_vel + dist.max(0.0) * inv_dt;
            let (rhs, cfm_factor) = if use_bias {
                // Not `clamp`: its `min <= max` assertion exits the kernel early, which breaks
                // the uniform control flow of the fused iterations' barriers on the web.
                let rhs_bias = (dist * erp_inv_dt).max(-max_corr_velocity).min(0.0);
                // Separated (speculative) points are solved rigidly.
                let cfm = if dist <= 0.0 { cfm_factor } else { 1.0 };
                (rhs_wo_bias + rhs_bias, cfm)
            } else {
                (rhs_wo_bias, 1.0)
            };

            let torque_dir_a = gcross(point.r_a, dir_a);
            let torque_dir_b = gcross(point.r_b, -dir_a);
            let dvel = dir_a.dot(solver_vel1.linear) + gdot(torque_dir_a, solver_vel1.angular)
                - dir_a.dot(solver_vel2.linear)
                + gdot(torque_dir_b, solver_vel2.angular)
                + rhs;
            let impulse = point.normal_impulse;
            let new_impulse = cfm_factor * (impulse - point.normal_mass * dvel).max(0.0);
            let delta_impulse = new_impulse - impulse;

            points.write_normal_impulse(k, new_impulse);

            solver_vel1.linear += dir_a * im_a * delta_impulse;
            solver_vel1.angular += self.ii_a_mul(torque_dir_a) * delta_impulse;

            solver_vel2.linear += dir_a * im_b * -delta_impulse;
            solver_vel2.angular += self.ii_b_mul(torque_dir_b) * delta_impulse;
        });

        // Friction is solved during the stabilization iteration, and during the
        // biased pass only when `friction_in_bias_pass` is set.
        if !solve_friction {
            return;
        }

        let friction_coeff = self.limit;
        let tangents = self.tangents();

        // Solve the tangent parts of the constraint.
        for_contact_point!(k, self.len, {
            let point = points.read_point(k);
            let limit = friction_coeff * point.normal_impulse;

            #[cfg(feature = "dim2")]
            {
                let t = tangents.read(0);
                let torque_dir_a = gcross(point.r_a, t);
                let torque_dir_b = gcross(point.r_b, -t);
                let dvel = t.dot(solver_vel1.linear) + gdot(torque_dir_a, solver_vel1.angular)
                    - t.dot(solver_vel2.linear)
                    + gdot(torque_dir_b, solver_vel2.angular);
                let impulse = point.tangent_impulse[0];
                // NOTE: don’t use clamp since it can panic.
                let new_impulse = (impulse - point.tangent_mass * dvel).max(-limit).min(limit);
                let delta_impulse = new_impulse - impulse;

                points.write_tangent_impulse(k, [new_impulse]);

                solver_vel1.linear += t * im_a * delta_impulse;
                solver_vel1.angular += self.ii_a_mul(torque_dir_a) * delta_impulse;

                solver_vel2.linear += t * im_b * -delta_impulse;
                solver_vel2.angular += self.ii_b_mul(torque_dir_b) * delta_impulse;
            }
            #[cfg(feature = "dim3")]
            {
                let t0 = tangents.read(0);
                let t1 = tangents.read(1);
                let torque_dir_a0 = gcross(point.r_a, t0);
                let torque_dir_b0 = gcross(point.r_b, -t0);
                let torque_dir_a1 = gcross(point.r_a, t1);
                let torque_dir_b1 = gcross(point.r_b, -t1);
                let dvel_0 = t0.dot(solver_vel1.linear) + gdot(torque_dir_a0, solver_vel1.angular)
                    - t0.dot(solver_vel2.linear)
                    + gdot(torque_dir_b0, solver_vel2.angular);
                let dvel_1 = t1.dot(solver_vel1.linear) + gdot(torque_dir_a1, solver_vel1.angular)
                    - t1.dot(solver_vel2.linear)
                    + gdot(torque_dir_b1, solver_vel2.angular);

                let k11 = point.tangent_k[0];
                let k22 = point.tangent_k[1];
                let k12 = point.tangent_k[2] * 0.5;
                let inv_det = maybe_inv(k11 * k22 - k12 * k12);
                let delta_impulse = Vec2::new(
                    (k22 * dvel_0 - k12 * dvel_1) * inv_det,
                    (k11 * dvel_1 - k12 * dvel_0) * inv_det,
                );
                let impulse = point.tangent_impulse;
                let new_impulse = cap_magnitude(impulse - delta_impulse, limit);
                let delta_impulse = new_impulse - impulse;
                points.write_tangent_impulse(k, new_impulse);

                let lin = t0 * delta_impulse.x + t1 * delta_impulse.y;
                solver_vel1.linear += lin * im_a;
                solver_vel1.angular += self
                    .ii_a_mul(torque_dir_a0 * delta_impulse.x + torque_dir_a1 * delta_impulse.y);

                solver_vel2.linear -= lin * im_b;
                solver_vel2.angular += self
                    .ii_b_mul(torque_dir_b0 * delta_impulse.x + torque_dir_b1 * delta_impulse.y);
            }
        });
    }
}

#[cfg(all(test, not(target_arch_is_gpu)))]
#[path = "../tests/contact_solver.rs"]
mod tests;

impl TwoBodyConstraint {
    #[inline(always)]
    pub fn init_header(&mut self, manifold: &IndexedManifold, mprops: &Slice<WorldMassProperties>) {
        manifold.contact_to_constraint_header(mprops, self);
    }
    /// Initializes the contact points of a constraint whose header was initialized from
    /// `manifold`.
    #[inline(always)]
    pub fn init_points_with_poses(
        &mut self,
        manifold: &IndexedManifold,
        collider_a: Pose,
        solver_a: Pose,
        solver_b: Pose,
        vels: &Slice<Velocity>,
    ) {
        manifold.contact_to_constraint_points(collider_a, solver_a, solver_b, vels, self);
    }
    /// Applies the velocity slots and mass scales chosen by mass splitting.
    #[inline(always)]
    pub fn apply_link(&mut self, link: &ContactLink) {
        self.vel_slot_a = link.vel_slot_a;
        self.vel_slot_b = link.vel_slot_b;
        self.restitution = link.restitution;
        self.im_a *= link.mass_scale_a;
        self.im_b *= link.mass_scale_b;
        #[cfg(feature = "dim2")]
        {
            self.ii_a *= link.mass_scale_a;
            self.ii_b *= link.mass_scale_b;
        }
        #[cfg(feature = "dim3")]
        {
            self.ii_a.scale(link.mass_scale_a);
            self.ii_b.scale(link.mass_scale_b);
        }
    }
    #[inline(always)]
    pub fn recycle_from(&mut self, old: &Self, vel1: &Velocity, vel2: &Velocity) {
        self.dir_a = old.dir_a;
        #[cfg(feature = "dim3")]
        {
            self.tangent_a = old.tangent_a;
        }
        self.len = old.len;
        for k in 0..old.len as usize {
            let mut point = *old.points.at(k);
            let v1 = vel1.linear + gcross_av(vel1.angular, point.r_a);
            let v2 = vel2.linear + gcross_av(vel2.angular, point.r_b);
            point.normal_vel = self.restitution * (v1 - v2).dot(old.dir_a);
            self.points.write(k, point);
        }
        self.compute_effective_masses();
    }
    #[inline(always)]
    pub fn local_anchors(&self, k: usize) -> (Vector, Vector) {
        let p = self.points.at(k);
        (p.local_pt_a, p.local_pt_b)
    }
    // Pass only the friction impulse through the matching loop, rather than
    // copying the entire old manifold for every matched point.
    #[cfg(feature = "dim3")]
    #[inline(always)]
    pub fn friction_warmstart(&self, k: usize) -> glamx::Vec4 {
        let t = self.points.at(k).tangent_impulse;
        (self.tangent_a * t.x + self.dir_a.cross(self.tangent_a) * t.y).extend(0.0)
    }
    #[cfg(feature = "dim2")]
    #[inline(always)]
    pub fn friction_warmstart(&self, k: usize) -> f32 {
        self.points.at(k).tangent_impulse[0]
    }
    #[cfg(feature = "dim3")]
    #[inline(always)]
    pub fn transfer_friction_point(&mut self, impulse: glamx::Vec4, k: usize) {
        let world = impulse.truncate();
        self.points.at_mut(k).tangent_impulse = Vec2::new(
            world.dot(self.tangent_a),
            world.dot(self.dir_a.cross(self.tangent_a)),
        );
    }
    #[cfg(feature = "dim2")]
    #[inline(always)]
    pub fn transfer_friction_point(&mut self, impulse: f32, k: usize) {
        self.points.at_mut(k).tangent_impulse = [impulse];
    }
    #[inline(always)]
    pub fn finish_friction_transfer(&mut self) {}
}
