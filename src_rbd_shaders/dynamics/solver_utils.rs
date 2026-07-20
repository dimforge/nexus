//! Main physics solver functions (PGS/Sequential Impulse)
//!
//! This module contains the core physics solver implementation using iterative
//! constraint-based methods, using the `Soft-TGS` approach (as in Rapier).

use super::body::{Velocity, WorldMassProperties};
use super::constraint::{
    SUB_LEN, TwoBodyConstraint, TwoBodyConstraintBuilder, TwoBodyConstraintNormalPart,
    TwoBodyConstraintTangentPart,
};
use super::sim_params::RbdSimParams;
use crate::{Pad, Pose, Vector, gcross, gdot};
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::{Vec2, Vec3};

use crate::queries::IndexedManifold;
use crate::utils::Slice;

/// Helper function for safe inverse.
fn inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

/// Helper function for maybe inverse with threshold.
#[cfg(feature = "dim3")]
fn maybe_inv(a: f32) -> f32 {
    const INV_EPSILON: f32 = 1.0e-20;
    if a < -INV_EPSILON || a > INV_EPSILON {
        1.0 / a
    } else {
        0.0
    }
}

/// Cap the magnitude of a 2D vector.
#[cfg(feature = "dim3")]
fn cap_magnitude(v: Vec2, limit: f32) -> Vec2 {
    let n = v.length();
    if n > limit { v * (limit / n) } else { v }
}

#[cfg(feature = "dim2")]
/// Computes an orthonormal vector perpendicular to the input (2D).
fn orthonormal_vector(vec: Vector) -> Vector {
    Vec2::new(-vec.y, vec.x)
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

#[cfg(feature = "dim2")]
/// Computes the tangent contact directions for friction (2D version).
fn compute_tangent_contact_directions(
    force_dir1: Vector,
    _linvel1: Vector,
    _linvel2: Vector,
) -> [Vector; SUB_LEN] {
    [orthonormal_vector(force_dir1)]
}

#[cfg(feature = "dim3")]
/// Computes the tangent contact directions for friction (3D version).
fn compute_tangent_contact_directions(
    force_dir1: Vector,
    linvel1: Vector,
    linvel2: Vector,
) -> [Vector; SUB_LEN] {
    // Compute the tangent direction. Pick the direction of
    // the linear relative velocity, if it is not too small.
    // Otherwise use a fallback direction.
    let relative_linvel = linvel1 - linvel2;
    let mut tangent_relative_linvel =
        relative_linvel - force_dir1 * force_dir1.dot(relative_linvel);

    let tangent_linvel_norm = tangent_relative_linvel.length();
    tangent_relative_linvel /= tangent_linvel_norm;

    const THRESHOLD: f32 = 1.0e-4;
    let use_fallback = tangent_linvel_norm < THRESHOLD;
    let tangent_fallback = orthonormal_vector(force_dir1);

    let tangent1 = if use_fallback {
        tangent_fallback
    } else {
        tangent_relative_linvel
    };
    let bitangent1 = force_dir1.cross(tangent1);

    [tangent1, bitangent1]
}

#[cfg(feature = "dim2")]
/// Cross product for angular velocity (2D).
fn gcross_(a: f32, b: Vector) -> Vector {
    a * b
}

#[cfg(feature = "dim3")]
/// Cross product for angular velocity (3D).
fn gcross_(a: Vector, b: Vector) -> Vector {
    a.cross(b)
}

impl IndexedManifold {
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
        params: &RbdSimParams,
        constraint: &mut TwoBodyConstraint,
        builder: &mut TwoBodyConstraintBuilder,
    ) {
        let id1 = self.colliders.x;
        let contact = &self.contact;
        let bid1 = self.bodies.x;
        let bid2 = self.bodies.y;

        let mprops1 = &mprops[bid1 as usize];
        let mprops2 = &mprops[bid2 as usize];
        // Contact features (`points_a`, `normal_a`) are stored in collider A's
        // local space, so only `cpose1` is needed to recover their world-space
        // forms; collider B's pose isn't read here.
        let cpose1 = collider_world_poses[id1 as usize];
        let spose1 = solver_body_poses[bid1 as usize];
        let spose2 = solver_body_poses[bid2 as usize];
        let vel1 = &vels[bid1 as usize];
        let vel2 = &vels[bid2 as usize];

        let force_dir1 = -(cpose1.rotation * contact.normal_a);

        let cfm_factor = params.contact_cfm_factor();
        let inv_dt_val = params.inv_dt();
        let erp_inv_dt = params.contact_erp_inv_dt();
        let allowed_lin_err = params.allowed_linear_error();
        let max_corr_velocity = params.max_corrective_velocity();

        // Combined friction / restitution of the two colliders, resolved at
        // narrow-phase time (rapier's `CoefficientCombineRule`) and stored in the
        // manifold.
        let friction = self.friction;
        let restitution = self.restitution;

        let tangents1 = compute_tangent_contact_directions(force_dir1, vel1.linear, vel2.linear);
        constraint.dir_a = force_dir1;
        constraint.im_a = mprops1.inv_mass;
        constraint.im_b = mprops2.inv_mass;
        constraint.cfm_factor = cfm_factor;
        constraint.limit = friction;
        constraint.solver_body_a = bid1;
        constraint.solver_body_b = bid2;

        #[cfg(feature = "dim3")]
        {
            constraint.tangent_a = tangents1.read(0);
        }

        for k in 0..(contact.len as usize) {
            let pt = cpose1
                * (contact.points_a.at(k).pt
                    + contact.normal_a * contact.points_a.at(k).dist / 2.0);
            // `mprops.com` and `solver_body_pose.translation` are equal (both are
            // the world COM); use the latter to mirror rapier's solver convention.
            let dp1 = pt - spose1.translation;
            let dp2 = pt - spose2.translation;
            let contact_vel1 = vel1.linear + gcross_(vel1.angular, dp1);
            let contact_vel2 = vel2.linear + gcross_(vel2.angular, dp2);

            //
            // Normal part:
            //
            let torque_dir1 = gcross(dp1, force_dir1);
            let torque_dir2 = gcross(dp2, -force_dir1);

            let ii_torque_dir1 = mprops1.inv_inertia_mul(torque_dir1);
            let ii_torque_dir2 = mprops2.inv_inertia_mul(torque_dir2);

            let imsum = mprops1.inv_mass + mprops2.inv_mass;
            let projected_mass = inv(force_dir1.dot(imsum * force_dir1)
                + gdot(ii_torque_dir1, torque_dir1)
                + gdot(ii_torque_dir2, torque_dir2));

            // TODO: handle is_bouncy?
            let dist = contact.points_a.at(k).dist;
            let normal_rhs_wo_bias = restitution * (contact_vel1 - contact_vel2).dot(force_dir1)
                + dist.max(0.0) * inv_dt_val;

            let rhs_bias = (erp_inv_dt * (dist + allowed_lin_err)).clamp(-max_corr_velocity, 0.0);

            constraint.elements.at_mut(k).normal_part = TwoBodyConstraintNormalPart {
                torque_dir_a: torque_dir1,
                ii_torque_dir_a: ii_torque_dir1,
                torque_dir_b: torque_dir2,
                ii_torque_dir_b: ii_torque_dir2,
                rhs: normal_rhs_wo_bias + rhs_bias,
                rhs_wo_bias: normal_rhs_wo_bias,
                impulse: 0.0,
                r: projected_mass,
                #[cfg(feature = "dim3")]
                _padding0: 0.0,
                #[cfg(feature = "dim3")]
                _padding1: 0.0,
                #[cfg(feature = "dim3")]
                _padding2: 0.0,
                #[cfg(feature = "dim3")]
                _padding3: 0.0,
            };

            //
            // Tangent part:
            //
            #[cfg(feature = "dim2")]
            {
                let t_torque_dir1 = gcross(dp1, tangents1.read(0));
                let t_torque_dir2 = gcross(dp2, -tangents1.read(0));
                let t_ii_torque_dir1 = mprops1.inv_inertia * t_torque_dir1;
                let t_ii_torque_dir2 = mprops2.inv_inertia * t_torque_dir2;
                let r = tangents1.read(0).dot(imsum * tangents1.read(0))
                    + gdot(t_ii_torque_dir1, t_torque_dir1)
                    + gdot(t_ii_torque_dir2, t_torque_dir2);
                let rhs_wo_bias = 0.0;

                constraint.elements.at_mut(k).tangent_part = TwoBodyConstraintTangentPart {
                    torque_dir_a: [Pad::new(t_torque_dir1)],
                    ii_torque_dir_a: [Pad::new(t_ii_torque_dir1)],
                    torque_dir_b: [Pad::new(t_torque_dir2)],
                    ii_torque_dir_b: [Pad::new(t_ii_torque_dir2)],
                    rhs: [rhs_wo_bias],
                    rhs_wo_bias: [rhs_wo_bias],
                    impulse: [0.0],
                    r: [inv(r)],
                };
            }

            #[cfg(feature = "dim3")]
            {
                let mut tangent_part = TwoBodyConstraintTangentPart::default();

                for j in 0..SUB_LEN {
                    let t_torque_dir1 = gcross(dp1, tangents1.read(j));
                    let t_torque_dir2 = gcross(dp2, -tangents1.read(j));
                    let t_ii_torque_dir1 = mprops1.inv_inertia_mul(t_torque_dir1);
                    let t_ii_torque_dir2 = mprops2.inv_inertia_mul(t_torque_dir2);
                    let r = tangents1.read(j).dot(imsum * tangents1.read(j))
                        + gdot(t_ii_torque_dir1, t_torque_dir1)
                        + gdot(t_ii_torque_dir2, t_torque_dir2);
                    let rhs_wo_bias = 0.0;

                    tangent_part.torque_dir_a.write(j, Pad::new(t_torque_dir1));
                    tangent_part
                        .ii_torque_dir_a
                        .write(j, Pad::new(t_ii_torque_dir1));
                    tangent_part.torque_dir_b.write(j, Pad::new(t_torque_dir2));
                    tangent_part
                        .ii_torque_dir_b
                        .write(j, Pad::new(t_ii_torque_dir2));
                    tangent_part.rhs.write(j, rhs_wo_bias);
                    tangent_part.rhs_wo_bias.write(j, rhs_wo_bias);
                    tangent_part.r.write(j, r);
                }

                tangent_part.impulse = Vec2::ZERO;

                // Compute r[2] for 3D (cross term)
                tangent_part.r.write(
                    2,
                    2.0 * ((**tangent_part.torque_dir_a.at(0))
                        .dot(**tangent_part.ii_torque_dir_a.at(1))
                        + (**tangent_part.torque_dir_b.at(0))
                            .dot(**tangent_part.ii_torque_dir_b.at(1))),
                );

                constraint.elements.at_mut(k).tangent_part = tangent_part;
            }

            // Builder info for warmstarting. Anchors are stored in the solver-body
            // (COM-centered) frame, matching rapier — `update_constraint` then
            // recovers the world point as `solver_body_pose * local_pt`.
            builder.infos.at_mut(k).local_pt_a = spose1.inverse_transform_point(pt);
            builder.infos.at_mut(k).local_pt_b = spose2.inverse_transform_point(pt);
            builder.infos.at_mut(k).dist = dist;
            builder.infos.at_mut(k).normal_vel = normal_rhs_wo_bias;
        }

        constraint.len = contact.len;
    }
}

impl TwoBodyConstraint {
    /// Updates constraint coefficients for a new substep.
    #[inline(always)]
    pub fn update_constraint(
        &mut self,
        builder: &TwoBodyConstraintBuilder,
        poses: &Slice<Pose>,
        params: &RbdSimParams,
    ) {
        let body1 = self.solver_body_a as usize;
        let body2 = self.solver_body_b as usize;

        let cfm_factor = params.contact_cfm_factor();
        let inv_dt_val = params.inv_dt();
        let allowed_lin_err = params.allowed_linear_error();
        let erp_inv_dt = params.contact_erp_inv_dt();
        let max_corr_velocity = params.max_corrective_velocity();
        let warmstart_coeff = params.warmstart_coefficient;

        let pose1 = poses[body1];
        let pose2 = poses[body2];
        let num_contacts = self.len as usize;

        #[cfg(feature = "dim2")]
        let tangents1 = Vec2::new(-self.dir_a.y, self.dir_a.x);

        #[cfg(feature = "dim3")]
        let tangents1 = [self.tangent_a, self.dir_a.cross(self.tangent_a)];

        for j in 0..num_contacts {
            // NOTE: the tangent velocity is equivalent to an additional movement of the first body's surface.
            let info = builder.infos.at(j);
            let p1 = pose1 * info.local_pt_a; // TODO (conveyor belts): + info.tangent_vel * solved_dt;
            let p2 = pose2 * info.local_pt_b;
            let dist = info.dist + (p1 - p2).dot(self.dir_a);

            // Normal part.
            {
                let rhs_wo_bias = info.normal_vel + dist.max(0.0) * inv_dt_val;
                let rhs_bias =
                    ((dist + allowed_lin_err) * erp_inv_dt).clamp(-max_corr_velocity, 0.0);
                let new_rhs = rhs_wo_bias + rhs_bias;

                self.elements.at_mut(j).normal_part.rhs_wo_bias = rhs_wo_bias;
                self.elements.at_mut(j).normal_part.rhs = new_rhs;
                self.elements.at_mut(j).normal_part.impulse *= warmstart_coeff;
            }

            // Tangent parts.
            {
                #[cfg(feature = "dim2")]
                {
                    let impulse_val =
                        self.elements.at(j).tangent_part.impulse.read(0) * warmstart_coeff;
                    let rhs_wo_bias_val = self.elements.at(j).tangent_part.rhs_wo_bias.read(0);
                    self.elements
                        .at_mut(j)
                        .tangent_part
                        .impulse
                        .write(0, impulse_val);
                    let bias = (p1 - p2).dot(tangents1) * inv_dt_val;
                    self.elements
                        .at_mut(j)
                        .tangent_part
                        .rhs
                        .write(0, rhs_wo_bias_val + bias);
                }
                #[cfg(feature = "dim3")]
                {
                    self.elements.at_mut(j).tangent_part.impulse *= warmstart_coeff;
                    let rhs_wo_bias_0 = self.elements.at(j).tangent_part.rhs_wo_bias.read(0);
                    let rhs_wo_bias_1 = self.elements.at(j).tangent_part.rhs_wo_bias.read(1);
                    let bias0 = (p1 - p2).dot(tangents1.read(0)) * inv_dt_val;
                    self.elements
                        .at_mut(j)
                        .tangent_part
                        .rhs
                        .write(0, rhs_wo_bias_0 + bias0);
                    let bias1 = (p1 - p2).dot(tangents1.read(1)) * inv_dt_val;
                    self.elements
                        .at_mut(j)
                        .tangent_part
                        .rhs
                        .write(1, rhs_wo_bias_1 + bias1);
                }
            }
        }

        self.cfm_factor = cfm_factor;
    }
}

/// Applies warmstart impulses to a single body (gather-style, no graph coloring required).
#[inline(always)]
pub fn warmstart_body(
    body_id: u32,
    body_constraint_counts: &Slice<u32>,
    body_constraint_ids: &Slice<u32>,
    constraints: &Slice<TwoBodyConstraint>,
    solver_vel: &mut Velocity,
) {
    let first_constraint_id = if body_id != 0 {
        body_constraint_counts[body_id as usize - 1] as usize
    } else {
        0
    };
    let last_constraint_id = body_constraint_counts[body_id as usize] as usize;

    for i in first_constraint_id..last_constraint_id {
        let cid = body_constraint_ids[i] as usize;
        let constraint = &constraints[cid];
        let solver_body_1 = constraint.solver_body_a;
        let dir_a = constraint.dir_a;
        let im_a = constraint.im_a;
        let im_b = constraint.im_b;
        let len = constraint.len as usize;

        #[cfg(feature = "dim2")]
        let tangent_a = Vec2::new(-dir_a.y, dir_a.x);
        #[cfg(feature = "dim3")]
        let tangent_a = constraint.tangent_a;

        for k in 0..len {
            // Warmstart the normal part of the constraint.
            {
                let c = &constraint.elements.at(k).normal_part;
                if solver_body_1 == body_id {
                    solver_vel.linear += dir_a * im_a * c.impulse;
                    solver_vel.angular += c.ii_torque_dir_a * c.impulse;
                } else {
                    solver_vel.linear += dir_a * im_b * -c.impulse;
                    solver_vel.angular += c.ii_torque_dir_b * c.impulse;
                }
            }

            // Warmstart the tangent parts of the constraint.
            {
                let c = &constraint.elements.at(k).tangent_part;

                #[cfg(feature = "dim2")]
                {
                    if solver_body_1 == body_id {
                        solver_vel.linear += tangent_a * im_a * c.impulse.read(0);
                        solver_vel.angular += c.ii_torque_dir_a.at(0).0 * c.impulse.read(0);
                    } else {
                        solver_vel.linear += tangent_a * im_b * -c.impulse.read(0);
                        solver_vel.angular += c.ii_torque_dir_b.at(0).0 * c.impulse.read(0);
                    }
                }
                #[cfg(feature = "dim3")]
                {
                    let tangents_a = [tangent_a, dir_a.cross(tangent_a)];
                    if solver_body_1 == body_id {
                        solver_vel.linear += (tangents_a.read(0) * c.impulse.x
                            + tangents_a.read(1) * c.impulse.y)
                            * im_a;
                        solver_vel.angular += **c.ii_torque_dir_a.at(0) * c.impulse.x
                            + **c.ii_torque_dir_a.at(1) * c.impulse.y;
                    } else {
                        solver_vel.linear += (tangents_a.read(0) * -c.impulse.x
                            + tangents_a.read(1) * -c.impulse.y)
                            * im_b;
                        solver_vel.angular += **c.ii_torque_dir_b.at(0) * c.impulse.x
                            + **c.ii_torque_dir_b.at(1) * c.impulse.y;
                    }
                }
            }
        }
    }
}

impl TwoBodyConstraint {
    /// Applies warmstart impulses to a constraint (scatter-style, requires graph coloring).
    pub fn warmstart_constraint(&self, solver_vel1: &mut Velocity, solver_vel2: &mut Velocity) {
        let dir_a = self.dir_a;
        let im_a = self.im_a;
        let im_b = self.im_b;

        #[cfg(feature = "dim2")]
        let tangent_a = Vec2::new(-dir_a.y, dir_a.x);
        #[cfg(feature = "dim3")]
        let tangent_a = self.tangent_a;

        for k in 0..(self.len as usize) {
            // Warmstart the normal part of the constraint.
            {
                let c = &self.elements.at(k).normal_part;
                solver_vel1.linear += dir_a * im_a * c.impulse;
                solver_vel1.angular += c.ii_torque_dir_a * c.impulse;
                solver_vel2.linear += dir_a * im_b * -c.impulse;
                solver_vel2.angular += c.ii_torque_dir_b * c.impulse;
            }

            // Warmstart the tangent parts of the constraint.
            {
                let c = &self.elements.at(k).tangent_part;

                #[cfg(feature = "dim2")]
                {
                    solver_vel1.linear += tangent_a * im_a * c.impulse.read(0);
                    solver_vel1.angular += **c.ii_torque_dir_a.at(0) * c.impulse.read(0);
                    solver_vel2.linear += tangent_a * im_b * -c.impulse.read(0);
                    solver_vel2.angular += **c.ii_torque_dir_b.at(0) * c.impulse.read(0);
                }
                #[cfg(feature = "dim3")]
                {
                    let tangents_a = [tangent_a, dir_a.cross(tangent_a)];
                    solver_vel1.linear += (tangents_a.read(0) * c.impulse.x
                        + tangents_a.read(1) * c.impulse.y)
                        * im_a;
                    solver_vel1.angular += **c.ii_torque_dir_a.at(0) * c.impulse.x
                        + **c.ii_torque_dir_a.at(1) * c.impulse.y;
                    solver_vel2.linear += (tangents_a.read(0) * -c.impulse.x
                        + tangents_a.read(1) * -c.impulse.y)
                        * im_b;
                    solver_vel2.angular += **c.ii_torque_dir_b.at(0) * c.impulse.x
                        + **c.ii_torque_dir_b.at(1) * c.impulse.y;
                }
            }
        }
    }

    /// Main constraint solver iteration (Projected Gauss-Seidel).
    #[inline(always)]
    pub fn solve_constraint_gauss_seidel(
        &mut self,
        solver_vel1: &mut Velocity,
        solver_vel2: &mut Velocity,
        use_bias: bool,
    ) {
        let dir_a = self.dir_a;
        let friction_coeff = self.limit;
        let im_a = self.im_a;
        let im_b = self.im_b;
        // The stabilization (no-bias) sweep uses the bias-free rhs and no CFM.
        // This replaces the former `remove_cfm_and_bias` full-buffer pass.
        let cfm_factor = if use_bias { self.cfm_factor } else { 1.0 };

        #[cfg(feature = "dim2")]
        let tangent_a = Vec2::new(-dir_a.y, dir_a.x);
        #[cfg(feature = "dim3")]
        let tangent_a = self.tangent_a;

        for k in 0..(self.len as usize) {
            // Solve the normal part of the constraint.
            let limit = {
                let c = &self.elements.at(k).normal_part;
                // Copy values we need after the assignment
                let ii_torque_dir_a = c.ii_torque_dir_a;
                let ii_torque_dir_b = c.ii_torque_dir_b;

                let rhs = if use_bias { c.rhs } else { c.rhs_wo_bias };
                let dvel = dir_a.dot(solver_vel1.linear)
                    + gdot(c.torque_dir_a, solver_vel1.angular)
                    - dir_a.dot(solver_vel2.linear)
                    + gdot(c.torque_dir_b, solver_vel2.angular)
                    + rhs;
                let new_impulse = cfm_factor * (c.impulse - c.r * dvel).max(0.0);
                let delta_impulse = new_impulse - c.impulse;

                self.elements.at_mut(k).normal_part.impulse = new_impulse;

                solver_vel1.linear += dir_a * im_a * delta_impulse;
                solver_vel1.angular += ii_torque_dir_a * delta_impulse;

                solver_vel2.linear += dir_a * im_b * -delta_impulse;
                solver_vel2.angular += ii_torque_dir_b * delta_impulse;
                new_impulse * friction_coeff // Friction impulse limit.
            };

            // Solve the tangent parts of the constraint.
            {
                let c = &self.elements.at(k).tangent_part;
                // Copy values we need after the assignment
                let ii_torque_dir_a = c.ii_torque_dir_a;
                let ii_torque_dir_b = c.ii_torque_dir_b;

                #[cfg(feature = "dim2")]
                {
                    let dvel = tangent_a.dot(solver_vel1.linear)
                        + gdot(**c.torque_dir_a.at(0), solver_vel1.angular)
                        - tangent_a.dot(solver_vel2.linear)
                        + gdot(**c.torque_dir_b.at(0), solver_vel2.angular)
                        + c.rhs.read(0);
                    let new_impulse =
                        cfm_factor * (c.impulse.read(0) - c.r.read(0) * dvel).clamp(-limit, limit);
                    let delta_impulse = new_impulse - c.impulse.read(0);

                    self.elements
                        .at_mut(k)
                        .tangent_part
                        .impulse
                        .write(0, new_impulse);

                    solver_vel1.linear += tangent_a * im_a * delta_impulse;
                    solver_vel1.angular += **ii_torque_dir_a.at(0) * delta_impulse;

                    solver_vel2.linear += tangent_a * im_b * -delta_impulse;
                    solver_vel2.angular += **ii_torque_dir_b.at(0) * delta_impulse;
                }
                #[cfg(feature = "dim3")]
                {
                    let tangents_a = [tangent_a, dir_a.cross(tangent_a)];
                    let dvel_0 = tangents_a.read(0).dot(solver_vel1.linear)
                        + gdot(**c.torque_dir_a.at(0), solver_vel1.angular)
                        - tangents_a.read(0).dot(solver_vel2.linear)
                        + gdot(**c.torque_dir_b.at(0), solver_vel2.angular)
                        + c.rhs.read(0);
                    let dvel_1 = tangents_a.read(1).dot(solver_vel1.linear)
                        + gdot(**c.torque_dir_a.at(1), solver_vel1.angular)
                        - tangents_a.read(1).dot(solver_vel2.linear)
                        + gdot(**c.torque_dir_b.at(1), solver_vel2.angular)
                        + c.rhs.read(1);

                    let dvel_00 = dvel_0 * dvel_0;
                    let dvel_11 = dvel_1 * dvel_1;
                    let dvel_01 = dvel_0 * dvel_1;
                    let inv_lhs = (dvel_00 + dvel_11)
                        * maybe_inv(
                            dvel_00 * c.r.read(0) + dvel_11 * c.r.read(1) + dvel_01 * c.r.read(2),
                        );
                    let delta_impulse = Vec2::new(inv_lhs * dvel_0, inv_lhs * dvel_1);
                    let mut new_impulse = c.impulse - delta_impulse;
                    new_impulse = cap_magnitude(new_impulse, limit);

                    let delta_impulse = new_impulse - c.impulse;
                    self.elements.at_mut(k).tangent_part.impulse = new_impulse;

                    solver_vel1.linear += (tangents_a.read(0) * delta_impulse.x
                        + tangents_a.read(1) * delta_impulse.y)
                        * im_a;
                    solver_vel1.angular += **ii_torque_dir_a.at(0) * delta_impulse.x
                        + **ii_torque_dir_a.at(1) * delta_impulse.y;

                    solver_vel2.linear += (tangents_a.read(0) * -delta_impulse.x
                        + tangents_a.read(1) * -delta_impulse.y)
                        * im_b;
                    solver_vel2.angular += **ii_torque_dir_b.at(0) * delta_impulse.x
                        + **ii_torque_dir_b.at(1) * delta_impulse.y;
                }
            }
        }
    }
}

