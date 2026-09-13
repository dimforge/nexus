//! Regression oracle: the pre-optimization point loop.
use super::*;
use crate::dynamics::MAX_CONSTRAINTS_PER_MANIFOLD;

fn reference_solve(
    c: &mut TwoBodyConstraint,
    poses: &Slice<Pose>,
    params: &RbdSimParams,
    solver_vel1: &mut Velocity,
    solver_vel2: &mut Velocity,
    use_bias: bool,
    solve_friction: bool,
) {
    let dir_a = c.dir_a;
    let im_a = c.im_a;
    let im_b = c.im_b;

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
    let pose1 = poses[c.solver_body_a as usize];
    let pose2 = poses[c.solver_body_b as usize];

    // Solve the normal parts of the constraint.
    for k in 0..(c.len as usize) {
        let point = c.points.at(k);
        let p1 = pose1 * point.local_pt_a;
        let p2 = pose2 * point.local_pt_b;
        let dist = point.dist + (p1 - p2).dot(dir_a);
        let rhs_wo_bias = point.normal_vel + dist.max(0.0) * inv_dt;
        let (rhs, cfm_factor) = if use_bias {
            // Not `clamp`: its `min <= max` assertion exits the kernel early, which breaks
            // the uniform control flow of the fused sweeps' barriers on the web.
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

        c.points.at_mut(k).normal_impulse = new_impulse;

        solver_vel1.linear += dir_a * im_a * delta_impulse;
        solver_vel1.angular += c.ii_a_mul(torque_dir_a) * delta_impulse;

        solver_vel2.linear += dir_a * im_b * -delta_impulse;
        solver_vel2.angular += c.ii_b_mul(torque_dir_b) * delta_impulse;
    }

    // Friction is solved during the stabilization sweep, and during the
    // biased pass only when `friction_in_bias_pass` is set.
    if !solve_friction {
        return;
    }

    let friction_coeff = c.limit;
    let tangents = c.tangents();

    // Solve the tangent parts of the constraint.
    for k in 0..(c.len as usize) {
        let point = c.points.at(k);
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

            c.points.at_mut(k).tangent_impulse = [new_impulse];

            solver_vel1.linear += t * im_a * delta_impulse;
            solver_vel1.angular += c.ii_a_mul(torque_dir_a) * delta_impulse;

            solver_vel2.linear += t * im_b * -delta_impulse;
            solver_vel2.angular += c.ii_b_mul(torque_dir_b) * delta_impulse;
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
            c.points.at_mut(k).tangent_impulse = new_impulse;

            let lin = t0 * delta_impulse.x + t1 * delta_impulse.y;
            solver_vel1.linear += lin * im_a;
            solver_vel1.angular +=
                c.ii_a_mul(torque_dir_a0 * delta_impulse.x + torque_dir_a1 * delta_impulse.y);

            solver_vel2.linear -= lin * im_b;
            solver_vel2.angular +=
                c.ii_b_mul(torque_dir_b0 * delta_impulse.x + torque_dir_b1 * delta_impulse.y);
        }
    }
}

#[test]
fn optimized_iterations_match_original_point_order() {
    let poses = [Pose::IDENTITY, Pose::from_translation(Vector::splat(0.25))];
    let poses = Slice(&poses, 0);
    let params = RbdSimParams::tgs_soft();
    for len in 0..=MAX_CONSTRAINTS_PER_MANIFOLD {
        for static_side in 0..3 {
            for locked in [false, true] {
                for use_bias in [false, true] {
                    for solve_friction in [false, true] {
                        let mut c = TwoBodyConstraint::default();
                        c.len = len as u32;
                        c.dir_a = Vector::splat(1.0).normalize();
                        #[cfg(feature = "dim3")]
                        {
                            c.tangent_a = compute_tangent_contact_directions(c.dir_a)[0];
                            c.ii_a = super::super::constraint::SymInertia {
                                diag: Vec3::new(0.7, 0.8, 0.9),
                                off: Vec3::splat(0.05),
                                ..Default::default()
                            };
                            c.ii_b = c.ii_a;
                        }
                        #[cfg(feature = "dim2")]
                        {
                            c.ii_a = 0.7;
                            c.ii_b = 0.8;
                        }
                        c.im_a = if static_side == 1 {
                            Vector::ZERO
                        } else {
                            Vector::splat(0.7)
                        };
                        c.im_b = if static_side == 2 {
                            Vector::ZERO
                        } else {
                            Vector::splat(0.4)
                        };
                        if locked {
                            c.im_a.x = 0.0;
                            c.im_b.y = 0.0;
                        }
                        c.solver_body_a = 0;
                        c.solver_body_b = 1;
                        c.limit = 0.6;
                        for k in 0..len {
                            let p = &mut c.points[k];
                            p.r_a = Vector::splat(0.1 * (k + 1) as f32);
                            p.r_b = -p.r_a;
                            p.local_pt_a = p.r_a;
                            p.local_pt_b = p.r_b;
                            p.dist = -0.1 + k as f32 * 0.05;
                            p.normal_vel = -0.03 * k as f32;
                            p.normal_impulse = 0.1 * k as f32;
                            #[cfg(feature = "dim3")]
                            {
                                p.tangent_impulse = Vec2::new(0.03, -0.01);
                            }
                            #[cfg(feature = "dim2")]
                            {
                                p.tangent_impulse = [0.03];
                            }
                        }
                        c.compute_effective_masses();
                        let mut expected = c;
                        let mut a = Velocity::default();
                        let mut b = Velocity::default();
                        a.linear = Vector::splat(0.3);
                        b.linear = Vector::splat(-0.2);
                        let (mut expected_a, mut expected_b) = (a, b);
                        reference_solve(
                            &mut expected,
                            &poses,
                            &params,
                            &mut expected_a,
                            &mut expected_b,
                            use_bias,
                            solve_friction,
                        );
                        super::super::coulomb_tiles::assert_tile_equivalent(
                            c,
                            &poses,
                            &params,
                            a,
                            b,
                            use_bias,
                            solve_friction,
                            &expected,
                            expected_a,
                            expected_b,
                        );
                        c.solve_constraint_gauss_seidel(
                            &poses,
                            &params,
                            &mut a,
                            &mut b,
                            use_bias,
                            solve_friction,
                        );
                        assert_eq!(a.linear, expected_a.linear);
                        assert_eq!(a.angular, expected_a.angular);
                        assert_eq!(b.linear, expected_b.linear);
                        assert_eq!(b.angular, expected_b.angular);
                        assert_eq!(bytemuck::bytes_of(&c), bytemuck::bytes_of(&expected));
                    }
                }
            }
        }
    }
}
