use super::*;
use crate::dynamics::{FrictionModel, RbdSimParams};
use crate::{Pose, utils::Slice};

fn patch(len: usize) -> TwistConstraint {
    let mut c = TwistConstraint::default();
    c.len = len as u32;
    c.dir_a = Vec3::Y;
    c.tangent_a = Vec3::X;
    c.im_a = Vec3::ONE;
    c.ii_a.diag = Vec3::ONE;
    c.solver_body_b = 1;
    c.limit = 0.5;
    for (i, x) in [-1.0, 1.0, -1.0, 1.0].into_iter().enumerate().take(len) {
        let p = &mut c.points[i];
        p.r_a = Vec3::new(x, -1.0, if i < 2 { -1.0 } else { 1.0 });
        p.normal_impulse = 1.0;
    }
    c.compute_effective_masses();
    c
}

#[test]
fn twist_radius_and_load_bound_the_angular_impulse() {
    // Four unit loads at radius sqrt(2), mu=0.5: max torque impulse=2sqrt(2).
    let mut c = patch(4);
    let mut a = Velocity::default();
    let mut b = Velocity::default();
    a.angular = Vec3::Y * 10.0;
    solve_friction_rows(&c.solver_data(), &mut c, &mut a, &mut b, Vec2::ZERO);
    let limit = 2.0 * 2.0f32.sqrt();
    assert!((c.friction.twist_impulse + limit).abs() < 1e-6);
    assert!((a.angular.y - (10.0 - limit)).abs() < 1e-6);
    assert_eq!(a.linear, Vec3::ZERO);
    assert_eq!(b.angular, Vec3::ZERO);
    // With an attainable target, the twist row cancels relative spin exactly.
    let mut c = patch(4);
    a.angular = Vec3::Y;
    solve_friction_rows(&c.solver_data(), &mut c, &mut a, &mut b, Vec2::ZERO);
    assert_eq!(a.angular, Vec3::ZERO);
    assert_eq!(c.friction.twist_impulse, -1.0);
}

#[test]
fn one_point_zero_load_and_zero_friction_do_not_add_twist_resistance() {
    for (len, mu, load) in [(1, 0.5, 1.0), (4, 0.0, 1.0), (4, 0.5, 0.0)] {
        let mut c = patch(len);
        c.limit = mu;
        for p in &mut c.points[..len] {
            p.normal_impulse = load;
        }
        let mut a = Velocity::default();
        let mut b = Velocity::default();
        a.angular = Vec3::Y * 10.0;
        solve_friction_rows(&c.solver_data(), &mut c, &mut a, &mut b, Vec2::ZERO);
        assert_eq!(c.friction.twist_impulse, 0.0);
        // A single offset point can still resist spin through its sliding row.
        if len > 1 {
            assert_eq!(a.angular.y, 10.0);
        }
    }
}

#[test]
fn central_friction_uses_total_load_and_warmstarts_once() {
    let mut c = patch(4);
    let mut a = Velocity::default();
    let mut b = Velocity::default();
    a.linear = Vec3::X * 10.0;
    solve_friction_rows(&c.solver_data(), &mut c, &mut a, &mut b, Vec2::ZERO);
    // Central effective mass=1/2, capped by mu*four unit normal impulses=2.
    assert_eq!(c.friction.tangent_impulse, Vec2::new(-2.0, 0.0));
    assert_eq!(a.linear.x, 8.0);
    assert_eq!(a.angular.z, -2.0);
    c.scale_impulses(0.25);
    let mut warm = Velocity::default();
    c.warmstart_constraint(&mut warm, &mut b);
    assert_eq!(warm.linear, Vec3::new(-0.5, 1.0, 0.0));
    assert_eq!(warm.angular.z, -0.5);
    assert_eq!(c.friction.tangent_impulse.x, -0.5);
}

#[test]
fn surviving_contact_warmstarts_are_averaged_and_single_point_twist_is_cleared() {
    let mut c = patch(4);
    let mut old = c;
    old.friction.tangent_impulse = Vec2::splat(2.0);
    old.friction.twist_impulse = 2.0;
    // Three matched points and one fresh point: average over all four.
    for k in 0..3 {
        c.transfer_friction_point(old.friction_warmstart(k), k);
    }
    c.finish_friction_transfer();
    assert_eq!(c.friction.tangent_impulse, Vec2::splat(1.5));
    assert_eq!(c.friction.twist_impulse, 1.5);
    c.len = 1;
    c.compute_effective_masses();
    assert_eq!(c.friction.twist_impulse, 0.0);
    assert_eq!(c.points[0].radius, 0.0);
}

#[test]
fn twist_tiles_and_cached_warmstarts_match_twist_constraints() {
    let poses_array = [Pose::IDENTITY; 2];
    let poses = Slice(&poses_array, 0);
    let params = RbdSimParams::tgs_soft();
    for len in 0..=4 {
        for locked in [false, true] {
            for biased in [false, true] {
                for friction in [false, true] {
                    let mut initial = patch(len);
                    initial.frame_a = Quat::from_rotation_x(0.13);
                    initial.frame_b = Quat::from_rotation_y(-0.08);
                    initial.im_b = Vec3::splat(0.4);
                    initial.ii_b.diag = Vec3::splat(0.3);
                    initial.ii_a.off = Vec3::splat(0.03);
                    if locked {
                        initial.im_a.x = 0.0;
                        initial.ii_a.diag = Vec3::ZERO;
                        initial.ii_a.off = Vec3::ZERO;
                    }
                    initial.compute_effective_masses();
                    let mut a = Velocity::default();
                    let mut b = Velocity::default();
                    a.linear = Vec3::new(0.2, -0.5, 0.4);
                    a.angular = Vec3::new(0.3, 0.6, 0.1);
                    b.angular = Vec3::splat(-0.2);
                    let (ia, ib) = (a, b);
                    let mut expected = initial;
                    expected.solve_constraint_gauss_seidel(
                        &poses, &params, &mut a, &mut b, biased, friction,
                    );
                    crate::dynamics::twist_tiles::assert_tile_equivalent(
                        initial, &poses, &params, ia, ib, biased, friction, &expected, a, b,
                    );
                }
            }
        }
    }
}

#[test]
fn layouts_and_default_remain_compatible() {
    assert_eq!(core::mem::size_of::<RbdSimParams>(), 76);
    assert_eq!(
        core::mem::size_of::<crate::dynamics::constraint::ContactPoint>(),
        96
    );
    assert_eq!(core::mem::size_of::<TwistConstraint>(), 384);
    assert_eq!(
        core::mem::size_of::<crate::dynamics::TwoBodyConstraint>(),
        544
    );
    assert_eq!(core::mem::size_of::<TwistContactPoint>(), 32);
    // 21 16-byte solver fields per constraint, and the 4 of its cached warmstart.
    assert_eq!(
        core::mem::size_of::<crate::dynamics::twist_tiles::TwistTile>(),
        (21 + 4) * 16 * crate::dynamics::contact_tiles::TILE_LEN
    );
    assert_eq!(
        RbdSimParams::default().friction_model,
        FrictionModel::Coulomb
    );
}

#[test]
fn twist_is_solved_before_the_off_center_sliding_row() {
    let mut c = patch(2);
    c.limit = 10.0;
    c.points[0].r_a = Vec3::new(0.0, -1.0, 0.0);
    c.points[1].r_a = Vec3::new(2.0, -1.0, 0.0);
    c.compute_effective_masses();
    let mut a = Velocity::default();
    let mut b = Velocity::default();
    a.angular = Vec3::Y;
    solve_friction_rows(&c.solver_data(), &mut c, &mut a, &mut b, Vec2::ZERO);
    assert_eq!(a.angular, Vec3::ZERO);
    assert_eq!(a.linear, Vec3::ZERO);
    assert_eq!(c.friction.tangent_impulse, Vec2::ZERO);
    assert_eq!(c.friction.twist_impulse, -1.0);
}

#[test]
fn twist_effective_mass_uses_both_bodies() {
    let mut c = patch(4);
    c.limit = 10.0;
    c.im_b = Vec3::ONE;
    c.ii_b.diag = Vec3::splat(3.0);
    c.compute_effective_masses();
    let mut a = Velocity::default();
    let mut b = Velocity::default();
    a.angular = Vec3::Y * 5.0;
    b.angular = Vec3::Y;
    solve_friction_rows(&c.solver_data(), &mut c, &mut a, &mut b, Vec2::ZERO);
    assert_eq!(a.angular, Vec3::Y * 4.0);
    assert_eq!(b.angular, a.angular);
    assert_eq!(c.friction.twist_impulse, -1.0);
}

#[test]
fn central_drift_bias_is_used_only_in_the_biased_friction_pass() {
    let poses = [Pose::IDENTITY, Pose::from_translation(Vec3::X * 0.1)];
    for biased in [false, true] {
        let mut c = patch(4);
        let mut a = Velocity::default();
        let mut b = Velocity::default();
        c.solve_constraint_gauss_seidel(
            &Slice(&poses, 0),
            &RbdSimParams::tgs_soft(),
            &mut a,
            &mut b,
            biased,
            true,
        );
        if biased {
            let limit = c.limit * c.points.iter().map(|p| p.normal_impulse).sum::<f32>();
            assert!(c.friction.tangent_impulse.x > 1.5);
            assert!((c.friction.tangent_impulse.length() - limit).abs() < 1e-6);
        } else {
            assert_eq!(c.friction.tangent_impulse, Vec2::ZERO);
        }
    }
}

#[test]
fn twist_constraints_recycle_or_partially_transfer_previous_contacts() {
    use crate::broad_phase::ContactPlan;
    use crate::dynamics::twist_tiles::{self, TwistTile};
    use crate::dynamics::{
        ContactLink, ContactRecycleOffsets, ContactRecycleState, gpu_init_contact_links,
        gpu_match_contact_links,
    };
    use crate::queries::IndexedManifold;
    use glamx::{UVec2, UVec3};
    let params = RbdSimParams::tgs_soft();
    let poses = [Pose::IDENTITY; 2];
    let velocities = [Velocity::default(); 2];
    let mut props = [WorldMassProperties::default(); 2];
    props[0].inv_mass = Vec3::ONE;
    props[0].inv_inertia = glamx::Mat4::IDENTITY;
    let mut manifold = IndexedManifold::default();
    manifold.bodies = UVec2::new(0, 1);
    manifold.colliders = UVec2::new(0, 1);
    manifold.contact.len = 4;
    manifold.contact.normal_a = -Vec3::Y;
    manifold.friction = 0.5;
    manifold.recycle_extent = 1.0;
    for k in 0..4 {
        manifold.contact.points_a[k].pt = Vec3::new((k % 2) as f32, 0.0, (k / 2) as f32);
    }
    let mut old = TwistConstraint::default();
    old.init_contact(
        &manifold,
        &Slice(&props, 0),
        &Slice(&poses, 0),
        &Slice(&poses, 0),
        &Slice(&velocities, 0),
    );
    for k in 0..4 {
        old.points[k].normal_impulse = (k + 1) as f32;
    }
    old.friction.tangent_impulse = Vec2::new(0.4, 0.2);
    old.friction.twist_impulse = 0.3;
    let mut old_tile = TwistTile::default();
    old_tile.write_constraint(0, &old);
    let mut old_links = [ContactLink::default()];
    gpu_init_contact_links(
        UVec3::ZERO,
        &[manifold],
        &mut old_links,
        &[0, 0],
        &ContactPlan {
            bound: 1,
            ..Default::default()
        },
    );
    let mut changed = manifold;
    changed.contact.points_a[3].pt += Vec3::X * 0.3;
    for recycle in [false, true] {
        let state = ContactRecycleState {
            pose_a: Pose::IDENTITY,
            pose_b: Pose::IDENTITY,
            colliders: manifold.colliders,
            max_extent: 1.0,
            max_drift: if recycle { 0.05 } else { 0.0 },
        };
        let plan = ContactPlan {
            bound: 1,
            ..Default::default()
        };
        let offsets = ContactRecycleOffsets {
            old_base: 0,
            new_base: 1,
            ..Default::default()
        };
        let mut states = [state; 2];
        let mut links = [ContactLink::default()];
        gpu_init_contact_links(UVec3::ZERO, &[changed], &mut links, &[0, 0], &plan);
        gpu_match_contact_links(
            UVec3::ZERO,
            &[changed],
            &mut links,
            &[1, 1],
            &[0],
            &old_links,
            &[0],
            &mut states,
            &poses,
            &plan,
            &params,
            &offsets,
        );
        assert_eq!(links[0].previous_constraint, 0);
        assert_eq!(links[0].recycled, u32::from(recycle));
        // The link of contact 0 is its own sorted link.
        let mut tiles = [TwistTile::default()];
        twist_tiles::gpu_prepare_constraints(
            UVec3::ZERO,
            &links,
            &[changed],
            &props,
            &poses,
            &velocities,
            &states,
            &[old_tile],
            &mut tiles,
            &plan,
            &offsets,
        );
        let new = tiles[0].constraint(0);
        let factor = if recycle { 1.0 } else { 0.75 };
        assert!((new.friction.twist_impulse - old.friction.twist_impulse * factor).abs() < 1e-6);
        assert!(
            (new.friction.tangent_impulse - old.friction.tangent_impulse * factor).length() < 1e-6
        );
        assert_eq!(
            new.points[3].normal_impulse,
            if recycle { 4.0 } else { 0.0 }
        );
        let (mut a, mut b) = (Velocity::default(), Velocity::default());
        new.warmstart_constraint(&mut a, &mut b);
        let [wa, wb] = tiles[0].warmstart.bodies[0];
        assert_eq!((wa.linear, wa.angular), (a.linear, a.angular));
        assert_eq!((wb.linear, wb.angular), (b.linear, b.angular));
    }
}

#[test]
fn projected_normals_match_world_anchor_reference() {
    // Compare against the unchanged world-anchor normal solver, including rotated
    // bodies, anisotropic/locked masses, restitution, and speculative contacts.
    for len in 0..=4 {
        for use_bias in [false, true] {
            for origin in [Vec3::ZERO, Vec3::new(42.5, -13.0, 20.0)] {
                for static_a in [false, true] {
                    let mut poses_array = [
                        Pose {
                            translation: origin,
                            rotation: glamx::Quat::from_scaled_axis(Vec3::new(0.2, -0.3, 0.1)),
                            ..Pose::IDENTITY
                        },
                        Pose {
                            translation: origin + Vec3::new(0.3, -0.7, 0.2),
                            rotation: glamx::Quat::from_scaled_axis(Vec3::new(-0.4, 0.1, 0.3)),
                            ..Pose::IDENTITY
                        },
                    ];
                    let mut c = patch(len);
                    c.dir_a = Vec3::new(0.2, 0.9, -0.3).normalize();
                    c.im_a = if static_a {
                        Vec3::ZERO
                    } else {
                        Vec3::new(0.0, 0.8, 1.2)
                    };
                    c.im_b = Vec3::new(0.3, 0.7, 0.2);
                    c.ii_a.diag = if static_a {
                        Vec3::ZERO
                    } else {
                        Vec3::new(0.6, 0.4, 0.8)
                    };
                    c.ii_b.diag = Vec3::new(0.4, 0.7, 0.9);
                    c.ii_b.off = Vec3::new(0.03, -0.04, 0.01);
                    c.offset_b = poses_array[0].translation - poses_array[1].translation;
                    c.frame_a = poses_array[0].rotation;
                    c.frame_b = poses_array[1].rotation;
                    for (k, p) in c.points.iter_mut().enumerate().take(len) {
                        p.r_a = poses_array[0].rotation * p.r_a;
                        p.normal_vel = [-0.1, 0.3, 0.0, -0.2][k];
                        p.dist = [-0.1, 0.08, -0.04, 0.2][k];
                    }
                    c.compute_effective_masses();
                    poses_array[0].rotation =
                        glamx::Quat::from_rotation_x(0.03) * poses_array[0].rotation;
                    poses_array[1].translation += Vec3::Y * 0.002;
                    let h = c.solver_data();
                    let mut reference = c;
                    let mut a = Velocity::default();
                    let mut b = Velocity::default();
                    a.linear = Vec3::new(0.2, -2.0, 0.4);
                    a.angular = Vec3::new(0.3, 0.6, 0.1);
                    b.angular = Vec3::splat(-0.2);
                    let (mut ra, mut rb) = (a, b);
                    let poses = Slice(&poses_array, 0);
                    let params = RbdSimParams::tgs_soft();
                    h.solve_points(
                        &mut reference,
                        &poses,
                        &params,
                        &mut ra,
                        &mut rb,
                        use_bias,
                        false,
                    );
                    solve_normals(&h, &mut c, &poses, &params, &mut a, &mut b, use_bias);
                    for (actual, expected) in [
                        (a.linear, ra.linear),
                        (a.angular, ra.angular),
                        (b.linear, rb.linear),
                        (b.angular, rb.angular),
                    ] {
                        assert!(
                            (actual - expected).length() < 5e-4,
                            "{actual:?} != {expected:?}"
                        );
                    }
                    for k in 0..len {
                        assert!(
                            (c.points[k].normal_impulse - reference.points[k].normal_impulse).abs()
                                < 5e-4
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn frozen_frames_survive_recycling_and_preserve_local_anchors() {
    let mut old = patch(4);
    old.frame_a = Quat::from_scaled_axis(Vec3::new(0.2, -0.4, 0.1));
    old.frame_b = Quat::from_scaled_axis(Vec3::new(-0.1, 0.3, 0.5));
    old.offset_b = Vec3::new(0.4, 0.7, -0.2);
    old.compute_effective_masses();
    let mut new = old;
    new.frame_a = Quat::from_rotation_y(0.7);
    new.frame_b = Quat::from_rotation_x(-0.4);
    new.offset_b += Vec3::splat(0.01);
    new.recycle_from(&old, &Velocity::default(), &Velocity::default());
    assert_eq!(new.frame_a, old.frame_a);
    assert_eq!(new.frame_b, old.frame_b);
    assert_eq!(new.offset_b, old.offset_b);
    for k in 0..4 {
        assert_eq!(new.local_anchors(k), old.local_anchors(k));
        let (a, b) = new.local_anchors(k);
        assert!((new.frame_a * a - new.points[k].r_a).length() < 1e-6);
        assert!((new.frame_b * b - new.points[k].r_a - new.offset_b).length() < 1e-6);
    }
}

#[test]
fn central_drift_matches_averaging_rotated_world_anchors() {
    for len in 1..=4 {
        let mut c = patch(len);
        c.frame_a = Quat::from_scaled_axis(Vec3::new(0.2, -0.4, 0.1));
        c.frame_b = Quat::from_scaled_axis(Vec3::new(-0.1, 0.3, 0.5));
        c.offset_b = Vec3::new(0.4, 0.7, -0.2);
        c.compute_effective_masses();
        let poses_array = [
            Pose::from_parts(Vec3::new(0.2, 0.3, 0.4), Quat::from_rotation_x(0.3)),
            Pose::from_parts(Vec3::new(0.4, 0.6, 0.2), Quat::from_rotation_y(-0.1)),
        ];
        let poses = Slice(&poses_array, 0);
        let params = RbdSimParams::tgs_soft();
        let h = c.solver_data();
        let mut reference = c;
        let (mut a, mut b) = (Velocity::default(), Velocity::default());
        let (mut ra, mut rb) = (a, b);
        // Independent old arithmetic: transform and sum every local anchor.
        h.solve_points(
            &mut reference,
            &poses,
            &params,
            &mut ra,
            &mut rb,
            true,
            false,
        );
        let mut drift = Vec3::ZERO;
        for k in 0..len {
            let (anchor_a, anchor_b) = reference.local_anchors(k);
            drift += poses_array[0] * anchor_a - poses_array[1] * anchor_b;
        }
        drift *= params.inv_dt() / len as f32;
        let rhs = Vec2::new(
            drift.dot(h.tangent_a),
            drift.dot(h.dir_a.cross(h.tangent_a)),
        );
        solve_friction_rows(&h, &mut reference, &mut ra, &mut rb, rhs);
        solve(&h, &mut c, &poses, &params, &mut a, &mut b, true, true);
        for (actual, expected) in [
            (a.linear, ra.linear),
            (a.angular, ra.angular),
            (b.linear, rb.linear),
            (b.angular, rb.angular),
        ] {
            assert!(
                (actual - expected).length() < 1e-4,
                "{actual:?} != {expected:?}"
            );
        }
        assert!((c.friction.tangent_impulse - reference.friction.tangent_impulse).length() < 1e-4);
    }
}
