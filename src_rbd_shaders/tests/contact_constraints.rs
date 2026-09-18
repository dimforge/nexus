//! The contact links' matching with the previous step, and the constraints built from them.
use super::*;
use crate::broad_phase::ContactPlan;
use crate::dynamics::{
    ContactLink, ContactRecycleOffsets, ContactRecycleState, MAX_CONSTRAINTS_PER_MANIFOLD,
    WorldMassProperties, gpu_init_contact_links, gpu_match_contact_links,
};
use crate::queries::{ContactManifold, IndexedManifold};
use glamx::{UVec2, UVec3};

fn plan(bound: u32) -> ContactPlan {
    ContactPlan {
        bound,
        ..Default::default()
    }
}

/// The links of `manifolds`, as initialized at the start of a step.
fn links_of(manifolds: &[IndexedManifold]) -> Vec<ContactLink> {
    let mut links = vec![ContactLink::default(); manifolds.len()];
    for i in 0..manifolds.len() {
        gpu_init_contact_links(
            UVec3::new(i as u32, 0, 0),
            manifolds,
            &mut links,
            &[0; 3],
            &plan(manifolds.len() as u32),
        );
    }
    links
}

#[test]
fn matching_distinguishes_subshapes_and_clears_stale_matches() {
    // Two manifolds of the same body/collider pair, in opposite adjacency order.
    let mut manifold = IndexedManifold::default();
    manifold.bodies = UVec2::new(1, 2);
    manifold.colliders = UVec2::new(1, 2);
    manifold.contact.len = 1;
    let mut old_manifolds = [manifold; 2];
    old_manifolds[1].subshape = 1;
    let old_links = links_of(&old_manifolds);
    let counts = [0, 2, 4];
    let ids = [1, 0, 0, 1];
    let offsets = ContactRecycleOffsets {
        old_base: 0,
        new_base: 2,
        ..Default::default()
    };
    let poses = [Pose::IDENTITY; 3];
    for subshape in 0..3 {
        let mut current = manifold;
        current.subshape = subshape;
        let mut links = links_of(&[current]);
        links[0].previous_constraint = 123;
        let mut states = [ContactRecycleState::default(); 4];
        gpu_match_contact_links(
            UVec3::ZERO,
            &[current],
            &mut links,
            &counts,
            &ids,
            &old_links,
            &[0, 1],
            &mut states,
            &poses,
            &plan(1),
            &RbdSimParams::tgs_soft(),
            &offsets,
        );
        assert_eq!(
            links[0].previous_constraint,
            if subshape < 2 { subshape } else { u32::MAX }
        );
        assert_eq!(
            links[0].previous_constraint_index,
            links[0].previous_constraint
        );
        assert_eq!(links[0].recycled, 0);
    }
}

#[test]
fn constraints_recycle_transfer_or_build_fresh_contacts() {
    let params = RbdSimParams::tgs_soft();
    let mut manifold = IndexedManifold {
        bodies: UVec2::new(0, 1),
        colliders: UVec2::new(0, 1),
        friction: 0.6,
        restitution: 0.2,
        subshape: 3,
        recycle_extent: 1.0,
        contact: ContactManifold {
            normal_a: Vector::Y,
            len: MAX_CONSTRAINTS_PER_MANIFOLD as u32,
            ..Default::default()
        },
    };
    for k in 0..MAX_CONSTRAINTS_PER_MANIFOLD {
        manifold.contact.points_a[k].pt = Vector::splat(0.05 * k as f32);
        manifold.contact.points_a[k].dist = -0.01;
    }
    let old_poses = [Pose::IDENTITY; 2];
    let vels = [Velocity::default(); 2];
    let mut mprops = [WorldMassProperties::default(); 2];
    let mut old = TwoBodyConstraint::default();
    manifold.contact_to_constraint(
        &Slice(&mprops, 0),
        &Slice(&old_poses, 0),
        &Slice(&old_poses, 0),
        &Slice(&vels, 0),
        &mut old,
    );
    for k in 0..MAX_CONSTRAINTS_PER_MANIFOLD {
        old.points[k].normal_impulse = 0.2 * k as f32;
    }
    let mut old_tiles = [CoulombTile::default()];
    old_tiles[0].write_constraint(0, &old);
    let old_links = links_of(&[manifold]);
    let old_state = ContactRecycleState {
        pose_a: old_poses[0],
        pose_b: old_poses[1],
        colliders: manifold.colliders,
        max_extent: 1.0,
        max_drift: 0.05,
    };
    let offsets = ContactRecycleOffsets {
        old_base: 0,
        new_base: 1,
        ..Default::default()
    };
    // Recycled contacts, fresh geometry, and a previously unseen manifold.
    for drift in [0.0, 0.001, 0.3] {
        for matched in [false, true] {
            for len in 1..=MAX_CONSTRAINTS_PER_MANIFOLD {
                let mut current = manifold;
                current.contact.len = len as u32;
                current.subshape += u32::from(!matched);
                let poses = [Pose::IDENTITY, Pose::from_translation(Vector::X * drift)];
                mprops[0].inv_mass = Vector::splat(0.3);
                mprops[1].inv_mass = Vector::new(
                    0.0,
                    0.6,
                    #[cfg(feature = "dim3")]
                    0.8,
                );
                let mut links = links_of(&[current]);
                let mut states = [old_state, ContactRecycleState::default()];
                gpu_match_contact_links(
                    UVec3::ZERO,
                    &[current],
                    &mut links,
                    &[1, 2],
                    &[0, 0],
                    &old_links,
                    &[0],
                    &mut states,
                    &poses,
                    &plan(1),
                    &params,
                    &offsets,
                );
                // The link of contact 0 is its own sorted link.
                let mut tiles = [CoulombTile::default()];
                gpu_prepare_constraints(
                    UVec3::ZERO,
                    &links,
                    &[current],
                    &mprops,
                    &poses,
                    &vels,
                    &states,
                    &old_tiles,
                    &mut tiles,
                    &plan(1),
                    &offsets,
                );

                // Recycling keeps the previous contacts with their impulses. Otherwise, the
                // drifted anchors are too far from the previous ones to inherit impulses.
                let recycled = matched && drift < 0.05;
                let mut expected = TwoBodyConstraint::default();
                current.contact_to_constraint(
                    &Slice(&mprops, 0),
                    &Slice(&poses, 0),
                    &Slice(&poses, 0),
                    &Slice(&vels, 0),
                    &mut expected,
                );
                if recycled {
                    expected.recycle_from(&old, &vels[0], &vels[1]);
                }
                if matched && !recycled && drift == 0.0 {
                    for k in 0..len {
                        expected.points[k].normal_impulse = old.points[k].normal_impulse;
                    }
                }
                let mut expected_tiles = [CoulombTile::default()];
                expected_tiles[0].write_constraint(0, &expected);
                // The cached warmstart is tagged with the body indices.
                let (mut a, mut b) = (Velocity::default(), Velocity::default());
                expected.warmstart_constraint(&mut a, &mut b);
                a.padding1 = 0;
                b.padding1 = 1;
                expected_tiles[0].warmstart.bodies[0] = [a, b];
                assert_eq!(tiles, expected_tiles, "drift {drift}, matched {matched}");
                assert_eq!(links[0].recycled, u32::from(recycled));
                let expected_state = if recycled {
                    old_state
                } else {
                    ContactRecycleState {
                        pose_a: poses[0],
                        pose_b: poses[1],
                        colliders: current.colliders,
                        max_extent: 1.0,
                        max_drift: params.contact_recycle_distance(),
                    }
                };
                assert_eq!(
                    bytemuck::bytes_of(&states[1]),
                    bytemuck::bytes_of(&expected_state)
                );
            }
        }
    }
}
