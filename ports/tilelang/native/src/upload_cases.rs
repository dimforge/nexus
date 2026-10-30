//! Native scene builders used by CPU export and Metal upload regression tests.
use nexus_rbd3d::{pipeline::RbdCapacities, rapier::prelude::*};

pub fn capacities(batches: u32) -> RbdCapacities {
    RbdCapacities {
        batches,
        body_capacity: 8,
        collisions_capacity: 8,
        mb_contact_constraints_capacity: 8,
        ..Default::default()
    }
}

pub fn scene(shift: f32, mesh: SharedShape) -> (RigidBodySet, ColliderSet, ImpulseJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let moving = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(shift, 2.0, 0.0))
            .linvel(Vec3::new(1.0 + shift, 0.0, 0.0))
            .angvel(Vec3::new(0.0, 0.3, 0.0))
            .can_sleep(false),
    );
    let fixed = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(shift, 5.0, 0.0)));
    colliders.insert_with_parent(
        ColliderBuilder::capsule_y(0.4, 0.2)
            .translation(Vec3::new(0.4, 0.0, 0.0))
            .density(2.0)
            .friction(0.7)
            .restitution(0.3)
            .friction_combine_rule(CoefficientCombineRule::Max)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_2,
                Group::GROUP_3,
                InteractionTestMode::Or,
            )),
        moving,
        &mut bodies,
    );
    colliders.insert_with_parent(
        ColliderBuilder::ball(0.3).translation(Vec3::new(-0.2, 0.1, 0.0)),
        moving,
        &mut bodies,
    );
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.5, 0.5, 0.5), fixed, &mut bodies);
    colliders.insert(ColliderBuilder::new(mesh).translation(Vec3::new(shift, -1.0, 0.0)));
    for (_, body) in bodies.iter_mut() {
        body.recompute_mass_properties_from_colliders(&colliders);
    }
    let mut joints = ImpulseJointSet::new();
    joints.insert(
        moving,
        fixed,
        PrismaticJointBuilder::new(Vec3::X)
            .local_anchor1(Vec3::new(0.2, 0.3, 0.4))
            .local_anchor2(Vec3::new(0.2, -2.7, 0.4))
            .limits([-0.25, 0.75])
            .motor_velocity(0.15, 0.8),
        true,
    );
    (bodies, colliders, joints)
}

/// Several workgroups of independent contacts, with shared read-only ground.
pub fn wide_scene(shift: f32) -> (RigidBodySet, ColliderSet, ImpulseJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(64.0, -0.5, shift)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(70.0, 0.5, 2.0), ground, &mut bodies);
    for i in 0..65 {
        let body = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vec3::new(i as f32 * 2.0, 0.49, shift))
                .linvel(Vec3::new(0.0, -0.1, 0.0))
                .can_sleep(false),
        );
        colliders.insert_with_parent(ColliderBuilder::ball(0.5), body, &mut bodies);
    }
    for (_, body) in bodies.iter_mut() {
        body.recompute_mass_properties_from_colliders(&colliders);
    }
    (bodies, colliders, ImpulseJointSet::new())
}

/// One ground contact per batch, used to exercise the inter-batch bucket hierarchy.
pub fn small_contact_scene(shift: f32) -> (RigidBodySet, ColliderSet, ImpulseJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, shift)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(2.0, 0.5, 2.0), ground, &mut bodies);
    let body = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(0.0, 0.49, shift))
            .linvel(Vec3::new(0.0, -0.1, 0.0))
            .can_sleep(false),
    );
    colliders.insert_with_parent(ColliderBuilder::ball(0.5), body, &mut bodies);
    for (_, body) in bodies.iter_mut() {
        body.recompute_mass_properties_from_colliders(&colliders);
    }
    (bodies, colliders, ImpulseJointSet::new())
}
