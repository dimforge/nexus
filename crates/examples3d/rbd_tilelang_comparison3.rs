//! Small same-input scene for visual solver comparisons (sleeping disabled).
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::new(NexusCapacities::default().rbd_collisions(256));
    let mut add = |body: RigidBodyBuilder, collider: ColliderBuilder| {
        let collider = collider.build();
        let shape = collider.shared_shape().clone();
        let handle =
            state.insert_rigid_body(body.can_sleep(false).build(), collider, RbdCoupling::None);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);
        handle
    };
    add(
        RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(5.0, 0.5, 4.0),
    );
    for y in 0..4 {
        for x in 0..3 {
            add(
                RigidBodyBuilder::dynamic().translation(Vec3::new(
                    x as f32 * 1.15 - 1.15,
                    0.52 + y as f32 * 1.05,
                    0.0,
                )),
                ColliderBuilder::cuboid(0.5, 0.5, 0.5),
            );
        }
        add(
            RigidBodyBuilder::dynamic().translation(Vec3::new(-2.8, 0.6 + y as f32 * 1.2, 1.7)),
            ColliderBuilder::ball(0.5),
        );
    }
    let anchor = add(
        RigidBodyBuilder::fixed().translation(Vec3::new(3.0, 4.0, 0.0)),
        ColliderBuilder::ball(0.15),
    );
    let bob = add(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(3.0, 2.5, 0.0))
            .linvel(Vec3::new(0.0, 0.0, 2.0)),
        ColliderBuilder::cuboid(0.3, 0.7, 0.3),
    );
    state.insert_impulse_joint(
        anchor,
        bob,
        SphericalJointBuilder::new().local_anchor2(Vec3::new(0.0, 1.5, 0.0)),
    );
    viewer.set_camera(Vec3::new(11.0, 8.0, 13.0), Vec3::new(0.0, 1.6, 0.0));
    viewer
        .scene3d_mut()
        .add_directional_light(Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;
    while viewer.render_frame().await {
        if viewer.simulating() {
            viewer.simulate(pipeline, &mut state, None).await?;
        }
        viewer.sync(&mut state, None).await?;
    }
    Ok(state)
}
