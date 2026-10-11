// DOCUSAURUS: basic_sim start
use nexus_viewer2d::NexusViewer;
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};
use nexus2d::rapier::prelude::*;

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let mut viewer = NexusViewer::new(Vec::new()).await.with_running();
        viewer.init_backend();
        let mut state = NexusState::default();
        let mut pipeline = NexusPipeline::default();

        /* Create the ground. */
        let ground = RigidBodyBuilder::fixed().build();
        let collider = ColliderBuilder::cuboid(5.0, 0.1).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(ground, collider, RbdCoupling::None);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);

        /* Create the falling ball. */
        let ball = RigidBodyBuilder::dynamic()
            .translation(Vec2::new(0.0, 3.0))
            .build();
        let collider = ColliderBuilder::ball(0.5).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(ball, collider, RbdCoupling::None);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);

        /* Upload the scene before rendering it. */
        state.finalize(viewer.backend()).await?;

        while viewer.render_frame().await {
            if viewer.simulating() {
                pipeline
                    .simulate(viewer.backend(), &mut state, None)
                    .await?;
            }
            viewer.sync(&mut state, None).await?;
        }
        Ok(())
    })
}
// DOCUSAURUS: basic_sim stop
