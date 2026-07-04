use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusPipeline, NexusState};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    /*
     * The ground
     */
    // let ground_size = 100.0;
    // let ground_height = 0.1;
    //
    // let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height - 5.0, 0.0));
    // let ground_handle = bodies.insert(rigid_body);
    // let collider = ColliderBuilder::cuboid(
    //     ground_size,
    //     ground_height,
    //     ground_size,
    // );
    // colliders.insert_with_parent(collider, ground_handle, &mut bodies);

    /*
     * A 3-link pendulum modeled with rapier's MultibodyJointSet.
     *
     * - A fixed root body is anchored at the origin.
     * - Three dynamic links hang from revolute joints about the X axis.
     * - Under gravity alone, the chain should swing in the YZ plane.
     *
     * The GPU pipeline picks up the multibody set from `SimulationState::environments`
     * and runs `GpuMultibodySolver::step` each frame — no contacts or constraints
     * with multibodies are involved.
     */
    let rad = 0.4;
    let link_len = 2.0;
    let num_links = 20;

    // Fixed root at origin.
    let root_body = RigidBodyBuilder::fixed().build();
    let root_collider = ColliderBuilder::cuboid(rad, rad, rad).build();
    let root_shape = root_collider.shared_shape().clone();
    let mut parent_handle = state.insert_rigid_body(root_body, root_collider);
    viewer.insert_shape(parent_handle, &root_shape, Pose::IDENTITY);

    for i in 0..num_links {
        // Each link hangs `link_len` below its parent.
        let x = (i as f32 + 1.0) * link_len;
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(Vec3::new(x, 0.0, 0.0))
            .build();
        let collider = ColliderBuilder::cuboid(link_len * 0.5, rad, rad)
            .collision_groups(InteractionGroups::none())
            .build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(rigid_body, collider);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);

        // Revolute joint about X: anchor on parent is at its bottom
        // (or at origin for the root), anchor on child is at its top.
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            // .limits([-0.1, 0.1])
            // .contacts_enabled(false)
            .build();
        state.insert_multibody_joint(parent_handle, handle, joint);

        parent_handle = handle;
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
