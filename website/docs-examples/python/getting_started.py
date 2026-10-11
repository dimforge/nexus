# DOCUSAURUS: basic_sim start
from nexus3d import (
    NexusViewer, NexusPipeline, NexusState, RbdCoupling,
    RigidBodyBuilder, ColliderBuilder, Vec3, Pose,
)


def main():
    viewer = NexusViewer().with_running()
    viewer.init_backend()
    state = NexusState()
    pipeline = NexusPipeline()

    # Create the ground.
    ground = RigidBodyBuilder.fixed().build()
    collider = ColliderBuilder.cuboid(5.0, 0.1, 5.0).build()
    handle = state.insert_rigid_body(ground, collider, RbdCoupling.NONE)
    viewer.insert_shape(handle, collider.shared_shape(), Pose.IDENTITY)

    # Create the falling ball.
    ball = RigidBodyBuilder.dynamic().translation(Vec3(0.0, 3.0, 0.0)).build()
    collider = ColliderBuilder.ball(0.5).build()
    handle = state.insert_rigid_body(ball, collider, RbdCoupling.NONE)
    viewer.insert_shape(handle, collider.shared_shape(), Pose.IDENTITY)

    state.finalize(viewer)
    viewer.add_directional_light(Vec3(1.0, -2.0, 3.0))
    while viewer.render_frame():
        if viewer.simulating():
            pipeline.simulate(viewer, state)
        viewer.sync(state)


if __name__ == "__main__":
    main()
# DOCUSAURUS: basic_sim stop
