"""Source of the Python guide snippets. Checked against the real extension."""
from nexus3d import (
    NexusState, NexusPipeline, RbdCoupling, RigidBodyBuilder, ColliderBuilder,
    InteractionGroups, RevoluteJointBuilder, Vec3, Pose, Particle, ParticleModel,
    SimulationParams, BoundaryCondition, JointAxis,
)


def bodies(state):
    # DOCUSAURUS: body start
    body = (RigidBodyBuilder.dynamic()
            .translation(Vec3(0.0, 3.0, 0.0))
            .rotation(Vec3(0.0, 0.0, 0.3))
            .linvel(Vec3(1.0, 0.0, 0.0))
            .angvel(Vec3(0.0, 0.0, 1.0))
            .build())
    collider = ColliderBuilder.ball(0.5).density(2.0).build()
    handle = state.insert_rigid_body(body, collider, RbdCoupling.NONE)
    # DOCUSAURUS: body stop
    return handle


def colliders(state):
    # DOCUSAURUS: colliders start
    body = RigidBodyBuilder.dynamic().translation(Vec3(0.0, 3.0, 0.0)).build()
    parent = state.insert_body(body, RbdCoupling.NONE)
    for x in [-0.5, 0.5]:
        collider = (ColliderBuilder.ball(0.5)
                    .translation(Vec3(x, 0.0, 0.0))
                    .density(1.0).friction(0.8).restitution(0.3).build())
        state.insert_collider_in(0, collider, parent)
    # DOCUSAURUS: colliders stop


def collision_groups():
    # DOCUSAURUS: groups start
    groups = InteractionGroups.new(0b0001, 0b0010)
    collider = ColliderBuilder.ball(0.5).collision_groups(groups).build()
    # DOCUSAURUS: groups stop
    return collider


def joints(state):
    body1 = state.insert_rigid_body(RigidBodyBuilder.fixed().build(),
                                    ColliderBuilder.ball(0.1).build(), RbdCoupling.NONE)
    body2 = bodies(state)
    # DOCUSAURUS: joint start
    joint = (RevoluteJointBuilder.new(Vec3.Z)
             .local_anchor1(Vec3(0.0, 0.0, 0.0))
             .local_anchor2(Vec3(0.0, 1.0, 0.0))
             .limits(-1.0, 1.0)
             .motor_velocity(1.0, 0.5)
             .motor_max_force(10.0).build())
    state.insert_impulse_joint(body1, body2, joint)
    # DOCUSAURUS: joint stop


def batches():
    # DOCUSAURUS: batches start
    state = NexusState()
    state.set_rbd_collisions_capacity(128)
    for i in range(4):
        env = 0 if i == 0 else state.add_environment()
        body = RigidBodyBuilder.dynamic().translation(Vec3(i * 2.0, 3.0, 0.0)).build()
        collider = ColliderBuilder.ball(0.5).build()
        state.insert_rigid_body_in(env, body, collider, RbdCoupling.NONE)
    # DOCUSAURUS: batches stop
    return state


def lifecycle(viewer, state):
    # DOCUSAURUS: lifecycle start
    pipeline = NexusPipeline()
    state.finalize(viewer)
    state.set_rbd_gravity(viewer, Vec3(0.0, -9.81, 0.0))
    pipeline.simulate(viewer, state)
    # DOCUSAURUS: lifecycle stop


def parameters(state):
    # DOCUSAURUS: parameters start
    state.set_rbd_timestep(1.0 / 120.0, 8)
    state.set_rbd_steps_per_frame(2)
    # DOCUSAURUS: parameters stop


def friction_model(state):
    # DOCUSAURUS: friction_model start
    state.set_rbd_solver_params(friction_model="simplified")
    # DOCUSAURUS: friction_model stop


def determinism(viewer, state):
    # DOCUSAURUS: determinism start
    state.set_deterministic(viewer, True)
    # DOCUSAURUS: determinism stop


def insertion(viewer):
    # DOCUSAURUS: insertion start
    state = NexusState()
    state.reserve_rigid_bodies(128)
    ground = RigidBodyBuilder.fixed().build()
    collider = ColliderBuilder.cuboid(5.0, 0.1, 5.0).build()
    state.insert_rigid_body(ground, collider, RbdCoupling.NONE)
    state.finalize(viewer)

    body = RigidBodyBuilder.dynamic().translation(Vec3(0.0, 3.0, 0.0)).build()
    state.add_rigid_bodies(viewer, [body], [ColliderBuilder.ball(0.5).build()], RbdCoupling.NONE)
    # DOCUSAURUS: insertion stop
    return state


def particles(viewer, state):
    # DOCUSAURUS: particles start
    cell_width = 0.5
    params = SimulationParams(Vec3(0.0, -9.81, 0.0), 1.0 / 60.0)
    state.set_mpm_params(viewer, params, cell_width)
    state.set_mpm_substeps(20)
    model = ParticleModel.elastic(1.0e4, 0.2)
    particles = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                position = Vec3(x * cell_width / 2.0,
                                2.0 + y * cell_width / 2.0,
                                z * cell_width / 2.0)
                particles.append(Particle(position, cell_width / 4.0, 1000.0, model))
    chunk = state.add_particles(viewer, particles)
    # DOCUSAURUS: particles stop
    return chunk


def coupling(state):
    # DOCUSAURUS: coupling start
    coupling = RbdCoupling.mpm_one_way(BoundaryCondition.separate(0.5))
    body = RigidBodyBuilder.fixed().build()
    collider = ColliderBuilder.cuboid(5.0, 0.1, 5.0).build()
    state.insert_rigid_body(body, collider, coupling)
    # DOCUSAURUS: coupling stop


def material_models():
    # DOCUSAURUS: materials start
    elastic = ParticleModel.elastic(1.0e4, 0.2)
    sand = ParticleModel.sand(1.0e5, 0.2)
    # DOCUSAURUS: materials stop
    return elastic, sand


def boundaries():
    # DOCUSAURUS: boundaries start
    sticky = BoundaryCondition.stick()
    sliding = BoundaryCondition.slip()
    separating = BoundaryCondition.separate(0.5)
    # DOCUSAURUS: boundaries stop
    return sticky, sliding, separating


def remove_particles(viewer, state, chunk):
    # DOCUSAURUS: chunks start
    model = ParticleModel.sand(1.0e5, 0.2)
    particle = Particle(Vec3(0.0, 3.0, 0.0), 0.125, 1000.0, model)
    state.extend_chunk(viewer, chunk, [particle])
    state.remove_chunk(viewer, chunk)
    # DOCUSAURUS: chunks stop


def multibody(state):
    # DOCUSAURUS: multibody start
    base = state.insert_body(RigidBodyBuilder.fixed().build(), RbdCoupling.NONE)
    body = RigidBodyBuilder.dynamic().translation(Vec3(0.0, -1.0, 0.0)).build()
    link = state.insert_rigid_body(body, ColliderBuilder.ball(0.3).build(), RbdCoupling.NONE)
    joint = (RevoluteJointBuilder.new(Vec3.Z).local_anchor2(Vec3.Y)
             .motor_velocity(0.0, 1.0).motor_max_force(10.0).build())
    joint_handle = state.insert_multibody_joint(base, link, joint)
    assert joint_handle is not None, "The joint must not create a cycle."
    # DOCUSAURUS: multibody stop
    return joint_handle


def motors(viewer, state):
    # DOCUSAURUS: motors start
    # In this two-link example the base is link 0 and the pendulum is link 1.
    state.set_multibody_motor_velocity(viewer, 0, 1, JointAxis.AngX, 1.0)
    # DOCUSAURUS: motors stop


def readback(viewer, state):
    # DOCUSAURUS: readback start
    coords, positions, quats, linvels, angvels = state.read_multibody_links(viewer)
    stride = state.multibody_links_per_env()
    positions_env0 = positions[:stride]
    print("Link positions:", positions_env0)
    # DOCUSAURUS: readback stop
    return positions_env0
