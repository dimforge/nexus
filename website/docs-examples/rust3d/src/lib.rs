use khal::backend::{Backend, GpuBackend};
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::*;
use nexus3d::rapier::prelude::*;
use nexus3d::rbd::dynamics::{FrictionModel, RbdSimParams};

pub fn bodies(state: &mut NexusState) -> RigidBodyHandle {
    // DOCUSAURUS: body start
    let body = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(0.0, 3.0, 0.0))
        .rotation(Vec3::new(0.0, 0.0, 0.3))
        .linvel(Vec3::new(1.0, 0.0, 0.0))
        .angvel(Vec3::new(0.0, 0.0, 1.0))
        .build();
    let collider = ColliderBuilder::ball(0.5).density(2.0).build();
    let handle = state.insert_rigid_body(body, collider, RbdCoupling::None);
    // DOCUSAURUS: body stop
    handle
}

pub fn colliders(state: &mut NexusState) {
    // DOCUSAURUS: colliders start
    let body = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(0.0, 3.0, 0.0))
        .build();
    let parent = state.insert_body(body, RbdCoupling::None);
    for x in [-0.5, 0.5] {
        let collider = ColliderBuilder::ball(0.5)
            .translation(Vec3::new(x, 0.0, 0.0))
            .density(1.0)
            .friction(0.8)
            .restitution(0.3)
            .build();
        state.insert_collider_in(0, collider, Some(parent));
    }
    // DOCUSAURUS: colliders stop
}

pub fn collision_groups() -> Collider {
    // DOCUSAURUS: groups start
    let groups = InteractionGroups::new(Group::GROUP_1, Group::GROUP_2, InteractionTestMode::And);
    let collider = ColliderBuilder::ball(0.5).collision_groups(groups).build();
    // DOCUSAURUS: groups stop
    collider
}

pub fn joints(state: &mut NexusState) {
    let body1 = state.insert_rigid_body(
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::ball(0.1).build(),
        RbdCoupling::None,
    );
    let body2 = bodies(state);
    // DOCUSAURUS: joint start
    let joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::new(0.0, 0.0, 0.0))
        .local_anchor2(Vec3::new(0.0, 1.0, 0.0))
        .limits([-1.0, 1.0])
        .motor_velocity(1.0, 0.5)
        .motor_max_force(10.0)
        .build();
    state.insert_impulse_joint(body1, body2, joint);
    // DOCUSAURUS: joint stop
}

pub fn batches() -> NexusState {
    // DOCUSAURUS: batches start
    let mut state = NexusState::default();
    state.set_rbd_collisions_capacity(128);
    for i in 0..4 {
        let env = if i == 0 { 0 } else { state.add_environment() };
        let body = RigidBodyBuilder::dynamic()
            .translation(Vec3::new(i as f32 * 2.0, 3.0, 0.0))
            .build();
        let collider = ColliderBuilder::ball(0.5).build();
        state.insert_rigid_body_in(env, body, collider, RbdCoupling::None);
    }
    // DOCUSAURUS: batches stop
    state
}

pub fn parameters(state: &mut NexusState) {
    // DOCUSAURUS: parameters start
    let params = RbdSimParams {
        dt: 1.0 / 120.0,
        num_solver_iterations: 8,
        ..RbdSimParams::default()
    };
    state.set_rbd_sim_params(0, params);
    state.set_rbd_steps_per_frame(2);
    // DOCUSAURUS: parameters stop
}

pub fn friction_model(state: &mut NexusState) {
    // DOCUSAURUS: friction_model start
    state.set_rbd_friction_model(FrictionModel::Simplified);
    // DOCUSAURUS: friction_model stop
}

pub fn determinism(backend: &GpuBackend, state: &mut NexusState) {
    // DOCUSAURUS: determinism start
    state.set_deterministic(backend, true);
    // DOCUSAURUS: determinism stop
}

pub async fn lifecycle(backend: &GpuBackend, state: &mut NexusState) -> anyhow::Result<()> {
    // DOCUSAURUS: lifecycle start
    let mut pipeline = NexusPipeline::default();
    pipeline.preload_pipelines(backend, NexusPipelineMask::RBD)?;
    state.finalize(backend).await?;
    state.set_rbd_gravity(backend, [0.0, -9.81, 0.0]);
    pipeline.simulate(backend, state, None).await?;
    // DOCUSAURUS: lifecycle stop
    Ok(())
}

pub async fn readback(
    backend: &GpuBackend,
    state: &NexusState,
    handle: RigidBodyHandle,
) -> anyhow::Result<Pose> {
    // DOCUSAURUS: readback start
    let rbd = state.rbd.as_ref().expect("Finalize the state first.");
    let mut poses = vec![Pose::IDENTITY; rbd.body_poses().len() as usize];
    backend
        .slow_read_buffer(rbd.body_poses().buffer(), &mut poses)
        .await?;
    let gpu_id = state.rbd2gpu[0].get(handle.0).unwrap().gpu_id as usize;
    let pose = poses[gpu_id];
    println!("Body altitude: {}", pose.translation.y);
    // DOCUSAURUS: readback stop
    Ok(pose)
}

pub async fn insertion(backend: &GpuBackend) -> anyhow::Result<NexusState> {
    // DOCUSAURUS: insertion start
    let mut state = NexusState::default();
    state.reserve_rigid_bodies(128);
    let ground = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::cuboid(5.0, 0.1, 5.0).build();
    state.insert_rigid_body(ground, collider, RbdCoupling::None);
    state.finalize(backend).await?;

    let body = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(0.0, 3.0, 0.0))
        .build();
    state.add_rigid_body(
        backend,
        body,
        ColliderBuilder::ball(0.5).build(),
        RbdCoupling::None,
    )?;
    // DOCUSAURUS: insertion stop
    Ok(state)
}

pub async fn particles(
    backend: &GpuBackend,
    state: &mut NexusState,
) -> anyhow::Result<NexusParticleChunk> {
    // DOCUSAURUS: particles start
    let cell_width = 0.5;
    let params = SimulationParams {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(backend, params, cell_width)?;
    state.set_mpm_substeps(20);

    let model = ParticleModel::elastic(1.0e4, 0.2);
    let mut particles = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let position = Vec3::new(
                    x as f32 * cell_width / 2.0,
                    2.0 + y as f32 * cell_width / 2.0,
                    z as f32 * cell_width / 2.0,
                );
                particles.push(Particle::new(position, cell_width / 4.0, 1000.0, model));
            }
        }
    }
    let chunk = state.add_particles(backend, particles)?;
    // DOCUSAURUS: particles stop
    Ok(chunk)
}

pub fn coupling(state: &mut NexusState) {
    // DOCUSAURUS: coupling start
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.5));
    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::cuboid(5.0, 0.1, 5.0).build();
    state.insert_rigid_body(body, collider, coupling);
    // DOCUSAURUS: coupling stop
}

pub fn material_models() -> [ParticleModel; 2] {
    // DOCUSAURUS: materials start
    let elastic = ParticleModel::elastic(1.0e4, 0.2);
    let sand = ParticleModel::sand(1.0e5, 0.2);
    // DOCUSAURUS: materials stop
    [elastic, sand]
}

pub fn boundaries() -> [BoundaryCondition; 3] {
    // DOCUSAURUS: boundaries start
    let sticky = BoundaryCondition::stick();
    let sliding = BoundaryCondition::slip();
    let separating = BoundaryCondition::separate(0.5);
    // DOCUSAURUS: boundaries stop
    [sticky, sliding, separating]
}

pub fn remove_particles(
    backend: &GpuBackend,
    state: &mut NexusState,
    chunk: NexusParticleChunk,
) -> anyhow::Result<()> {
    // DOCUSAURUS: chunks start
    let model = ParticleModel::sand(1.0e5, 0.2);
    let particle = Particle::new(Vec3::new(0.0, 3.0, 0.0), 0.125, 1000.0, model);
    state.extend_chunk(backend, chunk, vec![particle])?;
    state.remove_chunk(backend, chunk)?;
    // DOCUSAURUS: chunks stop
    Ok(())
}

pub fn multibody(state: &mut NexusState) -> MultibodyJointHandle {
    // DOCUSAURUS: multibody start
    let base = state.insert_body(RigidBodyBuilder::fixed().build(), RbdCoupling::None);
    let body = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(0.0, -1.0, 0.0))
        .build();
    let link = state.insert_rigid_body(body, ColliderBuilder::ball(0.3).build(), RbdCoupling::None);
    let joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor2(Vec3::Y)
        .motor_velocity(0.0, 1.0)
        .motor_max_force(10.0)
        .build();
    let joint_handle = state
        .insert_multibody_joint(base, link, joint)
        .expect("The joint must not create a cycle.");
    // DOCUSAURUS: multibody stop
    joint_handle
}

pub fn motors(
    backend: &GpuBackend,
    state: &mut NexusState,
    joint_handle: MultibodyJointHandle,
) -> anyhow::Result<()> {
    // DOCUSAURUS: motors start
    state.control_multibody_motors(backend, |_, world| {
        let (multibody, link_id) = world.multibody_joints.get_mut(joint_handle).unwrap();
        multibody
            .link_mut(link_id)
            .unwrap()
            .joint
            .data
            .set_motor_velocity(JointAxis::AngX, 1.0, 1.0);
    })?;
    // DOCUSAURUS: motors stop
    Ok(())
}

pub async fn reset(backend: &GpuBackend, state: &mut NexusState) {
    // DOCUSAURUS: reset start
    let rbd = state.rbd.as_mut().expect("Finalize the state first.");
    let initial_state = rbd.snapshot(backend).await;
    // Later, restore environment 0 to this state (with the same topology).
    rbd.reset_env_from_snapshot(backend, 0, &initial_state);
    // DOCUSAURUS: reset stop
}
