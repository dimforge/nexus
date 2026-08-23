//! Headless probe for MJCF-style joint dry friction (`frictionloss`).
//!
//! A single-link pendulum hinged at the origin, its rod lying along +X so
//! gravity applies a torque `m·g·l` about the hinge. MuJoCo models friction
//! loss as a constraint (a bound on the force friction may generate), not as a
//! `-f·sign(q̇)` force, so the expected behaviour is:
//!
//! * `frictionloss = 0`: the link falls freely.
//! * `frictionloss` above the gravity torque: the link sticks, exactly. A
//!   force-based implementation cannot do this; it chatters around `q̇ = 0`.
//! * `frictionloss` below the gravity torque: the link falls, but slower.
//!
//! Run with `cargo run --release --bin test_frictionloss --features metal`.

use khal::backend::{Backend, GpuBackend};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};
use nexus3d::rbd::dynamics::RbdSimParams;
use rapier3d::prelude::*;

const LINK_LEN: f32 = 1.0;
const RAD: f32 = 0.05;
const STEPS: usize = 60;

/// Builds the one-link pendulum and returns its state. `motor` adds a velocity
/// motor and a limit on the hinge, so the friction rows have to share the
/// constraint bank with them. `joint_frequency`, when given, softens the shared
/// joint constraint softness the friction rows draw their CFM from.
fn make_state(motor: bool, joint_frequency: Option<f32>) -> NexusState {
    let mut state = NexusState::default();
    if let Some(hz) = joint_frequency {
        let mut params = RbdSimParams::default();
        params.joint_natural_frequency = hz;
        state.set_rbd_sim_params(0, params);
    }
    let no_coupling = RbdCoupling::None;

    let root = RigidBodyBuilder::fixed().build();
    let root_collider = ColliderBuilder::cuboid(RAD, RAD, RAD)
        .collision_groups(InteractionGroups::none())
        .build();
    let root_handle = state.insert_rigid_body(root, root_collider, no_coupling);

    let link = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(LINK_LEN, 0.0, 0.0))
        .build();
    let collider = ColliderBuilder::cuboid(LINK_LEN * 0.5, RAD, RAD)
        .collision_groups(InteractionGroups::none())
        .build();
    let link_handle = state.insert_rigid_body(link, collider, no_coupling);

    // Hinge about Z at the origin, so gravity (-Y) torques the joint.
    let mut builder = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(Vec3::new(-LINK_LEN, 0.0, 0.0));
    if motor {
        builder = builder
            .limits([-2.0, 2.0])
            .motor_velocity(2.0, 0.0)
            .motor_max_force(100.0);
    }
    let joint = builder.build();
    state.insert_multibody_joint(root_handle, link_handle, joint);

    state
}

/// Steps the pendulum for `STEPS` frames and returns `(drop, |q̇|)`: how far
/// the link's centre of mass fell, and the joint velocity at the end.
async fn run_case(
    backend: &GpuBackend,
    frictionloss: f32,
    motor: bool,
    joint_frequency: Option<f32>,
) -> Result<(f32, f32), khal::backend::GpuBackendError> {
    let mut state = make_state(motor, joint_frequency);
    let mut pipeline = NexusPipeline::default();

    // The frictionloss slots are reserved on the first non-zero write, which
    // needs the GPU state to exist: finalize before setting it.
    state.finalize(backend).await?;
    let rbd = state.rbd.as_mut().expect("no rbd state");
    let ndofs = rbd.multibodies().dofs_per_batch() as usize;
    rbd.set_dof_frictionloss(backend, &vec![frictionloss; ndofs]);

    for _ in 0..STEPS {
        pipeline.simulate(backend, &mut state, None).await?;
    }

    let rbd = state.rbd.as_ref().expect("no rbd state");
    let poses: Vec<nexus3d::rbd::math::Pose> =
        backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let dof_state: Vec<f32> = backend
        .slow_read_vec(rbd.multibodies().dof_state().buffer())
        .await?;
    // Body 1 is the link; it starts level with the hinge, so any fall shows up
    // as a negative y.
    Ok((-poses[1].translation.y, dof_state[0].abs()))
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let backend =
        GpuBackend::Metal(khal::backend::Metal::new().map_err(|e| anyhow::anyhow!("{e:?}"))?);

    // Gravity torque about the hinge for a rod of half-length `LINK_LEN/2`.
    // The collider density is rapier's default (1.0).
    let mass = 2.0 * (LINK_LEN * 0.5) * (2.0 * RAD) * (2.0 * RAD);
    let gravity_torque = mass * 9.81 * LINK_LEN;
    println!("mass ≈ {mass:.5} kg, gravity torque ≈ {gravity_torque:.5} N·m\n");

    let cases = [
        ("free (fl = 0)", 0.0, false, None),
        ("weak (fl = 0.25 · τ_g)", 0.25 * gravity_torque, false, None),
        ("locked (fl = 4 · τ_g)", 4.0 * gravity_torque, false, None),
        // A force-based `-fl·sign(q̇)` blows up here; a bounded constraint row
        // simply never exceeds what it takes to stop the DoF.
        (
            "extreme (fl = 1000 · τ_g)",
            1000.0 * gravity_torque,
            false,
            None,
        ),
        // Friction rows sharing the constraint bank with a limit and a motor.
        ("motor + fl = 0.1 · τ_g", 0.1 * gravity_torque, true, None),
        // Same locked case, but with a compliant joint softness: the friction
        // row picks up CFM and the DoF creeps under load instead of sticking.
        (
            "locked, soft joints (2 Hz)",
            4.0 * gravity_torque,
            false,
            Some(2.0),
        ),
    ];

    let mut results = Vec::new();
    for (name, fl, motor, hz) in cases {
        let (drop, qd) = run_case(&backend, fl, motor, hz).await?;
        println!("{name:<26} drop = {drop:.6} m   |q̇| = {qd:.6} rad/s");
        results.push((name, drop, qd));
    }

    let (_, free_drop, _) = results[0];
    let (_, weak_drop, _) = results[1];
    let (_, locked_drop, locked_qd) = results[2];
    let (_, extreme_drop, extreme_qd) = results[3];
    let (_, _, motor_qd) = results[4];
    let (_, soft_drop, _) = results[5];

    println!();
    assert!(
        free_drop > 0.1,
        "frictionless pendulum should fall: drop = {free_drop}"
    );
    assert!(
        weak_drop < free_drop * 0.9,
        "sub-gravity friction should slow the fall: {weak_drop} vs {free_drop}"
    );
    assert!(
        locked_drop.abs() < 1.0e-4 && locked_qd < 1.0e-4,
        "friction above the gravity torque should hold the joint at rest: \
         drop = {locked_drop}, |q̇| = {locked_qd}"
    );
    assert!(
        extreme_drop.abs() < 1.0e-4 && extreme_qd < 1.0e-4,
        "an oversized friction bound must stay inert, not chatter: \
         drop = {extreme_drop}, |q̇| = {extreme_qd}"
    );
    assert!(
        (motor_qd - 2.0).abs() < 0.1,
        "the motor should still reach its 2 rad/s target through weak \
         friction: |q̇| = {motor_qd}"
    );
    assert!(
        soft_drop > 1.0e-3 && soft_drop < free_drop,
        "a compliant joint softness should let the held DoF creep, without \
         letting it fall freely: drop = {soft_drop}"
    );
    println!("OK");
    Ok(())
}
