//! End-to-end check that MJCF `<joint frictionloss>` reaches the GPU solver.
//!
//! The path is: `<joint frictionloss>` → rapier's `Multibody::frictions` (via
//! `rapier3d-mjcf`'s `add_frictionloss_to_multibody`) → the per-DoF friction
//! section of `dof_state` at build time → one `MB_JOINT_KIND_FRICTION` row per
//! non-zero DoF in `gpu_mb_init_joint_constraints`.
//!
//! Nothing calls `set_dof_frictionloss` here: the whole point is that loading
//! a model is enough.
//!
//! Run with `cargo run --release --bin test_frictionloss_mjcf --features metal`.

use khal::backend::{Backend, GpuBackend};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};
use rapier3d::prelude::*;
use rapier3d_mjcf::{MjcfLoaderOptions, MjcfMultibodyOptions, MjcfRobot};

const STEPS: usize = 15;

/// A hinge at the origin with a 1 m arm along +X, so gravity torques it.
/// `frictionloss` is substituted in.
fn model(frictionloss: f32) -> String {
    format!(
        r#"
<mujoco>
  <option gravity="0 -9.81 0"/>
  <worldbody>
    <body name="arm" pos="1 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="hinge" type="hinge" axis="0 0 1" pos="-1 0 0"
             frictionloss="{frictionloss}" damping="0"/>
      <geom type="box" size="1 0.05 0.05" density="0"/>
    </body>
  </worldbody>
</mujoco>
"#
    )
}

/// Loads the model into a `NexusState`, steps it, and returns how far the arm
/// fell plus the joint speed at the end.
async fn run_case(
    backend: &GpuBackend,
    frictionloss: f32,
) -> Result<(f32, f32), khal::backend::GpuBackendError> {
    let xml = model(frictionloss);
    let (robot, _) = MjcfRobot::from_str(&xml, MjcfLoaderOptions::default(), ".").unwrap();

    let mut state = NexusState::default();
    {
        let world = state.rbd_world_mut(0);
        robot.insert_using_multibody_joints(
            &mut world.bodies,
            &mut world.colliders,
            &mut world.multibody_joints,
            &mut world.impulse_joints,
            MjcfMultibodyOptions::empty(),
        );
    }

    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await?;

    // Confirm the value survived the loader before trusting the simulation.
    let loaded = state
        .rbd_world(0)
        .multibody_joints
        .multibodies()
        .flat_map(|mb| mb.frictions().iter().copied().collect::<Vec<_>>())
        .fold(0.0f32, f32::max);
    assert!(
        (loaded - frictionloss).abs() < 1.0e-6,
        "rapier's Multibody::frictions should carry the MJCF value: \
         got {loaded}, expected {frictionloss}"
    );

    for _ in 0..STEPS {
        pipeline.simulate(backend, &mut state, None).await?;
    }

    let rbd = state.rbd.as_ref().expect("no rbd state");
    let poses: Vec<nexus3d::rbd::math::Pose> =
        backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let dof: Vec<f32> = backend
        .slow_read_vec(rbd.multibodies().dof_state().buffer())
        .await?;
    let drop = poses
        .iter()
        .map(|p| -p.translation.y)
        .fold(0.0f32, f32::max);
    Ok((drop, dof[0].abs()))
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let backend =
        GpuBackend::Metal(khal::backend::Metal::new().map_err(|e| anyhow::anyhow!("{e:?}"))?);

    // Gravity torque about the hinge for a 1 kg arm with its centre 1 m out.
    let gravity_torque = 1.0 * 9.81 * 1.0;
    let cases = [
        ("frictionloss = 0", 0.0),
        ("frictionloss = 0.25 · τ_g", 0.25 * gravity_torque),
        ("frictionloss = 4 · τ_g", 4.0 * gravity_torque),
    ];

    let mut results = Vec::new();
    for (name, fl) in cases {
        let (drop, speed) = run_case(&backend, fl).await?;
        println!("{name:<28} drop = {drop:.6} m   |q̇| = {speed:.6} rad/s");
        results.push((drop, speed));
    }

    let (free, _) = results[0];
    let (weak, _) = results[1];
    let (locked, locked_speed) = results[2];

    println!();
    assert!(free > 0.02, "the arm should fall with no friction: {free}");
    assert!(
        weak < free * 0.95,
        "sub-gravity friction should slow the fall: {weak} vs {free}"
    );
    assert!(
        locked.abs() < 1.0e-4 && locked_speed < 1.0e-4,
        "friction above the gravity torque should hold the joint: \
         drop = {locked}, |q̇| = {locked_speed}"
    );
    println!("OK");
    Ok(())
}
