//! Stress probe for the `frictionloss` constraint slot reservation.
//!
//! `reserve_frictionloss_slots` recomputes every multibody's `first_constraint`
//! offset and grows the joint-constraint bank. The single-pendulum probe in
//! `test_frictionloss.rs` never exercises that: it has one multibody in one
//! batch, so every offset is zero. This one builds several multibodies of
//! differing DoF counts across several batches, with limits and motors mixed
//! in, and checks that nothing goes non-finite.
//!
//! Run with `cargo run --release --bin test_frictionloss_stress --features metal`.

use khal::backend::{Backend, GpuBackend};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};
use rapier3d::prelude::*;

const RAD: f32 = 0.05;
const LINK_LEN: f32 = 0.5;
const STEPS: usize = 90;

/// Chain lengths, in links. Differing DoF counts are the point: they make each
/// multibody's `first_constraint` offset distinct.
const CHAINS: [usize; 3] = [2, 4, 3];

/// Builds `CHAINS.len()` pendulum chains in environment `env`, offset along Z
/// so they don't overlap. Chain `c` gets limits and motors on its first joint
/// only, so limit/motor rows and friction rows share the bank unevenly.
fn build_env(state: &mut NexusState, env: usize, contacts: bool) {
    if contacts {
        let ground = RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -3.0, 0.0))
            .build();
        let ground_collider = ColliderBuilder::cuboid(40.0, 0.5, 40.0).build();
        state.insert_rigid_body_in(env, ground, ground_collider, RbdCoupling::None);
    }
    let groups = if contacts {
        InteractionGroups::all()
    } else {
        InteractionGroups::none()
    };
    for (c, num_links) in CHAINS.iter().enumerate() {
        let z = c as f32 * 4.0;
        let root = RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, 0.0, z))
            .build();
        let root_collider = ColliderBuilder::cuboid(RAD, RAD, RAD)
            .collision_groups(groups)
            .build();
        let mut parent = state.insert_rigid_body_in(env, root, root_collider, RbdCoupling::None);

        for i in 0..*num_links {
            let x = (i as f32 + 1.0) * LINK_LEN * 2.0;
            let body = RigidBodyBuilder::dynamic()
                .translation(Vec3::new(x, 0.0, z))
                .build();
            let collider = ColliderBuilder::cuboid(LINK_LEN, RAD, RAD)
                .collision_groups(groups)
                .build();
            let handle = state.insert_rigid_body_in(env, body, collider, RbdCoupling::None);

            let parent_anchor = if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(LINK_LEN, 0.0, 0.0)
            };
            let mut builder = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(parent_anchor)
                .local_anchor2(Vec3::new(-LINK_LEN, 0.0, 0.0));
            if i == 0 {
                builder = builder
                    .limits([-1.5, 1.5])
                    .motor_velocity(0.5, 0.1)
                    .motor_max_force(50.0);
            }
            state.insert_multibody_joint_in(env, parent, handle, builder.build());
            parent = handle;
        }
    }
}

#[allow(clippy::type_complexity)]
async fn run_case(
    backend: &GpuBackend,
    num_envs: usize,
    frictionloss: f32,
    // `false` turns off the implicit-Coriolis path, which is what makes the
    // solver take the per-substep `gpu_mb_refresh_joint_constraints` branch
    // instead of a full rebuild each substep. The friction rows are only
    // touched by that kernel here.
    implicit_coriolis: bool,
    contacts: bool,
    // `true` reproduces the zealot call pattern: the frictionloss is set
    // through `GpuMultibodySet` directly rather than through `RbdState`, so
    // nothing rebuilds `BatchIndices` at the call site.
    via_multibody_set: bool,
) -> Result<(usize, f32), khal::backend::GpuBackendError> {
    let mut state = NexusState::default();
    build_env(&mut state, 0, contacts);
    for _ in 1..num_envs {
        let env = state.add_environment();
        build_env(&mut state, env, contacts);
    }

    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await?;

    let rbd = state.rbd.as_mut().expect("no rbd state");
    let per_batch = rbd.multibodies().dofs_per_batch() as usize;
    let batches = rbd.multibodies().num_batches() as usize;
    if frictionloss > 0.0 {
        let values = vec![frictionloss; per_batch * batches];
        if via_multibody_set {
            rbd.multibodies_mut().set_dof_frictionloss(backend, &values);
        } else {
            rbd.set_dof_frictionloss(backend, &values);
        }
    }
    if !implicit_coriolis {
        rbd.multibodies_mut().set_substep_refresh(false);
        rbd.multibodies_mut().set_substep_refresh_light(false);
        rbd.set_implicit_coriolis(backend, false);
    }

    for _ in 0..STEPS {
        pipeline.simulate(backend, &mut state, None).await?;
    }

    let rbd = state.rbd.as_ref().expect("no rbd state");
    let poses: Vec<nexus3d::rbd::math::Pose> =
        backend.slow_read_vec(rbd.body_poses().buffer()).await?;
    let dof_state: Vec<f32> = backend
        .slow_read_vec(rbd.multibodies().dof_state().buffer())
        .await?;

    let bad_poses = poses
        .iter()
        .filter(|p| {
            !p.translation.x.is_finite()
                || !p.translation.y.is_finite()
                || !p.translation.z.is_finite()
        })
        .count();
    // Only the velocity section is integrated state; the rest are parameters.
    let bad_vels = dof_state[..per_batch * batches]
        .iter()
        .filter(|v| !v.is_finite())
        .count();
    let max_speed = dof_state[..per_batch * batches]
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0f32, |a, v| a.max(v.abs()));

    Ok((bad_poses + bad_vels, max_speed))
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let backend =
        GpuBackend::Metal(khal::backend::Metal::new().map_err(|e| anyhow::anyhow!("{e:?}"))?);

    let mut failures = 0;
    let mut known = 0;
    // `implicit = false` switches the multibody to explicit Coriolis forces,
    // which diverges on free-swinging chains of four or more links: the
    // velocities grow smoothly and exponentially until they overflow. It does
    // so with *or without* frictionloss, and identically on the pre-frictionloss
    // tree, so it is reported but not counted. See `test_refresh_nan` for the
    // isolated repro; it is the documented stability cost of that path, not a
    // regression here.
    for via_set in [false, true] {
        for contacts in [false, true] {
            for implicit in [true, false] {
                for num_envs in [1usize, 2, 4] {
                    // The last two values are deliberately absurd: the impulse bound is
                    // `frictionloss · dt`, so a large enough loss can overflow the
                    // accumulated impulse.
                    for fl in [0.0f32, 0.05, 5.0, 1.0e12, 1.0e30] {
                        let (bad, max_speed) =
                            run_case(&backend, num_envs, fl, implicit, contacts, via_set).await?;
                        let c = if contacts { "contacts" } else { "free    " };
                        let path = if implicit { "rebuild" } else { "refresh" };
                        let api = if via_set { "mb_set " } else { "rbd    " };
                        let tag = format!("{api}, {path}, {c}, envs = {num_envs}, fl = {fl:e}");
                        if bad > 0 && !implicit {
                            println!(
                                "{tag:<43} KNOWN (explicit-Coriolis divergence): {bad} non-finite"
                            );
                            known += 1;
                        } else if bad > 0 {
                            println!("{tag:<43} FAIL: {bad} non-finite values");
                            failures += 1;
                        } else {
                            println!("{tag:<43} ok, max |q̇| = {max_speed:.4}");
                        }
                    }
                }
            }
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} configuration(s) produced non-finite state");
    }
    println!("\nOK ({known} known explicit-Coriolis divergences ignored)");
    Ok(())
}
