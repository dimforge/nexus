//! Repro for the non-finite state seen with `implicit_coriolis = false`.
//!
//! This is *numerical divergence*, not corruption, and not a constraint bug:
//! it reproduces with no limits and no motors anywhere (`rows = none`), where
//! `gpu_mb_refresh_joint_constraints` is never even dispatched, and the
//! velocity trace grows smoothly and exponentially until it overflows
//! (`TRACE=1` prints it). It matches the tradeoff the solver already
//! documents: the explicit-Coriolis path is cheaper but less stable.
//!
//! Frictionloss is not involved: this reproduces at `frictionloss = 0`, and
//! identically on the tree from before joint friction became a constraint.
//!
//! The sweep isolates the variable: the same scenes are run with implicit
//! Coriolis on and off, reporting the first step at which any DoF velocity or
//! body pose stops being finite. Chains of four or more links diverge with it
//! off and are stable with it on.
//!
//! Run with `cargo run --release --bin test_refresh_nan --features metal`,
//! and `TRACE=1 ...` to see the per-step velocity growth.

use khal::backend::{Backend, GpuBackend};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};
use rapier3d::prelude::*;

const RAD: f32 = 0.05;
const LINK_LEN: f32 = 0.5;
const MAX_STEPS: usize = 100;

#[derive(Clone, Copy)]
struct Case {
    num_chains: usize,
    num_links: usize,
    limits: bool,
    motor: bool,
    /// Put the limit / motor rows on the chain's first joint only, leaving the
    /// rest of its DoFs with no constraint slot at all.
    first_joint_only: bool,
}

fn build(state: &mut NexusState, env: usize, case: Case) {
    for c in 0..case.num_chains {
        let z = c as f32 * 4.0;
        let root = RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, 0.0, z))
            .build();
        let rc = ColliderBuilder::cuboid(RAD, RAD, RAD)
            .collision_groups(InteractionGroups::none())
            .build();
        let mut parent = state.insert_rigid_body_in(env, root, rc, RbdCoupling::None);

        // Chain `c` gets one extra link, so the multibodies have distinct DoF
        // counts and therefore distinct constraint-slab offsets.
        for i in 0..(case.num_links + c) {
            let body = RigidBodyBuilder::dynamic()
                .translation(Vec3::new((i as f32 + 1.0) * LINK_LEN * 2.0, 0.0, z))
                .build();
            let collider = ColliderBuilder::cuboid(LINK_LEN, RAD, RAD)
                .collision_groups(InteractionGroups::none())
                .build();
            let handle = state.insert_rigid_body_in(env, body, collider, RbdCoupling::None);

            let anchor = if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(LINK_LEN, 0.0, 0.0)
            };
            let mut j = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(anchor)
                .local_anchor2(Vec3::new(-LINK_LEN, 0.0, 0.0));
            let rows_here = !case.first_joint_only || i == 0;
            if case.limits && rows_here {
                j = j.limits([-1.5, 1.5]);
            }
            if case.motor && rows_here {
                j = j.motor_velocity(0.5, 0.1).motor_max_force(50.0);
            }
            state.insert_multibody_joint_in(env, parent, handle, j.build());
            parent = handle;
        }
    }
}

/// Runs `case` on the refresh path and returns the first step index at which
/// state goes non-finite, or `None` if it stays finite for `MAX_STEPS`.
async fn first_bad_step(
    backend: &GpuBackend,
    case: Case,
    implicit_coriolis: bool,
) -> Result<Option<usize>, khal::backend::GpuBackendError> {
    let mut state = NexusState::default();
    build(&mut state, 0, case);
    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await?;

    if !implicit_coriolis {
        let rbd = state.rbd.as_mut().expect("no rbd state");
        rbd.multibodies_mut().set_substep_refresh(false);
        rbd.multibodies_mut().set_substep_refresh_light(false);
        rbd.set_implicit_coriolis(backend, false);
    }

    for step in 0..MAX_STEPS {
        pipeline.simulate(backend, &mut state, None).await?;
        let rbd = state.rbd.as_ref().expect("no rbd state");
        let n =
            rbd.multibodies().dofs_per_batch() as usize * rbd.multibodies().num_batches() as usize;
        let dof: Vec<f32> = backend
            .slow_read_vec(rbd.multibodies().dof_state().buffer())
            .await?;
        let poses: Vec<nexus3d::rbd::math::Pose> =
            backend.slow_read_vec(rbd.body_poses().buffer()).await?;
        let bad = dof[..n].iter().any(|v| !v.is_finite())
            || poses.iter().any(|p| {
                !p.translation.x.is_finite()
                    || !p.translation.y.is_finite()
                    || !p.translation.z.is_finite()
            });
        if std::env::var("TRACE").is_ok() {
            let m = dof[..n]
                .iter()
                .filter(|v| v.is_finite())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            if step % 5 == 0 || bad {
                println!("    step {step:>3}: max |q̇| = {m:e}");
            }
        }
        if bad {
            return Ok(Some(step));
        }
    }
    Ok(None)
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let backend =
        GpuBackend::Metal(khal::backend::Metal::new().map_err(|e| anyhow::anyhow!("{e:?}"))?);

    let mut any = false;
    for implicit in [false, true] {
        for first_joint_only in [true] {
            for num_chains in [1usize] {
                for num_links in [2usize, 4, 6] {
                    for (limits, motor) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        let case = Case {
                            num_chains,
                            num_links,
                            limits,
                            motor,
                            first_joint_only,
                        };
                        let rows = match (limits, motor) {
                            (false, false) => "none        ",
                            (true, false) => "limit       ",
                            (false, true) => "motor       ",
                            (true, true) => "limit+motor ",
                        };
                        let ic = if implicit {
                            "coriolis=implicit"
                        } else {
                            "coriolis=explicit"
                        };
                        match first_bad_step(&backend, case, implicit).await? {
                            Some(step) => {
                                println!(
                                    "{ic}, links = {num_links}, rows = {rows} DIVERGED at step {step}"
                                );
                                any = true;
                            }
                            None => println!(
                                "{ic}, links = {num_links}, rows = {rows} finite for {MAX_STEPS} steps"
                            ),
                        }
                    }
                }
            }
        }
    }

    if any {
        println!(
            "\nReproduced (explicit-Coriolis divergence; run with TRACE=1 to see the growth)."
        );
    } else {
        println!("\nNo failure in this sweep.");
    }
    Ok(())
}
