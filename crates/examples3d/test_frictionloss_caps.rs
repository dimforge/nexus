//! Checks that the `BatchIndices` uniform agrees with the joint-constraint
//! buffer sizes after `set_dof_frictionloss` grows them.
//!
//! Reserving the dry-friction slots reallocates `joint_constraints` /
//! `joint_constraint_columns` and changes their per-batch capacities. Those
//! capacities are mirrored into the shared `BatchIndices` uniform, which the
//! kernels use to locate each multibody's slab. If the uniform is not
//! re-uploaded, every kernel indexes the resized buffers with stale strides.
//!
//! This asserts on the uniform directly rather than watching for NaN, which
//! only shows up once the resulting garbage happens to be large.
//!
//! Run with `cargo run --release --bin test_frictionloss_caps --features metal`.

use khal::backend::{Backend, GpuBackend};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};
use nexus3d::rbd::shaders::utils::BatchIndices;
use rapier3d::prelude::*;

const RAD: f32 = 0.05;
const LINK_LEN: f32 = 0.5;

fn build_env(state: &mut NexusState, env: usize) {
    for (c, num_links) in [2usize, 4, 3].iter().enumerate() {
        let z = c as f32 * 4.0;
        let root = RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, 0.0, z))
            .build();
        let rc = ColliderBuilder::cuboid(RAD, RAD, RAD)
            .collision_groups(InteractionGroups::none())
            .build();
        let mut parent = state.insert_rigid_body_in(env, root, rc, RbdCoupling::None);
        for i in 0..*num_links {
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
            let joint = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(anchor)
                .local_anchor2(Vec3::new(-LINK_LEN, 0.0, 0.0))
                .limits([-1.5, 1.5])
                .motor_velocity(0.5, 0.1)
                .motor_max_force(50.0);
            state.insert_multibody_joint_in(env, parent, handle, joint.build());
            parent = handle;
        }
    }
}

/// `(uniform capacity, actual capacity)` for the joint-constraint bank and its
/// column buffer, after `steps` pipeline steps.
async fn caps(
    backend: &GpuBackend,
    via_multibody_set: bool,
    steps: usize,
) -> Result<((u32, u32), (u32, u32)), khal::backend::GpuBackendError> {
    let mut state = NexusState::default();
    build_env(&mut state, 0);
    for _ in 1..2 {
        let env = state.add_environment();
        build_env(&mut state, env);
    }
    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await?;

    let rbd = state.rbd.as_mut().expect("no rbd state");
    let n = rbd.multibodies().dofs_per_batch() as usize * rbd.multibodies().num_batches() as usize;
    let values = vec![0.1f32; n];
    if via_multibody_set {
        rbd.multibodies_mut().set_dof_frictionloss(backend, &values);
    } else {
        rbd.set_dof_frictionloss(backend, &values);
    }

    for _ in 0..steps {
        pipeline.simulate(backend, &mut state, None).await?;
    }

    let rbd = state.rbd.as_ref().expect("no rbd state");
    let bi: Vec<BatchIndices> = backend.slow_read_vec(rbd.batch_indices().buffer()).await?;
    let bi = bi[0];
    Ok((
        (
            bi.mb_joint_constraints_batch_capacity,
            rbd.multibodies().joint_constraints_per_batch(),
        ),
        (
            bi.mb_joint_constraint_columns_batch_capacity,
            rbd.multibodies().joint_constraint_columns_per_batch(),
        ),
    ))
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let backend =
        GpuBackend::Metal(khal::backend::Metal::new().map_err(|e| anyhow::anyhow!("{e:?}"))?);

    let mut bad = 0;
    for via_set in [false, true] {
        for steps in [0usize, 1] {
            let (rows, cols) = caps(&backend, via_set, steps).await?;
            let api = if via_set { "mb_set" } else { "rbd   " };
            let ok = rows.0 == rows.1 && cols.0 == cols.1;
            println!(
                "{api}, after {steps} step(s): rows uniform/actual = {}/{}, cols = {}/{}  {}",
                rows.0,
                rows.1,
                cols.0,
                cols.1,
                if ok { "ok" } else { "MISMATCH" }
            );
            // Before any step has run, the `mb_set` entry point is expected to
            // carry a stale uniform: the step is what re-uploads it.
            if !ok && steps > 0 {
                bad += 1;
            }
        }
    }

    if bad > 0 {
        anyhow::bail!("{bad} case(s) still stale after a step");
    }
    println!("\nOK");
    Ok(())
}
