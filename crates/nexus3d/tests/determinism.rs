//! Checks that two runs of the same scene give identical states in deterministic mode.
//! Needs a GPU: `cargo test -p nexus3d --release --features rbd,mpm,metal --test determinism -- --ignored`.

use khal::backend::{Backend, GpuBackend};
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use nexus3d::rbd::math::Pose;
use rapier3d::prelude::{
    ColliderBuilder, RevoluteJointBuilder, RigidBodyBuilder, SphericalJointBuilder, TriMeshFlags,
    Vec3,
};

/// Which scene a run simulates.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Scene {
    /// Rigid-bodies only: a box pyramid, with many contacts to order and color.
    Rbd,
    /// MPM only: elastic blobs landing on a floor.
    Mpm,
    /// Both, coupled: MPM material falling on rigid boxes.
    Mixed,
    /// Multibody chains falling on the floor.
    /// Catches contact order differences reaching the multibody solver.
    Multibody,
    /// Boxes on a triangle-mesh floor: one collider pair gives one manifold per triangle.
    Mesh,
    /// The `rbd_joint_ball3` demo: a jointed cloth and boxes, 93200 bodies in total.
    /// More than the 65535 workgroups of one dispatch dimension.
    Cloth,
}

/// The state read back after one step.
#[derive(Default)]
struct Snapshot {
    body_poses: Vec<f32>,
    particles: Vec<f32>,
}

impl Snapshot {
    /// Largest difference and number of different values, or `None` if both are identical.
    fn diff(&self, other: &Self) -> Option<(f32, usize)> {
        let mine = self.body_poses.iter().chain(self.particles.iter());
        let theirs = other.body_poses.iter().chain(other.particles.iter());

        let mut max_diff = 0.0f32;
        let mut count = 0;
        for (a, b) in mine.zip(theirs) {
            if a.to_bits() != b.to_bits() {
                count += 1;
                max_diff = max_diff.max((a - b).abs());
            }
        }

        (count > 0).then_some((max_diff, count))
    }
}

fn build_rbd(state: &mut NexusState) {
    let half = 0.5f32;
    state.insert_rigid_body(
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(40.0, 1.0, 40.0).build(),
        RbdCoupling::None,
    );

    // A pyramid: bodies with several contacts, so the coloring has work to do.
    // `STACK` makes it bigger, which some GPUs need to show the ordering races.
    let stack_height: usize = std::env::var("STACK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    for i in 0..stack_height {
        for j in i..stack_height {
            for k in i..stack_height {
                let (fi, fj, fk) = (i as f32, j as f32, k as f32);
                let x = (fi * 1.25) + (fk - fi) * 2.5 - stack_height as f32 * half;
                let y = 1.0 + fi * 1.05;
                let z = (fi * 1.25) + (fj - fi) * 2.5 - stack_height as f32 * half;
                state.insert_rigid_body(
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .build(),
                    ColliderBuilder::cuboid(half, half, half).build(),
                    RbdCoupling::None,
                );
            }
        }
    }
}

fn build_mpm(state: &mut NexusState, backend: &GpuBackend) {
    let cell_width = 0.5;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let blob_radius = 2.0f32;

    let mut particles = vec![];
    let n = (blob_radius / spacing).ceil() as i32;
    for (b, young_modulus) in [5.0e5f32, 5.0e6].iter().enumerate() {
        let center = Vec3::new(b as f32 * 5.0 - 2.5, 6.0, 0.0);
        let model = ParticleModel::elastic_neo_hookean(*young_modulus, 0.3);
        for i in -n..=n {
            for j in -n..=n {
                for k in -n..=n {
                    let offset = Vec3::new(i as f32, j as f32, k as f32) * spacing;
                    if offset.length() > blob_radius {
                        continue;
                    }
                    particles.push(Particle::with_group(
                        center + offset,
                        radius,
                        1000.0,
                        model,
                        b as u32,
                    ));
                }
            }
        }
    }

    let params = SimulationParams {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
    };
    state
        .set_mpm_params(backend, params, cell_width)
        .expect("MPM params");
    state.set_mpm_substeps(10);
    state
        .add_particles(backend, particles)
        .expect("MPM particles");

    // Floor, as an MPM boundary.
    state.insert_rigid_body(
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -1.0, 0.0))
            .build(),
        ColliderBuilder::cuboid(40.0, 1.0, 20.0).build(),
        RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.5)),
    );
}

/// A few multibody chains falling on the floor.
fn build_multibody(state: &mut NexusState) {
    state.insert_rigid_body(
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(40.0, 1.0, 40.0).build(),
        RbdCoupling::None,
    );

    let num_chains = 4;
    let num_links = 6;
    let half = 0.4f32;
    for c in 0..num_chains {
        let mut parent = None;
        for l in 0..num_links {
            let handle = state.insert_rigid_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(c as f32 * 2.0, 8.0 - l as f32 * 2.0 * half, 0.0))
                    .build(),
                ColliderBuilder::cuboid(half, half, half).build(),
                RbdCoupling::None,
            );
            if let Some(parent) = parent {
                let joint = RevoluteJointBuilder::new(Vec3::Z)
                    .local_anchor1(Vec3::new(0.0, -half, 0.0))
                    .local_anchor2(Vec3::new(0.0, half, 0.0));
                state.insert_multibody_joint(parent, handle, joint);
            }
            parent = Some(handle);
        }
    }
}

/// Boxes falling on a triangle-mesh floor.
fn build_mesh(state: &mut NexusState) {
    // A flat grid of triangles: a box touches several of them, so one pair gives several manifolds.
    let mut vertices = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let n = 8i32;
    let cell = 5.0f32;
    for i in 0..=n {
        for j in 0..=n {
            vertices.push(Vec3::new(
                i as f32 * cell - 20.0,
                0.0,
                j as f32 * cell - 20.0,
            ));
        }
    }
    let stride = (n + 1) as u32;
    for i in 0..n as u32 {
        for j in 0..n as u32 {
            let v = i * stride + j;
            indices.push([v, v + 1, v + stride]);
            indices.push([v + 1, v + stride + 1, v + stride]);
        }
    }

    state.insert_rigid_body(
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::trimesh_with_flags(
            vertices,
            indices,
            TriMeshFlags::MERGE_DUPLICATE_VERTICES,
        )
        .expect("valid trimesh")
        .build(),
        RbdCoupling::None,
    );

    for i in 0..6 {
        for j in 0..6 {
            state.insert_rigid_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(i as f32 * 2.0 - 5.0, 2.0 + j as f32 * 1.5, 0.0))
                    .build(),
                ColliderBuilder::cuboid(0.5, 0.5, 0.5).build(),
                RbdCoupling::None,
            );
        }
    }
}

/// The `rbd_joint_ball3` scene: a jointed cloth with boxes falling on it.
fn build_cloth(state: &mut NexusState) {
    let rad = 0.4;
    let ni = 200;
    let nk = 301;
    let shift = 1.0;
    let center = Vec3::new(nk as f32 * shift / 2.0, 0.0, ni as f32 * shift / 2.0);

    let mut body_handles = Vec::new();
    for k in 0..nk {
        for i in 0..ni {
            let fixed = ((i == 0 || i == ni - 1) && (k % 4 == 0 || k == ni - 1))
                || ((k == 0 || k == nk - 1) && (i % 4 == 0 || i == nk - 1));
            let body = if fixed {
                RigidBodyBuilder::fixed()
            } else {
                RigidBodyBuilder::dynamic()
            }
            .translation(Vec3::new(k as f32 * shift, 0.0, i as f32 * shift) - center)
            .build();
            let collider = if fixed {
                ColliderBuilder::cuboid(rad, rad, rad).build()
            } else {
                ColliderBuilder::ball(rad).density(10.0).build()
            };

            let child = state.insert_rigid_body(body, collider, RbdCoupling::None);

            if i > 0 {
                let parent = *body_handles.last().unwrap();
                let joint = SphericalJointBuilder::new().local_anchor2(Vec3::new(0.0, 0.0, -shift));
                state.insert_impulse_joint(parent, child, joint);
            }
            if k > 0 {
                let parent = body_handles[body_handles.len() - ni];
                let joint = SphericalJointBuilder::new().local_anchor2(Vec3::new(-shift, 0.0, 0.0));
                state.insert_impulse_joint(parent, child, joint);
            }

            body_handles.push(child);
        }
    }

    let (nj, nk, ni, rad) = (10, nk / 3, ni / 6, rad * 2.5);
    for k in 0..nk {
        for i in 0..ni {
            for j in 0..nj {
                state.insert_rigid_body(
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(
                            (k as f32 - nk as f32 / 2.0) * rad * 2.1,
                            j as f32 * rad * 2.1 + 2.0,
                            (i as f32 - ni as f32 / 2.0) * rad * 2.1,
                        ))
                        .build(),
                    ColliderBuilder::cuboid(rad, rad, rad).build(),
                    RbdCoupling::None,
                );
            }
        }
    }
}

/// Number of constraint pairs sharing a dynamic body and a color (must be zero).
/// Body 0 is the fixed floor and is skipped.
fn color_violations(state: &NexusState, backend: &GpuBackend) -> usize {
    let Some(rbd) = state.rbd.as_ref() else {
        return 0;
    };

    const STATIC_SLOT: u32 = 0;
    let constraints = rbd.debug_constraint_colors(backend);
    let mut violations = 0;
    for (i, (_, a1, b1, c1, _)) in constraints.iter().enumerate() {
        for (_, a2, b2, c2, _) in &constraints[i + 1..] {
            if c1 != c2 {
                continue;
            }
            let shared = [(a1, a2), (a1, b2), (b1, a2), (b1, b2)]
                .iter()
                .any(|(x, y)| x == y && **x != STATIC_SLOT);
            if shared {
                violations += 1;
            }
        }
    }
    violations
}

async fn test_backend() -> GpuBackend {
    #[cfg(feature = "metal")]
    {
        GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap())
    }
    #[cfg(not(feature = "metal"))]
    {
        GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap())
    }
}

/// Runs `scene` for `steps` steps in deterministic mode, and reads the state after each step.
async fn run_scene(backend: &GpuBackend, scene: Scene, steps: usize) -> Vec<Snapshot> {
    let collisions = if scene == Scene::Cloth {
        350_000
    } else {
        65_536
    };
    let capacities = NexusCapacities::default().rbd_collisions(collisions);
    let mut state = NexusState::new(capacities);
    // Set before adding anything: the flag is kept when the sub-states are created.
    state.set_deterministic(backend, true);

    match scene {
        Scene::Rbd => build_rbd(&mut state),
        Scene::Mpm => build_mpm(&mut state, backend),
        Scene::Mixed => {
            build_rbd(&mut state);
            build_mpm(&mut state, backend);
        }
        Scene::Multibody => build_multibody(&mut state),
        Scene::Mesh => build_mesh(&mut state),
        Scene::Cloth => build_cloth(&mut state),
    }

    let mut pipeline = NexusPipeline::default();
    state.finalize(backend).await.unwrap();

    let mut snapshots = Vec::with_capacity(steps);
    for _ in 0..steps {
        pipeline.simulate(backend, &mut state, None).await.unwrap();

        let mut snapshot = Snapshot::default();
        if let Some(rbd) = state.rbd.as_ref() {
            let poses: Vec<Pose> = backend
                .slow_read_vec(rbd.body_poses().buffer())
                .await
                .unwrap();
            for pose in &poses {
                snapshot.body_poses.extend_from_slice(&[
                    pose.translation.x,
                    pose.translation.y,
                    pose.translation.z,
                    pose.rotation.x,
                    pose.rotation.y,
                    pose.rotation.z,
                    pose.rotation.w,
                ]);
            }
        }
        if let Some(mpm) = state.mpm.as_ref() {
            for pt in mpm.particles.read_positions(backend).await.unwrap() {
                snapshot.particles.extend_from_slice(&[pt.x, pt.y, pt.z]);
            }
        }
        snapshots.push(snapshot);
    }

    // The UI shows this counter, and a mismatch is reported as a step index:
    // both must count the same thing.
    assert_eq!(state.steps(), steps as u64, "step counter");
    assert_eq!(
        color_violations(&state, backend),
        0,
        "{scene:?}: constraints sharing a body and a color"
    );

    snapshots
}

/// Runs `scene` twice and checks both runs are identical at every step.
/// `STEPS` sets the number of steps (default 120).
fn check_determinism(scene: Scene) {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    pollster::block_on(async {
        let backend = test_backend().await;
        let first = run_scene(&backend, scene, steps).await;
        let second = run_scene(&backend, scene, steps).await;

        for (step, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            if let Some((max_diff, count)) = a.diff(b) {
                panic!(
                    "{scene:?}: mismatch at step {step}: {count} values differ, by up to {max_diff:e}"
                );
            }
        }
    });
}

#[test]
#[ignore = "needs a GPU"]
fn rbd_is_deterministic() {
    check_determinism(Scene::Rbd);
}

#[test]
#[ignore = "needs a GPU"]
fn mpm_is_deterministic() {
    check_determinism(Scene::Mpm);
}

#[test]
#[ignore = "needs a GPU"]
fn mixed_is_deterministic() {
    check_determinism(Scene::Mixed);
}

#[test]
#[ignore = "needs a GPU"]
fn multibody_is_deterministic() {
    check_determinism(Scene::Multibody);
}

#[test]
#[ignore = "needs a GPU"]
fn mesh_is_deterministic() {
    check_determinism(Scene::Mesh);
}

#[test]
#[ignore = "needs a GPU, and takes minutes (93k bodies)"]
fn cloth_is_deterministic() {
    check_determinism(Scene::Cloth);
}
