//! 2D version of the 3D determinism test: runs each scene twice and compares the states.
//! Needs a GPU: `cargo test -p nexus2d --release --features rbd,mpm,metal --test determinism -- --ignored`.

use khal::backend::{Backend, GpuBackend};
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use nexus2d::rbd::math::Pose;
use rapier2d::prelude::{ColliderBuilder, RigidBodyBuilder, Vec2};

/// Which scene a run simulates.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Scene {
    /// A box pyramid: plenty of resting contacts.
    Rbd,
    /// Elastic blobs landing on a floor.
    Mpm,
    /// Boxes on a polyline floor: each segment gives its own manifold.
    Mesh,
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
    state.insert_rigid_body(
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(40.0, 1.0).build(),
        RbdCoupling::None,
    );

    let half = 0.5f32;
    let stack_height: usize = std::env::var("STACK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    for i in 0..stack_height {
        for j in i..stack_height {
            let fi = i as f32;
            let fj = j as f32;
            state.insert_rigid_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vec2::new(
                        (fi * 1.25) + (fj - fi) * 2.5 - stack_height as f32 * half,
                        1.0 + fi * 1.05,
                    ))
                    .build(),
                ColliderBuilder::cuboid(half, half).build(),
                RbdCoupling::None,
            );
        }
    }
}

fn build_mpm(state: &mut NexusState, backend: &GpuBackend) {
    let cell_width = 0.2;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let blob_radius = 2.0f32;

    let mut particles = vec![];
    let n = (blob_radius / spacing).ceil() as i32;
    for (b, young_modulus) in [5.0e5f32, 5.0e6].iter().enumerate() {
        let center = Vec2::new(b as f32 * 5.0 - 2.5, 6.0);
        let model = ParticleModel::elastic_neo_hookean(*young_modulus, 0.3);
        for i in -n..=n {
            for j in -n..=n {
                let offset = Vec2::new(i as f32, j as f32) * spacing;
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

    let params = SimulationParams {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        padding: 0.0,
    };
    state
        .set_mpm_params(backend, params, cell_width)
        .expect("MPM params");
    state.set_mpm_substeps(10);
    state
        .add_particles(backend, particles)
        .expect("MPM particles");

    state.insert_rigid_body(
        RigidBodyBuilder::fixed()
            .translation(Vec2::new(0.0, -1.0))
            .build(),
        ColliderBuilder::cuboid(40.0, 1.0).build(),
        RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.5)),
    );
}

fn build_mesh(state: &mut NexusState) {
    // A bit wavy: a flat polyline gives flat segment AABBs, which the BVH builder rejects.
    let vertices: Vec<_> = (0..=20)
        .map(|i| Vec2::new(i as f32 * 2.0 - 20.0, (i % 2) as f32 * 0.1 - 0.05))
        .collect();
    state.insert_rigid_body(
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::polyline(vertices, None).build(),
        RbdCoupling::None,
    );

    for i in 0..6 {
        for j in 0..6 {
            state.insert_rigid_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vec2::new(i as f32 * 2.0 - 5.0, 2.0 + j as f32 * 1.5))
                    .build(),
                ColliderBuilder::cuboid(0.5, 0.5).build(),
                RbdCoupling::None,
            );
        }
    }
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
    let capacities = NexusCapacities::default().rbd_collisions(65_536);
    let mut state = NexusState::new(capacities);
    state.set_deterministic(backend, true);

    match scene {
        Scene::Rbd => build_rbd(&mut state),
        Scene::Mpm => build_mpm(&mut state, backend),
        Scene::Mesh => build_mesh(&mut state),
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
                    pose.rotation.sin(),
                    pose.rotation.cos(),
                ]);
            }
        }
        if let Some(mpm) = state.mpm.as_ref() {
            for pt in mpm.particles.read_positions(backend).await.unwrap() {
                snapshot.particles.extend_from_slice(&[pt.x, pt.y]);
            }
        }
        snapshots.push(snapshot);
    }

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
fn mesh_is_deterministic() {
    check_determinism(Scene::Mesh);
}
