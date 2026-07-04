//! Headless benchmark mirroring the `joints3` demo (multibody variant).
//!
//! Builds the same scene as `crates/examples3d/joints3.rs` with
//! `use_articulations = true`: prismatic / revolute / fixed / spherical joint
//! chains, including motors and limits, all wired through `MultibodyJointSet`.
//! No rendering, just timed stepping.
//!
//! Usage:
//!
//! ```text
//! cargo run -p nexus_examples_3d --release \
//!     --bin bench_joints3 -- [num_batches] [num_warmup] [num_iters] [num_substeps]
//! ```
//!
//! Defaults: 1 batch, 10 warmup steps, 200 timed steps, 4 substeps.

use std::collections::HashMap;
use std::time::Duration;

use khal::backend::GpuBackend as KhalGpuBackend;
use khal::backend::WebGpu;
use khal::re_exports::wgpu;
use nexus_viewer3d::SimulationState;
use nexus_viewer3d::nexus::rbd::dynamics::RbdSimParams;
use nexus_viewer3d::rbd::BatchEnvironment;
use nexus_viewer3d::rbd::GpuBackend;
use nexus_viewer3d::rbd::backend::SimulationBackend;
use rapier3d::prelude::*;

fn create_prismatic_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
    num: usize,
) {
    let rad = 0.4;
    let shift = 2.0;

    let ground = RigidBodyBuilder::fixed().translation(origin);
    let mut curr_parent = bodies.insert(ground);
    let collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(collider, curr_parent, bodies);

    for i in 0..num {
        let z = origin.z + (i + 1) as f32 * shift;
        let rb = RigidBodyBuilder::dynamic().translation(Vector::new(origin.x, origin.y, z));
        let curr_child = bodies.insert(rb);
        let collider = ColliderBuilder::cuboid(rad, rad, rad);
        colliders.insert_with_parent(collider, curr_child, bodies);

        let axis = if i % 2 == 0 {
            Vector::new(1.0f32, 1.0, 0.0).normalize()
        } else {
            Vector::new(-1.0f32, 1.0, 0.0).normalize()
        };

        let prism = PrismaticJointBuilder::new(axis)
            .local_anchor1(Vector::new(0.0, 0.0, 0.0))
            .local_anchor2(Vector::new(0.0, 0.0, -shift))
            .limits([-2.0, 2.0]);

        multibody_joints.insert(curr_parent, curr_child, prism, true);
        curr_parent = curr_child;
    }
}

fn create_actuated_prismatic_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
    num: usize,
) {
    let rad = 0.4;
    let shift = 2.0;

    let ground = RigidBodyBuilder::fixed().translation(origin);
    let mut curr_parent = bodies.insert(ground);
    let collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(collider, curr_parent, bodies);

    for i in 0..num {
        let z = origin.z + (i + 1) as f32 * shift;
        let rb = RigidBodyBuilder::dynamic().translation(Vector::new(origin.x, origin.y, z));
        let curr_child = bodies.insert(rb);
        let collider = ColliderBuilder::cuboid(rad, rad, rad);
        colliders.insert_with_parent(collider, curr_child, bodies);

        let axis = if i % 2 == 0 {
            Vector::new(1.0, 1.0, 0.0).normalize()
        } else {
            Vector::new(-1.0, 1.0, 0.0).normalize()
        };

        let mut prism = PrismaticJointBuilder::new(axis)
            .local_anchor1(Vector::new(0.0, 0.0, shift))
            .local_anchor2(Vector::new(0.0, 0.0, 0.0))
            .build();

        if i == 0 {
            prism
                .set_motor_velocity(2.0, 1.0e5)
                .set_limits([-2.0, 5.0])
                .set_motor_max_force(100.0);
        } else if i == 1 {
            prism
                .set_limits([-Real::MAX, 5.0])
                .set_motor_velocity(6.0, 1.0e3)
                .set_motor_max_force(100.0);
        } else if i > 1 {
            prism
                .set_motor_position(2.0, 1.0e3, 1.0e2)
                .set_motor_max_force(60.0);
        }

        multibody_joints.insert(curr_parent, curr_child, prism, true);
        curr_parent = curr_child;
    }
}

fn create_revolute_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
    num: usize,
) {
    let rad = 0.4;
    let shift = 2.0;

    let ground = RigidBodyBuilder::fixed().translation(Vector::new(origin.x, origin.y, 0.0));
    let mut curr_parent = bodies.insert(ground);
    let collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(collider, curr_parent, bodies);

    for i in 0..num {
        let z = origin.z + i as f32 * shift * 2.0 + shift;
        let positions = [
            Pose::from_translation(Vector::new(origin.x, origin.y, z)),
            Pose::from_translation(Vector::new(origin.x + shift, origin.y, z)),
            Pose::from_translation(Vector::new(origin.x + shift, origin.y, z + shift)),
            Pose::from_translation(Vector::new(origin.x, origin.y, z + shift)),
        ];
        let mut handles = [curr_parent; 4];
        for k in 0..4 {
            let rb = RigidBodyBuilder::dynamic().pose(positions[k]);
            handles[k] = bodies.insert(rb);
            let collider = ColliderBuilder::cuboid(rad, rad, rad);
            colliders.insert_with_parent(collider, handles[k], bodies);
        }
        let x = Vector::X;
        let z = Vector::Z;
        let revs = [
            RevoluteJointBuilder::new(z).local_anchor2(Vector::new(0.0, 0.0, -shift)),
            RevoluteJointBuilder::new(x).local_anchor2(Vector::new(-shift, 0.0, 0.0)),
            RevoluteJointBuilder::new(z).local_anchor2(Vector::new(0.0, 0.0, -shift)),
            RevoluteJointBuilder::new(x).local_anchor2(Vector::new(shift, 0.0, 0.0)),
        ];
        multibody_joints.insert(curr_parent, handles[0], revs[0], true);
        multibody_joints.insert(handles[0], handles[1], revs[1], true);
        multibody_joints.insert(handles[1], handles[2], revs[2], true);
        multibody_joints.insert(handles[2], handles[3], revs[3], true);
        curr_parent = handles[3];
    }
}

fn create_revolute_joints_with_limits(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
) {
    let origin_v = origin;
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(origin_v));
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), ground, bodies);

    let shift = Vector::new(0.0, 0.0, 6.0);
    let platform1 = bodies.insert(RigidBodyBuilder::dynamic().translation(origin_v + shift));
    colliders.insert_with_parent(ColliderBuilder::cuboid(4.0, 0.2, 2.0), platform1, bodies);

    let platform2 = bodies.insert(RigidBodyBuilder::dynamic().translation(origin_v + shift * 2.0));
    colliders.insert_with_parent(ColliderBuilder::cuboid(4.0, 0.2, 2.0), platform2, bodies);

    let z = Vector::Z;
    let joint1 = RevoluteJointBuilder::new(z)
        .local_anchor1(shift)
        .limits([-0.2, 0.2]);
    multibody_joints.insert(ground, platform1, joint1, true);

    let joint2 = RevoluteJointBuilder::new(z)
        .local_anchor2(-shift)
        .limits([-0.2, 0.2]);
    multibody_joints.insert(platform1, platform2, joint2, true);

    let cuboid_body1 = bodies.insert(
        RigidBodyBuilder::dynamic().translation(origin_v + shift + Vector::new(-2.0, 4.0, 0.0)),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.6, 0.6, 0.6).friction(1.0),
        cuboid_body1,
        bodies,
    );
    let cuboid_body2 = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(origin_v + shift * 2.0 + Vector::new(2.0, 16.0, 0.0)),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.6, 0.6, 0.6).friction(1.0),
        cuboid_body2,
        bodies,
    );
}

fn create_fixed_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    impulse_joints: &mut ImpulseJointSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
    num: usize,
) {
    let rad = 0.4;
    let shift = 1.0;
    let mut body_handles = Vec::new();

    for i in 0..num {
        for k in 0..num {
            let fk = k as f32;
            let fi = i as f32;
            let status = if i == 0 && (k % 4 == 0 && k != num - 2 || k == num - 1) {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };
            let rb = RigidBodyBuilder::new(status).translation(Vector::new(
                origin.x + fk * shift,
                origin.y,
                origin.z + fi * shift,
            ));
            let child = bodies.insert(rb);
            let collider = ColliderBuilder::ball(rad);
            colliders.insert_with_parent(collider, child, bodies);

            if i > 0 {
                let parent_index = body_handles.len() - num;
                let parent_handle = body_handles[parent_index];
                let joint = FixedJointBuilder::new().local_anchor2(Vector::new(0.0, 0.0, -shift));
                multibody_joints.insert(parent_handle, child, joint, true);
            }

            if k > 0 {
                let parent_index = body_handles.len() - 1;
                let parent_handle = body_handles[parent_index];
                let joint = FixedJointBuilder::new().local_anchor2(Vector::new(-shift, 0.0, 0.0));
                impulse_joints.insert(parent_handle, child, joint, true);
            }

            body_handles.push(child);
        }
    }
}

fn create_spherical_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    impulse_joints: &mut ImpulseJointSet,
    multibody_joints: &mut MultibodyJointSet,
    num: usize,
) {
    let rad = 0.4;
    let shift = 1.0;
    let mut body_handles = Vec::new();

    for k in 0..num {
        for i in 0..num {
            let fk = k as f32;
            let fi = i as f32;
            let status = if i == 0 && (k % 4 == 0 || k == num - 1) {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };
            let rb = RigidBodyBuilder::new(status).translation(Vector::new(
                fk * shift,
                0.0,
                fi * shift * 2.0,
            ));
            let child = bodies.insert(rb);
            let collider = ColliderBuilder::capsule_z(rad * 1.25, rad);
            colliders.insert_with_parent(collider, child, bodies);

            if i > 0 {
                let parent = *body_handles.last().unwrap();
                let joint =
                    SphericalJointBuilder::new().local_anchor2(Vector::new(0.0, 0.0, -shift * 2.0));
                multibody_joints.insert(parent, child, joint, true);
            }
            if k > 0 {
                let parent = body_handles[body_handles.len() - num];
                let joint =
                    SphericalJointBuilder::new().local_anchor2(Vector::new(-shift, 0.0, 0.0));
                impulse_joints.insert(parent, child, joint, true);
            }
            body_handles.push(child);
        }
    }
}

fn create_spherical_joints_with_limits(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
) {
    let shift = Vector::new(0.0, 0.0, 3.0);
    let origin_v = origin;
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(origin_v));
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), ground, bodies);

    let ball1 = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(origin_v + shift)
            .linvel(Vector::new(20.0, 20.0, 0.0)),
    );
    colliders.insert_with_parent(ColliderBuilder::cuboid(1.0, 1.0, 1.0), ball1, bodies);

    let ball2 = bodies.insert(RigidBodyBuilder::dynamic().translation(origin_v + shift * 2.0));
    colliders.insert_with_parent(ColliderBuilder::cuboid(1.0, 1.0, 1.0), ball2, bodies);

    let joint1 = SphericalJointBuilder::new()
        .local_anchor2(-shift)
        .limits(JointAxis::LinX, [-0.2, 0.2])
        .limits(JointAxis::LinY, [-0.2, 0.2]);
    let joint2 = SphericalJointBuilder::new()
        .local_anchor2(-shift)
        .limits(JointAxis::LinX, [-0.3, 0.3])
        .limits(JointAxis::LinY, [-0.3, 0.3]);

    multibody_joints.insert(ground, ball1, joint1, true);
    multibody_joints.insert(ball1, ball2, joint2, true);
}

fn create_actuated_revolute_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
    num: usize,
) {
    let rad = 0.4;
    let shift = 2.0;
    let z = Vector::Z;
    let joint_template = RevoluteJointBuilder::new(z).local_anchor2(Vector::new(0.0, 0.0, -shift));
    let mut parent_handle = RigidBodyHandle::invalid();

    for i in 0..num {
        let fi = i as f32;
        let status = if i == 0 {
            RigidBodyType::Fixed
        } else {
            RigidBodyType::Dynamic
        };
        let shifty = (i >= 1) as u32 as f32 * -2.0;
        let rb = RigidBodyBuilder::new(status).translation(Vector::new(
            origin.x,
            origin.y + shifty,
            origin.z + fi * shift,
        ));
        let child = bodies.insert(rb);
        let collider = ColliderBuilder::cuboid(rad * 2.0, rad * 6.0 / (fi + 1.0), rad);
        colliders.insert_with_parent(collider, child, bodies);

        if i > 0 {
            let mut joint = joint_template.motor_model(MotorModel::AccelerationBased);
            if i % 3 == 1 {
                joint = joint.motor_velocity(-20.0, 100.0);
            } else if i == num - 1 {
                joint = joint.motor_position(std::f32::consts::FRAC_PI_2, 200.0, 100.0);
            }
            if i == 1 {
                joint = joint
                    .local_anchor2(Vector::new(0.0, 2.0, -shift))
                    .motor_velocity(-2.0, 1000.0);
            }
            multibody_joints.insert(parent_handle, child, joint, true);
        }
        parent_handle = child;
    }
}

fn create_actuated_spherical_joints(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    multibody_joints: &mut MultibodyJointSet,
    origin: Vector,
    num: usize,
) {
    let rad = 0.4;
    let shift = 2.0;
    let joint_template = SphericalJointBuilder::new().local_anchor1(Vector::new(0.0, 0.0, shift));
    let mut parent_handle = RigidBodyHandle::invalid();

    for i in 0..num {
        let fi = i as f32;
        let status = if i == 0 {
            RigidBodyType::Fixed
        } else {
            RigidBodyType::Dynamic
        };
        let rb = RigidBodyBuilder::new(status).translation(Vector::new(
            origin.x,
            origin.y,
            origin.z + fi * shift,
        ));
        let child = bodies.insert(rb);
        let collider = ColliderBuilder::capsule_y(rad * 2.0 / (fi + 1.0), rad);
        colliders.insert_with_parent(collider, child, bodies);

        if i > 0 {
            let mut joint = joint_template;
            if i == 1 {
                joint = joint
                    .motor_velocity(JointAxis::AngX, 0.0, 0.1)
                    .motor_velocity(JointAxis::AngY, 0.5, 0.1)
                    .motor_velocity(JointAxis::AngZ, -2.0, 0.1);
            } else if i == num - 1 {
                joint = joint
                    .motor_position(JointAxis::AngX, 0.0, 0.2, 1.0)
                    .motor_position(JointAxis::AngY, 1.0, 0.2, 1.0)
                    .motor_position(JointAxis::AngZ, std::f32::consts::FRAC_PI_2, 0.2, 1.0);
            }
            multibody_joints.insert(parent_handle, child, joint, true);
        }
        parent_handle = child;
    }
}

fn build_one_batch(num_substeps: u32) -> BatchEnvironment {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    create_prismatic_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(20.0, 5.0, 0.0),
        4,
    );
    create_actuated_prismatic_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(25.0, 5.0, 0.0),
        4,
    );
    create_revolute_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(20.0, 0.0, 0.0),
        3,
    );
    create_revolute_joints_with_limits(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(34.0, 0.0, 0.0),
    );
    create_fixed_joints(
        &mut bodies,
        &mut colliders,
        &mut impulse_joints,
        &mut multibody_joints,
        Vector::new(0.0, 10.0, 0.0),
        10,
    );
    create_actuated_revolute_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(20.0, 10.0, 0.0),
        6,
    );
    create_actuated_spherical_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(13.0, 10.0, 0.0),
        3,
    );
    create_spherical_joints(
        &mut bodies,
        &mut colliders,
        &mut impulse_joints,
        &mut multibody_joints,
        9,
    );
    create_spherical_joints_with_limits(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        Vector::new(-5.0, 0.0, 0.0),
    );

    let mut sim_params = RbdSimParams::default();
    sim_params.num_solver_iterations = num_substeps;
    BatchEnvironment {
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        sim_params,
        visuals: HashMap::new(),
    }
}

fn build_scene(num_batches: usize, num_substeps: u32) -> SimulationState {
    let envs = (0..num_batches.max(1))
        .map(|_| build_one_batch(num_substeps))
        .collect();
    SimulationState::from_environments(envs)
}

struct Sample {
    label: &'static str,
    per_step_avg: Duration,
    per_step_p50: Duration,
    per_step_min: Duration,
    per_step_max: Duration,
}

impl Sample {
    fn fmt_us(d: Duration) -> String {
        format!("{:>10.2} µs", d.as_secs_f64() * 1.0e6)
    }

    fn print(&self) {
        println!(
            "  {:<10}  avg {}  p50 {}  min {}  max {}",
            self.label,
            Self::fmt_us(self.per_step_avg),
            Self::fmt_us(self.per_step_p50),
            Self::fmt_us(self.per_step_min),
            Self::fmt_us(self.per_step_max),
        );
    }
}

async fn bench_backend(
    label: &'static str,
    backend: &KhalGpuBackend,
    state: &SimulationState,
    n_warmup: usize,
    n_iters: usize,
) -> Sample {
    let mut phys = GpuBackend::try_new(backend, state)
        .await
        .unwrap_or_else(|e| panic!("{label} backend init failed: {e}"));

    for _ in 0..n_warmup {
        let _ = phys.step(None).await;
    }

    let mut samples = Vec::with_capacity(n_iters);
    let mut last_stats = None;
    for i in 0..n_iters {
        let stats = phys.step(None).await;
        samples.push(stats.total_simulation_time_without_readback);
        if i == n_iters - 1 {
            last_stats = Some(stats);
        }
    }
    samples.sort();

    if let Some(stats) = last_stats {
        if !stats.gpu_pass_times.is_empty() {
            let mut passes = stats.gpu_pass_times.clone();
            passes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!(
                "    top passes ({} total, {:.3} ms):",
                passes.len(),
                stats.gpu_total_time
            );
            for (l, ms) in passes.iter().take(15) {
                println!("      {:>9.3} ms  {}", ms, l);
            }
        }
    }

    let total: Duration = samples.iter().sum();
    let per_step_avg = total / (n_iters as u32);
    let per_step_p50 = samples[n_iters / 2];
    let per_step_min = *samples.first().unwrap();
    let per_step_max = *samples.last().unwrap();

    Sample {
        label,
        per_step_avg,
        per_step_p50,
        per_step_min,
        per_step_max,
    }
}

async fn webgpu_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_000_000_000,
        max_storage_buffer_binding_size: 1_000_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut webgpu = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("Failed to initialize WebGPU backend");
    webgpu.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(webgpu)
}

async fn run(num_batches: usize, n_warmup: usize, n_iters: usize, num_substeps: u32) {
    println!(
        "Joints3 multibody benchmark — {num_batches} batches, num_substeps={num_substeps}, \
         {n_warmup} warmup + {n_iters} timed steps"
    );
    let state = build_scene(num_batches, num_substeps);
    let webgpu = webgpu_backend().await;
    let s = bench_backend("WebGPU", &webgpu, &state, n_warmup, n_iters).await;
    s.print();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_batches = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_warmup = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let num_substeps = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4u32);
    pollster::block_on(run(num_batches, n_warmup, n_iters, num_substeps));
}
