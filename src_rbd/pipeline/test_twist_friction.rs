use super::{RbdCapacities, RbdPipeline, RbdState};
use crate::dynamics::FrictionModel;
use crate::math::Pose;
use crate::rapier::prelude::*;
use crate::shaders::dynamics::{RbdSimParams, Velocity};
use khal::backend::{Backend, GpuBackend, WebGpu};

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn twist_friction_stops_loaded_spin_on_small_and_large_paths() {
    for (count, bindings) in [(1, 8), (4096, 8), (4096, 10), (4096, 16)] {
        let backend = GpuBackend::WebGpu(
            WebGpu::new(
                Default::default(),
                khal::re_exports::wgpu::Limits {
                    max_storage_buffers_per_shader_stage: bindings,
                    ..Default::default()
                },
            )
            .await
            .unwrap(),
        );
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let ground =
            bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(100.0, 0.5, 100.0).friction(0.8),
            ground,
            &mut bodies,
        );
        for i in 0..count {
            let body = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .can_sleep(false)
                    .translation(Vec3::new((i % 64) as f32, 0.21, (i / 64) as f32))
                    .angvel(Vec3::Y * 6.0),
            );
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.2, 0.2, 0.2)
                    .friction(if i % 2 == 0 { 0.8 } else { 0.0 })
                    .friction_combine_rule(CoefficientCombineRule::Min),
                body,
                &mut bodies,
            );
        }
        let mut params = RbdSimParams::tgs_soft();
        params.friction_model = FrictionModel::Simplified;
        params.warmstart_coefficient = if bindings == 16 { 1.0 } else { 0.75 };
        let mut state = RbdState::from_rapier(
            &backend,
            &[(
                &bodies,
                &colliders,
                &ImpulseJointSet::new(),
                &MultibodyJointSet::new(),
                &params,
            )],
            RbdCapacities {
                collisions_capacity: (count * 2).max(256),
                ..Default::default()
            },
        );
        let pipeline = RbdPipeline::new(&backend).unwrap();
        for step in 0..180 {
            // Both directions exercise stale-contact invalidation and graph changes.
            if step == 60 {
                state.set_friction_model(&backend, FrictionModel::Coulomb);
            }
            if step == 80 {
                state.set_friction_model(&backend, FrictionModel::Simplified);
            }
            pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
            pipeline.step(&backend, &mut state, None).unwrap();
            if [20, 70, 90].contains(&step) {
                let plan: Vec<crate::shaders::broad_phase::ContactPlan> = backend
                    .slow_read_vec(state.contact_plan.buffer())
                    .await
                    .unwrap();
                // step swaps buffers: old_constraints now holds the just-solved frame.
                let old = &state.old_constraints;
                if state.friction_model() == FrictionModel::Simplified {
                    assert!(old.tiles.coulomb().is_none());
                    let links: Vec<crate::shaders::dynamics::ContactLink> =
                        backend.slow_read_vec(old.links.buffer()).await.unwrap();
                    let constraint_indices: Vec<u32> = backend
                        .slow_read_vec(old.constraint_indices.buffer())
                        .await
                        .unwrap();
                    let tiles: Vec<crate::shaders::dynamics::twist_tiles::TwistTile> = backend
                        .slow_read_vec(old.tiles.simplified().unwrap().buffer())
                        .await
                        .unwrap();
                    let active: Vec<_> = (0..plan[0].bound as usize)
                        .filter(|&i| links[i].len > 0)
                        .map(|i| {
                            let (tile, lane) = crate::shaders::dynamics::contact_tiles::tile_lane(
                                constraint_indices[i] as usize,
                            );
                            tiles[tile].constraint(lane)
                        })
                        .collect();
                    assert!(!active.is_empty());
                    for c in &active {
                        let points = &c.points[..c.len as usize];
                        let twist_limit = c.limit
                            * points
                                .iter()
                                .map(|p| p.normal_impulse * p.radius)
                                .sum::<f32>();
                        let tangent_limit =
                            c.limit * points.iter().map(|p| p.normal_impulse).sum::<f32>();
                        assert!(c.friction.twist_impulse.abs() <= twist_limit + 1e-4);
                        assert!(c.friction.tangent_impulse.length() <= tangent_limit + 1e-4);
                    }
                } else {
                    assert!(old.tiles.simplified().is_none());
                    let snapshots = old
                        .read_impulses(&backend, plan[0].bound as usize)
                        .await
                        .unwrap();
                    assert!(snapshots.iter().any(|c| c.len > 0));
                }
            }
        }
        backend.synchronize().unwrap();
        assert_eq!(state.friction_model(), FrictionModel::Simplified);
        let velocities: Vec<Velocity> = backend.slow_read_vec(state.vels().buffer()).await.unwrap();
        let poses: Vec<Pose> = backend
            .slow_read_vec(state.body_poses().buffer())
            .await
            .unwrap();
        for i in 0..count as usize {
            let v = velocities[i + 1];
            let p = poses[i + 1];
            assert!(p.translation.is_finite() && v.angular.is_finite());
            assert!(
                (p.translation.y - 0.2).abs() < 0.015,
                "bindings={bindings} box={i} height={}",
                p.translation.y
            );
            if i % 2 == 0 {
                assert!(
                    v.angular.length() < 0.1,
                    "bindings={bindings} box={i} angular={:?}",
                    v.angular
                );
            } else {
                assert!(
                    v.angular.y > 5.0,
                    "frictionless box {i} lost spin: {:?}",
                    v.angular
                );
            }
        }
    }
}
