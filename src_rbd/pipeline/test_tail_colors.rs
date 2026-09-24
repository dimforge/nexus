//! Compares fused color tails with dispatch-by-dispatch GPU execution.
use crate::dynamics::contact_kernels::{
    ContactTiles, GpuSolveConstraints, GpuSolveConstraintsBiased, GpuSolveConstraintsTail,
    GpuSolveConstraintsTailBiased, GpuSolveConstraintsTailUnbiased,
    GpuSolveConstraintsTailUnbiasedCached, GpuSolveConstraintsUnbiased,
    GpuSolveConstraintsUnbiasedCached,
};
use crate::math::Pose;
use crate::shaders::dynamics::contact_tiles::{TAIL_COLOR, TILE_LEN, tile_lane};
use crate::shaders::dynamics::coulomb_tiles::CoulombTile;
use crate::shaders::dynamics::twist_tiles::TwistTile;
use crate::shaders::dynamics::*;
use crate::shaders::utils::BatchIndices;
use glamx::{Vec2, Vec3};
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use vortx::tensor::Tensor;

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn tail_colors_match_separate_color_dispatches() {
    for model in [FrictionModel::Coulomb, FrictionModel::Simplified] {
        check_tail_colors(model).await;
    }
}

async fn check_tail_colors(model: FrictionModel) {
    let backend = GpuBackend::WebGpu(
        WebGpu::new(Default::default(), Default::default())
            .await
            .unwrap(),
    );
    let dir = &crate::SPIRV_DIR;
    let biased = GpuSolveConstraintsBiased::from_dir(&backend, dir).unwrap();
    let generic = GpuSolveConstraints::from_dir(&backend, dir).unwrap();
    let warm = GpuSolveConstraintsUnbiasedCached::from_dir(&backend, dir).unwrap();
    let final_iteration = GpuSolveConstraintsUnbiased::from_dir(&backend, dir).unwrap();
    let tail_biased = GpuSolveConstraintsTailBiased::from_dir(&backend, dir).unwrap();
    let tail_generic = GpuSolveConstraintsTail::from_dir(&backend, dir).unwrap();
    let tail_warm = GpuSolveConstraintsTailUnbiasedCached::from_dir(&backend, dir).unwrap();
    let tail_final = GpuSolveConstraintsTailUnbiased::from_dir(&backend, dir).unwrap();
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let params = Tensor::scalar(&backend, RbdSimParams::tgs_soft(), BufferUsages::UNIFORM).unwrap();
    let colors: Vec<_> = (0..=64)
        .map(|c| Tensor::scalar(&backend, c, BufferUsages::UNIFORM).unwrap())
        .collect();
    // Alternate disjoint matchings of a body ring. Consecutive colors share bodies,
    // so missing inter-color visibility changes the result. Each occupied color has
    // more than 64 constraints; empty colors and an incomplete final tile are also covered.
    let mut input = Vec::new();
    let mut boundaries = vec![0u32];
    for color in 1..=64u32 {
        if [TAIL_COLOR, TAIL_COLOR + 1, 63, 64].contains(&color) {
            for index in 0..95u32 {
                let a = (index * 2 + color % 2) % 192;
                let b = (a + 1) % 192;
                let mut c = TwoBodyConstraint::default();
                c.len = 1 + index % 4;
                c.dir_a = Vec3::Y;
                c.tangent_a = Vec3::Z;
                c.limit = 0.7;
                c.im_a = Vec3::splat(0.4);
                c.im_b = Vec3::splat(0.6);
                c.ii_a.diag = Vec3::splat(0.2);
                c.ii_b.diag = Vec3::splat(0.3);
                c.solver_body_a = a;
                c.solver_body_b = b;
                c.vel_slot_a = a + if a % 7 == 0 { 192 } else { 0 };
                c.vel_slot_b = b + if b % 7 == 0 { 192 } else { 0 };
                for k in 0..c.len as usize {
                    let p = &mut c.points[k];
                    p.r_a = Vec3::new(k as f32 * 0.07, -0.2, 0.1);
                    p.r_b = Vec3::new(k as f32 * 0.07, 0.2, 0.1);
                    p.local_pt_a = p.r_a;
                    p.local_pt_b = p.r_b;
                    p.dist = -0.01;
                    p.normal_impulse = 0.04 * (k + 1) as f32;
                    p.tangent_impulse = Vec2::new(0.01, -0.02);
                }
                c.compute_effective_masses();
                input.push(c);
            }
        }
        boundaries.push(input.len() as u32);
    }
    boundaries.push(input.len() as u32);
    let poses = Tensor::vector(&backend, vec![Pose::IDENTITY; 192], storage).unwrap();
    let initial_vels: Vec<_> = (0..384)
        .map(|i| Velocity {
            linear: Vec3::new((i % 7) as f32 * 0.03, -0.1, 0.07),
            angular: Vec3::new(0.1, 0.03, (i % 3) as f32 * 0.02),
            ..Default::default()
        })
        .collect();
    for nb in [1u32, 3] {
        let buckets = Tensor::vector(
            &backend,
            boundaries
                .iter()
                .flat_map(|&end| vec![end; nb as usize])
                .collect::<Vec<_>>(),
            storage,
        )
        .unwrap();
        let ids = Tensor::scalar(
            &backend,
            BatchIndices {
                num_batches: nb,
                solver_color_buckets_stride: 67,
                ..Default::default()
            },
            BufferUsages::UNIFORM,
        )
        .unwrap();
        for variant in 0..4 {
            let mut results = Vec::new();
            for fused in [false, true] {
                // The constraints in tiles, zeroed so that their unused points compare equal.
                let num_tiles = input.len().div_ceil(TILE_LEN);
                let mut tiles = if model == FrictionModel::Simplified {
                    let compact: Vec<_> = input
                        .iter()
                        .map(|c| {
                            let mut t = TwistConstraint::default();
                            t.dir_a = c.dir_a;
                            t.tangent_a = c.tangent_a;
                            t.len = c.len;
                            t.limit = c.limit;
                            t.im_a = c.im_a;
                            t.im_b = c.im_b;
                            t.ii_a = c.ii_a;
                            t.ii_b = c.ii_b;
                            t.solver_body_a = c.solver_body_a;
                            t.solver_body_b = c.solver_body_b;
                            t.vel_slot_a = c.vel_slot_a;
                            t.vel_slot_b = c.vel_slot_b;
                            t.offset_b = c.points[0].r_b - c.points[0].r_a;
                            t.frame_a = glamx::Quat::from_rotation_x(0.13);
                            t.frame_b = glamx::Quat::from_rotation_y(-0.08);
                            for k in 0..t.len as usize {
                                let p = c.points[k];
                                t.points[k] = TwistContactPoint {
                                    r_a: p.r_a,
                                    dist: p.dist,
                                    normal_impulse: p.normal_impulse,
                                    normal_mass: 0.0,
                                    normal_vel: p.normal_vel,
                                    radius: 0.0,
                                };
                            }
                            t.friction.tangent_impulse = Vec2::new(0.01, -0.02);
                            t.friction.twist_impulse = if t.len > 1 { 0.01 } else { 0.0 };
                            t.compute_effective_masses();
                            t
                        })
                        .collect();
                    let mut tiles = vec![TwistTile::default(); num_tiles];
                    for (index, c) in compact.iter().enumerate() {
                        let (tile, lane) = tile_lane(index);
                        tiles[tile].write_constraint(lane, c);
                    }
                    ContactTiles::Simplified(Tensor::vector(&backend, tiles, storage).unwrap())
                } else {
                    let mut tiles = vec![CoulombTile::default(); num_tiles];
                    for (index, c) in input.iter().enumerate() {
                        let (tile, lane) = tile_lane(index);
                        tiles[tile].write_constraint(lane, c);
                    }
                    ContactTiles::Coulomb(Tensor::vector(&backend, tiles, storage).unwrap())
                };
                let mut vels = Tensor::vector(&backend, initial_vels.clone(), storage).unwrap();
                let mut encoder = backend.begin_encoding();
                {
                    let mut pass = encoder.begin_pass("tail-colors-regression", None);
                    let first = if fused { 64 } else { TAIL_COLOR };
                    for color in first..=64 {
                        macro_rules! call {
                            ($kernel:expr) => {
                                $kernel
                                    .call(
                                        &mut pass,
                                        if fused { 64u32 } else { 128u32 },
                                        &mut tiles,
                                        &mut vels,
                                        &buckets,
                                        &poses,
                                        &colors[color as usize],
                                        &ids,
                                        &colors[if variant == 1 {
                                            2
                                        } else if variant == 0 {
                                            1
                                        } else {
                                            0
                                        }],
                                        &params,
                                    )
                                    .unwrap()
                            };
                        }
                        match (fused, variant) {
                            (false, 0) => call!(biased),
                            (true, 0) => call!(tail_biased),
                            (false, 1) => call!(generic),
                            (true, 1) => call!(tail_generic),
                            (false, 2) => call!(warm),
                            (true, 2) => call!(tail_warm),
                            (false, _) => call!(final_iteration),
                            (true, _) => call!(tail_final),
                        }
                    }
                }
                backend.submit(encoder).unwrap();
                backend.synchronize().unwrap();
                let mut bytes: Vec<u8> = Vec::new();
                macro_rules! read {
                    ($tensor:expr, $ty:ty) => {
                        bytes.extend_from_slice(bytemuck::cast_slice(
                            &backend
                                .slow_read_vec::<$ty>($tensor.buffer())
                                .await
                                .unwrap(),
                        ))
                    };
                }
                match &tiles {
                    ContactTiles::Coulomb(x) => {
                        read!(x, CoulombTile);
                    }
                    ContactTiles::Simplified(x) => {
                        read!(x, TwistTile);
                    }
                }
                read!(vels, Velocity);
                results.push(bytes);
            }
            assert!(
                results[0] == results[1],
                "fused tail differs for variant {variant}, batches {nb}"
            );
        }
    }
}
