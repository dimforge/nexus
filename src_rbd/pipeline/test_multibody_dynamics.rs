//! Differential checks for tree dynamics and small-matrix Metal factorization.
use crate::math::{Pose, Quat, Vec3};
use crate::shaders::dynamics::*;
use crate::shaders::utils::BatchIndices;
use bytemuck::Zeroable;
use glamx::Vec4;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct Kernels {
    pre: GpuMbComputeDynamicsPre,
    kinematics: GpuMbKinematicsSerial,
    parallel: GpuMbComputeDynamicsParallel,
    reference: GpuMbGravityAndLu,
    recursive: GpuMbRecursiveForces,
    factor: GpuMbFactorSolveSimd,
}
fn close(a: f32, b: f32, context: &str) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= 3.0e-4 * (1.0 + a.abs().max(b.abs())),
        "{context}: {a} != {b}"
    );
}

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a Metal GPU"]
async fn test_tree_dynamics_equivalence() {
    let backend = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
    let kernels = Kernels::from_backend(&backend).unwrap();
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let batches = 5u32; // Tail slot in the packed 64-thread dispatch.
    let dt = 0.003f32;
    let gravity = Vec4::new(0.4, -9.7, 0.2, 0.0);
    let gpu_dt = Tensor::scalar(&backend, dt, BufferUsages::UNIFORM).unwrap();
    let gpu_g = Tensor::scalar(&backend, gravity, BufferUsages::UNIFORM).unwrap();
    let mut cases = 0;
    for links in [1u32, 2, 7, 13, 45] {
        for fixed_root in [false, true] {
            for branched in [false, true] {
                let mut topology: Vec<MultibodyLinkStatic> = vec![];
                let mut n = 0u32;
                for k in 0..links {
                    let parent = if k == 0 {
                        u32::MAX
                    } else if branched {
                        (k - 1) / 2
                    } else {
                        k - 1
                    };
                    let locked = if k == 0 {
                        if fixed_root { 63u32 } else { 0 }
                    } else {
                        match k % 4 {
                            0 => 63,
                            1 => 63 ^ (1 << 3),
                            2 => 63 ^ 2,
                            _ => 7,
                        }
                    };
                    let ndofs = 6 - locked.count_ones();
                    let mut ancestry = if k == 0 {
                        [0; 2]
                    } else {
                        topology[parent as usize].ancestor_dofs
                    };
                    for d in n..n + ndofs {
                        ancestry[d as usize / 32] |= 1 << (d % 32);
                    }
                    let mut stat = MultibodyLinkStatic::zeroed();
                    stat.rb_id = k;
                    stat.parent_link_id = parent;
                    stat.assembly_id = n;
                    stat.ndofs = ndofs;
                    stat.ancestor_dofs = ancestry;
                    stat.data.locked_axes = locked;
                    stat.data.local_frame_a = Pose::from_parts(
                        Vec3::new(0.1 * k as f32, 0.12, -0.04),
                        Quat::from_rotation_z(0.13 * k as f32),
                    );
                    stat.data.local_frame_b = Pose::from_parts(
                        Vec3::new(-0.03, 0.01, 0.04),
                        Quat::from_rotation_y(0.07 * k as f32),
                    );
                    stat.local_mprops = LocalMassProperties {
                        com: Vec3::new(0.02, -0.03, 0.05),
                        inv_mass: Vec3::splat(1.0 / (1.0 + k as f32 * 0.1)),
                        inv_principal_inertia: Vec3::new(3.0, 4.0, 5.0),
                        inertia_ref_frame: Quat::from_rotation_x(0.17),
                        ..Default::default()
                    };
                    if fixed_root && k == 0 {
                        stat.local_mprops.inv_mass = Vec3::ZERO;
                        stat.local_mprops.inv_principal_inertia = Vec3::ZERO;
                    }
                    topology.push(stat);
                    n += ndofs;
                }
                assert!(n <= 64);
                let cap = n.max(1);
                let params = BatchIndices {
                    num_batches: batches,
                    multibodies_len: 1,
                    multibodies_batch_capacity: 1,
                    links_batch_capacity: links,
                    dof_batch_capacity: cap,
                    coriolis_batch_capacity: links * 3 * cap,
                    mb_max_ndofs: n,
                    mb_max_links: links,
                    mb_pack_lanes: if n <= 1 {
                        1
                    } else {
                        n.next_power_of_two().clamp(8, 64)
                    },
                    ..Default::default()
                };
                let infos = vec![
                    MultibodyInfo {
                        ndofs: n,
                        num_links: links,
                        root_is_dynamic: u32::from(!fixed_root),
                        ..Default::default()
                    };
                    batches as usize
                ];
                let mut statics = vec![MultibodyLinkStatic::zeroed(); (links * batches) as usize];
                let mut ws = vec![Vec4::ZERO; (links * batches * WS_QUADS) as usize];
                let mut dofs = vec![0.0; (7 * cap * batches) as usize];
                for b in 0..batches {
                    let wa = WsAddr::new(0, batches, b);
                    for k in 0..links {
                        let mut stat = topology[k as usize];
                        stat.local_mprops.inv_mass *= 1.0 + b as f32 * 0.07;
                        statics[(k * batches + b) as usize] = stat;
                        ws_set_rot(
                            &mut ws,
                            wa,
                            k,
                            WS_JOINT_ROT,
                            Quat::from_rotation_x(0.05 * (k + b) as f32),
                        );
                        for axis in 0..6 {
                            ws_set_coord(&mut ws, wa, k, axis, 0.03 * (axis + k + b) as f32);
                        }
                        ws_set_ext_wrench(
                            &mut ws,
                            wa,
                            k,
                            Vec3::new(0.2 * k as f32, -0.1, 0.3),
                            Vec3::new(0.02, 0.01 * k as f32, -0.03),
                            0.7 + b as f32 * 0.1,
                        );
                    }
                    for d in 0..n {
                        dofs[(d * batches + b) as usize] = ((d + b) % 5) as f32 * 0.11 - 0.2;
                        dofs[((cap + d) * batches + b) as usize] = 0.1 + 0.01 * d as f32;
                        dofs[((2 * cap + d) * batches + b) as usize] = 0.05;
                        dofs[((3 * cap + d) * batches + b) as usize] = 0.2;
                        dofs[((4 * cap + d) * batches + b) as usize] = 0.01 * d as f32;
                        if b == 2 && d + 1 == n {
                            dofs[((5 * cap + d) * batches + b) as usize] = 1.0;
                        }
                    }
                }
                let gpu_infos = Tensor::vector(&backend, infos, storage).unwrap();
                let gpu_stat =
                    Tensor::vector(&backend, ls_soa_from_structs(&statics, batches), storage)
                        .unwrap();
                let gpu_dofs = Tensor::vector(&backend, dofs.clone(), storage).unwrap();
                let gpu_params = Tensor::scalar(&backend, params, BufferUsages::UNIFORM).unwrap();
                let mut outputs = vec![];
                for packed in [false, true] {
                    let mut gpu_ws = Tensor::vector(&backend, ws.clone(), storage).unwrap();
                    let mut gpu_pose = Tensor::vector(
                        &backend,
                        vec![Pose::default(); (links * batches) as usize],
                        storage,
                    )
                    .unwrap();
                    let mut gpu_jac = Tensor::vector(
                        &backend,
                        vec![0.0f32; (links * 6 * cap * batches) as usize],
                        storage,
                    )
                    .unwrap();
                    let mut gpu_mass = Tensor::vector(
                        &backend,
                        vec![0.0f32; (cap * cap * batches) as usize],
                        storage,
                    )
                    .unwrap();
                    let mut gpu_scratch = Tensor::vector(
                        &backend,
                        vec![0.0f32; ((2 * links * 3 * cap + 6 * cap) * batches) as usize],
                        storage,
                    )
                    .unwrap();
                    let mut enc = backend.begin_encoding();
                    {
                        let mut pass = enc.begin_pass("tree-dynamics", None);
                        if packed {
                            kernels
                                .kinematics
                                .call(
                                    &mut pass,
                                    batches,
                                    &gpu_infos,
                                    &gpu_stat,
                                    &mut gpu_ws,
                                    &mut gpu_pose,
                                    &gpu_dofs,
                                    &gpu_params,
                                )
                                .unwrap();
                        }
                        macro_rules! pre {
                            ($kernel:ident) => {
                                kernels
                                    .$kernel
                                    .call(
                                        &mut pass,
                                        [batches.div_ceil(64 / params.mb_pack_lanes) * 64, 1, 1],
                                        &gpu_infos,
                                        &gpu_stat,
                                        &mut gpu_ws,
                                        &mut gpu_pose,
                                        &mut gpu_jac,
                                        &mut gpu_mass,
                                        &mut gpu_scratch,
                                        &gpu_dofs,
                                        &gpu_dt,
                                        &gpu_params,
                                    )
                                    .unwrap()
                            };
                        }
                        if packed {
                            pre!(parallel);
                        } else {
                            pre!(pre);
                        }
                    }
                    backend.submit(enc).unwrap();
                    let result_jac: Vec<f32> =
                        backend.slow_read_vec(gpu_jac.buffer()).await.unwrap();
                    let result_mass: Vec<f32> =
                        backend.slow_read_vec(gpu_mass.buffer()).await.unwrap();
                    let result_ws: Vec<Vec4> =
                        backend.slow_read_vec(gpu_ws.buffer()).await.unwrap();
                    // Independent dense J^T I J reference, including damping, armature and springs.
                    for b in 0..batches {
                        let wa = WsAddr::new(0, batches, b);
                        let mut reference = vec![0.0f32; (n * n) as usize];
                        let mut reference_j = vec![[Vec3::ZERO; 2]; (links * n) as usize];
                        for k in 0..links {
                            let stat = statics[(k * batches + b) as usize];
                            let world = ws_pose(&result_ws, wa, k, WS_LTW);
                            let com = ws_vec(&result_ws, wa, k, WS_WORLD_COM);
                            if k > 0 {
                                let shift =
                                    com - ws_vec(&result_ws, wa, stat.parent_link_id, WS_WORLD_COM);
                                for d in 0..n {
                                    let [v, w] =
                                        reference_j[(stat.parent_link_id * n + d) as usize];
                                    reference_j[(k * n + d) as usize] = [v + w.cross(shift), w];
                                }
                            }
                            let parent_rot = if k == 0 {
                                Quat::IDENTITY
                            } else {
                                ws_pose(&result_ws, wa, stat.parent_link_id, WS_LTW).rotation
                            };
                            let rotation = parent_rot * stat.data.local_frame_a.rotation;
                            let shift = ws_vec(&result_ws, wa, k, WS_SHIFT23);
                            let mut d = stat.assembly_id;
                            for axis in 0..6 {
                                if stat.data.locked_axes & (1 << axis) == 0 {
                                    let basis = rotation
                                        * match axis % 3 {
                                            0 => Vec3::X,
                                            1 => Vec3::Y,
                                            _ => Vec3::Z,
                                        };
                                    let v = if axis < 3 { basis } else { basis.cross(shift) };
                                    let w = if axis < 3 { Vec3::ZERO } else { basis };
                                    reference_j[(k * n + d) as usize][0] += v;
                                    reference_j[(k * n + d) as usize][1] += w;
                                    d += 1;
                                }
                            }
                            let inertia = ws_world_inertia(&result_ws, wa, k, &stat.local_mprops);
                            let mass = if stat.local_mprops.inv_mass.x == 0.0 {
                                0.0
                            } else {
                                1.0 / stat.local_mprops.inv_mass.x
                            };
                            for i in 0..n {
                                for j in 0..n {
                                    let [vi, wi] = reference_j[(k * n + i) as usize];
                                    let [vj, wj] = reference_j[(k * n + j) as usize];
                                    reference[(j * n + i) as usize] +=
                                        mass * vi.dot(vj) + wi.dot(inertia * wj);
                                }
                            }
                            let base =
                                params.mb_region(b, 0, links * 6 * n) + k as usize * 6 * n as usize;
                            for d in 0..n {
                                let [v, w] = reference_j[(k * n + d) as usize];
                                let expected = [v.x, v.y, v.z, w.x, w.y, w.z];
                                for row in 0..6 {
                                    close(
                                        expected[row],
                                        result_jac[base + d as usize * 6 + row],
                                        "body Jacobian",
                                    );
                                }
                            }
                            let _ = world;
                        }
                        for d in 0..n {
                            reference[(d * n + d) as usize] +=
                                dofs[((cap + d) * batches + b) as usize] * dt
                                    + 0.05
                                    + 0.2 * dt * dt;
                        }
                        if b == 2 && n > 0 {
                            for d in 0..n {
                                reference[((n - 1) * n + d) as usize] = 0.0;
                                reference[(d * n + n - 1) as usize] = 0.0;
                            }
                            reference[(n * n - 1) as usize] = 1.0;
                        }
                        let base = params.mb_region(b, 0, n * n);
                        // Scale entrywise error by the two coordinate inertias. Long
                        // chains have nearly cancelling off-diagonal entries, for which
                        // relative-to-entry tolerances are ill-conditioned in f32.
                        let mut error2 = 0.0f64;
                        let mut norm2 = 0.0f64;
                        for (i, v) in reference.iter().enumerate() {
                            let actual = result_mass[base + i];
                            let row = i % n as usize;
                            let col = i / n as usize;
                            let scale = (reference[row * n as usize + row]
                                * reference[col * n as usize + col])
                                .abs()
                                .sqrt();
                            assert!(
                                actual.is_finite() && (actual - v).abs() <= 3.0e-5 * (1.0 + scale),
                                "mass: links={links}, fixed={fixed_root}, branched={branched}, batch={b}, index={i}: {actual} != {v}, scale={scale}"
                            );
                            error2 += f64::from(actual - v).powi(2);
                            norm2 += f64::from(*v).powi(2);
                        }
                        assert!(error2.sqrt() <= 2.0e-5 * (1.0 + norm2.sqrt()));
                    }
                    // Reference gravity/LU uses the original dense projection and shared-memory factorization.
                    if n > 0 && n <= 32 {
                        let mut accelerations = vec![];
                        for recursive in [false, true] {
                            let mut matrix =
                                Tensor::vector(&backend, result_mass.clone(), storage).unwrap();
                            let mut piv = Tensor::vector(
                                &backend,
                                vec![0u32; (cap * batches) as usize],
                                storage,
                            )
                            .unwrap();
                            let mut forces = Tensor::vector(
                                &backend,
                                vec![0.0f32; (cap * batches) as usize],
                                storage,
                            )
                            .unwrap();
                            let mut enc = backend.begin_encoding();
                            {
                                let mut pass = enc.begin_pass("force-equivalence", None);
                                if recursive {
                                    kernels
                                        .recursive
                                        .call(
                                            &mut pass,
                                            batches,
                                            &gpu_infos,
                                            &gpu_stat,
                                            &mut gpu_ws,
                                            &gpu_jac,
                                            &mut forces,
                                            &mut gpu_scratch,
                                            &gpu_dofs,
                                            &gpu_g,
                                            &gpu_params,
                                            &gpu_dt,
                                        )
                                        .unwrap();
                                    kernels
                                        .factor
                                        .call(
                                            &mut pass,
                                            [32, batches, 1],
                                            &gpu_infos,
                                            &mut matrix,
                                            &mut piv,
                                            &mut forces,
                                            &gpu_params,
                                        )
                                        .unwrap();
                                } else {
                                    kernels
                                        .reference
                                        .call(
                                            &mut pass,
                                            [64, batches, 1],
                                            &gpu_infos,
                                            &gpu_stat,
                                            &mut gpu_ws,
                                            &gpu_jac,
                                            &mut forces,
                                            &mut matrix,
                                            &mut piv,
                                            &gpu_dofs,
                                            &gpu_g,
                                            &gpu_params,
                                            &gpu_dt,
                                        )
                                        .unwrap();
                                }
                            }
                            backend.submit(enc).unwrap();
                            accelerations
                                .push(backend.slow_read_vec::<f32>(forces.buffer()).await.unwrap());
                        }
                        for (a, b) in accelerations[0].iter().zip(&accelerations[1]) {
                            close(*a, *b, "acceleration");
                        }
                    }
                    outputs.push((result_jac, result_mass));
                }
                for (a, b) in outputs[0].0.iter().zip(&outputs[1].0) {
                    close(*a, *b, "split/fused Jacobian");
                }
                for (a, b) in outputs[0].1.iter().zip(&outputs[1].1) {
                    close(*a, *b, "split/fused mass");
                }
                cases += 1;
            }
        }
    }
    println!("{cases} tree configurations × {batches} environments passed");
}
