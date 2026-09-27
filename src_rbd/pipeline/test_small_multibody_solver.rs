//! Compare cooperative and native Metal PGS, including heterogeneous batches,
//! partially occupied groups, upper DOFs, coupling, friction, and fused iterations.
use crate::shaders::dynamics::*;
use crate::shaders::utils::BatchIndices;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct Solvers {
    wide: GpuMbSolveConstraints,
    narrow: GpuMbSolveConstraints32,
    #[cfg(feature = "metal")]
    simd: GpuMbSolveConstraintsSimd,
    #[cfg(feature = "metal")]
    packed: GpuMbSolveConstraintsPacked,
}

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a GPU device"]
async fn test_small_multibody_solver_equivalence() {
    #[cfg(feature = "metal")]
    let backend = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
    #[cfg(not(feature = "metal"))]
    let backend = GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap());
    let kernels = Solvers::from_backend(&backend).unwrap();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    // Qualified spirv attributes are not parsed by spirv_bindgen for host
    // dispatch metadata. Catch accidental one-thread workgroups explicitly.
    #[cfg(feature = "metal")]
    assert_eq!(
        <GpuMbSolveConstraintsPackedArgs<'static> as khal::shader::ShaderArgsType>::WORKGROUP_SIZE,
        [32, 1, 1]
    );
    let batches = 5u32;
    let joints = 6u32;
    let contacts = 6u32;
    let mut cases = 0;
    for ndofs in [0u32, 1, 8, 18, 31, 32] {
        for heterogeneous in [false, true] {
            let stride = ndofs.max(1) + 3;
            let dofs = |b: u32| {
                if heterogeneous {
                    match b % 3 {
                        0 => ndofs,
                        1 => ndofs.saturating_sub(3),
                        _ => ndofs / 2,
                    }
                } else {
                    ndofs
                }
            };
            let infos: Vec<_> = (0..batches)
                .map(|b| MultibodyInfo {
                    ndofs: dofs(b),
                    max_constraints: if heterogeneous && b % 2 == 1 {
                        3
                    } else {
                        joints
                    },
                    contact_constraint_start: b * contacts,
                    contact_constraint_count: if heterogeneous {
                        (b % 3) * (crate::math::DIM as u32)
                    } else {
                        contacts
                    },
                    ..Default::default()
                })
                .collect();
            let mut jc = vec![MultibodyJointConstraint::default(); (joints * batches) as usize];
            let mut cc = vec![MultibodyContactConstraint::default(); (contacts * batches) as usize];
            for b in 0..batches {
                for s in 0..joints {
                    jc[(b * joints + s) as usize] = MultibodyJointConstraint {
                        kind: [
                            MB_JOINT_KIND_MOTOR,
                            MB_JOINT_KIND_LIMIT,
                            MB_JOINT_KIND_COUPLING,
                            MB_JOINT_KIND_FRICTION,
                            0,
                            MB_JOINT_KIND_LIMIT_INACTIVE,
                        ][s as usize],
                        dof_id: (s * 7 + 3) % dofs(b).max(1),
                        dof2_id: (s * 11 + 7) % dofs(b).max(1),
                        coupling_coeff: if s == 2 { 0.35 } else { 0.0 },
                        rhs: -0.13,
                        rhs_wo_bias: -0.03,
                        inv_lhs: 0.65,
                        impulse: 0.1,
                        impulse_lo: -0.5,
                        impulse_hi: 0.75,
                        cfm_gain: 0.02,
                        ..Default::default()
                    };
                }
                for s in 0..contacts {
                    cc[(b * contacts + s) as usize] = MultibodyContactConstraint {
                        kind: if s % crate::math::DIM as u32 == 0 {
                            MB_CONTACT_KIND_NORMAL
                        } else {
                            MB_CONTACT_KIND_TANGENT
                        },
                        free_body_id: if b == 1 { u32::MAX } else { b },
                        normal_constraint_slot: s / crate::math::DIM as u32
                            * crate::math::DIM as u32,
                        inv_lhs: 0.55,
                        rhs: -0.21,
                        rhs_wo_bias: -0.06,
                        impulse: 0.12,
                        cfm_factor: 0.98,
                        friction_coeff: 0.6,
                        free_body_im: 0.05,
                        #[cfg(feature = "dim3")]
                        lin_jac: crate::math::Vec3::new(0.3, 0.6, -0.2),
                        #[cfg(feature = "dim2")]
                        lin_jac: crate::math::Vec2::new(0.3, 0.6),
                        #[cfg(feature = "dim3")]
                        ang_jac: crate::math::Vec3::new(-0.1, 0.2, 0.3),
                        #[cfg(feature = "dim2")]
                        ang_jac: 0.3,
                        #[cfg(feature = "dim3")]
                        ii_ang_jac: crate::math::Vec3::new(-0.02, 0.04, 0.06),
                        #[cfg(feature = "dim2")]
                        ii_ang_jac: 0.06,
                        ..Default::default()
                    };
                }
            }
            let jcols: Vec<f32> = (0..joints * batches * stride)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.017)
                .collect();
            let ccols: Vec<f32> = (0..contacts * batches * stride * 2)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.013)
                .collect();
            let velocities: Vec<f32> = (0..stride * batches)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.07)
                .collect();
            let gpu_infos = Tensor::vector(&backend, infos, usage).unwrap();
            let gpu_jcols = Tensor::vector(&backend, jcols, usage).unwrap();
            let gpu_ccols = Tensor::vector(&backend, ccols, usage).unwrap();
            let params = Tensor::scalar(
                &backend,
                BatchIndices {
                    num_batches: batches,
                    multibodies_len: 1,
                    dof_batch_capacity: stride,
                    mb_joint_constraints_batch_capacity: joints,
                    mb_joint_constraint_columns_batch_capacity: joints * stride,
                    mb_max_joint_constraints: joints,
                    ..Default::default()
                },
                BufferUsages::UNIFORM,
            )
            .unwrap();
            let contact_bound = Tensor::scalar(&backend, contacts, BufferUsages::UNIFORM).unwrap();
            for bias in [0u32, 1, 2] {
                for iteration_count in [1u32, 4] {
                    let mode = Tensor::scalar(&backend, bias, BufferUsages::UNIFORM).unwrap();
                    let mut outputs = vec![];
                    for variant in 0..if cfg!(feature = "metal") { 6 } else { 2 } {
                        let mut gpu_jc = Tensor::vector(&backend, jc.clone(), usage).unwrap();
                        let mut gpu_cc = Tensor::vector(&backend, cc.clone(), usage).unwrap();
                        let mut gpu_velocities =
                            Tensor::vector(&backend, velocities.clone(), usage).unwrap();
                        let mut gpu_free = Tensor::vector(
                            &backend,
                            vec![Velocity::default(); batches as usize],
                            usage,
                        )
                        .unwrap();
                        let iterations = Tensor::scalar(
                            &backend,
                            if variant == 3 || variant == 5 {
                                iteration_count
                            } else {
                                1u32
                            },
                            BufferUsages::UNIFORM,
                        )
                        .unwrap();
                        let mut encoder = backend.begin_encoding();
                        {
                            let mut pass = encoder.begin_pass("solver-equivalence", None);
                            for _ in 0..if variant == 3 || variant == 5 {
                                1
                            } else {
                                iteration_count
                            } {
                                macro_rules! solve {
                                    ($kernel:ident, $lanes:expr) => {{
                                        let dispatch = if variant >= 4 {
                                            [batches * 8, 2, 1]
                                        } else {
                                            [2 * $lanes, batches, 1]
                                        };
                                        kernels
                                            .$kernel
                                            .call(
                                                &mut pass,
                                                dispatch,
                                                &gpu_infos,
                                                &mut gpu_jc,
                                                &gpu_jcols,
                                                &mut gpu_cc,
                                                &gpu_ccols,
                                                &mode,
                                                &params,
                                                &contact_bound,
                                                &mut gpu_velocities,
                                                &mut gpu_free,
                                                &iterations,
                                            )
                                            .unwrap();
                                    }};
                                }
                                match variant {
                                    #[cfg(feature = "metal")]
                                    4 | 5 => solve!(packed, 8),
                                    #[cfg(feature = "metal")]
                                    2 | 3 => solve!(simd, 32),
                                    1 => solve!(narrow, 32),
                                    _ => solve!(wide, 64),
                                }
                            }
                        }
                        backend.submit(encoder).unwrap();
                        let joint: Vec<MultibodyJointConstraint> =
                            backend.slow_read_vec(gpu_jc.buffer()).await.unwrap();
                        let contact: Vec<MultibodyContactConstraint> =
                            backend.slow_read_vec(gpu_cc.buffer()).await.unwrap();
                        let velocity: Vec<f32> = backend
                            .slow_read_vec(gpu_velocities.buffer())
                            .await
                            .unwrap();
                        let free: Vec<Velocity> =
                            backend.slow_read_vec(gpu_free.buffer()).await.unwrap();
                        assert!(velocity.iter().all(|v| v.is_finite()));
                        outputs.push((
                            bytemuck::cast_slice::<_, u8>(&joint).to_vec(),
                            bytemuck::cast_slice::<_, u8>(&contact).to_vec(),
                            velocity,
                            bytemuck::cast_slice::<_, u8>(&free).to_vec(),
                        ));
                    }
                    assert_eq!(outputs[0], outputs[1], "ndofs={ndofs} bias={bias}");
                    #[cfg(feature = "metal")]
                    {
                        for target in [2, 4, 5] {
                            for (&a, &b) in outputs[0].2.iter().zip(&outputs[target].2) {
                                close(a, b);
                            }
                            compare_constraints(
                                &outputs[0].0,
                                &outputs[target].0,
                                core::mem::size_of::<MultibodyJointConstraint>(),
                                core::mem::offset_of!(MultibodyJointConstraint, impulse),
                            );
                            compare_constraints(
                                &outputs[0].1,
                                &outputs[target].1,
                                core::mem::size_of::<MultibodyContactConstraint>(),
                                core::mem::offset_of!(MultibodyContactConstraint, impulse),
                            );
                            for (a, b) in outputs[0]
                                .3
                                .chunks_exact(4)
                                .zip(outputs[target].3.chunks_exact(4))
                            {
                                close(
                                    f32::from_ne_bytes(a.try_into().unwrap()),
                                    f32::from_ne_bytes(b.try_into().unwrap()),
                                );
                            }
                        }
                        assert!(
                            outputs[2] == outputs[3],
                            "fused SIMD ndofs={ndofs} bias={bias} heterogeneous={heterogeneous}"
                        );
                        assert!(
                            outputs[4] == outputs[5],
                            "fused packed ndofs={ndofs} bias={bias} heterogeneous={heterogeneous}"
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    println!("Solver equivalence passed in {cases} cases");
}

#[cfg(feature = "metal")]
fn close(a: f32, b: f32) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= 1.0e-6 * (1.0 + a.abs()),
        "{a} vs {b}"
    );
}

#[cfg(feature = "metal")]
fn compare_constraints(a: &[u8], b: &[u8], stride: usize, impulse: usize) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.chunks_exact(stride).zip(b.chunks_exact(stride)) {
        // IDs, kind, Jacobians, padding, and all immutable coefficients must
        // remain bit-for-bit identical. Only the accumulated impulse may round.
        assert!(a[..impulse] == b[..impulse]);
        assert!(a[impulse + 4..] == b[impulse + 4..]);
        close(
            f32::from_ne_bytes(a[impulse..impulse + 4].try_into().unwrap()),
            f32::from_ne_bytes(b[impulse..impulse + 4].try_into().unwrap()),
        );
    }
}
