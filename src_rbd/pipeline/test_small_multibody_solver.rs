//! The narrow workgroup must preserve the full solver, including friction and coupling.
use crate::shaders::dynamics::*;
use crate::shaders::utils::BatchIndices;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct Solvers {
    wide: GpuMbSolveConstraints,
    narrow: GpuMbSolveConstraints32,
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
    let batches = 3u32;
    let joints = 6u32;
    let contacts = 6u32;
    let mut cases = 0;
    for ndofs in [0u32, 1, 8, 18, 31, 32] {
        let stride = ndofs.max(1);
        let infos: Vec<_> = (0..batches)
            .map(|b| MultibodyInfo {
                ndofs,
                max_constraints: joints,
                contact_constraint_start: b * contacts,
                contact_constraint_count: contacts,
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
                    dof_id: s % stride,
                    dof2_id: (s + 1) % stride,
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
                    kind: if s % 3 == 0 {
                        MB_CONTACT_KIND_NORMAL
                    } else {
                        MB_CONTACT_KIND_TANGENT
                    },
                    free_body_id: if b == 1 { u32::MAX } else { b },
                    normal_constraint_slot: s / 3 * 3,
                    inv_lhs: 0.55,
                    rhs: -0.21,
                    rhs_wo_bias: -0.06,
                    impulse: 0.12,
                    cfm_factor: 0.98,
                    friction_coeff: 0.6,
                    free_body_im: 0.05,
                    lin_jac: crate::math::Vec3::new(0.3, 0.6, -0.2),
                    ang_jac: crate::math::Vec3::new(-0.1, 0.2, 0.3),
                    ii_ang_jac: crate::math::Vec3::new(-0.02, 0.04, 0.06),
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
            let mode = Tensor::scalar(&backend, bias, BufferUsages::UNIFORM).unwrap();
            let mut outputs = vec![];
            for narrow in [false, true] {
                let mut gpu_jc = Tensor::vector(&backend, jc.clone(), usage).unwrap();
                let mut gpu_cc = Tensor::vector(&backend, cc.clone(), usage).unwrap();
                let mut gpu_velocities =
                    Tensor::vector(&backend, velocities.clone(), usage).unwrap();
                let mut gpu_free =
                    Tensor::vector(&backend, vec![Velocity::default(); batches as usize], usage)
                        .unwrap();
                let mut encoder = backend.begin_encoding();
                {
                    let mut pass = encoder.begin_pass("solver-equivalence", None);
                    for _ in 0..4 {
                        macro_rules! solve {
                            ($kernel:ident, $lanes:expr) => {
                                // Include an inactive multibody workgroup tail.
                                kernels
                                    .$kernel
                                    .call(
                                        &mut pass,
                                        [2 * $lanes, batches, 1],
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
                                    )
                                    .unwrap();
                            };
                        }
                        if narrow {
                            solve!(narrow, 32);
                        } else {
                            solve!(wide, 64);
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
                let free: Vec<Velocity> = backend.slow_read_vec(gpu_free.buffer()).await.unwrap();
                assert!(velocity.iter().all(|v| v.is_finite()));
                outputs.push((
                    bytemuck::cast_slice::<_, u8>(&joint).to_vec(),
                    bytemuck::cast_slice::<_, u8>(&contact).to_vec(),
                    velocity,
                    bytemuck::cast_slice::<_, u8>(&free).to_vec(),
                ));
            }
            assert_eq!(outputs[0], outputs[1], "ndofs={ndofs} bias={bias}");
            cases += 1;
        }
    }
    println!("32-thread solver matches 64-thread solver in {cases} cases");
}
