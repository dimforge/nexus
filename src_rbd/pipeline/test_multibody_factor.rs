//! LU and constraint-column checks against an independent f64 linear solve.
use crate::shaders::dynamics::*;
use crate::shaders::utils::BatchIndices;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct Kernels {
    factor: GpuMbFactorSolveSimd,
    finalize: GpuMbFinalizeJointConstraints,
}
fn solve(matrix: &[f32], rhs: &[f32]) -> Vec<f64> {
    let n = rhs.len();
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|r| (0..n).map(|c| matrix[c * n + r] as f64).collect())
        .collect();
    let mut x: Vec<f64> = rhs.iter().map(|v| *v as f64).collect();
    for k in 0..n {
        let p = (k..n)
            .max_by(|a0, b| a[*a0][k].abs().total_cmp(&a[*b][k].abs()))
            .unwrap();
        a.swap(k, p);
        x.swap(k, p);
        for r in k + 1..n {
            let f = a[r][k] / a[k][k];
            for c in k..n {
                a[r][c] -= f * a[k][c];
            }
            x[r] -= f * x[k];
        }
    }
    for r in (0..n).rev() {
        for c in r + 1..n {
            x[r] -= a[r][c] * x[c];
        }
        x[r] /= a[r][r];
    }
    x
}
fn close(a: f64, b: f32) {
    assert!(
        b.is_finite() && (a - b as f64).abs() < 2e-5 * (1.0 + a.abs()),
        "{a} != {b}"
    );
}
#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a Metal GPU"]
async fn test_factor_and_columns_equivalence() {
    let backend = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
    let kernels = Kernels::from_backend(&backend).unwrap();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    for batches in [5u32, 129] {
        let joints = 10u32;
        for n in [0u32, 1, 8, 18, 31, 32] {
            let cap = n.max(1);
            let stride = cap + 3;
            let params = BatchIndices {
                num_batches: batches,
                multibodies_len: 2,
                dof_batch_capacity: 2 * stride,
                mb_max_ndofs: n,
                mb_joint_constraints_batch_capacity: 2 * joints,
                mb_joint_constraint_columns_batch_capacity: 2 * joints * 2 * stride,
                ..Default::default()
            };
            let mut infos = vec![MultibodyInfo::default(); (2 * batches) as usize];
            let mut matrices = vec![0.0f32; (2 * cap * cap * batches) as usize];
            let mut rhs = vec![0.0f32; (2 * stride * batches) as usize];
            let mut constraints =
                vec![MultibodyJointConstraint::default(); (2 * joints * batches) as usize];
            let mut originals = vec![];
            for mb in 0..2 {
                for b in 0..batches {
                    infos[params.mbi(b, mb as usize)] = MultibodyInfo {
                        ndofs: n,
                        first_dof: mb * stride,
                        mass_matrix_offset: mb * cap * cap,
                        first_constraint: mb * joints,
                        max_constraints: joints,
                        ..Default::default()
                    };
                    let m0 = params.mb_region(b, mb * cap * cap, n * n);
                    let r0 = params.mb_region(b, mb * stride, n);
                    // Diagonally dominant matrix; reverse the rows for half the cases
                    // to force pivot swaps at many elimination stages.
                    for c in 0..n {
                        for r in 0..n {
                            let rr = if (b + mb) % 2 == 0 { r } else { n - 1 - r };
                            matrices[m0 + (c * n + r) as usize] = if rr == c {
                                3.0 + c as f32 * 0.1
                            } else {
                                ((rr * 7 + c * 3 + b) % 13) as f32 * 0.006 - 0.03
                            };
                        }
                    }
                    for d in 0..n {
                        rhs[r0 + d as usize] = 0.2 + d as f32 * 0.03;
                    }
                    originals.push((
                        matrices[m0..m0 + (n * n) as usize].to_vec(),
                        rhs[r0..r0 + n as usize].to_vec(),
                    ));
                    for s in 0..joints {
                        let kind = [
                            MB_JOINT_KIND_MOTOR,
                            MB_JOINT_KIND_LIMIT,
                            MB_JOINT_KIND_MOTOR,
                            MB_JOINT_KIND_LIMIT_INACTIVE,
                            MB_JOINT_KIND_COUPLING,
                            MB_JOINT_KIND_FRICTION,
                            0,
                            MB_JOINT_KIND_MOTOR,
                            MB_JOINT_KIND_LIMIT,
                            MB_JOINT_KIND_LIMIT,
                        ][s as usize];
                        constraints[(b * 2 * joints + mb * joints + s) as usize] =
                            MultibodyJointConstraint {
                                kind,
                                dof_id: match s {
                                    0 | 1 => n.saturating_sub(1),
                                    2 | 3 => n / 2,
                                    7 => 0,
                                    8 => 1 % cap,
                                    _ => s % cap,
                                },
                                dof2_id: (s + 1) % cap,
                                coupling_coeff: if s == 4 { 0.35 } else { 0.0 },
                                cfm_coeff: 0.01 * (s + 1) as f32,
                                cfm_gain: 0.03 * s as f32,
                                rhs: 0.7,
                                impulse: 0.13,
                                ..Default::default()
                            };
                    }
                }
            }
            let gpu_info = Tensor::vector(&backend, infos, usage).unwrap();
            let gpu_params = Tensor::scalar(&backend, params, BufferUsages::UNIFORM).unwrap();
            let mut gpu_matrix = Tensor::vector(&backend, matrices, usage).unwrap();
            let mut gpu_rhs = Tensor::vector(&backend, rhs, usage).unwrap();
            let mut gpu_piv =
                Tensor::vector(&backend, vec![0u32; (2 * stride * batches) as usize], usage)
                    .unwrap();
            let mut gpu_constraints = Tensor::vector(&backend, constraints.clone(), usage).unwrap();
            let mut gpu_cols = Tensor::vector(
                &backend,
                vec![0.0f32; (2 * joints * 2 * stride * batches) as usize],
                usage,
            )
            .unwrap();
            let mut enc = backend.begin_encoding();
            {
                let mut pass = enc.begin_pass("factor-column-equivalence", None);
                kernels
                    .factor
                    .call(
                        &mut pass,
                        [64, batches, 1],
                        &gpu_info,
                        &mut gpu_matrix,
                        &mut gpu_piv,
                        &mut gpu_rhs,
                        &gpu_params,
                    )
                    .unwrap();
                kernels
                    .finalize
                    .call(
                        &mut pass,
                        if batches >= 128 {
                            [(2 * batches).div_ceil(8) * 64, 1, 1]
                        } else {
                            [128, batches, 1]
                        },
                        &gpu_info,
                        &mut gpu_constraints,
                        &mut gpu_cols,
                        &gpu_matrix,
                        &gpu_piv,
                        &gpu_params,
                    )
                    .unwrap();
            }
            backend.submit(enc).unwrap();
            let actual_rhs: Vec<f32> = backend.slow_read_vec(gpu_rhs.buffer()).await.unwrap();
            let actual_cols: Vec<f32> = backend.slow_read_vec(gpu_cols.buffer()).await.unwrap();
            let actual_cons: Vec<MultibodyJointConstraint> = backend
                .slow_read_vec(gpu_constraints.buffer())
                .await
                .unwrap();
            for mb in 0..2 {
                for b in 0..batches {
                    let (matrix, rhs) = &originals[(mb * batches + b) as usize];
                    let r0 = params.mb_region(b, mb * stride, n);
                    for (i, v) in solve(matrix, rhs).iter().enumerate() {
                        close(*v, actual_rhs[r0 + i]);
                    }
                    for s in 0..joints {
                        let ci = (b * 2 * joints + mb * joints + s) as usize;
                        let c = constraints[ci];
                        let actual = actual_cons[ci];
                        assert_eq!(actual.kind, c.kind);
                        assert_eq!(actual.dof_id, c.dof_id);
                        assert_eq!(actual.impulse, c.impulse);
                        assert_eq!(actual.rhs, c.rhs);
                        if n == 0 || c.kind == 0 {
                            assert_eq!(actual.inv_lhs, 0.0);
                            continue;
                        }
                        let mut unit = vec![0.0; n as usize];
                        unit[c.dof_id as usize] = 1.0;
                        unit[c.dof2_id as usize] -= c.coupling_coeff;
                        let col = solve(matrix, &unit);
                        let c0 = ((b * 2 * joints + mb * joints + s) * 2 * stride) as usize;
                        for (i, v) in col.iter().enumerate() {
                            close(*v, actual_cols[c0 + i]);
                        }
                        let lhs = col[c.dof_id as usize]
                            - c.coupling_coeff as f64 * col[c.dof2_id as usize];
                        let cfm = lhs * c.cfm_coeff as f64 + c.cfm_gain as f64;
                        close(cfm, actual.cfm_gain);
                        close(1.0 / (lhs + cfm), actual.inv_lhs);
                    }
                }
            }
        }
    }
    println!(
        "1608 LU/column configurations passed, with small and large batch dispatch, pivoted matrices and motor/limit/coupling/friction columns"
    );
}
