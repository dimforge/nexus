//! CPU tests for the `utils::linalg` primitives.
//!
//! These mirror the operations the multibody kernels depend on:
//!   - matrix views (`MatSlice`) and their indexing,
//!   - `fill`, `copy_from`,
//!   - `gemm`, `gemm_mat3_lhs`,
//!   - `axpy_mat`,
//!   - `quadform_spatial` (block-diagonal spatial-mass CRBA step),
//!   - `gemv_tr_spatial`,
//!   - `skew_tr`,
//!   - `lu_decompose` + `lu_solve_in_place`, including reuse of the same LU factor
//!     with multiple right-hand sides.

use crate::utils::linalg::{
    MatSlice, axpy_mat, copy_from, fill, gemm, gemm_mat3_lhs, gemm_tr, gemv_tr_spatial,
    lu_decompose, lu_solve_in_place, quadform_spatial, skew, skew_tr,
};
use glamx::{Mat3, Vec3};

const EPS: f32 = 1.0e-5;

fn approx_eq(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= EPS * (1.0 + a.abs().max(b.abs())),
        "expected {a} ≈ {b} (diff = {})",
        (a - b).abs(),
    );
}

fn assert_slice_eq(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= EPS * (1.0 + g.abs().max(w.abs())),
            "mismatch at index {i}: got {g}, want {w}",
        );
    }
}

//
// MatSlice indexing.
//

#[test]
fn matslice_indexing_dense() {
    let m = MatSlice::dense(7, 4, 3);
    // Column-major: m[r, c] is at offset + c * rows + r.
    assert_eq!(m.idx(0, 0), 7);
    assert_eq!(m.idx(3, 0), 7 + 3);
    assert_eq!(m.idx(0, 1), 7 + 4);
    assert_eq!(m.idx(3, 2), 7 + 2 * 4 + 3);
}

#[test]
fn matslice_fixed_rows_columns() {
    let m = MatSlice::dense(0, 6, 4);

    let top = m.fixed_rows(0, 3);
    assert_eq!((top.rows, top.cols, top.lead), (3, 4, 6));
    assert_eq!(top.idx(2, 3), 6 * 3 + 2);

    let bot = m.fixed_rows(3, 3);
    assert_eq!((bot.rows, bot.cols, bot.lead), (3, 4, 6));
    assert_eq!(bot.idx(0, 0), 3);
    assert_eq!(bot.idx(2, 2), 6 * 2 + 3 + 2);

    let cols = m.columns(1, 2);
    assert_eq!((cols.rows, cols.cols, cols.lead), (6, 2, 6));
    assert_eq!(cols.idx(0, 0), 6);
    assert_eq!(cols.idx(5, 1), 6 * 2 + 5);
}

#[test]
fn matslice_rows_range_pair_split() {
    let m = MatSlice::dense(0, 6, 3);
    let (a, b) = m.rows_range_pair(0, 3, 3, 3);
    assert_eq!(a.idx(2, 0), 2);
    assert_eq!(b.idx(0, 0), 3);
    assert_eq!(b.idx(2, 2), 6 * 2 + 5);
}

//
// fill / copy_from.
//

#[test]
fn fill_sets_every_entry() {
    let mut buf = vec![-1.0f32; 12]; // 4×3
    let m = MatSlice::dense(0, 4, 3);
    fill(&mut buf, m, 2.5);
    for &v in &buf {
        approx_eq(v, 2.5);
    }
}

#[test]
fn fill_respects_sub_view() {
    // Fill only the central 2×2 of a 4×4 without touching the surround.
    let mut buf = vec![9.0f32; 16];
    let m = MatSlice::dense(0, 4, 4);
    let center = m.view(1, 1, 2, 2);
    fill(&mut buf, center, 0.0);
    // Column-major: untouched entries keep 9.0.
    let expected = [
        9.0, 9.0, 9.0, 9.0, // col 0
        9.0, 0.0, 0.0, 9.0, // col 1
        9.0, 0.0, 0.0, 9.0, // col 2
        9.0, 9.0, 9.0, 9.0, // col 3
    ];
    assert_slice_eq(&buf, &expected);
}

#[test]
fn copy_from_copies_a_disjoint_block() {
    // Two 3×2 blocks side by side in one buffer; copy right block from left.
    let mut buf = vec![0.0f32; 12];
    // Fill the left block (offset 0, 3 rows × 2 cols).
    let left = MatSlice::dense(0, 3, 2);
    for c in 0..2u32 {
        for r in 0..3u32 {
            buf[left.idx(r, c)] = (c * 3 + r) as f32;
        }
    }
    let right = MatSlice::dense(6, 3, 2);
    copy_from(&mut buf, right, left);
    for c in 0..2u32 {
        for r in 0..3u32 {
            approx_eq(buf[right.idx(r, c)], buf[left.idx(r, c)]);
        }
    }
}

//
// gemm.
//

#[test]
fn gemm_beta_zero_computes_c_eq_alpha_ab() {
    //  A (2×3) = [[1, 2, 3],
    //             [4, 5, 6]]
    //  B (3×2) = [[1, 4],
    //             [2, 5],
    //             [3, 6]]
    //  A·B = [[14, 32],
    //         [32, 77]]
    // Layout: A at offset 0 (6 entries), B at offset 6 (6 entries), C at offset 12.
    let mut buf = vec![0.0f32; 16];
    let a = MatSlice::dense(0, 2, 3);
    let b = MatSlice::dense(6, 3, 2);
    let c = MatSlice::dense(12, 2, 2);
    // A column-major:
    buf[a.idx(0, 0)] = 1.0;
    buf[a.idx(1, 0)] = 4.0;
    buf[a.idx(0, 1)] = 2.0;
    buf[a.idx(1, 1)] = 5.0;
    buf[a.idx(0, 2)] = 3.0;
    buf[a.idx(1, 2)] = 6.0;
    // B column-major:
    buf[b.idx(0, 0)] = 1.0;
    buf[b.idx(1, 0)] = 2.0;
    buf[b.idx(2, 0)] = 3.0;
    buf[b.idx(0, 1)] = 4.0;
    buf[b.idx(1, 1)] = 5.0;
    buf[b.idx(2, 1)] = 6.0;

    gemm(&mut buf, c, 1.0, a, b, 0.0);

    approx_eq(buf[c.idx(0, 0)], 14.0);
    approx_eq(buf[c.idx(1, 0)], 32.0);
    approx_eq(buf[c.idx(0, 1)], 32.0);
    approx_eq(buf[c.idx(1, 1)], 77.0);
}

#[test]
fn gemm_accumulates_with_beta_and_alpha() {
    // c := 2·c + 3·(A·B) with A·B = [[1]] (1×1 case).
    let mut buf = vec![5.0f32; 3]; // A=buf[0], B=buf[1], C=buf[2]
    let a = MatSlice::dense(0, 1, 1);
    let b = MatSlice::dense(1, 1, 1);
    let c = MatSlice::dense(2, 1, 1);
    buf[a.idx(0, 0)] = 2.0;
    buf[b.idx(0, 0)] = 3.0;
    buf[c.idx(0, 0)] = 1.0; // initial c
    gemm(&mut buf, c, 3.0, a, b, 2.0);
    approx_eq(buf[c.idx(0, 0)], 2.0 * 1.0 + 3.0 * (2.0 * 3.0));
}

//
// gemm_mat3_lhs.
//

#[test]
fn gemm_mat3_lhs_matches_dense_gemm() {
    // Verify gemm_mat3_lhs against a reference computation.
    let a = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 10.0),
    );
    // B (3×2) random-ish.
    let mut buf = vec![0.0f32; 3 * 2 + 3 * 2]; // B then C
    let b = MatSlice::dense(0, 3, 2);
    let c = MatSlice::dense(6, 3, 2);
    buf[b.idx(0, 0)] = 1.0;
    buf[b.idx(1, 0)] = 0.0;
    buf[b.idx(2, 0)] = -1.0;
    buf[b.idx(0, 1)] = 2.0;
    buf[b.idx(1, 1)] = 1.0;
    buf[b.idx(2, 1)] = 0.5;
    // Set initial C to something nonzero to exercise β.
    buf[c.idx(0, 0)] = 10.0;
    buf[c.idx(1, 0)] = 20.0;
    buf[c.idx(2, 0)] = 30.0;
    buf[c.idx(0, 1)] = -1.0;
    buf[c.idx(1, 1)] = -2.0;
    buf[c.idx(2, 1)] = -3.0;

    // Snapshot B and initial C.
    let b_vals = [[1.0f32, 0.0, -1.0], [2.0, 1.0, 0.5]];
    let c0 = [[10.0f32, 20.0, 30.0], [-1.0, -2.0, -3.0]];

    let alpha = 0.5f32;
    let beta = 2.0f32;
    gemm_mat3_lhs(&mut buf, c, alpha, a, b, beta);

    // Reference: c[:, j] = beta * c0[:, j] + alpha * A · B[:, j]
    for j in 0..2usize {
        let bcol = Vec3::new(b_vals[j][0], b_vals[j][1], b_vals[j][2]);
        let prod = a.x_axis * bcol.x + a.y_axis * bcol.y + a.z_axis * bcol.z;
        let want = Vec3::new(
            beta * c0[j][0] + alpha * prod.x,
            beta * c0[j][1] + alpha * prod.y,
            beta * c0[j][2] + alpha * prod.z,
        );
        approx_eq(buf[c.idx(0, j as u32)], want.x);
        approx_eq(buf[c.idx(1, j as u32)], want.y);
        approx_eq(buf[c.idx(2, j as u32)], want.z);
    }
}

//
// axpy_mat.
//

#[test]
fn axpy_mat_adds_alpha_scaled_source() {
    let mut dst = vec![1.0f32; 6]; // 3×2 = 6
    let src = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let dst_view = MatSlice::dense(0, 3, 2);
    let src_view = MatSlice::dense(0, 3, 2);
    axpy_mat(&mut dst, dst_view, 2.0, &src, src_view);
    let expected = [
        1.0 + 2.0 * 10.0,
        1.0 + 2.0 * 20.0,
        1.0 + 2.0 * 30.0,
        1.0 + 2.0 * 40.0,
        1.0 + 2.0 * 50.0,
        1.0 + 2.0 * 60.0,
    ];
    assert_slice_eq(&dst, &expected);
}

//
// skew_tr.
//

#[test]
fn skew_tr_maps_v_to_v_cross_t() {
    let t = Vec3::new(1.0, 2.0, 3.0);
    let m = skew_tr(t);
    let v = Vec3::new(4.0, -1.0, 5.0);
    // skew_tr(t) · v = v × t
    let lhs = m.x_axis * v.x + m.y_axis * v.y + m.z_axis * v.z;
    let rhs = v.cross(t);
    approx_eq(lhs.x, rhs.x);
    approx_eq(lhs.y, rhs.y);
    approx_eq(lhs.z, rhs.z);
}

#[test]
fn skew_maps_v_to_t_cross_v() {
    let t = Vec3::new(1.0, 2.0, 3.0);
    let m = skew(t);
    let v = Vec3::new(4.0, -1.0, 5.0);
    // skew(t) · v = t × v
    let lhs = m.x_axis * v.x + m.y_axis * v.y + m.z_axis * v.z;
    let rhs = t.cross(v);
    approx_eq(lhs.x, rhs.x);
    approx_eq(lhs.y, rhs.y);
    approx_eq(lhs.z, rhs.z);
}

//
// gemm_tr.
//

#[test]
fn gemm_tr_matches_reference() {
    // Compute C (k×n) = α · Aᵀ · B + β · C with A (m×k), B (m×n).
    // Put A in its own buffer, B in its own buffer, C in its own buffer.
    let (m, k, n) = (4u32, 3u32, 2u32);
    let mut buf_a = vec![0.0f32; (m * k) as usize];
    let mut buf_b = vec![0.0f32; (m * n) as usize];
    let mut buf_c = vec![0.0f32; (k * n) as usize];
    let a_view = MatSlice::dense(0, m, k);
    let b_view = MatSlice::dense(0, m, n);
    let c_view = MatSlice::dense(0, k, n);

    // Fill A column-major.
    let a_rows: &[&[f32]] = &[
        &[1.0, 0.5, -1.0],
        &[2.0, 1.5, 0.0],
        &[3.0, -0.5, 1.0],
        &[0.0, 2.0, 2.0],
    ];
    for r in 0..m {
        for c in 0..k {
            buf_a[a_view.idx(r, c)] = a_rows[r as usize][c as usize];
        }
    }
    let b_rows: &[&[f32]] = &[&[1.0, 2.0], &[0.0, 1.0], &[-1.0, 0.5], &[0.5, -1.0]];
    for r in 0..m {
        for c in 0..n {
            buf_b[b_view.idx(r, c)] = b_rows[r as usize][c as usize];
        }
    }
    let c_initial = [[7.0f32, 11.0], [13.0, 17.0], [19.0, 23.0]];
    for r in 0..k {
        for c in 0..n {
            buf_c[c_view.idx(r, c)] = c_initial[r as usize][c as usize];
        }
    }

    let alpha = 2.0f32;
    let beta = 3.0f32;
    gemm_tr(
        &mut buf_c, c_view, alpha, &buf_a, a_view, &buf_b, b_view, beta,
    );

    // Reference: C[i, j] = β · C₀[i, j] + α · Σ_p A[p, i] · B[p, j].
    for i in 0..k as usize {
        for j in 0..n as usize {
            let mut s = 0.0f32;
            for p in 0..m as usize {
                s += a_rows[p][i] * b_rows[p][j];
            }
            let want = beta * c_initial[i][j] + alpha * s;
            approx_eq(buf_c[c_view.idx(i as u32, j as u32)], want);
        }
    }
}

//
// quadform_spatial  — M = α·Jᵀ·diag(m·I₃, I)·J + β·M.
//

/// Reference dense implementation of the same CRBA step.
fn reference_quadform(j: &[[f32; 6]], mass: f32, inertia: Mat3) -> Vec<Vec<f32>> {
    // j is a list of ndofs column vectors (each length 6).
    let ndofs = j.len();
    let mut m = vec![vec![0.0f32; ndofs]; ndofs];
    // W = diag(mass·I₃, inertia).
    for r in 0..ndofs {
        for c in 0..ndofs {
            let jv_r = Vec3::new(j[r][0], j[r][1], j[r][2]);
            let jw_r = Vec3::new(j[r][3], j[r][4], j[r][5]);
            let jv_c = Vec3::new(j[c][0], j[c][1], j[c][2]);
            let jw_c = Vec3::new(j[c][3], j[c][4], j[c][5]);
            let i_jw = inertia.x_axis * jw_c.x + inertia.y_axis * jw_c.y + inertia.z_axis * jw_c.z;
            m[r][c] = mass * jv_r.dot(jv_c) + jw_r.dot(i_jw);
        }
    }
    m
}

#[test]
fn quadform_spatial_matches_reference() {
    // Random-ish 6×2 jacobian.
    let j_cols: Vec<[f32; 6]> = vec![
        [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
        [0.5, 1.0, -1.0, 1.0, 0.0, 0.5],
    ];
    let ndofs = j_cols.len() as u32;
    let mass = 3.0;
    let inertia = Mat3::from_cols(
        Vec3::new(2.0, 0.1, 0.0),
        Vec3::new(0.1, 3.0, 0.2),
        Vec3::new(0.0, 0.2, 4.0),
    );

    // Pack J into a flat buffer (column-major, 6 rows).
    let mut buf_j = vec![0.0f32; 6 * ndofs as usize];
    let j_view = MatSlice::dense(0, 6, ndofs);
    for (c, col) in j_cols.iter().enumerate() {
        for r in 0..6 {
            buf_j[j_view.idx(r as u32, c as u32)] = col[r];
        }
    }

    // Initial M = 0; compute M += 1·Jᵀ·W·J.
    let mut buf_m = vec![0.0f32; (ndofs * ndofs) as usize];
    let m_view = MatSlice::dense(0, ndofs, ndofs);
    quadform_spatial(&mut buf_m, m_view, 1.0, mass, inertia, &buf_j, j_view, 0.0);

    let want = reference_quadform(&j_cols, mass, inertia);
    for r in 0..ndofs as usize {
        for c in 0..ndofs as usize {
            approx_eq(buf_m[m_view.idx(r as u32, c as u32)], want[r][c]);
        }
    }

    // Now run again with α=2, β=3 to check the blend:
    // expected new M = 3·M_prev + 2·J^T·W·J = 3·prev + 2·prev = 5·prev.
    let prev: Vec<f32> = buf_m.clone();
    quadform_spatial(&mut buf_m, m_view, 2.0, mass, inertia, &buf_j, j_view, 3.0);
    for (i, &p) in prev.iter().enumerate() {
        approx_eq(buf_m[i], 5.0 * p);
    }
}

//
// gemv_tr_spatial — y := β·y + α·Aᵀ·x, with x a 6-vector.
//

#[test]
fn gemv_tr_spatial_matches_reference() {
    // A is 6 × 3. y has length 3. x is the 6-vector.
    let a_cols: [[f32; 6]; 3] = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.5, -1.0, 1.5, -2.0, 2.5, -3.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ];
    let mut buf_a = vec![0.0f32; 6 * 3];
    let a_view = MatSlice::dense(0, 6, 3);
    for (c, col) in a_cols.iter().enumerate() {
        for r in 0..6 {
            buf_a[a_view.idx(r as u32, c as u32)] = col[r];
        }
    }

    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut buf_y = vec![10.0f32, 20.0, 30.0];

    gemv_tr_spatial(&mut buf_y, 0, 2.0, &buf_a, a_view, x, 0.5);

    // Reference: y[c] = 0.5·y_prev[c] + 2·Σ_r A[r, c]·x[r].
    let y_prev = [10.0f32, 20.0, 30.0];
    for c in 0..3 {
        let mut s = 0.0f32;
        for r in 0..6 {
            s += a_cols[c][r] * x[r];
        }
        approx_eq(buf_y[c], 0.5 * y_prev[c] + 2.0 * s);
    }
}

//
// LU decompose + solve.
//

/// Write a row-major list of rows into a column-major flat buffer view.
fn pack_matrix(buf: &mut [f32], m: MatSlice, rows: &[&[f32]]) {
    assert_eq!(rows.len() as u32, m.rows);
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.len() as u32, m.cols);
        for (c, &v) in row.iter().enumerate() {
            buf[m.idx(r as u32, c as u32)] = v;
        }
    }
}

/// Dense matrix-vector multiply for test verification.
fn matvec(m_rows: &[&[f32]], x: &[f32]) -> Vec<f32> {
    let n_rows = m_rows.len();
    let n_cols = m_rows[0].len();
    assert_eq!(x.len(), n_cols);
    let mut y = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let mut s = 0.0f32;
        for c in 0..n_cols {
            s += m_rows[r][c] * x[c];
        }
        y[r] = s;
    }
    y
}

#[test]
fn lu_solve_identity_recovers_rhs() {
    let n = 4u32;
    let mut buf_m = vec![0.0f32; (n * n) as usize];
    let m = MatSlice::dense(0, n, n);
    for i in 0..n {
        buf_m[m.idx(i, i)] = 1.0;
    }
    let mut pivots = vec![0u32; n as usize];
    lu_decompose(&mut buf_m, m, &mut pivots, 0);

    let mut rhs = vec![7.0, -3.0, 2.5, 1.25];
    let want = rhs.clone();
    lu_solve_in_place(&buf_m, m, &pivots, 0, &mut rhs, 0);
    assert_slice_eq(&rhs, &want);
}

#[test]
fn lu_solve_spd_matrix() {
    // A symmetric positive-definite 3×3 (mimics a small mass matrix).
    //  M = [[4, 1, 2],
    //       [1, 5, 3],
    //       [2, 3, 6]]
    let rows: &[&[f32]] = &[&[4.0, 1.0, 2.0], &[1.0, 5.0, 3.0], &[2.0, 3.0, 6.0]];
    let n = 3u32;
    let mut buf = vec![0.0f32; (n * n) as usize];
    let m = MatSlice::dense(0, n, n);
    pack_matrix(&mut buf, m, rows);

    // Decompose.
    let mut pivots = vec![0u32; n as usize];
    lu_decompose(&mut buf, m, &mut pivots, 0);

    // Solve M · x = b with b = [1, 2, 3].
    let mut rhs = vec![1.0f32, 2.0, 3.0];
    lu_solve_in_place(&buf, m, &pivots, 0, &mut rhs, 0);

    // Verify by multiplying back: M · x ≈ b.
    let mx = matvec(rows, &rhs);
    assert_slice_eq(&mx, &[1.0, 2.0, 3.0]);
}

#[test]
fn lu_solve_requires_pivoting() {
    // This matrix has a zero top-left entry, so Doolittle without pivoting would fail.
    // Partial pivoting should swap rows and produce a correct factor + solve.
    //  M = [[0, 2, 1],
    //       [1, 0, 3],
    //       [4, 1, 0]]
    let rows: &[&[f32]] = &[&[0.0, 2.0, 1.0], &[1.0, 0.0, 3.0], &[4.0, 1.0, 0.0]];
    let n = 3u32;
    let mut buf = vec![0.0f32; (n * n) as usize];
    let m = MatSlice::dense(0, n, n);
    pack_matrix(&mut buf, m, rows);

    let mut pivots = vec![0u32; n as usize];
    lu_decompose(&mut buf, m, &mut pivots, 0);

    // Pick a known solution, compute b = M·x, then confirm the solve recovers x.
    let x_true = [1.5f32, -0.5, 2.0];
    let b = matvec(rows, &x_true);

    let mut rhs = b.clone();
    lu_solve_in_place(&buf, m, &pivots, 0, &mut rhs, 0);
    assert_slice_eq(&rhs, &x_true);
}

#[test]
fn lu_factor_reused_across_multiple_rhs() {
    // The point of splitting `lu_decompose` and `lu_solve` is that we should be able
    // to solve multiple right-hand sides without re-factoring. Verify that here.
    let rows: &[&[f32]] = &[
        &[6.0, 2.0, 1.0, 0.0],
        &[2.0, 5.0, 2.0, 1.0],
        &[1.0, 2.0, 4.0, 1.0],
        &[0.0, 1.0, 1.0, 3.0],
    ];
    let n = 4u32;
    let mut buf = vec![0.0f32; (n * n) as usize];
    let m = MatSlice::dense(0, n, n);
    pack_matrix(&mut buf, m, rows);

    // Decompose once.
    let mut pivots = vec![0u32; n as usize];
    lu_decompose(&mut buf, m, &mut pivots, 0);

    // Solve three different RHSes with the same factorization.
    let rhss = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, -2.0, 3.0, 4.0],
    ];
    for b in &rhss {
        let mut rhs = b.to_vec();
        lu_solve_in_place(&buf, m, &pivots, 0, &mut rhs, 0);
        // Verify: M · rhs ≈ b.
        let mx = matvec(rows, &rhs);
        assert_slice_eq(&mx, b);
    }
}

#[test]
fn lu_solve_respects_offsets() {
    // Run two independent systems in one set of buffers using nonzero offsets —
    // this is how the multibody kernels share a flat buffer across multibodies.
    let rows_a: &[&[f32]] = &[&[3.0, 1.0], &[1.0, 2.0]];
    let rows_b: &[&[f32]] = &[&[5.0, 2.0], &[2.0, 4.0]];
    let n = 2u32;

    // Layout: [A (4) | B (4)] for matrices, pivots similarly, RHS similarly.
    let mut buf_m = vec![0.0f32; 8];
    let m_a = MatSlice::dense(0, n, n);
    let m_b = MatSlice::dense(4, n, n);
    pack_matrix(&mut buf_m, m_a, rows_a);
    pack_matrix(&mut buf_m, m_b, rows_b);

    let mut pivots = vec![0u32; 4];
    lu_decompose(&mut buf_m, m_a, &mut pivots, 0);
    lu_decompose(&mut buf_m, m_b, &mut pivots, 2);

    let x_a_true = [1.0f32, -1.0];
    let x_b_true = [2.0f32, 0.5];
    let b_a = matvec(rows_a, &x_a_true);
    let b_b = matvec(rows_b, &x_b_true);

    let mut buf_rhs = vec![b_a[0], b_a[1], b_b[0], b_b[1]];
    lu_solve_in_place(&buf_m, m_a, &pivots, 0, &mut buf_rhs, 0);
    lu_solve_in_place(&buf_m, m_b, &pivots, 2, &mut buf_rhs, 2);

    assert_slice_eq(&buf_rhs[0..2], &x_a_true);
    assert_slice_eq(&buf_rhs[2..4], &x_b_true);
}
