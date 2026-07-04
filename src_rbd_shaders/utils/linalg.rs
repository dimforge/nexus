//! Flat-buffer linear-algebra primitives.
//!
//! A GPU-friendly subset of nalgebra's matrix-view / BLAS-style API. Mirrors the
//! operations used by rapier's reduced-coordinates multibody (`fill`, `copy_from`,
//! `gemm`, `quadform`, `gemv_tr`, …) but on flat `&[f32]` / `&mut [f32]` buffers
//! addressed by `(offset, rows, cols, lead)` tuples — no pointer arithmetic,
//! SPIR-V compatible.
//!
//! All matrices are dense column-major. Views can overlap the same buffer; the
//! caller is responsible for avoiding write-write conflicts.

#[cfg(feature = "dim2")]
use glamx::Vec2;
use glamx::{Mat3, Vec3};
use khal_std::index::MaybeIndexUnchecked;

/// Maximum number of DOFs per multibody that operations stack-allocating scratch
/// space assume. If a multibody exceeds this, callers must split or extend the
/// scratch arrays.
pub const MAX_MB_DOFS: usize = 64;

/// A column-major matrix view into a flat f32 buffer.
#[derive(Copy, Clone)]
pub struct MatSlice {
    /// Offset (in f32 entries) of the (0, 0) element inside the backing buffer.
    pub offset: usize,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
    /// Leading dimension — distance between columns, in f32 entries.
    pub lead: u32,
}

impl MatSlice {
    /// Dense (`lead = rows`) view at `offset`.
    #[inline]
    pub fn dense(offset: usize, rows: u32, cols: u32) -> Self {
        Self {
            offset,
            rows,
            cols,
            lead: rows,
        }
    }

    /// Flat index of element `(r, c)`.
    #[inline]
    pub fn idx(&self, r: u32, c: u32) -> usize {
        self.offset + (c * self.lead + r) as usize
    }

    /// Sub-view starting at `(r0, c0)` with shape `(nr × nc)`. Inherits `lead`.
    #[inline]
    pub fn view(&self, r0: u32, c0: u32, nr: u32, nc: u32) -> Self {
        Self {
            offset: self.offset + (c0 * self.lead + r0) as usize,
            rows: nr,
            cols: nc,
            lead: self.lead,
        }
    }

    /// `n` consecutive rows starting at `start`.
    #[inline]
    pub fn fixed_rows(&self, start: u32, n: u32) -> Self {
        self.view(start, 0, n, self.cols)
    }

    /// `n` consecutive columns starting at `start`.
    #[inline]
    pub fn columns(&self, start: u32, n: u32) -> Self {
        self.view(0, start, self.rows, n)
    }

    /// Two disjoint row ranges in the same view.
    #[inline]
    pub fn rows_range_pair(&self, r0a: u32, na: u32, r0b: u32, nb: u32) -> (Self, Self) {
        (self.fixed_rows(r0a, na), self.fixed_rows(r0b, nb))
    }
}

/// `m := val` (element-wise).
#[inline]
pub fn fill(buf: &mut [f32], m: MatSlice, val: f32) {
    for c in 0..m.cols {
        for r in 0..m.rows {
            buf.write(m.idx(r, c), val);
        }
    }
}

/// `dst := src`. `dst` and `src` must be disjoint in memory.
#[inline]
pub fn copy_from(buf: &mut [f32], dst: MatSlice, src: MatSlice) {
    for c in 0..dst.cols {
        for r in 0..dst.rows {
            let v = buf.read(src.idx(r, c));
            buf.write(dst.idx(r, c), v);
        }
    }
}

/// `c := beta * c + alpha * a * b`. `a`, `b`, `c` are all views into `buf`;
/// `c` must be disjoint from `a` and `b`.
#[inline]
pub fn gemm(buf: &mut [f32], c: MatSlice, alpha: f32, a: MatSlice, b: MatSlice, beta: f32) {
    let kmax = a.cols;
    for j in 0..c.cols {
        for i in 0..c.rows {
            let mut s = 0.0f32;
            for k in 0..kmax {
                s += buf.read(a.idx(i, k)) * buf.read(b.idx(k, j));
            }
            let cur = buf.read(c.idx(i, j));
            buf.write(c.idx(i, j), beta * cur + alpha * s);
        }
    }
}

/// `c := beta * c + alpha * A * b` where `A` is an inline 3×3 (stack) and `b`, `c`
/// are views with `rows == 3`. Matches rapier's `link_j_v.gemm(1.0, &shift_tr, &J_w, 1.0)`.
#[inline]
pub fn gemm_mat3_lhs(buf: &mut [f32], c: MatSlice, alpha: f32, a: Mat3, b: MatSlice, beta: f32) {
    for j in 0..c.cols {
        let bx = buf.read(b.idx(0, j));
        let by = buf.read(b.idx(1, j));
        let bz = buf.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf.write(i0, beta * buf.read(i0) + alpha * p.x);
        buf.write(i1, beta * buf.read(i1) + alpha * p.y);
        buf.write(i2, beta * buf.read(i2) + alpha * p.z);
    }
}

/// `dst += alpha * src`. `src` lives in a separate buffer (stack scratch or another tensor).
#[inline]
pub fn axpy_mat(buf_dst: &mut [f32], dst: MatSlice, alpha: f32, buf_src: &[f32], src: MatSlice) {
    for c in 0..dst.cols {
        for r in 0..dst.rows {
            let cur = buf_dst.read(dst.idx(r, c));
            let s = buf_src.read(src.idx(r, c));
            buf_dst.write(dst.idx(r, c), cur + alpha * s);
        }
    }
}

/// `m := beta * m + alpha * Jᵀ · diag(mass·I₃, I_world) · J` (3D).
///
/// The spatial mass matrix is block-diagonal: a scalar `mass` on the linear rows
/// and a 3×3 world-space `inertia` on the angular rows. `j` is `6 × ndofs`, `m`
/// is `ndofs × ndofs`.
#[cfg(feature = "dim3")]
#[inline]
pub fn quadform_spatial(
    buf_m: &mut [f32],
    m: MatSlice,
    alpha: f32,
    mass: f32,
    inertia: Mat3,
    buf_j: &[f32],
    j: MatSlice,
    beta: f32,
) {
    let ndofs = m.rows;
    // For each column `cc`, compute the matching WJ column (6 floats in registers)
    // and accumulate `M[:, cc] += alpha · Jᵀ · wj_col + beta · M[:, cc]`.
    for cc in 0..ndofs {
        let jvc = Vec3::new(
            buf_j.read(j.idx(0, cc)),
            buf_j.read(j.idx(1, cc)),
            buf_j.read(j.idx(2, cc)),
        );
        let jwc = Vec3::new(
            buf_j.read(j.idx(3, cc)),
            buf_j.read(j.idx(4, cc)),
            buf_j.read(j.idx(5, cc)),
        );
        let wjv = jvc * mass;
        let wjw = inertia.x_axis * jwc.x + inertia.y_axis * jwc.y + inertia.z_axis * jwc.z;

        for rr in 0..ndofs {
            let jvr = Vec3::new(
                buf_j.read(j.idx(0, rr)),
                buf_j.read(j.idx(1, rr)),
                buf_j.read(j.idx(2, rr)),
            );
            let jwr = Vec3::new(
                buf_j.read(j.idx(3, rr)),
                buf_j.read(j.idx(4, rr)),
                buf_j.read(j.idx(5, rr)),
            );
            let s = jvr.x * wjv.x
                + jvr.y * wjv.y
                + jvr.z * wjv.z
                + jwr.x * wjw.x
                + jwr.y * wjw.y
                + jwr.z * wjw.z;
            let idx = m.idx(rr, cc);
            buf_m.write(idx, beta * buf_m.read(idx) + alpha * s);
        }
    }
}

/// `m := beta * m + alpha * Jᵀ · diag(mass·I₂, inertia) · J` (2D).
///
/// The spatial mass matrix is diagonal: `(mass, mass, inertia)`. `j` is
/// `3 × ndofs`, `m` is `ndofs × ndofs`.
#[cfg(feature = "dim2")]
#[inline]
pub fn quadform_spatial(
    buf_m: &mut [f32],
    m: MatSlice,
    alpha: f32,
    mass: f32,
    inertia: f32,
    buf_j: &[f32],
    j: MatSlice,
    beta: f32,
) {
    let ndofs = m.rows;
    for cc in 0..ndofs {
        let wj0 = buf_j.read(j.idx(0, cc)) * mass;
        let wj1 = buf_j.read(j.idx(1, cc)) * mass;
        let wj2 = buf_j.read(j.idx(2, cc)) * inertia;

        for rr in 0..ndofs {
            let s = buf_j.read(j.idx(0, rr)) * wj0
                + buf_j.read(j.idx(1, rr)) * wj1
                + buf_j.read(j.idx(2, rr)) * wj2;
            let idx = m.idx(rr, cc);
            buf_m.write(idx, beta * buf_m.read(idx) + alpha * s);
        }
    }
}

/// `y := beta * y + alpha * Aᵀ · x` where `A` is a view into `buf_a` and `x` is a
/// 6-vector. `y_offset` is where `y` starts in `buf_y`; `y` has length `a.cols`.
#[inline]
pub fn gemv_tr_spatial<const XDIM: usize>(
    buf_y: &mut [f32],
    y_offset: usize,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    x: [f32; XDIM], // TODO: use const-generics instead of hard-coding 6
    beta: f32,
) {
    for c in 0..a.cols {
        let mut s = 0.0f32;
        for r in 0..XDIM as u32 {
            s += buf_a.read(a.idx(r, c)) * x[r as usize];
        }
        let idx = y_offset + c as usize;
        let cur = buf_y.read(idx);
        buf_y.write(idx, beta * cur + alpha * s);
    }
}

/// `y := beta * y + alpha * Aᵀ · (x_lin, x_ang)` for a `SPATIAL_DIM × ndofs`
/// jacobian split into its linear and angular blocks.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemv_tr_spatial_split(
    buf_y: &mut [f32],
    y_offset: usize,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    x_lin: Vec3,
    x_ang: Vec3,
    beta: f32,
) {
    for c in 0..a.cols {
        let s = buf_a.read(a.idx(0, c)) * x_lin.x
            + buf_a.read(a.idx(1, c)) * x_lin.y
            + buf_a.read(a.idx(2, c)) * x_lin.z
            + buf_a.read(a.idx(3, c)) * x_ang.x
            + buf_a.read(a.idx(4, c)) * x_ang.y
            + buf_a.read(a.idx(5, c)) * x_ang.z;
        let idx = y_offset + c as usize;
        let cur = buf_y.read(idx);
        buf_y.write(idx, beta * cur + alpha * s);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemv_tr_spatial_split(
    buf_y: &mut [f32],
    y_offset: usize,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    x_lin: Vec2,
    x_ang: f32,
    beta: f32,
) {
    for c in 0..a.cols {
        let s = buf_a.read(a.idx(0, c)) * x_lin.x
            + buf_a.read(a.idx(1, c)) * x_lin.y
            + buf_a.read(a.idx(2, c)) * x_ang;
        let idx = y_offset + c as usize;
        let cur = buf_y.read(idx);
        buf_y.write(idx, beta * cur + alpha * s);
    }
}

/// `[t]_×ᵀ`, the transpose of the cross-product matrix: `skew_tr(t) · v = v × t`.
#[inline]
pub fn skew_tr(t: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, -t.z, t.y),
        Vec3::new(t.z, 0.0, -t.x),
        Vec3::new(-t.y, t.x, 0.0),
    )
}

/// `[t]_×`, the cross-product matrix: `skew(t) · v = t × v`.
#[inline]
pub fn skew(t: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, t.z, -t.y),
        Vec3::new(-t.z, 0.0, t.x),
        Vec3::new(t.y, -t.x, 0.0),
    )
}

/// `c := beta * c + alpha * [t]_×ᵀ · b`, with `b`, `c` views into different
/// flat buffers. `b` has `ANG_DIM` rows, `c` has `DIM` rows; `t` is a `Vector`.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_skew_tr_lhs_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    let a = skew_tr(t);
    for j in 0..c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_skew_tr_lhs_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec2,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    for j in 0..c.cols {
        let bw = buf_b.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * (t.y * bw));
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * (-t.x * bw));
    }
}

/// `c := beta * c + alpha * [t]_× · b` with `b`, `c` views into different
/// flat buffers. `b` has `DIM` rows, `c` has `DIM` rows; `t` is an
/// `AngVector`.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_skew_lhs_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    let a = skew(t);
    for j in 0..c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_skew_lhs_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    omega: f32,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    for j in 0..c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * (-omega * by));
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * (omega * bx));
    }
}

/// `c := beta * c + alpha * inertia · b` (cross-buffer), where `inertia`
/// is the world-space rigid-body inertia (Mat3 in 3D, scalar in 2D), and
/// `b`, `c` have `ANG_DIM` rows.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_inertia_lhs_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    inertia: Mat3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    for j in 0..c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = inertia.x_axis * bx + inertia.y_axis * by + inertia.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_inertia_lhs_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    inertia: f32,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    for j in 0..c.cols {
        let bw = buf_b.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * (inertia * bw));
    }
}

/// `c := beta * c + alpha * (parent_w · skew_tr(shift)) · b` — a fused
/// `[parent_w]_× · [shift]_×ᵀ` left-multiply, used by the Coriolis term
/// `coriolis_v += (parent_w · shift_cross_tr) · parent_j_w`.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_omega_skew_tr_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    parent_w: Vec3,
    shift: Vec3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    let combined = skew(parent_w) * skew_tr(shift);
    for j in 0..c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = combined.x_axis * bx + combined.y_axis * by + combined.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_omega_skew_tr_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    parent_w: f32,
    shift: Vec2,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    // [ω]_× · skew_tr(shift) acting on a scalar bw:
    // skew_tr(shift) · bw = (shift.y · bw, -shift.x · bw)
    // [ω]_× · (a, b) = (-ω · b, ω · a) = (ω · shift.x · bw, ω · shift.y · bw)
    for j in 0..c.cols {
        let bw = buf_b.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf_c.write(
            i0,
            beta * buf_c.read(i0) + alpha * (parent_w * shift.x * bw),
        );
        buf_c.write(
            i1,
            beta * buf_c.read(i1) + alpha * (parent_w * shift.y * bw),
        );
    }
}

/// Same-buffer variant of [`gemm_skew_tr_lhs_cross_buf`] (`b` and `c` are
/// disjoint views into the same flat buffer).
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_skew_tr_lhs(buf: &mut [f32], c: MatSlice, alpha: f32, t: Vec3, b: MatSlice, beta: f32) {
    let a = skew_tr(t);
    for j in 0..c.cols {
        let bx = buf.read(b.idx(0, j));
        let by = buf.read(b.idx(1, j));
        let bz = buf.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf.write(i0, beta * buf.read(i0) + alpha * p.x);
        buf.write(i1, beta * buf.read(i1) + alpha * p.y);
        buf.write(i2, beta * buf.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_skew_tr_lhs(buf: &mut [f32], c: MatSlice, alpha: f32, t: Vec2, b: MatSlice, beta: f32) {
    for j in 0..c.cols {
        let bw = buf.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf.write(i0, beta * buf.read(i0) + alpha * (t.y * bw));
        buf.write(i1, beta * buf.read(i1) + alpha * (-t.x * bw));
    }
}

/// `dst := alpha * src + dst`, but `src` lives in its own flat buffer.
/// Same as [`axpy_mat`] but kept under a name that makes the reuse across buffers explicit.
#[inline]
pub fn axpy_mat_scaled(
    buf_dst: &mut [f32],
    dst: MatSlice,
    alpha: f32,
    buf_src: &[f32],
    src: MatSlice,
) {
    axpy_mat(buf_dst, dst, alpha, buf_src, src);
}

/// `c := beta * c + alpha * aᵀ * b`. `a` and `b` live in independent flat buffers,
/// `c` in a third. Shapes: `a` is `m × k`, `b` is `m × n`, `c` is `k × n`.
///
/// Matches rapier's `acc_augmented_mass.gemm_tr(1.0, rb_j, &i_coriolis_dt, 1.0)`.
#[inline]
pub fn gemm_tr(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    // a^T is a.cols × a.rows. So we want c[i, j] = sum_k a[k, i] * b[k, j] for
    // i in 0..c.rows, j in 0..c.cols, k in 0..a.rows.
    let kmax = a.rows;
    for j in 0..c.cols {
        for i in 0..c.rows {
            let mut s = 0.0f32;
            for k in 0..kmax {
                s += buf_a.read(a.idx(k, i)) * buf_b.read(b.idx(k, j));
            }
            let idx = c.idx(i, j);
            let cur = buf_c.read(idx);
            buf_c.write(idx, beta * cur + alpha * s);
        }
    }
}

//
// LU decomposition + solve, split so the factorization can be reused with
// multiple right-hand sides (mirrors `nalgebra::LU` / `LU::solve_mut`).
//

/// In-place LU factorization with partial pivoting (Doolittle form).
///
/// Overwrites `m` (a square `n × n` dense column-major view) with the LU factors:
/// strictly-below-diagonal entries hold `L` (with implicit unit diagonal), the
/// diagonal and above hold `U`. Row pivots are written to `pivots[0..n]` —
/// `pivots[k]` is the row that was swapped with row `k` during elimination step
/// `k`. `pivots_offset` is where this multibody's pivot slot starts in `buf_pivots`.
#[inline]
pub fn lu_decompose(buf_m: &mut [f32], m: MatSlice, buf_pivots: &mut [u32], pivots_offset: usize) {
    let n = m.rows;
    for k in 0..n {
        // Partial pivot: find max |M[i, k]| for i in k..n.
        let mut pivot_row = k;
        let mut pivot_val = {
            let v = buf_m.read(m.idx(k, k));
            if v >= 0.0 { v } else { -v }
        };
        for i in (k + 1)..n {
            let v = buf_m.read(m.idx(i, k));
            let av = if v >= 0.0 { v } else { -v };
            if av > pivot_val {
                pivot_val = av;
                pivot_row = i;
            }
        }
        buf_pivots.write(pivots_offset + k as usize, pivot_row);

        // Row swap k ↔ pivot_row (full row since we haven't computed past col k).
        if pivot_row != k {
            for c in 0..n {
                let a = buf_m.read(m.idx(k, c));
                let b = buf_m.read(m.idx(pivot_row, c));
                buf_m.write(m.idx(k, c), b);
                buf_m.write(m.idx(pivot_row, c), a);
            }
        }

        // Scale sub-column below the pivot: M[i, k] /= M[k, k].
        let akk = buf_m.read(m.idx(k, k));
        let inv_akk = if akk != 0.0 { 1.0 / akk } else { 0.0 };
        for r in (k + 1)..n {
            let v = buf_m.read(m.idx(r, k)) * inv_akk;
            buf_m.write(m.idx(r, k), v);
        }

        // Trailing sub-matrix update: M[i, j] -= M[i, k] * M[k, j].
        for j in (k + 1)..n {
            let akj = buf_m.read(m.idx(k, j));
            for i2 in (k + 1)..n {
                let lik = buf_m.read(m.idx(i2, k));
                let v = buf_m.read(m.idx(i2, j)) - lik * akj;
                buf_m.write(m.idx(i2, j), v);
            }
        }
    }
}

/// Solve `M · x = rhs` in-place, using LU factors produced by
/// [`lu_decompose`] (and its pivot array). The result overwrites `rhs`.
///
/// `m` and `pivots` must be the exact outputs of a previous `lu_decompose` call.
/// `rhs` is an `n`-element column vector; `rhs_offset` is where it starts in `buf_rhs`.
#[inline]
pub fn lu_solve_in_place(
    buf_m: &[f32],
    m: MatSlice,
    buf_pivots: &[u32],
    pivots_offset: usize,
    buf_rhs: &mut [f32],
    rhs_offset: usize,
) {
    let n = m.rows;

    // Permute rhs in place according to the recorded pivots.
    for k in 0..n {
        let p = buf_pivots.read(pivots_offset + k as usize);
        if p != k {
            let ki = rhs_offset + k as usize;
            let pi = rhs_offset + p as usize;
            let a = buf_rhs.read(ki);
            let b = buf_rhs.read(pi);
            buf_rhs.write(ki, b);
            buf_rhs.write(pi, a);
        }
    }

    // Forward substitution: L · y = P · rhs (L is unit-lower — implicit diag = 1).
    for i in 0..n {
        let mut s = buf_rhs.read(rhs_offset + i as usize);
        for j in 0..i {
            s -= buf_m.read(m.idx(i, j)) * buf_rhs.read(rhs_offset + j as usize);
        }
        buf_rhs.write(rhs_offset + i as usize, s);
    }

    // Back substitution: U · x = y (reverse iteration — equivalent to `for ii in (0..n).rev()`).
    for step in 0..n {
        let ii = n - 1 - step;
        let mut s = buf_rhs.read(rhs_offset + ii as usize);
        for j in (ii + 1)..n {
            s -= buf_m.read(m.idx(ii, j)) * buf_rhs.read(rhs_offset + j as usize);
        }
        let u = buf_m.read(m.idx(ii, ii));
        buf_rhs.write(rhs_offset + ii as usize, if u != 0.0 { s / u } else { 0.0 });
    }
}

//
// Workgroup-parallel variants. Mirror the sequential primitives above but
// partition each iteration's work across `lanes` lanes of a SIMT workgroup.
// All control flow is uniform (every lane runs the same outer loops); the
// barriers are placed so every lane reaches them, preventing deadlocks.
//
// Conventions:
// - `lane` is `local_invocation_id.x` in the calling kernel; `0 ≤ lane < lanes`.
// - `lanes` is the workgroup width along x (e.g. `32` for the LU + mass-matrix
//   kernels).
// - Pivot/scratch broadcasts go through a `&mut u32` parameter that the kernel
//   declares with `#[spirv(workgroup)]`; we write from lane 0, barrier, then
//   read from every lane.
//

/// Same-buffer parallel variant of [`gemm_skew_tr_lhs`] (b and c are disjoint
/// views into the same flat buffer).
///
/// Each lane handles at most one column: `cols ≤ MAX_MB_DOFS = 64 = lanes`,
/// so an `if` guard suffices and we avoid `while` loops (rust-gpu lowers `for`
/// loops to structured SPIR-V cleanly, but `while` loops can produce
/// unstructured control flow that is silently miscompiled).
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_skew_tr_lhs_par(
    buf: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec3,
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let a = skew_tr(t);
    let j = lane;
    if j < c.cols {
        let bx = buf.read(b.idx(0, j));
        let by = buf.read(b.idx(1, j));
        let bz = buf.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf.write(i0, beta * buf.read(i0) + alpha * p.x);
        buf.write(i1, beta * buf.read(i1) + alpha * p.y);
        buf.write(i2, beta * buf.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_skew_tr_lhs_par(
    buf: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec2,
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        // TODO(perf): since matrices are stored column-major,
        //             the workgroup memory access into buf are not coalesced here.
        //             (we probably just need to switch storage to row-major to match
        //              the convention from vortx)
        let bw = buf.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf.write(i0, beta * buf.read(i0) + alpha * (t.y * bw));
        buf.write(i1, beta * buf.read(i1) + alpha * (-t.x * bw));
    }
}

/// `m := val` — parallel across columns.
#[inline]
pub fn fill_par(buf: &mut [f32], m: MatSlice, val: f32, lane: u32, _lanes: u32) {
    let c = lane;
    if c < m.cols {
        // TODO(perf): not a good memory access pattern if the matrix is column-major.
        for r in 0..m.rows {
            buf.write(m.idx(r, c), val);
        }
    }
}

/// `dst := src` — parallel across columns.
#[inline]
pub fn copy_from_par(buf: &mut [f32], dst: MatSlice, src: MatSlice, lane: u32, _lanes: u32) {
    // TODO(perf): memory access patern isn’t ideal for column-major matrix
    //             since `c` is the workgroup thread index.
    //             (we probably just need to switch storage to row-major to match
    //              the convention from vortx)
    let c = lane;
    if c < dst.cols {
        for r in 0..dst.rows {
            let v = buf.read(src.idx(r, c));
            buf.write(dst.idx(r, c), v);
        }
    }
}

/// Parallel `quadform_spatial` — each lane owns one column `cc = lane`
/// (cols ≤ MAX_MB_DOFS = lanes), so writes to `M[*, cc]` are race-free
/// without further synchronisation.
#[cfg(feature = "dim3")]
#[inline]
pub fn quadform_spatial_par(
    buf_m: &mut [f32],
    m: MatSlice,
    alpha: f32,
    mass: f32,
    inertia: Mat3,
    buf_j: &[f32],
    j: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let ndofs = m.rows;
    let cc = lane;
    if cc < ndofs {
        let jvc = Vec3::new(
            buf_j.read(j.idx(0, cc)),
            buf_j.read(j.idx(1, cc)),
            buf_j.read(j.idx(2, cc)),
        );
        let jwc = Vec3::new(
            buf_j.read(j.idx(3, cc)),
            buf_j.read(j.idx(4, cc)),
            buf_j.read(j.idx(5, cc)),
        );
        let wjv = jvc * mass;
        let wjw = inertia.x_axis * jwc.x + inertia.y_axis * jwc.y + inertia.z_axis * jwc.z;

        for rr in 0..ndofs {
            let jvr = Vec3::new(
                buf_j.read(j.idx(0, rr)),
                buf_j.read(j.idx(1, rr)),
                buf_j.read(j.idx(2, rr)),
            );
            let jwr = Vec3::new(
                buf_j.read(j.idx(3, rr)),
                buf_j.read(j.idx(4, rr)),
                buf_j.read(j.idx(5, rr)),
            );
            let s = jvr.x * wjv.x
                + jvr.y * wjv.y
                + jvr.z * wjv.z
                + jwr.x * wjw.x
                + jwr.y * wjw.y
                + jwr.z * wjw.z;
            let idx = m.idx(rr, cc);
            buf_m.write(idx, beta * buf_m.read(idx) + alpha * s);
        }
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn quadform_spatial_par(
    buf_m: &mut [f32],
    m: MatSlice,
    alpha: f32,
    mass: f32,
    inertia: f32,
    buf_j: &[f32],
    j: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let ndofs = m.rows;
    let cc = lane;
    if cc < ndofs {
        let wj0 = buf_j.read(j.idx(0, cc)) * mass;
        let wj1 = buf_j.read(j.idx(1, cc)) * mass;
        let wj2 = buf_j.read(j.idx(2, cc)) * inertia;

        for rr in 0..ndofs {
            let s = buf_j.read(j.idx(0, rr)) * wj0
                + buf_j.read(j.idx(1, rr)) * wj1
                + buf_j.read(j.idx(2, rr)) * wj2;
            let idx = m.idx(rr, cc);
            buf_m.write(idx, beta * buf_m.read(idx) + alpha * s);
        }
    }
}

/// Parallel `gemm_tr`: `C := beta·C + alpha·Aᵀ·B` partitioned across columns of `C`.
#[inline]
pub fn gemm_tr_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let kmax = a.rows;
    let j = lane;
    if j < c.cols {
        for i in 0..c.rows {
            let mut s = 0.0f32;
            for kk in 0..kmax {
                s += buf_a.read(a.idx(kk, i)) * buf_b.read(b.idx(kk, j));
            }
            let idx = c.idx(i, j);
            let cur = buf_c.read(idx);
            buf_c.write(idx, beta * cur + alpha * s);
        }
    }
}

/// Parallel variant of [`gemm_skew_tr_lhs_cross_buf`].
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_skew_tr_lhs_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let a = skew_tr(t);
    let j = lane;
    if j < c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_skew_tr_lhs_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec2,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        let bw = buf_b.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * (t.y * bw));
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * (-t.x * bw));
    }
}

/// Parallel variant of [`gemm_skew_lhs_cross_buf`].
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_skew_lhs_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    t: Vec3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let a = skew(t);
    let j = lane;
    if j < c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

/// Same-buffer variant of [`gemm_inertia_lhs_cross_buf_par`] — `b` and `c`
/// are disjoint views of the same flat buffer.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_inertia_lhs_par(
    buf: &mut [f32],
    c: MatSlice,
    alpha: f32,
    inertia: Mat3,
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        let bx = buf.read(b.idx(0, j));
        let by = buf.read(b.idx(1, j));
        let bz = buf.read(b.idx(2, j));
        let p = inertia.x_axis * bx + inertia.y_axis * by + inertia.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf.write(i0, beta * buf.read(i0) + alpha * p.x);
        buf.write(i1, beta * buf.read(i1) + alpha * p.y);
        buf.write(i2, beta * buf.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_inertia_lhs_par(
    buf: &mut [f32],
    c: MatSlice,
    alpha: f32,
    inertia: f32,
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        let bw = buf.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        buf.write(i0, beta * buf.read(i0) + alpha * (inertia * bw));
    }
}

/// Parallel variant of [`gemm_inertia_lhs_cross_buf`].
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_inertia_lhs_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    inertia: Mat3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = inertia.x_axis * bx + inertia.y_axis * by + inertia.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_inertia_lhs_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    inertia: f32,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        let bw = buf_b.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * (inertia * bw));
    }
}

/// Parallel variant of [`gemm_omega_skew_tr_cross_buf`].
#[cfg(feature = "dim3")]
#[inline]
pub fn gemm_omega_skew_tr_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    parent_w: Vec3,
    shift: Vec3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let combined = skew(parent_w) * skew_tr(shift);
    let j = lane;
    if j < c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = combined.x_axis * bx + combined.y_axis * by + combined.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemm_omega_skew_tr_cross_buf_par(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    parent_w: f32,
    shift: Vec2,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let j = lane;
    if j < c.cols {
        let bw = buf_b.read(b.idx(0, j));
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        buf_c.write(
            i0,
            beta * buf_c.read(i0) + alpha * (parent_w * shift.x * bw),
        );
        buf_c.write(
            i1,
            beta * buf_c.read(i1) + alpha * (parent_w * shift.y * bw),
        );
    }
}

/// Parallel `Aᵀ·(x_lin,x_ang)` for the gravity kernel.
#[cfg(feature = "dim3")]
#[inline]
pub fn gemv_tr_spatial_split_par(
    buf_y: &mut [f32],
    y_offset: usize,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    x_lin: Vec3,
    x_ang: Vec3,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let c = lane;
    if c < a.cols {
        let s = buf_a.read(a.idx(0, c)) * x_lin.x
            + buf_a.read(a.idx(1, c)) * x_lin.y
            + buf_a.read(a.idx(2, c)) * x_lin.z
            + buf_a.read(a.idx(3, c)) * x_ang.x
            + buf_a.read(a.idx(4, c)) * x_ang.y
            + buf_a.read(a.idx(5, c)) * x_ang.z;
        let idx = y_offset + c as usize;
        let cur = buf_y.read(idx);
        buf_y.write(idx, beta * cur + alpha * s);
    }
}

#[cfg(feature = "dim2")]
#[inline]
pub fn gemv_tr_spatial_split_par(
    buf_y: &mut [f32],
    y_offset: usize,
    alpha: f32,
    buf_a: &[f32],
    a: MatSlice,
    x_lin: Vec2,
    x_ang: f32,
    beta: f32,
    lane: u32,
    _lanes: u32,
) {
    let c = lane;
    if c < a.cols {
        let s = buf_a.read(a.idx(0, c)) * x_lin.x
            + buf_a.read(a.idx(1, c)) * x_lin.y
            + buf_a.read(a.idx(2, c)) * x_ang;
        let idx = y_offset + c as usize;
        let cur = buf_y.read(idx);
        buf_y.write(idx, beta * cur + alpha * s);
    }
}

// Note: parallel `lu_decompose_par` / `lu_solve_in_place_par` variants were
// explored but did not pay off at typical multibody sizes. The kernels in
// `super::dynamics::multibody::lu` use the sequential primitives instead.
