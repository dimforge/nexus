//! Register-resident LU for a small dense matrix, one row per SIMD lane.
use crate::utils::linalg::{MatSlice, VSlice};
use glamx::Vec4;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::subgroup_f_max;

// Constant register indices avoid private-array spills and dynamic register selection.
macro_rules! forward_dofs {($k:ident,$n:expr,$body:block)=>{{
    {let $k=0u32;if $k<$n $body}
    {let $k=1u32;if $k<$n $body}
    {let $k=2u32;if $k<$n $body}
    {let $k=3u32;if $k<$n $body}
    {let $k=4u32;if $k<$n $body}
    {let $k=5u32;if $k<$n $body}
    {let $k=6u32;if $k<$n $body}
    {let $k=7u32;if $k<$n $body}
    {let $k=8u32;if $k<$n $body}
    {let $k=9u32;if $k<$n $body}
    {let $k=10u32;if $k<$n $body}
    {let $k=11u32;if $k<$n $body}
    {let $k=12u32;if $k<$n $body}
    {let $k=13u32;if $k<$n $body}
    {let $k=14u32;if $k<$n $body}
    {let $k=15u32;if $k<$n $body}
    {let $k=16u32;if $k<$n $body}
    {let $k=17u32;if $k<$n $body}
    {let $k=18u32;if $k<$n $body}
    {let $k=19u32;if $k<$n $body}
    {let $k=20u32;if $k<$n $body}
    {let $k=21u32;if $k<$n $body}
    {let $k=22u32;if $k<$n $body}
    {let $k=23u32;if $k<$n $body}
    {let $k=24u32;if $k<$n $body}
    {let $k=25u32;if $k<$n $body}
    {let $k=26u32;if $k<$n $body}
    {let $k=27u32;if $k<$n $body}
    {let $k=28u32;if $k<$n $body}
    {let $k=29u32;if $k<$n $body}
    {let $k=30u32;if $k<$n $body}
    {let $k=31u32;if $k<$n $body}
}};}
macro_rules! backward_dofs {($k:ident,$n:expr,$body:block)=>{{
    {let $k=31u32;if $k<$n $body}
    {let $k=30u32;if $k<$n $body}
    {let $k=29u32;if $k<$n $body}
    {let $k=28u32;if $k<$n $body}
    {let $k=27u32;if $k<$n $body}
    {let $k=26u32;if $k<$n $body}
    {let $k=25u32;if $k<$n $body}
    {let $k=24u32;if $k<$n $body}
    {let $k=23u32;if $k<$n $body}
    {let $k=22u32;if $k<$n $body}
    {let $k=21u32;if $k<$n $body}
    {let $k=20u32;if $k<$n $body}
    {let $k=19u32;if $k<$n $body}
    {let $k=18u32;if $k<$n $body}
    {let $k=17u32;if $k<$n $body}
    {let $k=16u32;if $k<$n $body}
    {let $k=15u32;if $k<$n $body}
    {let $k=14u32;if $k<$n $body}
    {let $k=13u32;if $k<$n $body}
    {let $k=12u32;if $k<$n $body}
    {let $k=11u32;if $k<$n $body}
    {let $k=10u32;if $k<$n $body}
    {let $k=9u32;if $k<$n $body}
    {let $k=8u32;if $k<$n $body}
    {let $k=7u32;if $k<$n $body}
    {let $k=6u32;if $k<$n $body}
    {let $k=5u32;if $k<$n $body}
    {let $k=4u32;if $k<$n $body}
    {let $k=3u32;if $k<$n $body}
    {let $k=2u32;if $k<$n $body}
    {let $k=1u32;if $k<$n $body}
    {let $k=0u32;if $k<$n $body}
}};}

struct Row {
    a: Vec4,
    b: Vec4,
    c: Vec4,
    d: Vec4,
    e: Vec4,
    f: Vec4,
    g: Vec4,
    h: Vec4,
}
impl Row {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            a: Vec4::ZERO,
            b: Vec4::ZERO,
            c: Vec4::ZERO,
            d: Vec4::ZERO,
            e: Vec4::ZERO,
            f: Vec4::ZERO,
            g: Vec4::ZERO,
            h: Vec4::ZERO,
        }
    }
    #[inline(always)]
    fn get(&self, index: u32) -> f32 {
        match index {
            0 => self.a.x,
            1 => self.a.y,
            2 => self.a.z,
            3 => self.a.w,
            4 => self.b.x,
            5 => self.b.y,
            6 => self.b.z,
            7 => self.b.w,
            8 => self.c.x,
            9 => self.c.y,
            10 => self.c.z,
            11 => self.c.w,
            12 => self.d.x,
            13 => self.d.y,
            14 => self.d.z,
            15 => self.d.w,
            16 => self.e.x,
            17 => self.e.y,
            18 => self.e.z,
            19 => self.e.w,
            20 => self.f.x,
            21 => self.f.y,
            22 => self.f.z,
            23 => self.f.w,
            24 => self.g.x,
            25 => self.g.y,
            26 => self.g.z,
            27 => self.g.w,
            28 => self.h.x,
            29 => self.h.y,
            30 => self.h.z,
            _ => self.h.w,
        }
    }
    #[inline(always)]
    fn set(&mut self, index: u32, value: f32) {
        match index {
            0 => self.a.x = value,
            1 => self.a.y = value,
            2 => self.a.z = value,
            3 => self.a.w = value,
            4 => self.b.x = value,
            5 => self.b.y = value,
            6 => self.b.z = value,
            7 => self.b.w = value,
            8 => self.c.x = value,
            9 => self.c.y = value,
            10 => self.c.z = value,
            11 => self.c.w = value,
            12 => self.d.x = value,
            13 => self.d.y = value,
            14 => self.d.z = value,
            15 => self.d.w = value,
            16 => self.e.x = value,
            17 => self.e.y = value,
            18 => self.e.z = value,
            19 => self.e.w = value,
            20 => self.f.x = value,
            21 => self.f.y = value,
            22 => self.f.z = value,
            23 => self.f.w = value,
            24 => self.g.x = value,
            25 => self.g.y = value,
            26 => self.g.z = value,
            27 => self.g.w = value,
            28 => self.h.x = value,
            29 => self.h.y = value,
            30 => self.h.z = value,
            _ => self.h.w = value,
        }
    }
}
#[inline(always)]
fn shuffle(value: f32, lane: u32) -> f32 {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::subgroup_shuffle(value, lane)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = lane;
        value
    }
}

#[inline(always)]
pub(super) fn factor_solve(
    lane: u32,
    n: u32,
    matrix: &mut [f32],
    m: MatSlice,
    pivots: &mut [u32],
    piv: VSlice,
    forces: &mut [f32],
    rhs: VSlice,
) {
    let mut row = Row::zero();
    if lane < n && 0 < n {
        row.a.x = matrix.read(m.idx(lane, 0));
    }
    if lane < n && 1 < n {
        row.a.y = matrix.read(m.idx(lane, 1));
    }
    if lane < n && 2 < n {
        row.a.z = matrix.read(m.idx(lane, 2));
    }
    if lane < n && 3 < n {
        row.a.w = matrix.read(m.idx(lane, 3));
    }
    if lane < n && 4 < n {
        row.b.x = matrix.read(m.idx(lane, 4));
    }
    if lane < n && 5 < n {
        row.b.y = matrix.read(m.idx(lane, 5));
    }
    if lane < n && 6 < n {
        row.b.z = matrix.read(m.idx(lane, 6));
    }
    if lane < n && 7 < n {
        row.b.w = matrix.read(m.idx(lane, 7));
    }
    if lane < n && 8 < n {
        row.c.x = matrix.read(m.idx(lane, 8));
    }
    if lane < n && 9 < n {
        row.c.y = matrix.read(m.idx(lane, 9));
    }
    if lane < n && 10 < n {
        row.c.z = matrix.read(m.idx(lane, 10));
    }
    if lane < n && 11 < n {
        row.c.w = matrix.read(m.idx(lane, 11));
    }
    if lane < n && 12 < n {
        row.d.x = matrix.read(m.idx(lane, 12));
    }
    if lane < n && 13 < n {
        row.d.y = matrix.read(m.idx(lane, 13));
    }
    if lane < n && 14 < n {
        row.d.z = matrix.read(m.idx(lane, 14));
    }
    if lane < n && 15 < n {
        row.d.w = matrix.read(m.idx(lane, 15));
    }
    if lane < n && 16 < n {
        row.e.x = matrix.read(m.idx(lane, 16));
    }
    if lane < n && 17 < n {
        row.e.y = matrix.read(m.idx(lane, 17));
    }
    if lane < n && 18 < n {
        row.e.z = matrix.read(m.idx(lane, 18));
    }
    if lane < n && 19 < n {
        row.e.w = matrix.read(m.idx(lane, 19));
    }
    if lane < n && 20 < n {
        row.f.x = matrix.read(m.idx(lane, 20));
    }
    if lane < n && 21 < n {
        row.f.y = matrix.read(m.idx(lane, 21));
    }
    if lane < n && 22 < n {
        row.f.z = matrix.read(m.idx(lane, 22));
    }
    if lane < n && 23 < n {
        row.f.w = matrix.read(m.idx(lane, 23));
    }
    if lane < n && 24 < n {
        row.g.x = matrix.read(m.idx(lane, 24));
    }
    if lane < n && 25 < n {
        row.g.y = matrix.read(m.idx(lane, 25));
    }
    if lane < n && 26 < n {
        row.g.z = matrix.read(m.idx(lane, 26));
    }
    if lane < n && 27 < n {
        row.g.w = matrix.read(m.idx(lane, 27));
    }
    if lane < n && 28 < n {
        row.h.x = matrix.read(m.idx(lane, 28));
    }
    if lane < n && 29 < n {
        row.h.y = matrix.read(m.idx(lane, 29));
    }
    if lane < n && 30 < n {
        row.h.z = matrix.read(m.idx(lane, 30));
    }
    if lane < n && 31 < n {
        row.h.w = matrix.read(m.idx(lane, 31));
    }
    let mut x = if lane < n {
        forces.read(rhs.at(lane))
    } else {
        0.0
    };
    forward_dofs!(k, n, {
        let a = row.get(k);
        let abs = if a >= 0.0 { a } else { -a };
        let maximum = subgroup_f_max(if lane >= k && lane < n { abs } else { -1.0 });
        let p = (-subgroup_f_max(if lane >= k && lane < n && abs == maximum {
            -(lane as f32)
        } else {
            -32.0
        })) as u32;
        if lane == 0 {
            pivots.write(piv.at(k), p);
        }
        if p != k {
            let src = if lane == k {
                p
            } else if lane == p {
                k
            } else {
                lane
            };
            if 0 < n {
                row.a.x = shuffle(row.a.x, src);
            }
            if 1 < n {
                row.a.y = shuffle(row.a.y, src);
            }
            if 2 < n {
                row.a.z = shuffle(row.a.z, src);
            }
            if 3 < n {
                row.a.w = shuffle(row.a.w, src);
            }
            if 4 < n {
                row.b.x = shuffle(row.b.x, src);
            }
            if 5 < n {
                row.b.y = shuffle(row.b.y, src);
            }
            if 6 < n {
                row.b.z = shuffle(row.b.z, src);
            }
            if 7 < n {
                row.b.w = shuffle(row.b.w, src);
            }
            if 8 < n {
                row.c.x = shuffle(row.c.x, src);
            }
            if 9 < n {
                row.c.y = shuffle(row.c.y, src);
            }
            if 10 < n {
                row.c.z = shuffle(row.c.z, src);
            }
            if 11 < n {
                row.c.w = shuffle(row.c.w, src);
            }
            if 12 < n {
                row.d.x = shuffle(row.d.x, src);
            }
            if 13 < n {
                row.d.y = shuffle(row.d.y, src);
            }
            if 14 < n {
                row.d.z = shuffle(row.d.z, src);
            }
            if 15 < n {
                row.d.w = shuffle(row.d.w, src);
            }
            if 16 < n {
                row.e.x = shuffle(row.e.x, src);
            }
            if 17 < n {
                row.e.y = shuffle(row.e.y, src);
            }
            if 18 < n {
                row.e.z = shuffle(row.e.z, src);
            }
            if 19 < n {
                row.e.w = shuffle(row.e.w, src);
            }
            if 20 < n {
                row.f.x = shuffle(row.f.x, src);
            }
            if 21 < n {
                row.f.y = shuffle(row.f.y, src);
            }
            if 22 < n {
                row.f.z = shuffle(row.f.z, src);
            }
            if 23 < n {
                row.f.w = shuffle(row.f.w, src);
            }
            if 24 < n {
                row.g.x = shuffle(row.g.x, src);
            }
            if 25 < n {
                row.g.y = shuffle(row.g.y, src);
            }
            if 26 < n {
                row.g.z = shuffle(row.g.z, src);
            }
            if 27 < n {
                row.g.w = shuffle(row.g.w, src);
            }
            if 28 < n {
                row.h.x = shuffle(row.h.x, src);
            }
            if 29 < n {
                row.h.y = shuffle(row.h.y, src);
            }
            if 30 < n {
                row.h.z = shuffle(row.h.z, src);
            }
            if 31 < n {
                row.h.w = shuffle(row.h.w, src);
            }
            x = shuffle(x, src);
        }
        let akk = shuffle(row.get(k), k);
        let inv = if akk != 0.0 { 1.0 / akk } else { 0.0 };
        let lower = row.get(k) * inv;
        if 1 > k && 1 < n {
            let u = shuffle(row.a.y, k);
            if lane > k && lane < n {
                row.a.y -= lower * u;
            }
        }
        if 2 > k && 2 < n {
            let u = shuffle(row.a.z, k);
            if lane > k && lane < n {
                row.a.z -= lower * u;
            }
        }
        if 3 > k && 3 < n {
            let u = shuffle(row.a.w, k);
            if lane > k && lane < n {
                row.a.w -= lower * u;
            }
        }
        if 4 > k && 4 < n {
            let u = shuffle(row.b.x, k);
            if lane > k && lane < n {
                row.b.x -= lower * u;
            }
        }
        if 5 > k && 5 < n {
            let u = shuffle(row.b.y, k);
            if lane > k && lane < n {
                row.b.y -= lower * u;
            }
        }
        if 6 > k && 6 < n {
            let u = shuffle(row.b.z, k);
            if lane > k && lane < n {
                row.b.z -= lower * u;
            }
        }
        if 7 > k && 7 < n {
            let u = shuffle(row.b.w, k);
            if lane > k && lane < n {
                row.b.w -= lower * u;
            }
        }
        if 8 > k && 8 < n {
            let u = shuffle(row.c.x, k);
            if lane > k && lane < n {
                row.c.x -= lower * u;
            }
        }
        if 9 > k && 9 < n {
            let u = shuffle(row.c.y, k);
            if lane > k && lane < n {
                row.c.y -= lower * u;
            }
        }
        if 10 > k && 10 < n {
            let u = shuffle(row.c.z, k);
            if lane > k && lane < n {
                row.c.z -= lower * u;
            }
        }
        if 11 > k && 11 < n {
            let u = shuffle(row.c.w, k);
            if lane > k && lane < n {
                row.c.w -= lower * u;
            }
        }
        if 12 > k && 12 < n {
            let u = shuffle(row.d.x, k);
            if lane > k && lane < n {
                row.d.x -= lower * u;
            }
        }
        if 13 > k && 13 < n {
            let u = shuffle(row.d.y, k);
            if lane > k && lane < n {
                row.d.y -= lower * u;
            }
        }
        if 14 > k && 14 < n {
            let u = shuffle(row.d.z, k);
            if lane > k && lane < n {
                row.d.z -= lower * u;
            }
        }
        if 15 > k && 15 < n {
            let u = shuffle(row.d.w, k);
            if lane > k && lane < n {
                row.d.w -= lower * u;
            }
        }
        if 16 > k && 16 < n {
            let u = shuffle(row.e.x, k);
            if lane > k && lane < n {
                row.e.x -= lower * u;
            }
        }
        if 17 > k && 17 < n {
            let u = shuffle(row.e.y, k);
            if lane > k && lane < n {
                row.e.y -= lower * u;
            }
        }
        if 18 > k && 18 < n {
            let u = shuffle(row.e.z, k);
            if lane > k && lane < n {
                row.e.z -= lower * u;
            }
        }
        if 19 > k && 19 < n {
            let u = shuffle(row.e.w, k);
            if lane > k && lane < n {
                row.e.w -= lower * u;
            }
        }
        if 20 > k && 20 < n {
            let u = shuffle(row.f.x, k);
            if lane > k && lane < n {
                row.f.x -= lower * u;
            }
        }
        if 21 > k && 21 < n {
            let u = shuffle(row.f.y, k);
            if lane > k && lane < n {
                row.f.y -= lower * u;
            }
        }
        if 22 > k && 22 < n {
            let u = shuffle(row.f.z, k);
            if lane > k && lane < n {
                row.f.z -= lower * u;
            }
        }
        if 23 > k && 23 < n {
            let u = shuffle(row.f.w, k);
            if lane > k && lane < n {
                row.f.w -= lower * u;
            }
        }
        if 24 > k && 24 < n {
            let u = shuffle(row.g.x, k);
            if lane > k && lane < n {
                row.g.x -= lower * u;
            }
        }
        if 25 > k && 25 < n {
            let u = shuffle(row.g.y, k);
            if lane > k && lane < n {
                row.g.y -= lower * u;
            }
        }
        if 26 > k && 26 < n {
            let u = shuffle(row.g.z, k);
            if lane > k && lane < n {
                row.g.z -= lower * u;
            }
        }
        if 27 > k && 27 < n {
            let u = shuffle(row.g.w, k);
            if lane > k && lane < n {
                row.g.w -= lower * u;
            }
        }
        if 28 > k && 28 < n {
            let u = shuffle(row.h.x, k);
            if lane > k && lane < n {
                row.h.x -= lower * u;
            }
        }
        if 29 > k && 29 < n {
            let u = shuffle(row.h.y, k);
            if lane > k && lane < n {
                row.h.y -= lower * u;
            }
        }
        if 30 > k && 30 < n {
            let u = shuffle(row.h.z, k);
            if lane > k && lane < n {
                row.h.z -= lower * u;
            }
        }
        if 31 > k && 31 < n {
            let u = shuffle(row.h.w, k);
            if lane > k && lane < n {
                row.h.w -= lower * u;
            }
        }
        if lane > k && lane < n {
            row.set(k, lower);
        }
    });
    forward_dofs!(k, n, {
        let y = shuffle(x, k);
        if lane > k && lane < n {
            x -= row.get(k) * y;
        }
    });
    backward_dofs!(k, n, {
        if lane == k {
            let u = row.get(k);
            x = if u != 0.0 { x / u } else { 0.0 };
        }
        let y = shuffle(x, k);
        if lane < k {
            x -= row.get(k) * y;
        }
    });
    if lane < n {
        forces.write(rhs.at(lane), x);
    }
    if lane < n && 0 < n {
        matrix.write(m.idx(lane, 0), row.a.x);
    }
    if lane < n && 1 < n {
        matrix.write(m.idx(lane, 1), row.a.y);
    }
    if lane < n && 2 < n {
        matrix.write(m.idx(lane, 2), row.a.z);
    }
    if lane < n && 3 < n {
        matrix.write(m.idx(lane, 3), row.a.w);
    }
    if lane < n && 4 < n {
        matrix.write(m.idx(lane, 4), row.b.x);
    }
    if lane < n && 5 < n {
        matrix.write(m.idx(lane, 5), row.b.y);
    }
    if lane < n && 6 < n {
        matrix.write(m.idx(lane, 6), row.b.z);
    }
    if lane < n && 7 < n {
        matrix.write(m.idx(lane, 7), row.b.w);
    }
    if lane < n && 8 < n {
        matrix.write(m.idx(lane, 8), row.c.x);
    }
    if lane < n && 9 < n {
        matrix.write(m.idx(lane, 9), row.c.y);
    }
    if lane < n && 10 < n {
        matrix.write(m.idx(lane, 10), row.c.z);
    }
    if lane < n && 11 < n {
        matrix.write(m.idx(lane, 11), row.c.w);
    }
    if lane < n && 12 < n {
        matrix.write(m.idx(lane, 12), row.d.x);
    }
    if lane < n && 13 < n {
        matrix.write(m.idx(lane, 13), row.d.y);
    }
    if lane < n && 14 < n {
        matrix.write(m.idx(lane, 14), row.d.z);
    }
    if lane < n && 15 < n {
        matrix.write(m.idx(lane, 15), row.d.w);
    }
    if lane < n && 16 < n {
        matrix.write(m.idx(lane, 16), row.e.x);
    }
    if lane < n && 17 < n {
        matrix.write(m.idx(lane, 17), row.e.y);
    }
    if lane < n && 18 < n {
        matrix.write(m.idx(lane, 18), row.e.z);
    }
    if lane < n && 19 < n {
        matrix.write(m.idx(lane, 19), row.e.w);
    }
    if lane < n && 20 < n {
        matrix.write(m.idx(lane, 20), row.f.x);
    }
    if lane < n && 21 < n {
        matrix.write(m.idx(lane, 21), row.f.y);
    }
    if lane < n && 22 < n {
        matrix.write(m.idx(lane, 22), row.f.z);
    }
    if lane < n && 23 < n {
        matrix.write(m.idx(lane, 23), row.f.w);
    }
    if lane < n && 24 < n {
        matrix.write(m.idx(lane, 24), row.g.x);
    }
    if lane < n && 25 < n {
        matrix.write(m.idx(lane, 25), row.g.y);
    }
    if lane < n && 26 < n {
        matrix.write(m.idx(lane, 26), row.g.z);
    }
    if lane < n && 27 < n {
        matrix.write(m.idx(lane, 27), row.g.w);
    }
    if lane < n && 28 < n {
        matrix.write(m.idx(lane, 28), row.h.x);
    }
    if lane < n && 29 < n {
        matrix.write(m.idx(lane, 29), row.h.y);
    }
    if lane < n && 30 < n {
        matrix.write(m.idx(lane, 30), row.h.z);
    }
    if lane < n && 31 < n {
        matrix.write(m.idx(lane, 31), row.h.w);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_factor_solve_simd(
    #[spirv(workgroup_id)] wid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] lid: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    info: &[super::types::MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] matrix: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] forces: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &crate::utils::BatchIndices,
) {
    let b = wid.y;
    let mb = wid.x;
    if b >= batch_ids.num_batches || mb >= batch_ids.multibodies_len {
        return;
    }
    let info = info.read(batch_ids.mbi(b, mb as usize));
    let n = info.ndofs;
    if n == 0 {
        return;
    }
    let m = MatSlice::dense(batch_ids.mb_region(b, info.mass_matrix_offset, n * n), n, n);
    let rhs = VSlice::dense(batch_ids.mb_region(b, info.first_dof, n));
    if n <= 32 && cfg!(target_arch = "spirv") && khal_std::sync::subgroup_f_add(1.0) == 32.0 {
        factor_solve(lid.x, n, matrix, m, pivots, rhs, forces, rhs);
    } else if lid.x == 0 {
        crate::utils::linalg::lu_decompose(matrix, m, pivots, rhs);
        crate::utils::linalg::lu_solve_in_place(matrix, m, pivots, rhs, forces, rhs);
    }
}
