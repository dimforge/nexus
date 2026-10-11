//! 32 floats held in registers, and loops over them.
//!
//! rust-gpu neither inlines nor constant-folds at the MIR level, so a local array or an index
//! `match` costs memory traffic or a switch per access. Instead, [`each_reg`] expands an
//! operation once per register with its index as a literal and its field spelled out.

use glamx::Vec4;

/// 32 floats as eight quads: register `i` is lane `i % 4` of quad `i / 4`.
#[derive(Copy, Clone)]
pub(super) struct Regs32 {
    pub a: Vec4,
    pub b: Vec4,
    pub c: Vec4,
    pub d: Vec4,
    pub e: Vec4,
    pub f: Vec4,
    pub g: Vec4,
    pub h: Vec4,
}

impl Regs32 {
    #[inline(always)]
    pub(super) fn zero() -> Self {
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
}

/// Invokes `$m!(args.., i, quad, lane)` for each register `i` of a [`Regs32`], in increasing
/// order, or decreasing order with `rev`: register `i` is `regs.quad.lane`.
macro_rules! each_reg {
    (rev $m:ident $(, $arg:tt)*) => {
        $m!($($arg,)* 31, h, w); $m!($($arg,)* 30, h, z); $m!($($arg,)* 29, h, y); $m!($($arg,)* 28, h, x);
        $m!($($arg,)* 27, g, w); $m!($($arg,)* 26, g, z); $m!($($arg,)* 25, g, y); $m!($($arg,)* 24, g, x);
        $m!($($arg,)* 23, f, w); $m!($($arg,)* 22, f, z); $m!($($arg,)* 21, f, y); $m!($($arg,)* 20, f, x);
        $m!($($arg,)* 19, e, w); $m!($($arg,)* 18, e, z); $m!($($arg,)* 17, e, y); $m!($($arg,)* 16, e, x);
        $m!($($arg,)* 15, d, w); $m!($($arg,)* 14, d, z); $m!($($arg,)* 13, d, y); $m!($($arg,)* 12, d, x);
        $m!($($arg,)* 11, c, w); $m!($($arg,)* 10, c, z); $m!($($arg,)* 9, c, y); $m!($($arg,)* 8, c, x);
        $m!($($arg,)* 7, b, w); $m!($($arg,)* 6, b, z); $m!($($arg,)* 5, b, y); $m!($($arg,)* 4, b, x);
        $m!($($arg,)* 3, a, w); $m!($($arg,)* 2, a, z); $m!($($arg,)* 1, a, y); $m!($($arg,)* 0, a, x);
    };
    ($m:ident $(, $arg:tt)*) => {
        $m!($($arg,)* 0, a, x); $m!($($arg,)* 1, a, y); $m!($($arg,)* 2, a, z); $m!($($arg,)* 3, a, w);
        $m!($($arg,)* 4, b, x); $m!($($arg,)* 5, b, y); $m!($($arg,)* 6, b, z); $m!($($arg,)* 7, b, w);
        $m!($($arg,)* 8, c, x); $m!($($arg,)* 9, c, y); $m!($($arg,)* 10, c, z); $m!($($arg,)* 11, c, w);
        $m!($($arg,)* 12, d, x); $m!($($arg,)* 13, d, y); $m!($($arg,)* 14, d, z); $m!($($arg,)* 15, d, w);
        $m!($($arg,)* 16, e, x); $m!($($arg,)* 17, e, y); $m!($($arg,)* 18, e, z); $m!($($arg,)* 19, e, w);
        $m!($($arg,)* 20, f, x); $m!($($arg,)* 21, f, y); $m!($($arg,)* 22, f, z); $m!($($arg,)* 23, f, w);
        $m!($($arg,)* 24, g, x); $m!($($arg,)* 25, g, y); $m!($($arg,)* 26, g, z); $m!($($arg,)* 27, g, w);
        $m!($($arg,)* 28, h, x); $m!($($arg,)* 29, h, y); $m!($($arg,)* 30, h, z); $m!($($arg,)* 31, h, w);
    };
}
pub(super) use each_reg;
