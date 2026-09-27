//! The kiss3d side of rapier's debug-render pipeline.

use crate::rapier::pipeline::{DebugColor, DebugRenderBackend, DebugRenderObject};
use nexus::rbd::math::Vector;

/// A world-space segment to draw, with its RGBA color.
#[derive(Copy, Clone, Debug)]
pub struct DebugLine {
    pub a: Vector,
    pub b: Vector,
    pub color: [f32; 4],
}

/// A world-space point to draw, with its RGBA color.
#[derive(Copy, Clone, Debug)]
pub struct DebugPoint {
    pub point: Vector,
    pub color: [f32; 4],
}

/// Collects the segments from rapier's `DebugRenderPipeline`.
/// rapier gives HSLA colors, so they are converted to RGBA here.
#[derive(Default)]
pub struct LineCollector {
    pub lines: Vec<DebugLine>,
}

impl LineCollector {
    /// Adds a segment with an RGBA color, for what nexus draws itself (e.g. the contacts).
    pub fn push_rgba(&mut self, a: Vector, b: Vector, color: [f32; 4]) {
        self.lines.push(DebugLine { a, b, color });
    }
}

impl DebugRenderBackend for LineCollector {
    fn draw_line(&mut self, _: DebugRenderObject, a: Vector, b: Vector, color: DebugColor) {
        self.lines.push(DebugLine {
            a,
            b,
            color: hsla_to_rgba(color),
        });
    }
}

/// Converts a rapier `[hue, saturation, lightness, alpha]` color to RGBA.
pub fn hsla_to_rgba([h, s, l, a]: DebugColor) -> [f32; 4] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    // `h / 60` picks the sextant; `x` is the ramp within it.
    let h6 = (h / 60.0).rem_euclid(6.0);
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [r + m, g + m, b + m, a]
}

/// Adds the edges of an axis-aligned box: 4 segments in 2D, 12 in 3D.
#[cfg(feature = "dim2")]
pub(super) fn push_box(mins: Vector, maxs: Vector, color: [f32; 4], lines: &mut Vec<DebugLine>) {
    let corners = [
        mins,
        Vector::new(maxs.x, mins.y),
        maxs,
        Vector::new(mins.x, maxs.y),
    ];
    for i in 0..4 {
        lines.push(DebugLine {
            a: corners[i],
            b: corners[(i + 1) % 4],
            color,
        });
    }
}

/// Adds the edges of an axis-aligned box: 4 segments in 2D, 12 in 3D.
#[cfg(feature = "dim3")]
pub(super) fn push_box(mins: Vector, maxs: Vector, color: [f32; 4], lines: &mut Vec<DebugLine>) {
    // Corner `i` takes x/y/z from `maxs` where bit 0/1/2 of `i` is set.
    // Two corners share an edge if `i ^ j` has a single bit.
    let corner = |i: usize| {
        Vector::new(
            if i & 1 != 0 { maxs.x } else { mins.x },
            if i & 2 != 0 { maxs.y } else { mins.y },
            if i & 4 != 0 { maxs.z } else { mins.z },
        )
    };
    for i in 0..8usize {
        for bit in [1, 2, 4] {
            let j = i ^ bit;
            if j > i {
                lines.push(DebugLine {
                    a: corner(i),
                    b: corner(j),
                    color,
                });
            }
        }
    }
}
