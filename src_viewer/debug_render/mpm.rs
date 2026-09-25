//! Debug rendering of the MPM state: the particles and the sparse grid.

use super::backend::{DebugLine, DebugPoint};
use bitflags::bitflags;
use nexus::mpm::debug::{DebugGrid, DebugParticle};
use nexus::rbd::math::Vector;

bitflags! {
    /// Which parts of the MPM state to draw (like rapier's `DebugRenderMode`).
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct MpmDebugRenderMode: u32 {
        /// A point per particle, red if the particle is pinned.
        const PARTICLES = 1 << 0;
        /// A segment per particle along its velocity.
        const PARTICLE_VELOCITIES = 1 << 1;
        /// A segment per particle along its CPIC contact normal.
        const PARTICLE_CDF = 1 << 2;
        /// The box of each allocated grid block.
        const GRID_BLOCKS = 1 << 3;
        /// A point per grid node with mass.
        const GRID_NODES = 1 << 4;
        /// A segment per grid node with mass along its velocity.
        const GRID_VELOCITIES = 1 << 5;
    }
}

impl MpmDebugRenderMode {
    /// Whether the particles must be read back.
    pub fn needs_particles(self) -> bool {
        self.intersects(Self::PARTICLES | Self::PARTICLE_VELOCITIES | Self::PARTICLE_CDF)
    }

    /// Whether the grid must be read back.
    pub fn needs_grid(self) -> bool {
        self.intersects(Self::GRID_BLOCKS | Self::GRID_NODES | Self::GRID_VELOCITIES)
    }
}

/// Dark colors that are easy to see on the bright background.
const PARTICLE_COLOR: [f32; 4] = [0.0, 0.35, 0.8, 1.0];
const PARTICLE_FIXED_COLOR: [f32; 4] = [0.8, 0.0, 0.0, 1.0];
const PARTICLE_VELOCITY_COLOR: [f32; 4] = [0.0, 0.55, 0.25, 1.0];
const PARTICLE_CDF_COLOR: [f32; 4] = [0.7, 0.55, 0.0, 1.0];
/// Blocks with their own particles, and blocks that only exist because of
/// a neighbor's particles (lighter).
const BLOCK_COLOR: [f32; 4] = [0.25, 0.25, 0.45, 0.9];
const BLOCK_EMPTY_COLOR: [f32; 4] = [0.6, 0.5, 0.8, 0.7];
const NODE_COLOR: [f32; 4] = [0.1, 0.1, 0.1, 1.0];
const NODE_VELOCITY_COLOR: [f32; 4] = [0.85, 0.25, 0.0, 1.0];

/// Draws the particles into `lines`/`points`.
/// `velocity_scale` is the time (in seconds) a velocity segment stands for.
pub fn render_particles(
    particles: &[DebugParticle],
    mode: MpmDebugRenderMode,
    velocity_scale: f32,
    lines: &mut Vec<DebugLine>,
    points: &mut Vec<DebugPoint>,
) {
    for particle in particles {
        if !particle.enabled {
            continue;
        }

        if mode.contains(MpmDebugRenderMode::PARTICLES) {
            points.push(DebugPoint {
                point: particle.position,
                color: if particle.fixed {
                    PARTICLE_FIXED_COLOR
                } else {
                    PARTICLE_COLOR
                },
            });
        }

        if mode.contains(MpmDebugRenderMode::PARTICLE_VELOCITIES) {
            lines.push(DebugLine {
                a: particle.position,
                b: particle.position + particle.velocity * velocity_scale,
                color: PARTICLE_VELOCITY_COLOR,
            });
        }

        // The normal is only set when a collider is close (it is zero otherwise).
        if mode.contains(MpmDebugRenderMode::PARTICLE_CDF) && particle.cdf_affinity != 0 {
            lines.push(DebugLine {
                a: particle.position,
                b: particle.position + particle.cdf_normal * particle.radius * 4.0,
                color: PARTICLE_CDF_COLOR,
            });
        }
    }
}

/// Draws the sparse grid into `lines`/`points`.
/// `velocity_scale` is the time (in seconds) a velocity segment stands for.
pub fn render_grid(
    grid: &DebugGrid,
    mode: MpmDebugRenderMode,
    velocity_scale: f32,
    lines: &mut Vec<DebugLine>,
    points: &mut Vec<DebugPoint>,
) {
    if mode.contains(MpmDebugRenderMode::GRID_BLOCKS) {
        for block in &grid.blocks {
            let color = if block.num_particles > 0 {
                BLOCK_COLOR
            } else {
                BLOCK_EMPTY_COLOR
            };
            push_box(block.mins, block.maxs, color, lines);
        }
    }

    if !mode.intersects(MpmDebugRenderMode::GRID_NODES | MpmDebugRenderMode::GRID_VELOCITIES) {
        return;
    }

    for node in &grid.nodes {
        if mode.contains(MpmDebugRenderMode::GRID_NODES) {
            points.push(DebugPoint {
                point: node.position,
                color: NODE_COLOR,
            });
        }
        if mode.contains(MpmDebugRenderMode::GRID_VELOCITIES) {
            lines.push(DebugLine {
                a: node.position,
                b: node.position + node.velocity * velocity_scale,
                color: NODE_VELOCITY_COLOR,
            });
        }
    }
}

/// Adds the edges of an axis-aligned box: 4 segments in 2D, 12 in 3D.
#[cfg(feature = "dim2")]
fn push_box(mins: Vector, maxs: Vector, color: [f32; 4], lines: &mut Vec<DebugLine>) {
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
fn push_box(mins: Vector, maxs: Vector, color: [f32; 4], lines: &mut Vec<DebugLine>) {
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
