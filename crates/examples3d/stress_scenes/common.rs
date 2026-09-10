//! Pieces shared by several stress scenes.

use super::builder::Scene;
use rapier3d::prelude::*;

/// A ground slab sized to hold `extent` meters of content, its top face at z = 0.5.
pub fn ground(s: &mut Scene, extent: f32) -> usize {
    let size = (4.0 * extent).max(200.0);
    s.rigid([size, size, 1.0], 0.0, 0.5, [0.0; 3])
}

/// A 1 × 0.5 × 0.5 brick turned `angle` about z.
pub fn brick(s: &mut Scene, x: f32, y: f32, z: f32, angle: f32) -> usize {
    let id = s.rigid([1.0, 0.5, 0.5], 1.0, 0.5, [x, y, z]);
    s.set_rotation(id, Rotation::from_rotation_z(angle));
    id
}

/// A sphere resting on the ground at (x, y), rolling along +y at `speed`.
pub fn rolling_ball(s: &mut Scene, r: f32, density: f32, x: f32, y: f32, speed: f32) -> usize {
    let ball = s.sphere(r, density, 0.5, [x, y, 0.5 + r]);
    s.set_linvel(ball, [0.0, speed, 0.0]);
    s.set_angvel(ball, [-speed / r, 0.0, 0.0]);
    ball
}

/// An `n` × `n` grid of thin plates ball-jointed at their edges, static where `pinned`.
pub fn plate_grid(
    s: &mut Scene,
    n: usize,
    pitch: f32,
    z: f32,
    friction: f32,
    pinned: impl Fn(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let mut plates = vec![];
    for x in 0..n {
        let mut column = vec![];
        for y in 0..n {
            let density = if pinned(x, y) { 0.0 } else { 1.0 };
            let pos = [
                (x as f32 - (n - 1) as f32 / 2.0) * pitch,
                (y as f32 - (n - 1) as f32 / 2.0) * pitch,
                z,
            ];
            column.push(s.rigid([pitch * 0.9, pitch * 0.9, 0.1], density, friction, pos));
        }
        plates.push(column);
    }
    let h = pitch / 2.0;
    for x in 0..n {
        for y in 0..n {
            if x > 0 {
                s.ball_joint(
                    plates[x - 1][y],
                    plates[x][y],
                    [h, 0.0, 0.0],
                    [-h, 0.0, 0.0],
                );
            }
            if y > 0 {
                s.ball_joint(
                    plates[x][y - 1],
                    plates[x][y],
                    [0.0, h, 0.0],
                    [0.0, -h, 0.0],
                );
            }
        }
    }
    plates
}
