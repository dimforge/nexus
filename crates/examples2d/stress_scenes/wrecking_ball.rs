//! A brick wall hit by a heavy, fast block.

use super::builder::{Camera, Scene};

/// A wall of `cols` × `rows` bricks hit by a heavy, fast block.
fn wrecking_ball(camera: Camera, cols: usize, rows: usize) -> Scene {
    let mut s = Scene::new(camera);
    s.rigid([cols as f32 * 3.0 + 200.0, 1.0], 0.0, 0.6, [0.0, -0.5, 0.0]);
    for y in 0..rows {
        let offset = if y % 2 == 0 { 0.0 } else { 0.5 };
        for x in 0..cols {
            let pos = [
                x as f32 + offset - cols as f32 / 2.0,
                0.25 + y as f32 * 0.5,
                0.0,
            ];
            s.rigid([1.0, 0.5], 1.0, 0.6, pos);
        }
    }
    let size = (rows as f32 * 0.15).max(3.0);
    let pos = [
        -(cols as f32) / 2.0 - size * 2.0 - 5.0,
        rows as f32 * 0.25,
        0.0,
    ];
    let block = s.rigid([size, size], 20.0, 0.5, pos);
    s.set_velocity(block, [40.0, 0.0, 0.0]);
    s
}

pub fn wrecking_ball_100x40() -> Scene {
    wrecking_ball(Camera::new(-10.0, 10.0, 8.0), 100, 40)
}

pub fn wrecking_ball_400x100() -> Scene {
    wrecking_ball(Camera::new(-80.0, 25.0, 2.2), 400, 100)
}
