//! Rows of triangular brick walls smashed by two heavy spheres.

use super::builder::{Camera, Scene, hsl};
use super::common::{brick, ground, rolling_ball};

/// `columns` × `rows` triangular brick walls, `base` bricks wide, smashed by two heavy spheres
/// rolling down the columns `ball_columns`.
fn brick_walls(
    camera: Camera,
    columns: usize,
    rows: usize,
    base: usize,
    ball_columns: [usize; 2],
) -> Scene {
    let mut s = Scene::new(camera);
    let (pitch_x, pitch_y) = ((base + 2) as f32, 5.0);
    ground(
        &mut s,
        (columns as f32 * pitch_x).max(rows as f32 * pitch_y) / 2.0,
    );
    for cx in 0..columns {
        for cy in 0..rows {
            let x0 = (cx as f32 - (columns - 1) as f32 / 2.0) * pitch_x;
            let y = (cy as f32 - (rows - 1) as f32 / 2.0) * pitch_y;
            for k in 0..base {
                let m = base - k;
                for i in 0..m {
                    let x = x0 + (i as f32 - (m - 1) as f32 / 2.0) * 1.01;
                    brick(&mut s, x, y, 0.75 + 0.5 * k as f32, 0.0);
                }
            }
        }
    }
    let r = base as f32 / 7.0;
    for cx in ball_columns {
        let x = (cx as f32 - (columns - 1) as f32 / 2.0) * pitch_x;
        let y = -((rows - 1) as f32 / 2.0) * pitch_y - r - 12.0;
        rolling_ball(&mut s, r, 50.0, x, y, 30.0);
    }
    s
}

/// 8 × 16 walls of 210 bricks.
pub fn brick_walls_27k() -> Scene {
    let (columns, rows, base) = (8, 16, 20);
    let ball_columns = [2, 5];
    let camera = Camera::new(86.0, [15.0, 0.0, 2.0])
        .azimuth(-69.0)
        .elevation(0.12);
    brick_walls(camera, columns, rows, base, ball_columns)
}

/// 16 × 68 walls of 465 bricks (505,920 bricks) and two rolling spheres.
pub fn pyramids_500k() -> Scene {
    let camera = Camera::new(650.0, [0.0, -25.0, 5.0])
        .azimuth(-69.0)
        .elevation(0.55);
    let mut scene = brick_walls(camera, 16, 68, 30, [6, 9]);
    scene.set_color(0, 0x808080);
    for i in 1..scene.bodies.len() - 2 {
        // A repeatable palette makes individual bricks visible in the dense field.
        scene.set_color(i, hsl((i * 37 % 101) as f32 / 101.0, 0.45, 0.65));
    }
    let n = scene.bodies.len();
    scene.set_color(n - 2, 0xad70d6);
    scene.set_color(n - 1, 0xd677a7);
    scene
}
