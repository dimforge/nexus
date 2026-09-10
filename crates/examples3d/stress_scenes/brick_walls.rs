//! Rows of triangular brick walls smashed by two heavy spheres.

use super::builder::{Camera, Scene};
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
