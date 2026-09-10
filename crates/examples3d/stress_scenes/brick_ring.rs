//! A ring wall of bricks smashed by a sphere rolling in from outside.

use super::builder::{Camera, Scene};
use super::common::{brick, ground, rolling_ball};
use std::f32::consts::PI;

/// `courses` courses in running bond on rings from `radius` outwards, `rows` bricks deep at the
/// bottom and one fewer every `tier` courses.
fn brick_ring(camera: Camera, radius: f32, courses: usize, tier: usize, rows: usize) -> Scene {
    let mut s = Scene::new(camera);
    ground(&mut s, radius + rows as f32);
    for c in 0..courses {
        let depth = rows.saturating_sub(c / tier).max(1);
        for j in 0..depth {
            let r = radius + 0.25 + 0.52 * j as f32;
            let n = (2.0 * PI * (r - 0.25) / 1.02).floor() as usize;
            for i in 0..n {
                let a = (i as f32 + (c % 2) as f32 * 0.5) / n as f32 * 2.0 * PI;
                brick(
                    &mut s,
                    r * a.cos(),
                    r * a.sin(),
                    0.75 + 0.5 * c as f32,
                    a + PI / 2.0,
                );
            }
        }
    }
    let r = courses as f32 / 5.0;
    rolling_ball(
        &mut s,
        r,
        10.0,
        0.0,
        -(radius + 0.52 * rows as f32 + r + 2.0),
        30.0,
    );
    s
}

/// The 110k-brick ring at a quarter of the bricks (same proportions).
pub fn brick_ring_28k() -> Scene {
    let camera = Camera::new(110.0, [0.0, -8.0, 0.0])
        .azimuth(-100.0)
        .elevation(0.3);
    brick_ring(camera, 40.0, 20, 2, 10)
}

pub fn brick_ring_110k() -> Scene {
    let camera = Camera::new(215.0, [0.0, -15.0, 0.0])
        .azimuth(-100.0)
        .elevation(0.3);
    brick_ring(camera, 80.0, 40, 4, 10)
}
