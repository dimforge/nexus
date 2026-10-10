//! A forest of box columns.

use super::builder::{Camera, Scene};
use super::common::ground;

/// `n` × `n` columns of `h` unit boxes, each resting on the one below.
pub fn box_columns_100k() -> Scene {
    let (n, h) = (100, 10);
    let mut s = Scene::new(Camera::new(190.0, [0.0, 0.0, 5.0]).elevation(0.5));
    ground(&mut s, n as f32 * 1.5);
    for x in 0..n {
        for y in 0..n {
            for z in 0..h {
                let pos = [
                    (x as f32 - n as f32 / 2.0) * 1.5,
                    (y as f32 - n as f32 / 2.0) * 1.5,
                    1.0 + z as f32,
                ];
                s.rigid([1.0, 1.0, 1.0], 1.0, 0.5, pos);
            }
        }
    }
    s
}
