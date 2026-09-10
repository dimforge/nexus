//! A large pyramid of boxes.

use super::builder::{Camera, Scene};

/// A pyramid of `size` rows of 1 × 0.5 boxes.
fn pyramid(camera: Camera, size: usize) -> Scene {
    let mut s = Scene::new(camera);
    let ground_width = (size as f32 * 1.5).max(100.0);
    s.rigid([ground_width, 0.5], 0.0, 0.5, [0.0, -2.0, 0.0]);
    for y in 0..size {
        for x in 0..size - y {
            let pos = [
                x as f32 * 1.1 + y as f32 * 0.5 - size as f32 / 2.0,
                y as f32 * 0.85,
                0.0,
            ];
            s.rigid([1.0, 0.5], 1.0, 0.5, pos);
        }
    }
    s
}

pub fn pyramid_50() -> Scene {
    pyramid(Camera::new(0.0, 10.0, 12.0), 50)
}

pub fn pyramid_100() -> Scene {
    pyramid(Camera::new(0.0, 22.0, 6.0), 100)
}

pub fn pyramid_200() -> Scene {
    pyramid(Camera::new(0.0, 45.0, 2.8), 200)
}
