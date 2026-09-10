//! A grid of boxes dropped into a walled container.

use super::builder::{Camera, Random, Scene};

/// A `cols` × `rows` grid of boxes (slightly jittered sizes) dropped into a walled container.
fn box_rain(camera: Camera, cols: usize, rows: usize) -> Scene {
    let mut s = Scene::new(camera);
    let width = cols as f32 * 1.2 + 4.0;
    let wall_height = rows as f32 * 1.4 + 10.0;
    s.rigid([width + 2.0, 1.0], 0.0, 0.5, [0.0, -0.5, 0.0]);
    s.rigid(
        [1.0, wall_height],
        0.0,
        0.5,
        [-width / 2.0, wall_height / 2.0, 0.0],
    );
    s.rigid(
        [1.0, wall_height],
        0.0,
        0.5,
        [width / 2.0, wall_height / 2.0, 0.0],
    );
    let mut rand = Random::new(1);
    for y in 0..rows {
        for x in 0..cols {
            let w = 0.6 + rand.next() * 0.5;
            let h = 0.6 + rand.next() * 0.5;
            let pos = [
                x as f32 * 1.2 - (cols - 1) as f32 * 0.6,
                2.0 + y as f32 * 1.4,
                rand.next() * 0.5,
            ];
            s.rigid([w, h], 1.0, 0.5, pos);
        }
    }
    s
}

pub fn box_rain_40x25() -> Scene {
    box_rain(Camera::new(0.0, 15.0, 12.0), 40, 25)
}

pub fn box_rain_100x50() -> Scene {
    box_rain(Camera::new(0.0, 30.0, 5.5), 100, 50)
}

pub fn box_rain_900x100() -> Scene {
    box_rain(Camera::new(0.0, 60.0, 0.7), 900, 100)
}
