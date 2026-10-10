//! A block of randomly sized and turned boxes dropped into a pile.

use super::builder::{Camera, Random, Scene};
use super::common::ground;
use rapier3d::prelude::*;

/// An `n` × `n` × `h` block of boxes.
fn box_pile(camera: Camera, n: usize, h: usize) -> Scene {
    let mut s = Scene::new(camera);
    ground(&mut s, n as f32 * 1.6);
    let mut rand = Random::new(12345);
    for z in 0..h {
        for x in 0..n {
            for y in 0..n {
                let size = [
                    0.5 + rand.next() * 0.7,
                    0.5 + rand.next() * 0.7,
                    0.5 + rand.next() * 0.7,
                ];
                let pos = [
                    (x as f32 - n as f32 / 2.0) * 1.6,
                    (y as f32 - n as f32 / 2.0) * 1.6,
                    2.0 + z as f32 * 1.6,
                ];
                let id = s.rigid(size, 1.0, 0.5, pos);
                let q = Rotation::from_xyzw(
                    rand.next() - 0.5,
                    rand.next() - 0.5,
                    rand.next() - 0.5,
                    rand.next() - 0.5,
                );
                s.set_rotation(id, q);
            }
        }
    }
    s
}

pub fn box_pile_4k() -> Scene {
    let camera = Camera::new(55.0, [0.0, 0.0, 4.0]).elevation(0.45);
    box_pile(camera, 20, 10)
}

pub fn box_pile_32k() -> Scene {
    let camera = Camera::new(110.0, [0.0, 0.0, 6.0]).elevation(0.45);
    box_pile(camera, 40, 20)
}
