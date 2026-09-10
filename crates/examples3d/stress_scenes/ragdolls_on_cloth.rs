//! A block of ragdolls dropped onto a cloth of jointed plates.

use super::builder::{Camera, Random, Scene, hsl};
use super::common::{ground, plate_grid};
use rapier3d::prelude::*;
use std::f32::consts::PI;

/// A ragdoll standing on z = 0 in its own frame: rods (from, to, width) and a head.
const RAGDOLL_RODS: [([f32; 3], [f32; 3], f32); 9] = [
    ([0.0, 0.0, 0.5], [0.0, 0.0, 0.93], 0.24),     // torso
    ([0.0, 0.19, 0.9], [0.0, 0.19, 0.66], 0.09),   // upper arms
    ([0.0, -0.19, 0.9], [0.0, -0.19, 0.66], 0.09), //
    ([0.0, 0.19, 0.66], [0.0, 0.19, 0.42], 0.085), // forearms
    ([0.0, -0.19, 0.66], [0.0, -0.19, 0.42], 0.085),
    ([0.0, 0.08, 0.5], [0.0, 0.08, 0.27], 0.11), // thighs
    ([0.0, -0.08, 0.5], [0.0, -0.08, 0.27], 0.11),
    ([0.0, 0.08, 0.27], [0.0, 0.08, 0.03], 0.1), // shins
    ([0.0, -0.08, 0.27], [0.0, -0.08, 0.03], 0.1),
];
const RAGDOLL_HEAD: ([f32; 3], f32) = ([0.0, 0.0, 1.1], 0.12);
/// Ball joints: (part, part, point), parts indexing the rods then the head (9).
const RAGDOLL_JOINTS: [(usize, usize, [f32; 3]); 9] = [
    (0, 9, [0.0, 0.0, 1.0]),
    (0, 1, [0.0, 0.19, 0.9]),
    (0, 2, [0.0, -0.19, 0.9]),
    (1, 3, [0.0, 0.19, 0.66]),
    (2, 4, [0.0, -0.19, 0.66]),
    (0, 5, [0.0, 0.08, 0.5]),
    (0, 6, [0.0, -0.08, 0.5]),
    (5, 7, [0.0, 0.08, 0.27]),
    (6, 8, [0.0, -0.08, 0.27]),
];
/// The ragdoll's middle (its frame is turned about this).
const RAGDOLL_CENTER: [f32; 3] = [0.0, 0.0, 0.6];

/// The rotation taking the x axis onto the unit vector `d`.
fn align_x(d: Vector) -> Rotation {
    if 1.0 + d.x < 1.0e-9 {
        Rotation::from_xyzw(0.0, 0.0, 1.0, 0.0)
    } else {
        Rotation::from_rotation_arc(Vector::X, d)
    }
}

/// A floppy ragdoll (ball joints) at `origin`, turned by `q`, in one color.
fn ragdoll(s: &mut Scene, origin: Vector, q: Rotation, color: u32) {
    let center = Vector::from(RAGDOLL_CENTER);
    let place = |c: Vector| (origin + q * (c - center)).into();
    // Each part's body, center and rotation in the ragdoll's frame.
    let mut parts = vec![];
    for (from, to, width) in RAGDOLL_RODS {
        let (from, to) = (Vector::from(from), Vector::from(to));
        let len = (to - from).length();
        let c = (from + to) / 2.0;
        let rot = align_x((to - from) / len);
        let body = s.rigid([len + width, width, width], 1.0, 0.6, place(c));
        s.set_rotation(body, q * rot);
        s.set_color(body, color);
        parts.push((body, c, rot));
    }
    let head_center = Vector::from(RAGDOLL_HEAD.0);
    let head = s.sphere(RAGDOLL_HEAD.1, 1.0, 0.6, place(head_center));
    s.set_rotation(head, q);
    s.set_color(head, color);
    parts.push((head, head_center, Rotation::IDENTITY));

    // Anchors in each part's own frame (the same in the ragdoll's frame as in the world's).
    let local = |k: usize, point: Vector| {
        let (_, c, rot) = parts[k];
        (rot.inverse() * (point - c)).into()
    };
    for (a, b, point) in RAGDOLL_JOINTS {
        let point = Vector::from(point);
        s.ball_joint(parts[a].0, parts[b].0, local(a, point), local(b, point));
    }
}

/// A block of ragdolls (`side` × `side` × `layers`, colored in a rainbow) dropped onto a cloth
/// of plates pinned at its four corners.
pub fn ragdolls_on_cloth() -> Scene {
    let (side, layers, cloth) = (12, 10, 96);
    let camera = Camera::new(52.0, [0.0, 0.0, 6.0])
        .azimuth(-115.0)
        .elevation(0.3);
    let mut s = Scene::new(camera);
    let pitch = 0.3;
    ground(&mut s, cloth as f32 * pitch * 0.5);
    let z_cloth = 8.0;
    let corner = |x: usize, y: usize| (x == 0 || x == cloth - 1) && (y == 0 || y == cloth - 1);
    let plates = plate_grid(&mut s, cloth, pitch, z_cloth, 0.6, corner);
    for id in plates.into_iter().flatten() {
        s.set_color(id, 0xe8e2d4);
    }

    let mut rand = Random::new(1414);
    let spacing = 1.4;
    for layer in 0..layers {
        for gx in 0..side {
            for gy in 0..side {
                // A random orientation, uniform over rotations.
                let (u1, u2, u3) = (rand.next(), rand.next(), rand.next());
                let q = Rotation::from_xyzw(
                    (1.0 - u1).sqrt() * (2.0 * PI * u2).sin(),
                    (1.0 - u1).sqrt() * (2.0 * PI * u2).cos(),
                    u1.sqrt() * (2.0 * PI * u3).sin(),
                    u1.sqrt() * (2.0 * PI * u3).cos(),
                );
                let origin = Vector::new(
                    (gx as f32 - (side - 1) as f32 / 2.0) * spacing,
                    (gy as f32 - (side - 1) as f32 / 2.0) * spacing,
                    z_cloth + 1.5 + layer as f32 * spacing,
                );
                let color = hsl(0.82 * gx as f32 / (side - 1).max(1) as f32, 0.72, 0.6);
                ragdoll(&mut s, origin, q, color);
            }
        }
    }
    s
}
