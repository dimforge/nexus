//! A lattice of welded boxes hanging from its two top corners.

use super::builder::{Camera, Scene};

/// A `w` × `h` lattice of unit boxes welded to their neighbors.
fn joint_lattice(camera: Camera, w: usize, h: usize) -> Scene {
    let mut s = Scene::new(camera);
    let mut grid = vec![];
    for x in 0..w {
        let mut column = vec![];
        for y in 0..h {
            let pinned = y == h - 1 && (x == 0 || x == w - 1);
            let density = if pinned { 0.0 } else { 1.0 };
            column.push(s.rigid([1.0, 1.0], density, 0.5, [x as f32, y as f32, 0.0]));
        }
        grid.push(column);
    }
    for (left, right) in grid.iter().zip(&grid[1..]) {
        for (&a, &b) in left.iter().zip(right) {
            s.fixed_joint(a, b, [0.5, 0.0], [-0.5, 0.0]);
        }
    }
    for column in &grid {
        for (&below, &above) in column.iter().zip(&column[1..]) {
            s.fixed_joint(below, above, [0.0, 0.5], [0.0, -0.5]);
        }
    }
    for x in 1..w {
        for y in 1..h {
            s.ignore_collision(grid[x - 1][y - 1], grid[x][y]);
            s.ignore_collision(grid[x][y - 1], grid[x - 1][y]);
        }
    }
    s
}

pub fn joint_lattice_64x64() -> Scene {
    joint_lattice(Camera::new(32.0, 20.0, 7.0), 64, 64)
}

pub fn joint_lattice_320x320() -> Scene {
    joint_lattice(Camera::new(160.0, 120.0, 1.6), 320, 320)
}

pub fn joint_lattice_512x512() -> Scene {
    joint_lattice(Camera::new(256.0, 180.0, 1.0), 512, 512)
}
