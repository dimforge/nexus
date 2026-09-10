//! Jointed blocks of cubes dropped onto a chain-mail net.

use super::builder::{Camera, Scene};
use super::common::{ground, plate_grid};

/// Plates of 5 × 5 × 2 small cubes ball-jointed at their shared face centers, `grid` × `grid`
/// per layer, drop in `layers` staggered layers onto a chain-mail net pinned along its border.
pub fn jointed_drop() -> Scene {
    let (grid, layers, net) = (10, 6, 64);
    let camera = Camera::new(55.0, [0.0, 0.0, 4.0])
        .azimuth(-120.0)
        .elevation(0.5);
    let mut s = Scene::new(camera);
    ground(&mut s, net as f32 * 0.5);
    let z_net = 4.0;
    let edge = |x: usize, y: usize| x == 0 || y == 0 || x == net - 1 || y == net - 1;
    let links = plate_grid(&mut s, net, 0.5, z_net, 0.5, edge);
    for id in links.into_iter().flatten() {
        s.set_color(id, 0xb8bcc2);
    }

    let c = 0.4;
    let (nx, ny, nz) = (5, 5, 2);
    let pitch = 3.0;
    for layer in 0..layers {
        let offset = if layer % 2 == 0 { 0.0 } else { pitch / 2.0 };
        for gx in 0..grid {
            for gy in 0..grid {
                let origin = [
                    (gx as f32 - (grid - 1) as f32 / 2.0) * pitch + offset - 0.75,
                    (gy as f32 - (grid - 1) as f32 / 2.0) * pitch + offset - 0.75,
                    z_net + 3.0 + layer as f32 * 2.5,
                ];
                let mut cells = vec![];
                for x in 0..nx {
                    for y in 0..ny {
                        for z in 0..nz {
                            let pos = [
                                origin[0] + (x as f32 - (nx - 1) as f32 / 2.0) * c,
                                origin[1] + (y as f32 - (ny - 1) as f32 / 2.0) * c,
                                origin[2] + z as f32 * c,
                            ];
                            cells.push(s.rigid([c; 3], 1.0, 0.5, pos));
                        }
                    }
                }
                let at = |x: usize, y: usize, z: usize| cells[(x * ny + y) * nz + z];
                let h = c / 2.0;
                for x in 0..nx {
                    for y in 0..ny {
                        for z in 0..nz {
                            if x > 0 {
                                s.ball_joint(
                                    at(x - 1, y, z),
                                    at(x, y, z),
                                    [h, 0.0, 0.0],
                                    [-h, 0.0, 0.0],
                                );
                            }
                            if y > 0 {
                                s.ball_joint(
                                    at(x, y - 1, z),
                                    at(x, y, z),
                                    [0.0, h, 0.0],
                                    [0.0, -h, 0.0],
                                );
                            }
                            if z > 0 {
                                s.ball_joint(
                                    at(x, y, z - 1),
                                    at(x, y, z),
                                    [0.0, 0.0, h],
                                    [0.0, 0.0, -h],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    s
}
