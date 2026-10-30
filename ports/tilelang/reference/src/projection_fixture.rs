use nexus_rbd_shaders3d::{
    glamx::{Quat, Vec3},
    shapes::{Segment, Tetrahedron, Triangle},
};
use serde_json::json;
use std::{fs, path::Path};

pub fn fixture(dir: &Path, kind: &str, count: usize) -> serde_json::Value {
    fs::create_dir_all(dir).unwrap();
    let storage = count + 3;
    let guard = 0xdeadbeefu32;
    let mut inputs = vec![[0u32; 24]; storage];
    let mut output = vec![[guard; 16]; storage];
    for (i, row) in inputs.iter_mut().enumerate() {
        let mode = i % 24;
        let f = (i % 7) as f32;
        let rotation = if i % 3 == 0 {
            Quat::IDENTITY
        } else {
            Quat::from_scaled_axis(Vec3::new(0.13 * f, -0.07 * f, 0.09 * f))
        };
        let shift = Vec3::new(0.2 * f, -0.11 * f, 0.17 * f);
        let scale = [0.25, 1., 4.][i % 3];
        let mut points = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, Vec3::ZERO];
        if kind == "segment_projection" {
            points[0] = -Vec3::X;
            points[4] = match mode % 6 {
                0 => Vec3::new(-2., 0., 0.),
                1 => Vec3::new(2., 0., 0.),
                2 => Vec3::new(0.2, 0.7, -0.3),
                3 => Vec3::ZERO,
                4 => Vec3::X,
                _ => -Vec3::X,
            };
            if mode == 23 {
                points[1] = points[0];
            }
        }
        if kind == "triangle_projection" {
            points[4] = match mode % 10 {
                0 => Vec3::new(-1., -1., 0.2),
                1 => Vec3::new(2., -0.5, 0.2),
                2 => Vec3::new(-0.5, 2., 0.2),
                3 => Vec3::new(0.3, -0.5, 0.2),
                4 => Vec3::new(-0.5, 0.3, 0.2),
                5 => Vec3::new(0.7, 0.7, 0.2),
                6 => Vec3::new(0.2, 0.3, 1.),
                7 => Vec3::new(0.2, 0.3, -1.),
                8 => Vec3::new(0.2, 0.3, 0.),
                _ => Vec3::ZERO,
            };
            if mode == 22 {
                points[2] = Vec3::X * 2.;
            }
            if mode == 23 {
                points[1] = Vec3::ZERO;
                points[2] = Vec3::ZERO;
            }
        }
        if kind == "tetrahedron_projection" {
            points[4] = match mode % 15 {
                0 => Vec3::splat(-1.),
                1 => Vec3::new(2., -0.5, -0.5),
                2 => Vec3::new(-0.5, 2., -0.5),
                3 => Vec3::new(-0.5, -0.5, 2.),
                4 => Vec3::new(0.5, -0.5, -0.5),
                5 => Vec3::new(-0.5, 0.5, -0.5),
                6 => Vec3::new(-0.5, -0.5, 0.5),
                7 => Vec3::new(0.7, 0.7, -0.5),
                8 => Vec3::new(0.7, -0.5, 0.7),
                9 => Vec3::new(-0.5, 0.7, 0.7),
                10 => Vec3::new(0.2, 0.2, -1.),
                11 => Vec3::new(0.2, -1., 0.2),
                12 => Vec3::new(-1., 0.2, 0.2),
                13 => Vec3::splat(0.7),
                _ => Vec3::splat(0.2),
            };
            if mode == 23 {
                points[3] = Vec3::X + Vec3::Y;
            }
        }
        for (j, point) in points.iter().enumerate() {
            let p = rotation * (*point * scale) + shift;
            for (axis, value) in p.to_array().iter().enumerate() {
                row[j * 4 + axis] = value.to_bits();
            }
            row[j * 4 + 3] = guard;
        }
        row[20] = (i % 2) as u32;
        row[21] = guard;
        row[22] = guard;
        row[23] = guard;
    }
    fs::write(dir.join("Inputs.bin"), bytemuck::cast_slice(&inputs)).unwrap();
    fs::write(dir.join("Output.bin"), bytemuck::cast_slice(&output)).unwrap();
    for (i, row) in inputs.iter().take(count).enumerate() {
        let p = (0..5)
            .map(|j| {
                Vec3::new(
                    f32::from_bits(row[j * 4]),
                    f32::from_bits(row[j * 4 + 1]),
                    f32::from_bits(row[j * 4 + 2]),
                )
            })
            .collect::<Vec<_>>();
        let solid = row[20] != 0;
        let result = match kind {
            "segment_projection" => {
                Segment::new(p[0], p[1]).project_local_point_and_get_location(p[4], solid)
            }
            "triangle_projection" => {
                Triangle::new(p[0], p[1], p[2]).project_local_point_and_get_location(p[4], solid)
            }
            _ => Tetrahedron::new(p[0], p[1], p[2], p[3])
                .project_local_point_and_get_location(p[4], solid),
        };
        let bary = result.barycentric_coordinates();
        for d in 0..3 {
            output[i][d] = result.point.to_array()[d].to_bits();
            output[i][4 + d] = result.bcoords.to_array()[d].to_bits();
            output[i][12 + d] = bary.to_array()[d].to_bits();
        }
        output[i][3] = result.inside as u32;
        output[i][7] = result.feature_type;
        output[i][8] = result.id;
        output[i][9..12].fill(0);
    }
    fs::write(
        dir.join("Output.expected.bin"),
        bytemuck::cast_slice(&output),
    )
    .unwrap();
    let case = json!({"name":dir.file_name().unwrap().to_str().unwrap(),"kind":kind,"count":count,"storage":storage,"groups":1,"iterations":1,"buffer_words":{"Inputs":storage*24,"Output":storage*16}});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&case).unwrap(),
    )
    .unwrap();
    case
}
