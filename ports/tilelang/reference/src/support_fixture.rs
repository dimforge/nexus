use crate::queries::gjk::{
    VoronoiSimplex,
    gjk::{closest_points, cso_point_from_shapes},
};
use glamx::{Quat, Vec3};
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    shapes::{Shape, Triangle},
};
use serde_json::{Value, json};
use std::{fs, path::Path};
fn shape(ty: usize) -> Shape {
    match ty {
        0 => Shape::ball(0.5),
        1 => Shape::cuboid(Vec3::new(0.5, 0.7, 0.3)),
        2 => Shape::capsule(Vec3::new(0.1, -0.5, 0.0), Vec3::new(-0.1, 0.5, 0.1), 0.2),
        3 => Shape::cone(0.7, 0.5),
        4 => Shape::cylinder(0.7, 0.5),
        5 => Shape::convex_poly(0, 8, 0, 0),
        _ => Shape::from_triangle(&Triangle::new(
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.1),
        )),
    }
}
use crate::simplex_fixture::pack;

pub fn fixture(dir: &Path, kind: &str, n: usize) -> Value {
    fs::create_dir_all(dir).unwrap();
    let storage = n + 3;
    let guard = 0xdeadbeefu32;
    let mut shapes = vec![shape(0); 2 * storage];
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut input = vec![[guard; 8]; storage];
    let mut output = vec![[guard; 16]; storage];
    let mut states = vec![[guard; 64]; storage];
    let mut simplexes = vec![VoronoiSimplex::default(); storage];
    let vertices = (0..8)
        .map(|i| {
            PaddedVector::new(Vec3::new(
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.7 } else { 0.7 },
                if i & 4 == 0 { -0.3 } else { 0.3 },
            ))
        })
        .collect::<Vec<_>>();
    for i in 0..n {
        shapes[2 * i] = shape(i % 7);
        shapes[2 * i + 1] = shape((i / 7) % 7);
        let profile = (i / 49) % 7;
        let scale = (i % 5) as f32;
        poses[i] = Pose {
            rotation: if profile % 2 == 0 {
                Quat::IDENTITY
            } else {
                Quat::from_scaled_axis(Vec3::new(0.17, -0.11, 0.23) * scale)
            },
            translation: match profile {
                0 => Vec3::new(3.0, 0.1, 0.0),
                1 => Vec3::new(1.3, 0.2, -0.1),
                2 => Vec3::new(0.2, -0.1, 0.1),
                3 => Vec3::ZERO,
                4 => Vec3::new(0.9, 0.1, 0.0),
                5 => Vec3::new(0.1, 2.0, 0.3),
                _ => Vec3::new(-0.4, 0.2, -1.2),
            },
            ..Pose::IDENTITY
        };
        let direction = match i % 8 {
            0 => Vec3::X,
            1 => Vec3::new(-0.0, 1.0, 0.0),
            2 => -Vec3::Y,
            3 => Vec3::Z,
            4 => Vec3::new(1.0, 2.0, -0.5),
            5 => Vec3::new(-0.2, 0.7, 1.0),
            6 => Vec3::new(-1.0, -1.0, 0.5),
            _ => Vec3::new(0.2, -0.3, -0.7),
        };
        for (dest, value) in input[i][..3].iter_mut().zip(direction.to_array()) {
            *dest = value.to_bits();
        }
        input[i][3] = [0.0f32, 0.1, 1.0, 10.0][i % 4].to_bits();
        input[i][4] = (i % 2) as u32;
        simplexes[i] = VoronoiSimplex::init(cso_point_from_shapes(
            poses[i],
            &shapes[2 * i],
            &shapes[2 * i + 1],
            direction,
            &vertices,
        ));
        pack(&simplexes[i], &mut states[i]);
    }
    macro_rules! write {
        ($name:expr,$values:expr) => {
            fs::write(dir.join($name), bytemuck::cast_slice($values)).unwrap()
        };
    }
    write!("Shapes.bin", &shapes);
    write!("Poses.bin", &poses);
    write!("Vertices.bin", &vertices);
    write!("Inputs.bin", &input);
    write!("Output.bin", &output);
    write!("State.bin", &states);
    for i in 0..n {
        let direction = Vec3::new(
            f32::from_bits(input[i][0]),
            f32::from_bits(input[i][1]),
            f32::from_bits(input[i][2]),
        );
        let values = if kind == "support_local" {
            vec![shapes[2 * i].local_support_point(direction, &vertices)]
        } else if kind == "support_world" {
            vec![shapes[2 * i].support_point(poses[i], direction, &vertices)]
        } else if kind == "cso_support" {
            let p = cso_point_from_shapes(
                poses[i],
                &shapes[2 * i],
                &shapes[2 * i + 1],
                direction,
                &vertices,
            );
            vec![p.point, p.orig_a, p.orig_b]
        } else {
            let r = closest_points(
                poses[i],
                &shapes[2 * i],
                &shapes[2 * i + 1],
                f32::from_bits(input[i][3]),
                input[i][4] != 0,
                &mut simplexes[i],
                &vertices,
            );
            output[i][9] = r.status;
            pack(&simplexes[i], &mut states[i]);
            vec![r.a, r.b, r.dir]
        };
        for (field, v) in values.iter().enumerate() {
            for d in 0..3 {
                output[i][field * 3 + d] = v.to_array()[d].to_bits();
            }
        }
    }
    write!("Output.expected.bin", &output);
    write!("State.expected.bin", &states);
    let sizes = [
        (
            "Shapes",
            fs::metadata(dir.join("Shapes.bin")).unwrap().len() / 4,
        ),
        (
            "Poses",
            fs::metadata(dir.join("Poses.bin")).unwrap().len() / 4,
        ),
        (
            "Vertices",
            fs::metadata(dir.join("Vertices.bin")).unwrap().len() / 4,
        ),
        ("Inputs", storage as u64 * 8),
        ("Output", storage as u64 * 16),
        ("State", storage as u64 * 64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), json!(v)))
    .collect::<serde_json::Map<_, _>>();
    let case = json!({"name":format!("n{n}"),"kind":kind,"count":n,"storage":storage,"groups":1,"iterations":1,"buffer_words":sizes});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&case).unwrap(),
    )
    .unwrap();
    case
}
