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
        5 => Shape::convex_poly(2, 10, 3, 42),
        _ => Shape::from_triangle(&Triangle::new(
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.1),
        )),
    }
}

pub fn fixture(dir: &Path, kind: &str, n: usize) -> Value {
    fixture_impl(dir, kind, n, false)
}
pub fn triangle_fixture(dir: &Path) -> Value {
    fixture_impl(dir, "pfm_manifold", 2, true)
}
fn fixture_impl(dir: &Path, kind: &str, n: usize, triangle_case: bool) -> Value {
    fs::create_dir_all(dir).unwrap();
    let storage = n + 3;
    let guard = 0xdeadbeefu32;
    let mut shapes = vec![shape(0); storage * 2];
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut input = vec![[guard; 8]; storage];
    let mut output = vec![[guard; 32]; storage];
    let mut subshapes = vec![[guard; 12]; storage * 2];
    let mut radii = vec![[guard; 4]; storage];
    let mut vertices = vec![PaddedVector::new(Vec3::splat(19.0)); 2];
    vertices.extend((0..8).map(|i| {
        PaddedVector::new(Vec3::new(
            if i & 1 == 0 { -0.5 } else { 0.5 },
            if i & 2 == 0 { -0.7 } else { 0.7 },
            if i & 4 == 0 { -0.3 } else { 0.3 },
        ))
    }));
    vertices.push(PaddedVector::new(Vec3::splat(23.0)));
    let indices = vec![
        guard, guard, guard, 0, 0, 0, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6,
        7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, guard,
    ];
    for i in 0..n {
        shapes[i * 2] = shape(i % 7);
        shapes[i * 2 + 1] = shape((i / 7) % 7);
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
        input[i][0] = 0.02f32.to_bits();
    }
    if triangle_case {
        let f = f32::from_bits;
        let pose = Pose::from_parts(
            Vec3::new(f(1067869798), f(1045220559), f(3184315596)),
            Quat::from_xyzw(f(1048315297), f(3189969584), f(1051360326), f(1063611899)),
        );
        let a = Vec3::new(-0.6, -0.5, 0.0);
        let b = Vec3::new(0.6, -0.5, 0.0);
        let c = Vec3::new(0.6, 0.5, 0.0);
        let d = Vec3::new(-0.6, 0.5, 0.0);
        shapes[0] = Shape::from_triangle(&Triangle::new(a, b, c));
        shapes[1] = shape(5);
        shapes[2] = Shape::from_triangle(&Triangle::new(a, c, d));
        shapes[3] = shape(5);
        poses[0] = pose;
        poses[1] = pose;
    }
    macro_rules! write {
        ($name:expr,$values:expr) => {
            fs::write(dir.join($name), bytemuck::cast_slice($values)).unwrap()
        };
    }
    write!("Shapes.bin", &shapes);
    write!("Poses.bin", &poses);
    write!("Inputs.bin", &input);
    write!("Output.bin", &output);
    write!("Vertices.bin", &vertices);
    write!("Indices.bin", &indices);
    write!("SubShapes.bin", &subshapes);
    write!("Radii.bin", &radii);
    for i in 0..n {
        let a = shapes[i * 2].pfm_subshape();
        let b = shapes[i * 2 + 1].pfm_subshape();
        subshapes[i * 2].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&a.shape)));
        subshapes[i * 2 + 1].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&b.shape)));
        radii[i] = [
            a.thickness.to_bits(),
            a.valid as u32,
            b.thickness.to_bits(),
            b.valid as u32,
        ];
        if kind == "pfm_manifold" {
            let manifold = if a.valid && b.valid {
                nexus_rbd_shaders3d::queries::pfm_pfm(
                    poses[i],
                    &a.shape,
                    a.thickness,
                    &b.shape,
                    b.thickness,
                    f32::from_bits(input[i][0]),
                    &vertices,
                    &indices,
                )
            } else {
                nexus_rbd_shaders3d::queries::ContactManifold::default()
            };
            for v in 0..4 {
                for d in 0..3 {
                    output[i][v * 4 + d] = manifold.points_a[v].pt.to_array()[d].to_bits();
                }
                output[i][v * 4 + 3] = manifold.points_a[v].dist.to_bits();
            }
            for d in 0..3 {
                output[i][16 + d] = manifold.normal_a.to_array()[d].to_bits();
            }
            output[i][19] = manifold.len;
        }
    }
    write!("Output.expected.bin", &output);
    write!("SubShapes.expected.bin", &subshapes);
    write!("Radii.expected.bin", &radii);
    let sizes = [
        ("Shapes", storage * 24),
        ("Poses", storage * 8),
        ("Inputs", storage * 8),
        ("Output", storage * 32),
        ("Vertices", vertices.len() * 4),
        ("Indices", indices.len()),
        ("SubShapes", storage * 24),
        ("Radii", storage * 4),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), json!(v)))
    .collect::<serde_json::Map<_, _>>();
    let c = json!({"name":if triangle_case {"triangle_fallback".to_owned()} else {format!("n{n}")},"kind":kind,"count":n,"storage":storage,"groups":1,"iterations":1,"buffer_words":sizes});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&c).unwrap(),
    )
    .unwrap();
    c
}
