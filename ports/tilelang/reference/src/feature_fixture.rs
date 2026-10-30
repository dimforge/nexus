use glamx::Vec3;
use nexus_rbd_shaders3d::{
    PaddedVector,
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

pub fn fixture(dir: &Path, n: usize) -> Value {
    fs::create_dir_all(dir).unwrap();
    let storage = n + 3;
    let guard = 0xdeadbeefu32;
    let mut shapes = vec![shape(0); storage];
    let mut input = vec![[guard; 8]; storage];
    let mut output = vec![[guard; 16]; storage];
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
    let directions = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        -Vec3::X,
        -Vec3::Y,
        -Vec3::Z,
        Vec3::ONE,
        Vec3::new(-0.0, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 2.0),
        Vec3::new(0.4, 0.49999997, 0.2),
        Vec3::new(0.4, 0.5, 0.2),
        Vec3::new(0.4, -0.49999997, 0.2),
        Vec3::new(0.4, -0.5, 0.2),
        Vec3::new(1.0e-8, -1.0, 1.0e-8),
        Vec3::new(1.0e-6, 1.0, 0.0),
        Vec3::new(0.1, 0.2, 0.3),
        Vec3::new(-0.2, 0.3, 0.1),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.5),
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(0.2, -0.7, 0.3),
    ];
    for i in 0..n {
        shapes[i] = shape(i % 7);
        if i % 7 == 2 && (i / 7) % 3 == 0 {
            shapes[i] = Shape::capsule(Vec3::new(0.1, -0.5, 0.0), Vec3::new(-0.1, 0.5, 0.1), 0.0);
        }
        for (dest, value) in input[i][..3]
            .iter_mut()
            .zip(directions[(i / 7) % 24].to_array())
        {
            *dest = value.to_bits();
        }
    }
    macro_rules! write {
        ($name:expr,$values:expr) => {
            fs::write(dir.join($name), bytemuck::cast_slice($values)).unwrap()
        };
    }
    write!("Shapes.bin", &shapes);
    write!("Inputs.bin", &input);
    write!("Output.bin", &output);
    write!("Vertices.bin", &vertices);
    write!("Indices.bin", &indices);
    for i in 0..n {
        let d = Vec3::new(
            f32::from_bits(input[i][0]),
            f32::from_bits(input[i][1]),
            f32::from_bits(input[i][2]),
        );
        let face = shapes[i].support_face(d, &vertices, &indices);
        for v in 0..4 {
            for axis in 0..3 {
                output[i][v * 3 + axis] = face.vertices[v].to_array()[axis].to_bits();
            }
        }
        output[i][12] = face.num_vertices;
    }
    write!("Output.expected.bin", &output);
    let sizes = [
        ("Shapes", storage * std::mem::size_of::<Shape>() / 4),
        ("Inputs", storage * 8),
        ("Output", storage * 16),
        ("Vertices", vertices.len() * 4),
        ("Indices", indices.len()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), json!(v)))
    .collect::<serde_json::Map<_, _>>();
    let c = json!({"name":format!("n{n}"),"kind":"support_face","count":n,"storage":storage,"groups":1,"iterations":1,"buffer_words":sizes});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&c).unwrap(),
    )
    .unwrap();
    c
}
