use glamx::{Quat, Vec3};
use nexus_rbd_shaders3d::Pose;
use serde_json::{Value, json};
use std::{fs, path::Path};
fn feature(count: u32, degenerate: bool) -> nexus_rbd_shaders3d::queries::PolygonalFeature {
    let mut f = nexus_rbd_shaders3d::queries::PolygonalFeature {
        num_vertices: count,
        ..Default::default()
    };
    let v = match count {
        0 => [Vec3::ZERO; 4],
        1 => [
            Vec3::new(0.1, -0.1, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ],
        2 => [
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
        ],
        3 => [
            Vec3::new(-0.6, -0.4, 0.0),
            Vec3::new(0.6, -0.4, 0.0),
            Vec3::new(0.0, 0.6, 0.0),
            Vec3::ZERO,
        ],
        _ => [
            Vec3::new(-0.6, -0.4, 0.0),
            Vec3::new(0.6, -0.4, 0.0),
            Vec3::new(0.6, 0.4, 0.0),
            Vec3::new(-0.6, 0.4, 0.0),
        ],
    };
    f.vertices = v;
    if degenerate && count == 2 {
        f.vertices[1] = f.vertices[0];
    }
    f
}
pub fn fixture(dir: &Path, n: usize) -> Value {
    fixture_impl(dir, n, false)
}
pub fn witness_fixture(dir: &Path) -> Value {
    fixture_impl(dir, 24, true)
}
fn fixture_impl(dir: &Path, n: usize, witness_case: bool) -> Value {
    fs::create_dir_all(dir).unwrap();
    let storage = n + 3;
    let guard = 0xdeadbeefu32;
    let mut features = vec![[guard; 16]; 2 * storage];
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut input = vec![[guard; 8]; storage];
    let mut output = vec![[guard; 32]; storage];
    let mut fs = vec![];
    for i in 0..n {
        let profile = (i / 25) % 6;
        let a = feature((i % 5) as u32, profile == 1);
        let b = feature(((i / 5) % 5) as u32, profile == 1);
        fs.push((a, b));
        for (side, f) in [a, b].iter().enumerate() {
            for v in 0..4 {
                for d in 0..3 {
                    features[i * 2 + side][v * 3 + d] = f.vertices[v].to_array()[d].to_bits();
                }
            }
            features[i * 2 + side][12] = f.num_vertices;
        }
        poses[i] = Pose {
            rotation: match profile {
                2 => Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                3 => Quat::from_rotation_z(std::f32::consts::PI),
                4 => Quat::from_rotation_z(0.3),
                5 => Quat::from_scaled_axis(Vec3::new(-0.2, 0.1, 0.4)),
                _ => Quat::IDENTITY,
            },
            translation: if profile == 1 {
                Vec3::new(3.0, 0.0, 0.01)
            } else {
                Vec3::new(0.1, -0.1, if i % 3 == 0 { -0.02 } else { 0.01 })
            },
            ..Pose::IDENTITY
        };
        input[i][0] = 0;
        input[i][1] = 0;
        input[i][2] = 1.0f32.to_bits();
        input[i][3] = if i % 3 == 0 { 0.0f32 } else { 0.02f32 }.to_bits();
        input[i][4] = ((i / 150) % 2) as u32;
    }
    if witness_case {
        for i in 0..n {
            let scale = [0.001, 1.0, 1000.0][i / 8];
            let mode = i % 4;
            let direction = if mode == 3 { Vec3::Y } else { Vec3::X };
            let a = feature(2, false);
            let mut b = a;
            b.vertices[0] = -direction;
            b.vertices[1] = direction;
            if mode == 1 {
                b.vertices.swap(0, 1);
            }
            let mut a = a;
            a.vertices[0] = -Vec3::X;
            a.vertices[1] = Vec3::X;
            for f in [&mut a, &mut b] {
                for v in &mut f.vertices {
                    *v *= scale;
                }
            }
            fs[i] = (a, b);
            for (side, f) in [a, b].iter().enumerate() {
                for v in 0..4 {
                    for d in 0..3 {
                        features[i * 2 + side][v * 3 + d] = f.vertices[v].to_array()[d].to_bits();
                    }
                }
                features[i * 2 + side][12] = 2;
            }
            poses[i] = Pose::from_translation(
                Vec3::new(0.0, if mode < 2 { 0.1 } else { 0.0 }, -0.1) * scale,
            );
            input[i][3] = (0.02f32 * scale).to_bits();
            input[i][4] = ((i / 4) % 2) as u32;
        }
    }
    macro_rules! write {
        ($name:expr,$values:expr) => {
            std::fs::write(dir.join($name), bytemuck::cast_slice($values)).unwrap()
        };
    }
    write!("Features.bin", &features);
    write!("Poses.bin", &poses);
    write!("Inputs.bin", &input);
    write!("Output.bin", &output);
    for i in 0..n {
        let normal = Vec3::Z;
        let inv = poses[i].inverse();
        let manifold = nexus_rbd_shaders3d::queries::contacts(
            poses[i],
            inv,
            normal,
            inv.transform_vector(-normal),
            &fs[i].0,
            &fs[i].1,
            f32::from_bits(input[i][3]),
            input[i][4] != 0,
        );
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
    write!("Output.expected.bin", &output);
    let sizes = [
        ("Features", storage * 32),
        ("Poses", storage * 8),
        ("Inputs", storage * 8),
        ("Output", storage * 32),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), json!(v)))
    .collect::<serde_json::Map<_, _>>();
    let c = json!({"name":if witness_case { "edge_witnesses".to_owned() } else {format!("n{n}")},"kind":"polygonal_contacts","count":n,"storage":storage,"groups":1,"iterations":1,"buffer_words":sizes});
    std::fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&c).unwrap(),
    )
    .unwrap();
    c
}
