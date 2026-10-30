use crate::queries::gjk::{VoronoiSimplex, cso_point::CsoPoint};
use nexus_rbd_shaders3d::glamx::Vec3;
use serde_json::json;
use std::{fs, path::Path};

pub(crate) fn pack(s: &VoronoiSimplex, row: &mut [u32; 64]) {
    for i in 0..4 {
        for (field, v) in [
            s.vertices[i].point,
            s.vertices[i].orig_a,
            s.vertices[i].orig_b,
        ]
        .iter()
        .enumerate()
        {
            for (d, x) in v.to_array().iter().enumerate() {
                row[i * 9 + field * 3 + d] = x.to_bits();
            }
        }
    }
    for d in 0..3 {
        row[36 + d] = s.proj.to_array()[d].to_bits();
        row[39 + d] = s.prev_proj.to_array()[d].to_bits();
    }
    row[42] = s.dim;
    row[43] = s.prev_dim;
    row[44..48].copy_from_slice(&s.prev_vertices);
}
fn point(v: Vec3, i: usize) -> CsoPoint {
    let a = v * 0.5 + Vec3::splat(i as f32 * 0.03);
    CsoPoint::new(v, a, a - v)
}
fn unpack(row: &[u32; 64]) -> VoronoiSimplex {
    let vector = |at| {
        Vec3::new(
            f32::from_bits(row[at]),
            f32::from_bits(row[at + 1]),
            f32::from_bits(row[at + 2]),
        )
    };
    VoronoiSimplex {
        vertices: std::array::from_fn(|i| {
            CsoPoint::new(vector(i * 9), vector(i * 9 + 3), vector(i * 9 + 6))
        }),
        proj: vector(36),
        prev_proj: vector(39),
        dim: row[42],
        prev_dim: row[43],
        prev_vertices: row[44..48].try_into().unwrap(),
    }
}
pub fn fixture(dir: &Path, op: &str, count: usize, source: &Path) -> serde_json::Value {
    fs::create_dir_all(dir).unwrap();
    let bytes = fs::read(source.join("tetrahedron_projection/n129/Inputs.bin")).unwrap();
    let geom: &[[u32; 24]] = bytemuck::cast_slice(&bytes);
    let storage = count + 3;
    let guard = 0xdeadbeefu32;
    let mut values = vec![[guard; 64]; storage];
    for (i, row) in values.iter_mut().take(count).enumerate() {
        let coords = &geom[i];
        let v = |j| {
            Vec3::new(
                f32::from_bits(coords[j * 4]),
                f32::from_bits(coords[j * 4 + 1]),
                f32::from_bits(coords[j * 4 + 2]),
            )
        };
        let offset = v(4);
        let dim = (i % if op == "simplex_add" { 3 } else { 4 }) as u32;
        let simplex = VoronoiSimplex {
            vertices: std::array::from_fn(|k| point(v(k) - offset, k)),
            proj: Vec3::new(0.2, 0.3, 0.5),
            prev_proj: Vec3::new(0.1, 0.2, 0.7),
            dim,
            prev_dim: dim.min(2),
            prev_vertices: [2, 0, 3, 1],
        };
        pack(&simplex, row);
        let next = if i % 4 == 0 {
            simplex.vertices[0]
        } else {
            point(v((dim + 1) as usize % 4) - offset, 5)
        };
        for (field, v) in [next.point, next.orig_a, next.orig_b].iter().enumerate() {
            for (d, x) in v.to_array().iter().enumerate() {
                row[48 + field * 3 + d] = x.to_bits();
            }
        }
    }
    fs::write(dir.join("State.bin"), bytemuck::cast_slice(&values)).unwrap();
    for row in values.iter_mut().take(count) {
        let mut s = unpack(row);
        let vec = |at| {
            Vec3::new(
                f32::from_bits(row[at]),
                f32::from_bits(row[at + 1]),
                f32::from_bits(row[at + 2]),
            )
        };
        let next = CsoPoint::new(vec(48), vec(51), vec(54));
        if op == "simplex_init" {
            s = VoronoiSimplex::init(next);
        } else if op == "simplex_add" {
            s = s.add_point(next);
        } else {
            let projected = s.project_origin_and_reduce();
            s = projected.simplex;
            for d in 0..3 {
                row[57 + d] = projected.point.to_array()[d].to_bits();
            }
        }
        pack(&s, row);
    }
    fs::write(
        dir.join("State.expected.bin"),
        bytemuck::cast_slice(&values),
    )
    .unwrap();
    let case = json!({"name":dir.file_name().unwrap().to_str().unwrap(),"kind":op,"count":count,"storage":storage,"groups":1,"iterations":1,"buffer_words":{"State":storage*64}});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&case).unwrap(),
    )
    .unwrap();
    case
}
