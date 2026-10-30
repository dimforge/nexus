//! Live public mesh projection, pseudo-normal and ray evaluation oracles.
use nexus_rbd_shaders3d::{
    PaddedVector, bounding_volumes::Aabb, glamx::Vec3, queries::Ray, shapes::TriMesh,
};
use serde_json::json;
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Query {
    point: [f32; 3],
    parameter: f32,
    direction: [f32; 3],
    triangle: u32,
    feature: u32,
    feature_id: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Result {
    point: [f32; 3],
    found: u32,
    inside: u32,
    padding: u32,
}
fn write<T: bytemuck::NoUninit>(d: &Path, n: &str, v: &[T]) {
    fs::write(d.join(n), bytemuck::cast_slice(v)).unwrap();
}
fn main() {
    let out = std::env::args().nth(1).expect("output");
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    let l = json!({"vertex_words":size_of::<PaddedVector>()/4,"vertex_vector":offset_of!(PaddedVector,0)/4,"query_words":size_of::<Query>()/4,"query":{"point":offset_of!(Query,point)/4,"parameter":offset_of!(Query,parameter)/4,"direction":offset_of!(Query,direction)/4,"triangle":offset_of!(Query,triangle)/4,"feature":offset_of!(Query,feature)/4,"feature_id":offset_of!(Query,feature_id)/4},"result_words":size_of::<Result>()/4,"result":{"point":offset_of!(Result,point)/4,"found":offset_of!(Result,found)/4,"inside":offset_of!(Result,inside)/4,"padding":offset_of!(Result,padding)/4}});
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&l).unwrap(),
    )
    .unwrap();
    let index_root = 5;
    let vertex_root = 2;
    let nodes = 3;
    let vertices_count = 4;
    let triangles = 2;
    let bounds = Aabb::new(Vec3::ZERO, Vec3::new(1., 1., 0.));
    let mesh = TriMesh::new(
        vertex_root,
        index_root,
        nodes,
        triangles,
        vertices_count,
        bounds,
    );
    let mut indices = vec![0xdeadbeef; index_root as usize];
    indices.extend([1, 3, 0, u32::MAX, 2, 0, u32::MAX, 3, 1, 0, 1, 2, 1, 3, 2]);
    indices.extend([0xdeadbeef; 3]);
    let mut vertices = vec![PaddedVector::from(Vec3::splat(13.)); vertex_root as usize];
    for _ in 0..nodes {
        vertices.extend([Vec3::ZERO, Vec3::new(1., 1., 0.)].map(PaddedVector::from));
    }
    vertices.extend([Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1., 1., 0.)].map(PaddedVector::from));
    for j in 0..vertices_count {
        vertices.push(Vec3::new(0., 0., 2. + j as f32).into());
    }
    for j in 0..triangles * 3 {
        vertices.push(Vec3::new(0., 0., 7. + j as f32).into());
    }
    vertices.extend([PaddedVector::from(Vec3::splat(19.)); 3]);
    let points = [
        Vec3::new(0.2, 0.3, 0.5),
        Vec3::new(0.8, 0.7, -0.5),
        Vec3::new(0.5, -0.375, 0.25),
        Vec3::new(-0.375, 0.5, -0.25),
        Vec3::new(1.4, 1.3, 0.2),
        Vec3::new(-0.2, -0.3, -0.1),
        Vec3::new(0.5, 0.5, 0.),
        Vec3::new(0., 1., 0.25),
        Vec3::new(1.4, 0.3, 0.),
        Vec3::new(0.4, 1.3, -0.2),
        Vec3::new(0.25, 0.25, 0.5),
        Vec3::new(0.25, 0.25, -0.5),
    ];
    for kind in ["mesh_projection", "mesh_pseudo_normal", "ray_point"] {
        let folder = root.join(kind);
        fs::create_dir_all(&folder).unwrap();
        let mut cases = Vec::new();
        for n in [0usize, 1, 31, 32, 33, 63, 64, 65, 129] {
            let name = format!("n{n}");
            let d = folder.join(&name);
            fs::create_dir_all(&d).unwrap();
            let storage = n + 3;
            let queries: Vec<_> = (0..storage)
                .map(|i| Query {
                    point: points[i % points.len()].to_array(),
                    parameter: if kind == "ray_point" {
                        (i as i32 % 7 - 3) as f32 * 0.25
                    } else {
                        [10., 0., 0.25, 0.5, 0.75, 2.][(i / points.len()) % 6]
                    },
                    direction: if i % 4 == 0 {
                        [0.; 3]
                    } else {
                        [0.2, -0.5, 1.2]
                    },
                    triangle: (i % 2) as u32,
                    feature: [0, 1, 2, 3, 99][(i / 6) % 5],
                    feature_id: ((i / 2) % 3) as u32,
                })
                .collect();
            let mut result = vec![
                Result {
                    point: [f32::from_bits(0xdeadbeef); 3],
                    found: 0xdeadbeef,
                    inside: 0xdeadbeef,
                    padding: 0xdeadbeef
                };
                storage
            ];
            let count = [n as u32, 0x12345678];
            write(&d, "Queries.bin", &queries);
            write(&d, "Vertices.bin", &vertices);
            write(&d, "Indices.bin", &indices);
            write(&d, "Count.bin", &count);
            write(&d, "Results.bin", &result);
            for i in 0..n {
                let q = queries[i];
                let pt = Vec3::from_array(q.point);
                let (point, found, inside) = match kind {
                    "mesh_projection" => {
                        let (p, f) = mesh.project_local_point(&indices, &vertices, pt, q.parameter);
                        (p.point, f, p.is_inside)
                    }
                    "mesh_pseudo_normal" => (
                        mesh.pseudo_normal(
                            &indices,
                            &vertices,
                            q.triangle,
                            q.feature,
                            q.feature_id,
                        ),
                        true,
                        false,
                    ),
                    _ => (
                        Ray::new(pt, Vec3::from_array(q.direction)).point_at(q.parameter),
                        true,
                        false,
                    ),
                };
                result[i].point = point.to_array();
                result[i].found = found as u32;
                result[i].inside = inside as u32;
            }
            write(&d, "Results.expected.bin", &result);
            cases.push(json!({"name":name,"kind":kind,"count":n,"storage":storage,"index_root":index_root,"vertex_root":vertex_root,"nodes":nodes,"vertices_count":vertices_count,"triangles":triangles,"groups":n.div_ceil(64).max(1),"iterations":1,"buffer_words":{"Queries":queries.len()*size_of::<Query>()/4,"Vertices":vertices.len()*size_of::<PaddedVector>()/4,"Indices":indices.len(),"Count":2,"Results":result.len()*size_of::<Result>()/4}}));
        }
        fs::write(
            folder.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
