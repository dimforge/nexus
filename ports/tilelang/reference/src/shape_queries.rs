//! Native public Shape projection wrapper fixtures, including native singular behavior.
use nexus_rbd_shaders3d::{
    Pose,
    glamx::{Quat, Vec3},
    shapes::{
        SHAPE_TYPE_BALL, SHAPE_TYPE_CAPSULE, SHAPE_TYPE_CONE, SHAPE_TYPE_CUBOID,
        SHAPE_TYPE_CYLINDER, Shape,
    },
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
    shape_index: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Result {
    point: [f32; 3],
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
    let layout = json!({"shape_words":size_of::<Shape>()/4,"shape":{"a":offset_of!(Shape,a)/4,"b":offset_of!(Shape,b)/4,"c":offset_of!(Shape,c)/4},"shape_types":{"ball":SHAPE_TYPE_BALL,"cuboid":SHAPE_TYPE_CUBOID,"capsule":SHAPE_TYPE_CAPSULE,"cone":SHAPE_TYPE_CONE,"cylinder":SHAPE_TYPE_CYLINDER},"pose_words":size_of::<Pose>()/4,"pose_rotation":offset_of!(Pose,rotation)/4,"pose_translation":offset_of!(Pose,translation)/4,"query_words":size_of::<Query>()/4,"query":{"point":offset_of!(Query,point)/4,"shape_index":offset_of!(Query,shape_index)/4},"result_words":size_of::<Result>()/4,"result":{"point":offset_of!(Result,point)/4,"inside":offset_of!(Result,inside)/4,"padding":offset_of!(Result,padding)/4}});
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&layout).unwrap(),
    )
    .unwrap();
    let mut fallback = Shape::ball(0.5);
    fallback.a[3] = f32::from_bits(987654);
    let shapes = [
        Shape::ball(0.6),
        Shape::cuboid(Vec3::new(0.7, 1.1, 0.5)),
        Shape::capsule(Vec3::new(-0.2, -0.7, 0.1), Vec3::new(0.2, 0.7, -0.1), 0.35),
        Shape::capsule(Vec3::ZERO, Vec3::ZERO, 0.35),
        Shape::cone(1.2, 0.6),
        Shape::cylinder(1.2, 0.6),
        fallback,
    ];
    let points = [
        Vec3::ZERO,
        Vec3::new(0.1, 0.15, 0.05),
        Vec3::new(1.4, 0.2, 0.4),
        Vec3::new(0.2, 2.1, -0.1),
        Vec3::new(-0.3, -2.2, 0.2),
        Vec3::new(1.1, -2., 0.7),
        Vec3::new(-1.2, 1.8, -0.9),
        Vec3::new(0., 0.45, 0.),
        Vec3::new(0., -0.85, 0.),
        Vec3::new(0.3, -0.1, -0.2),
        Vec3::new(-0.4, 0.2, 0.1),
        Vec3::new(0.8, -0.7, 0.6),
    ];
    for kind in [
        "shape_project_local",
        "shape_project_local_boundary",
        "shape_project_world",
        "shape_project_world_boundary",
    ] {
        let folder = root.join(kind);
        fs::create_dir_all(&folder).unwrap();
        let mut cases = Vec::new();
        let world = kind.contains("world");
        let boundary = kind.ends_with("boundary");
        for n in [0usize, 1, 31, 32, 33, 63, 64, 65, 129] {
            let name = format!("n{n}");
            let d = folder.join(&name);
            fs::create_dir_all(&d).unwrap();
            let storage = n + 3;
            let poses: Vec<_> = (0..storage)
                .map(|i| {
                    Pose::from_parts(
                        Vec3::new(-0.25, 0.4 * (i % 3) as f32, 0.7),
                        if i % 3 == 0 {
                            Quat::IDENTITY
                        } else {
                            Quat::from_scaled_axis(Vec3::new(0.13, -0.27, 0.21))
                        },
                    )
                })
                .collect();
            let queries: Vec<_> = (0..storage)
                .map(|i| {
                    let pt = points[(i / shapes.len()) % points.len()];
                    Query {
                        point: if world {
                            (poses[i] * pt).to_array()
                        } else {
                            pt.to_array()
                        },
                        shape_index: (i % shapes.len()) as u32,
                    }
                })
                .collect();
            let mut result = vec![
                Result {
                    point: [f32::from_bits(0xdeadbeef); 3],
                    inside: 0xdeadbeef,
                    padding: 0xdeadbeef
                };
                storage
            ];
            let count = [n as u32, 0x12345678];
            write(&d, "Shapes.bin", &shapes);
            write(&d, "Poses.bin", &poses);
            write(&d, "Queries.bin", &queries);
            write(&d, "Count.bin", &count);
            write(&d, "Results.bin", &result);
            for i in 0..n {
                let shape = shapes[queries[i].shape_index as usize];
                let pt = Vec3::from_array(queries[i].point);
                let (point, inside) = if boundary {
                    let projected = if world {
                        shape.project_point_on_boundary(poses[i], pt)
                    } else {
                        shape.project_local_point_on_boundary(pt)
                    };
                    (projected.point, projected.is_inside)
                } else {
                    (
                        if world {
                            shape.project_point(poses[i], pt)
                        } else {
                            shape.project_local_point(pt)
                        },
                        false,
                    )
                };
                result[i].point = point.to_array();
                result[i].inside = inside as u32;
            }
            write(&d, "Results.expected.bin", &result);
            cases.push(json!({"name":name,"kind":kind,"count":n,"storage":storage,"shapes":shapes.len(),"groups":n.div_ceil(64).max(1),"iterations":1,"buffer_words":{"Shapes":shapes.len()*size_of::<Shape>()/4,"Poses":poses.len()*size_of::<Pose>()/4,"Queries":queries.len()*size_of::<Query>()/4,"Count":2,"Results":result.len()*size_of::<Result>()/4}}));
        }
        fs::write(
            folder.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
