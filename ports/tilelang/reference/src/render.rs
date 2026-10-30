//! Native direct-to-renderer instance preparation oracle with tight f32 output strides.
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::{RbdInstanceDesc, gpu_rbd_prep_render},
    glamx::{Quat, UVec3, Vec3, Vec4},
};
use serde_json::json;
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn write<T: bytemuck::NoUninit>(d: &Path, n: &str, v: &[T]) {
    fs::write(d.join(n), bytemuck::cast_slice(v)).unwrap();
}
fn main() {
    let out = std::env::args().nth(1).expect("output");
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    let layout = json!({"pose_words":size_of::<Pose>()/4,"pose_rotation":offset_of!(Pose,rotation)/4,"pose_translation":offset_of!(Pose,translation)/4,
 "descriptor_words":size_of::<RbdInstanceDesc>()/4,"descriptor":{"color":offset_of!(RbdInstanceDesc,color)/4,"local_pose":offset_of!(RbdInstanceDesc,local_pose)/4,"scale":offset_of!(RbdInstanceDesc,scale)/4,"pose_index":offset_of!(RbdInstanceDesc,pose_index)/4}});
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&layout).unwrap(),
    )
    .unwrap();
    let root = root.join("prep_render");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for n in [0usize, 1, 31, 32, 33, 63, 64, 65, 129] {
        let name = format!("n{n}");
        let d = root.join(&name);
        fs::create_dir_all(&d).unwrap();
        let storage = n + 3;
        let bodies = n.max(1) / 2 + 3;
        let poses: Vec<_> = (0..bodies)
            .map(|i| {
                Pose::from_parts(
                    Vec3::new(i as f32 * 0.12, -0.2, 0.3),
                    Quat::from_scaled_axis(Vec3::new(0.1, -0.07 * (i % 5) as f32, 0.2)),
                )
            })
            .collect();
        let desc: Vec<_> = (0..storage)
            .map(|i| RbdInstanceDesc {
                color: Vec4::new(-0., 0.25 * (i % 3) as f32, 0.7, 1.),
                local_pose: Pose::from_parts(
                    Vec3::new(-0.03, 0.04 * (i % 7) as f32, 0.15),
                    Quat::from_scaled_axis(Vec3::new(0.03 * (i % 3) as f32, -0.1, 0.04)),
                ),
                scale: Vec3::new(if i % 3 == 0 { 0. } else { -1.2 }, 0.5, 1.7),
                pose_index: ((bodies - 1 - i % bodies) % bodies) as u32,
            })
            .collect();
        let mut positions = vec![f32::from_bits(0xdeadbeef); storage * 3];
        let mut deformations = vec![f32::from_bits(0xdeadbeef); storage * 9];
        let mut colors = vec![f32::from_bits(0xdeadbeef); storage * 4];
        let count = [n as u32, 0x12345678];
        write(&d, "Poses.bin", &poses);
        write(&d, "Descriptors.bin", &desc);
        write(&d, "Count.bin", &count);
        write(&d, "Positions.bin", &positions);
        write(&d, "Deformations.bin", &deformations);
        write(&d, "Colors.bin", &colors);
        for i in 0..n.div_ceil(64).max(1) * 64 {
            gpu_rbd_prep_render(
                UVec3::new(i as u32, 0, 0),
                &mut positions,
                &mut deformations,
                &mut colors,
                &poses,
                &desc,
                &count[0],
            );
        }
        write(&d, "Positions.expected.bin", &positions);
        write(&d, "Deformations.expected.bin", &deformations);
        write(&d, "Colors.expected.bin", &colors);
        cases.push(json!({"name":name,"kind":"prep_render","count":n,"storage":storage,"bodies":bodies,"groups":n.div_ceil(64).max(1),"iterations":1,"buffer_words":{"Poses":poses.len()*size_of::<Pose>()/4,"Descriptors":desc.len()*size_of::<RbdInstanceDesc>()/4,"Count":2,"Positions":positions.len(),"Deformations":deformations.len(),"Colors":colors.len()}}));
    }
    fs::write(
        root.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}
