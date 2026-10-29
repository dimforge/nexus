//! Live native narrow-phase oracle for the native analytic first pass.
#[allow(dead_code)]
#[path = "contacts.rs"]
mod contacts;
use nexus_rbd_shaders3d::{
    Pose,
    broad_phase::{CollisionPair, ContactPlan, gpu_narrow_phase_shape_shape},
    dynamics::RbdSimParams,
    glamx::{Quat, UVec2, UVec3, Vec3},
    queries::{ColliderMaterial, IndexedManifold},
    shapes::*,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn layout() -> Value {
    let mut l = contacts::layout();
    macro_rules! fields {($ty:ty,$($field:ident),+)=>{json!({$(stringify!($field):offset_of!($ty,$field)/4),+})};}
    l["shape_words"] = json!(size_of::<Shape>() / 4);
    l["shape"] = fields!(Shape, a, b, c);
    l["shape_types"] = json!({"ball":SHAPE_TYPE_BALL,"cuboid":SHAPE_TYPE_CUBOID,"capsule":SHAPE_TYPE_CAPSULE,"cone":SHAPE_TYPE_CONE,"cylinder":SHAPE_TYPE_CYLINDER});
    l["contact"]["recycle_extent"] = json!(offset_of!(IndexedManifold, recycle_extent) / 4);
    l["plan_words"] = json!(size_of::<ContactPlan>() / 4);
    l["plan"] = fields!(ContactPlan, bound, pfm_base, pfm_len);
    l["pair_words"] = json!(size_of::<CollisionPair>() / 4);
    l["pair_colliders"] = json!(offset_of!(CollisionPair, colliders) / 4);
    l["material_words"] = json!(size_of::<ColliderMaterial>() / 4);
    l["material"] = fields!(
        ColliderMaterial,
        friction,
        restitution,
        friction_combine_rule,
        restitution_combine_rule
    );
    l["rbd_params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["rbd_params"] = fields!(RbdSimParams, normalized_prediction_distance, length_unit);
    l
}
fn shape(ty: usize) -> Shape {
    match ty {
        0 => Shape::ball(0.45),
        1 => Shape::cuboid(Vec3::new(0.5, 0.8, 0.3)),
        2 => Shape::capsule(Vec3::new(0., -0.7, 0.), Vec3::new(0., 0.7, 0.), 0.3),
        3 => Shape::cone(0.8, 0.5),
        4 => Shape::cylinder(0.8, 0.5),
        _ => Shape::trimesh(0, 0, 0, 0, 0, Vec3::ZERO, Vec3::ONE),
    }
}
fn fixture(root: &Path, n: usize, groups: usize, mode: &str) -> Value {
    let name = format!("n{n}-g{groups}-{mode}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let storage = n.max(1) + 3;
    let colliders = (n * 2).max(1) + 3;
    let mut pairs = vec![CollisionPair::default(); storage];
    let mut poses = vec![Pose::IDENTITY; colliders];
    let mut shapes = vec![shape(0); colliders];
    let mut parents: Vec<u32> = (0..colliders as u32).collect();
    let mut mats = vec![ColliderMaterial::default(); colliders];
    let mut output = vec![IndexedManifold::default(); storage];
    for m in &mut output {
        bytemuck::cast_slice_mut::<_, u32>(std::slice::from_mut(m)).fill(0xdeadbeef);
    }
    let mut p = RbdSimParams::tgs_soft();
    p.normalized_prediction_distance = 0.1;
    p.length_unit = if mode == "scaled" { 2. } else { 1. };
    for (i, pair) in pairs.iter_mut().enumerate().take(n) {
        let a = i * 2;
        let b = a + 1;
        pair.colliders = UVec2::new(a as u32, b as u32);
        let ty = if mode == "materials" {
            0
        } else if mode.contains("capsule") {
            2
        } else {
            mode.strip_prefix("shape-")
                .map(|x| x.parse::<usize>().unwrap())
                .unwrap_or((i / 2) % 6)
        };
        let flip = i % 2 == 1;
        shapes[a] = shape(if flip { 0 } else { ty });
        shapes[b] = shape(if flip { ty } else { 0 });
        if mode.starts_with("boxes") {
            shapes[a] = Shape::cuboid(Vec3::new(0.5, 0.8, 0.3));
            shapes[b] = Shape::cuboid(Vec3::new(0.6, 0.4, 0.7));
        }
        if mode == "boxes-degenerate" {
            shapes[a] = Shape::cuboid(Vec3::ZERO);
        }
        if mode == "boxes-thin" {
            shapes[a] = Shape::cuboid(Vec3::new(0.5, 0.0001, 0.3));
        }
        if mode == "degenerate-capsule" || mode == "zero-radius-capsule" {
            let capsule = Shape::capsule(
                Vec3::ZERO,
                Vec3::ZERO,
                if mode == "zero-radius-capsule" {
                    0.
                } else {
                    0.3
                },
            );
            if flip {
                shapes[b] = capsule;
            } else {
                shapes[a] = capsule;
            }
        }
        let t = match mode.strip_prefix("boxes-").unwrap_or(mode) {
            "coincident" | "degenerate-capsule" | "zero-radius-capsule" => Vec3::ZERO,
            "materials" => Vec3::new(0.6, 0.2, -0.17),
            "negative-zero" => Vec3::new(-0.0, 0., -0.0),
            "axis" => Vec3::new(0., 0.3, 0.),
            "boundary" => Vec3::new(0.5, 0., 0.),
            "inside" => Vec3::new(0.07, -0.2, 0.06),
            "far" => Vec3::splat(3.),
            "threshold" => Vec3::new(1., 0., 0.),
            "touching" => Vec3::new(1.1, 0., 0.),
            "prediction" => Vec3::new(1.2, 0., 0.),
            "edge" => Vec3::new(0.9, 1., 0.7),
            _ => match (i / 12) % 8 {
                0 => Vec3::new(0.6, 0.2, -0.17),
                1 => Vec3::new(0.04, 0.1, 0.03),
                2 => Vec3::new(0.1, 1., 0.1),
                3 => Vec3::new(0.1, -1., 0.1),
                4 => Vec3::new(0.8, 1., 0.8),
                5 => Vec3::new(0.8, -1., 0.8),
                6 => Vec3::splat(3.),
                _ => Vec3::new(0., 0.3, 0.),
            },
        };
        if mode == "rotated"
            || mode == "mixed"
            || mode == "scaled"
            || mode == "boxes"
            || mode == "boxes-rotated"
            || mode == "boxes-edge"
        {
            let base = Pose::from_parts(
                Vec3::new(-0.3, 0.4, 0.1),
                Quat::from_scaled_axis(Vec3::new(0.3, 0.1, -0.2)),
            );
            poses[a] = base;
            poses[b] =
                base * Pose::from_parts(t, Quat::from_scaled_axis(Vec3::new(-0.2, 0.1, 0.05)));
        } else {
            poses[b] = Pose::from_translation(t);
        }
        poses[a].padding = 0xabcdef12;
        poses[b].padding = 0x12345678;
        if mode == "same-body" || (mode == "mixed" && i % 17 == 0) {
            parents[b] = parents[a];
        }
        if mode == "materials" {
            mats[a] = ColliderMaterial {
                friction: -0.25,
                restitution: 0.4,
                friction_combine_rule: (i / 6) as u32,
                restitution_combine_rule: (i % 6) as u32,
            };
            mats[b] = ColliderMaterial {
                friction: 0.75,
                restitution: 0.8,
                friction_combine_rule: (i % 6) as u32,
                restitution_combine_rule: (i / 6) as u32,
            };
        }
    }
    let plan = ContactPlan {
        bound: n as u32,
        pfm_base: n as u32,
        pfm_len: 0,
    };
    macro_rules! w {
        ($n:expr,$d:expr) => {
            contacts::pilot::write_buffer(&dir, $n, $d)
        };
    }
    w!("Pairs.bin", &pairs);
    w!("Plan.bin", &[plan]);
    w!("Poses.bin", &poses);
    w!("Shapes.bin", &shapes);
    w!("Contacts.bin", &output);
    w!("Parents.bin", &parents);
    w!("Materials.bin", &mats);
    w!("RbdParams.bin", &[p]);
    for gid in 0..groups * 64 {
        gpu_narrow_phase_shape_shape(
            UVec3::new(gid as u32, 0, 0),
            UVec3::new(groups as u32, 1, 1),
            &pairs,
            &plan,
            &poses,
            &shapes,
            &mut output,
            &parents,
            &mats,
            &p,
        );
    }
    w!("Contacts.expected.bin", &output);
    let sizes: serde_json::Map<String, Value> = [
        "Pairs",
        "Plan",
        "Poses",
        "Shapes",
        "Contacts",
        "Parents",
        "Materials",
        "RbdParams",
    ]
    .into_iter()
    .map(|name| {
        (
            name.to_owned(),
            json!(fs::metadata(dir.join(format!("{name}.bin"))).unwrap().len() / 4),
        )
    })
    .collect();
    json!({"name":name,"kind":"analytic_contacts","count":n,"storage":storage,"colliders":colliders,"groups":groups,"iterations":1,"buffer_words":sizes,"mode":mode})
}
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    contacts::pilot::write_json(root.join("layout.json"), &layout());
    let cases_root = root.join("analytic_contacts");
    fs::create_dir_all(&cases_root).unwrap();
    let root = cases_root.as_path();
    let mut cases = Vec::new();
    for (n, g) in [
        (0, 1),
        (1, 1),
        (31, 1),
        (32, 2),
        (33, 1),
        (63, 2),
        (64, 1),
        (65, 1),
        (129, 1),
    ] {
        cases.push(fixture(root, n, g, "mixed"));
    }
    for mode in [
        "coincident",
        "negative-zero",
        "axis",
        "boundary",
        "inside",
        "far",
        "threshold",
        "rotated",
        "scaled",
        "same-body",
        "materials",
        "degenerate-capsule",
        "zero-radius-capsule",
    ] {
        cases.push(fixture(
            root,
            if mode == "materials" { 36 } else { 96 },
            1,
            mode,
        ));
    }
    for ty in 0..6 {
        cases.push(fixture(root, 32, 1, &format!("shape-{ty}")));
    }
    for mode in [
        "boxes",
        "boxes-coincident",
        "boxes-far",
        "boxes-rotated",
        "boxes-degenerate",
        "boxes-thin",
        "boxes-touching",
        "boxes-prediction",
        "boxes-edge",
    ] {
        cases.push(fixture(root, 96, 1, mode));
    }
    contacts::pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_export_preserves_saved_outputs() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nexus-narrow-export-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let first = fixture(&root, 5, 1, "mixed");
        let d = root.join(first["name"].as_str().unwrap());
        let state = fs::read(d.join("Contacts.expected.bin")).unwrap();
        fs::write(d.join("Contacts.actual.bin"), b"runtime artifact").unwrap();
        let second = fixture(&root, 5, 1, "mixed");
        let after = fs::read(d.join("Contacts.expected.bin")).unwrap();
        let artifact = fs::read(d.join("Contacts.actual.bin")).unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(first, second);
        assert_eq!(state, after);
        assert_eq!(artifact, b"runtime artifact");
    }
}
