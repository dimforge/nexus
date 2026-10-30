//! Live native ball/box scenes with device-stage-equivalent persistence.
use crate::{bucket_program, coloring_program, prefix_program, radix};
use bytemuck::Zeroable;
use khal::backend::GpuBuffer;
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    bounding_volumes::Aabb,
    broad_phase::*,
    dynamics::*,
    glamx::{UVec2, UVec3, Vec3, Vec4},
    queries::{ColliderMaterial, IndexedManifold},
    rapier::geometry::InteractionGroups,
    shapes::*,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
#[allow(dead_code)]
#[path = "lbvh_runtime.rs"]
mod lbvh;
#[path = "scene_hubs.rs"]
mod scene_hubs;
#[path = "scene_model.rs"]
mod scene_model;
#[allow(dead_code)]
#[path = "scene_upload.rs"]
pub(crate) mod scene_upload;

fn write<T: bytemuck::NoUninit>(dir: &Path, name: &str, data: &[T]) {
    fs::write(dir.join(name), bytemuck::cast_slice(data)).unwrap();
}
pub fn fixture(
    dir: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    frames: u32,
) -> Value {
    fixture_with_snapshots(dir, label, columns, levels, nb, frames, &[])
}
pub fn fixture_with_snapshots(
    dir: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    frames: u32,
    snapshots: &[u32],
) -> Value {
    fixture_config(dir, label, columns, levels, nb, frames, snapshots, false)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixture_config(
    dir: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    frames: u32,
    snapshots: &[u32],
    use_lbvh: bool,
) -> Value {
    fixture_storage(
        dir, label, columns, levels, nb, frames, snapshots, use_lbvh, None, true,
    )
}

/// Export native typed initial storage for a capacity specialization. Smaller
/// variants must not execute the reference without the host retry controller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixture_storage(
    dir: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    frames: u32,
    snapshots: &[u32],
    use_lbvh: bool,
    pair_capacity_override: Option<usize>,
    execute: bool,
) -> Value {
    scene_storage(
        dir,
        label,
        columns,
        levels,
        nb,
        frames,
        snapshots,
        use_lbvh,
        pair_capacity_override,
        execute,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scene_storage(
    dir: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    frames: u32,
    snapshots: &[u32],
    use_lbvh: bool,
    pair_capacity_override: Option<usize>,
    execute: bool,
    upload: Option<&scene_upload::SceneUpload>,
) -> Value {
    match upload.map_or(FrictionModel::Coulomb, |u| u.params.friction_model) {
        _ if upload.is_none_or(|u| u.params.friction_model != FrictionModel::Simplified) => {
            scene_storage_model::<scene_model::Coulomb>(
                dir,
                label,
                columns,
                levels,
                nb,
                frames,
                snapshots,
                use_lbvh,
                pair_capacity_override,
                execute,
                upload,
            )
        }
        _ => scene_storage_model::<scene_model::Twist>(
            dir,
            label,
            columns,
            levels,
            nb,
            frames,
            snapshots,
            use_lbvh,
            pair_capacity_override,
            execute,
            upload,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn scene_storage_model<M: scene_model::Model>(
    dir: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    frames: u32,
    snapshots: &[u32],
    use_lbvh: bool,
    pair_capacity_override: Option<usize>,
    execute: bool,
    upload: Option<&scene_upload::SceneUpload>,
) -> Value {
    assert!(execute || snapshots.is_empty());
    assert!(pair_capacity_override.is_none() || !execute || upload.is_some());
    fs::create_dir_all(dir).unwrap();
    let guard = 0xdeadbeefu32;
    let per = if let Some(u) = upload {
        u.batch.colliders_len as usize
    } else if label == "empty" {
        0
    } else {
        1 + columns * levels
    };
    let body_per = upload.map_or(per, |u| u.batch.bodies_len as usize);
    let body_capacity = upload.map_or(per, |u| u.batch.colliders_batch_capacity as usize) + 3;
    let n = body_per * nb;
    let collider_count = per * nb;
    let storage = body_capacity * nb;
    let general_scene = upload.is_some() || label.starts_with("general_");
    let traced = true;
    let pair_capacity =
        pair_capacity_override.unwrap_or(per * per * nb * if general_scene { 2 } else { 1 });
    let capacity = pair_capacity * 2;
    let cs = capacity + 3;
    let groups = if upload.is_some() {
        (capacity as u32).div_ceil(64).max(1)
    } else {
        1u32
    };
    let bg = (n as u32).div_ceil(64).max(1);
    let color_stride = upload.map_or(8, |u| {
        u.batch
            .solver_color_buckets_stride
            .max(if u.coloring_rounds.is_some() { 3 } else { 4 }) as usize
    });
    let rounds = if let Some(upload) = upload {
        upload
            .coloring_rounds
            .map_or((color_stride - 3).max(2), |n| n as usize)
    } else {
        16
    };
    let contact_reduction = upload.is_none_or(|u| u.contact_reduction);
    let joint_scene =
        upload.is_some_and(|u| u.batch.impulse_joints_len != 0) || label.starts_with("joint_");
    let joint_per = upload.map_or(if joint_scene { columns * levels } else { 0 }, |u| {
        u.batch.impulse_joints_len as usize
    });
    let nj = joint_per * nb;
    let js = nj + 3;
    let batch = BatchIndices {
        impulse_joints_len: joint_per as u32,
        num_batches: nb as u32,
        bodies_len: body_per as u32,
        colliders_len: per as u32,
        colliders_batch_capacity: body_capacity as u32,
        collision_pairs_capacity: pair_capacity as u32,
        contacts_capacity: capacity as u32,
        solver_color_buckets_stride: color_stride as u32,
        deterministic: 1,
        contact_sort_collider_shift: 0,
        ..Default::default()
    };
    let mut lbvh = use_lbvh.then(|| lbvh::SceneLbvh::new(batch, storage));
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut local_poses = vec![Pose::IDENTITY; storage];
    let mut collider_poses = poses.clone();
    let mut solver_poses = poses.clone();
    let mut vels = vec![Velocity::default(); storage];
    let native_body_slots = upload.map_or(0, |u| {
        (u.batch.colliders_batch_capacity * u.batch.num_batches) as usize
    });
    let hub_pool = if upload.is_some() {
        native_body_slots.max(1024)
    } else {
        0
    };
    let mut solver_vels =
        vec![Velocity::default(); storage + hub_pool + if hub_pool > 0 { 3 } else { 0 }];
    let mut inc = vels.clone();
    let mut local = vec![LocalMassProperties::default(); storage];
    let mut world = vec![WorldMassProperties::default(); storage];
    let mut shapes = vec![Shape::ball(0.5); storage];
    let mut materials = vec![
        ColliderMaterial {
            friction: 0.5,
            restitution: if label == "impact" { 0.8 } else { 0. },
            ..Default::default()
        };
        storage
    ];
    let mut parents = (0..storage as u32).collect::<Vec<_>>();
    let mut filters = parents.iter().map(|i| [*i, 0]).collect::<Vec<_>>();
    let mut interaction = vec![InteractionGroups::all(); storage];
    let mut vertices = vec![PaddedVector::new(Vec3::ZERO)];
    let mut geometry_indices = vec![guard];
    if upload.is_none() {
        for b in 0..nb {
            poses[b].translation = Vec3::new(0., -0.5, b as f32 * 4.);
            shapes[b] = Shape::cuboid(Vec3::new(20., 0.5, 20.));
        }
        for i in 0..storage {
            local[i].inv_mass = Vec3::ZERO;
            local[i].inv_principal_inertia = Vec3::ZERO;
            if i >= nb && i < n {
                let body = i / nb;
                let level = (body - 1) % levels;
                let column = (body - 1) / levels;
                poses[i].translation = Vec3::new(
                    column as f32 * 2.,
                    0.495 + level as f32 * 0.995,
                    (i % nb) as f32 * 4.,
                );
                if label == "free" || (label == "capacity_growth" && column > 0) {
                    poses[i].translation.y += 3.;
                }
                local[i].inv_mass = Vec3::ONE;
                local[i].inv_principal_inertia = Vec3::splat(10.);
                if label == "boxes" {
                    shapes[i] = Shape::cuboid(Vec3::splat(0.5));
                    local[i].inv_principal_inertia = Vec3::splat(6.);
                }
                vels[i].linear = Vec3::new(
                    if label == "friction" { 2. } else { 0. },
                    if label == "impact" { -2. } else { 0. },
                    0.,
                );
            }
        }
        if general_scene {
            // Shared convex cube, then a two-triangle ground mesh or two thick segments.
            vertices.extend((0..8).map(|i| {
                PaddedVector::new(Vec3::new(
                    if i & 1 == 0 { -0.5 } else { 0.5 },
                    if i & 2 == 0 { -0.5 } else { 0.5 },
                    if i & 4 == 0 { -0.5 } else { 0.5 },
                ))
            }));
            geometry_indices.extend([
                0, 0, 0, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0,
                2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6,
            ]);
            let vb = vertices.len() as u32;
            let ib = geometry_indices.len() as u32;
            let lo = Vec3::new(-20., 0., -20.);
            let hi = Vec3::new(20., 0., 20.);
            for _ in 0..3 {
                vertices.push(PaddedVector::new(lo));
                vertices.push(PaddedVector::new(hi));
            }
            for p in [lo, Vec3::new(-20., 0., 20.), hi, Vec3::new(20., 0., -20.)] {
                vertices.push(PaddedVector::new(p));
            }
            geometry_indices.extend([1, 3, 0, u32::MAX, 2, 0, u32::MAX, 3, 1, 0, 1, 2, 0, 2, 3]);
            let pib = geometry_indices.len() as u32;
            geometry_indices.extend([1, 3, 0, u32::MAX, 2, 0, u32::MAX, 3, 1, 0, 3, 1, 2]);
            let pvb = vertices.len() as u32;
            for _ in 0..3 {
                vertices.push(PaddedVector::new(Vec3::new(-20., 0., -0.1)));
                vertices.push(PaddedVector::new(Vec3::new(20., 0., 0.1)));
            }
            for p in [
                Vec3::new(-20., 0., -0.1),
                Vec3::new(-20., 0., 0.1),
                Vec3::new(20., 0., 0.1),
                Vec3::new(20., 0., -0.1),
            ] {
                vertices.push(PaddedVector::new(p));
            }
            for b in 0..nb {
                if label == "general_mesh" {
                    shapes[b] = Shape::trimesh(vb, ib, 3, 2, 4, lo, hi);
                    poses[b].translation.y = 0.;
                } else if label == "general_polyline" {
                    shapes[b] = Shape::polyline(
                        pvb,
                        pib,
                        3,
                        Vec3::new(-20., 0., -0.1),
                        Vec3::new(20., 0., 0.1),
                    );
                    poses[b].translation.y = -0.4;
                }
            }
            for i in nb..n {
                let kind = match label {
                    "general_capsule" => 2,
                    "general_cone" => 3,
                    "general_cylinder" => 4,
                    "general_convex" | "general_mesh" => 5,
                    "general_triangle" => 6,
                    "general_polyline" => 0,
                    _ => (i / nb - 1) % 7,
                };
                shapes[i] = match kind {
                    0 => Shape::ball(0.5),
                    1 => Shape::cuboid(Vec3::splat(0.5)),
                    2 => Shape::capsule(Vec3::new(0., -0.3, 0.), Vec3::new(0., 0.3, 0.), 0.2),
                    3 => Shape::cone(0.5, 0.5),
                    4 => Shape::cylinder(0.5, 0.5),
                    5 => Shape::convex_poly(1, 9, 1, 40),
                    _ => Shape::from_triangle(&Triangle::new(
                        Vec3::new(-0.5, -0.5, -0.4),
                        Vec3::new(0.5, -0.5, -0.4),
                        Vec3::new(0., 0.5, 0.4),
                    )),
                };
                local[i].inv_principal_inertia = Vec3::splat(6.);
                if label == "general_triangle" {
                    poses[i].translation.y += 0.5;
                }
            }
        }
    }
    if let Some(u) = upload {
        macro_rules! copy {
            ($dst:ident, $src:ident) => {
                $dst[..u.$src.len()].copy_from_slice(&u.$src);
            };
        }
        copy!(poses, poses);
        copy!(vels, velocities);
        copy!(local, local);
        copy!(local_poses, local_poses);
        copy!(parents, parents);
        copy!(shapes, shapes);
        copy!(materials, materials);
        copy!(interaction, groups);
        copy!(filters, filters);
        vertices = u.vertices.clone();
        geometry_indices = u.geometry_indices.clone();
    }
    let mut joints = vec![ImpulseJoint::zeroed(); js];
    let mut builders = vec![JointConstraintBuilder::zeroed(); js];
    let mut joint_constraints = vec![JointConstraint::default(); js];
    for con in &mut joint_constraints {
        con.padding = [guard; 3];
        for el in &mut con.elements {
            el.joint_id = guard;
        }
    }
    let mut unsorted = Vec::new();
    let mut masks = vec![0u128; per];
    let mut assignments = Vec::new();
    for k in 0..if upload.is_none() { joint_per } else { 0 } {
        let body = k + 1;
        let chain = label == "joint_chain";
        let previous = if chain && k % levels != 0 {
            body - 1
        } else {
            0
        };
        let column = k / levels;
        let level = k % levels;
        let height = if chain {
            (levels - level) as f32
        } else if label == "joint_impact" {
            0.495
        } else if matches!(label, "joint_ball" | "joint_revolute") {
            1.
        } else {
            2.
        };
        for b in 0..nb {
            poses[body * nb + b].translation = Vec3::new(column as f32 * 2., height, b as f32 * 4.);
            vels[body * nb + b].linear =
                if matches!(label, "joint_ball" | "joint_revolute" | "joint_chain") {
                    Vec3::X
                } else if label == "joint_prismatic" {
                    Vec3::X * 1.5
                } else {
                    Vec3::ZERO
                };
        }
        let mut data = GenericJoint::zeroed();
        data.local_frame_a = Pose::IDENTITY;
        data.local_frame_b = Pose::IDENTITY;
        data.locked_axes = match label {
            "joint_ball" | "joint_chain" => 7,
            "joint_revolute" => 31,
            "joint_prismatic" | "joint_motor" => 62,
            _ => 63,
        };
        if previous == 0 {
            let lever = if chain || matches!(label, "joint_ball" | "joint_revolute") {
                1.
            } else {
                0.
            };
            data.local_frame_a.translation =
                Vec3::new(column as f32 * 2., height + lever + 0.5, 0.);
            data.local_frame_b.translation = Vec3::new(0., lever, 0.);
        } else {
            data.local_frame_a.translation = Vec3::new(0., -0.5, 0.);
            data.local_frame_b.translation = Vec3::new(0., 0.5, 0.);
        }
        if label == "joint_prismatic" {
            data.limit_axes = 1;
            data.limits[0] = JointLimits {
                min: -0.4,
                max: 0.4,
                impulse: 0.,
            };
        }
        if label == "joint_motor" {
            data.motor_axes = 1;
            data.motors[0] = JointMotor {
                target_pos: 0.25,
                target_vel: 0.,
                stiffness: 40.,
                damping: 8.,
                max_force: 10.,
                impulse: 0.,
                model: 0,
            };
        }
        let color = (masks[previous] | masks[body]).trailing_ones();
        assert!(color < 3);
        masks[previous] |= 1 << color;
        masks[body] |= 1 << color;
        assignments.push(color);
        unsorted.push(ImpulseJoint {
            body_a: previous as u32,
            body_b: body as u32,
            padding: [0; 2],
            data,
        });
    }
    // Same stable greedy endpoint coloring and color grouping as the Rust host.
    let mut order = (0..if upload.is_none() { joint_per } else { 0 }).collect::<Vec<_>>();
    order.sort_by_key(|k| assignments[*k]);
    for (slot, k) in order.iter().enumerate() {
        for b in 0..nb {
            joints[slot * nb + b] = unsorted[*k];
        }
    }
    let mut joint_colors = vec![
        assignments.iter().filter(|c| **c == 0).count() as u32,
        assignments.iter().filter(|c| **c <= 1).count() as u32,
        joint_per as u32,
        guard,
    ];
    if let Some(u) = upload {
        joints[..nj].copy_from_slice(&u.joints[..nj]);
        joint_colors = u.joint_colors.clone();
        joint_colors.push(guard);
    }
    let joint_color_count = upload.map_or(3, |u| u.joint_colors.len());
    let joint_groups = (nj as u32).div_ceil(64).max(1);
    let motor_params = vec![guard; js * 36];
    let trace_stride = n * (std::mem::size_of::<Pose>() + std::mem::size_of::<Velocity>()) / 4;
    let mut trajectory = vec![guard; frames as usize * trace_stride + 3];
    let mut frame_index = [0u32, guard];
    let mut p = RbdSimParams::tgs_soft();
    let mut substeps = 4u32;
    let mut pgs = 2u32;
    p.dt = 1. / 60. / substeps as f32;
    p.num_solver_iterations = substeps;
    p.num_internal_pgs_iterations = pgs;
    p.normalized_prediction_distance = 0.08;
    p.friction_in_bias_pass = 1;
    p.warmstart_coefficient = 1.;
    if let Some(u) = upload {
        p = u.params;
        substeps = p.num_solver_iterations;
        pgs = p.num_internal_pgs_iterations;
    }
    let mode = if p.friction_in_bias_pass != 0 {
        2u32
    } else {
        1u32
    };
    let gravity = upload.map_or(Vec4::new(0., -9.81, 0., 0.), |u| u.gravity);
    let mut aabbs = vec![Aabb::new(Vec3::ZERO, Vec3::ZERO); storage];
    let mut pairs = vec![
        CollisionPair {
            colliders: UVec2::splat(guard)
        };
        pair_capacity + 3
    ];
    let mut pair_count = if use_lbvh { [0; 3] } else { [0, guard, guard] };
    let mut pfm_count = [0, guard, guard];
    let mut plan = [ContactPlan::default(); 2];
    let mut pfm_sort_len = [guard; 3];
    let mut contacts_grid = [[guard; 3]; 2];
    let mut pfm_grid = [[guard; 3]; 2];
    let mut mb_grid = [[guard; 3]; 2];
    let mut raw = vec![IndexedManifold::default(); cs];
    let mut contacts = raw.clone();
    let mut counts = vec![0u32; storage];
    let mut ids = vec![guard; 2 * capacity + 3];
    let mut sorted_ids = ids.clone();
    let is_multibody = vec![0u32; storage];
    let body_group = (0..storage as u32).collect::<Vec<_>>();
    let mut first = vec![guard; storage];
    let mut hubs = HubCounts {
        base: storage as u32,
        pool: hub_pool as u32,
        ..Default::default()
    };
    let mut split = scene_hubs::Storage::new(hub_pool);
    let mut links = vec![ContactLink::default(); cs];
    let mut old_links = links.clone();
    let mut old_counts = vec![0u32; storage];
    let mut old_ids = ids.clone();
    let mut old_indices = vec![guard; cs];
    let mut old_colors = vec![0u32; cs];
    let mut states = vec![ContactRecycleState::default(); cs * 2];
    let recycle = ContactRecycleOffsets {
        old_base: 0,
        new_base: cs as u32,
        ..Default::default()
    };
    let mut colors = vec![guard; cs];
    let rands = colors.clone();
    let mut colored = colors.clone();
    let mut pending = colors.clone();
    let mut num_colors = [1, guard];
    let mut coloring_grid = [[guard; 3]; 3];
    let mut tiles = vec![M::Tile::default(); cs.div_ceil(32)];
    let mut old_tiles = tiles.clone();
    let mut sorted_links = links.clone();
    let mut indices = vec![guard; cs];
    let mut bucket_ends = vec![0u32; color_stride * nb];
    let mut stats = vec![
        ColorStats {
            sizes: [0; 64],
            highest_color: 0,
            max_bucket: 0
        };
        1
    ];
    let mut indirect = vec![[guard; 3]; color_stride];
    let (strides, levels_scan) = prefix_program::hierarchy(if hub_pool > 0 { storage } else { n });
    let mut aux: Vec<_> = strides[1..]
        .iter()
        .map(|s| vec![guard; *s as usize + 3])
        .collect();
    let (bucket_strides, bucket_levels) = prefix_program::hierarchy(color_stride * nb);
    let mut bucket_aux: Vec<_> = bucket_strides[1..]
        .iter()
        .map(|s| vec![guard; *s as usize + 3])
        .collect();
    let mut order: Vec<u32> = (0..cs)
        .map(|i| {
            if i < n {
                (n - 1 - i) as u32
            } else {
                0xdeadbeef
            }
        })
        .collect();
    let mut keys = vec![0xdeadbeefu32; cs];
    let mut values = keys.clone();
    let mut count = [0xabcdef12u32; 1];
    let mut sorting = radix::Arrays::new();
    for key in [
        "SortKeys",
        "PongKeys",
        "PongValues",
        "RadixAux",
        "RadixOutAux",
        "RadixBatchIds",
    ] {
        sorting.insert(key.into(), GpuBuffer::Cpu(vec![0xdeadbeef; cs]));
    }
    for (key, len) in [
        ("Counts", 16 * cs.div_ceil(1024) + 3),
        ("Reduced", 1024 + 3),
        ("SortGrid", 6),
        ("ReduceGrid", 6),
    ] {
        sorting.insert(key.into(), GpuBuffer::Cpu(vec![0xdeadbeef; len]));
    }
    for (key, shift) in [("Config".into(), 0), ("InitConfig".into(), 1)]
        .into_iter()
        .chain((0..8).map(|i| (format!("Config{i}"), i * 4)))
    {
        sorting.insert(key, GpuBuffer::Cpu(vec![shift, cs as u32, 0]));
    }

    let mut pfm_pairs = vec![NarrowPhasePfmPair::default(); pair_capacity + 3];
    bytemuck::cast_slice_mut::<_, u32>(&mut pfm_pairs).fill(guard);
    let pfm_identity: Vec<u32> = (0..cs as u32).collect();
    let mut pfm_order = vec![guard; cs];
    let mut pfm_keys = vec![guard; cs];
    let mut pfm_sorted_keys = vec![guard; cs];
    macro_rules! buffers {
        ($suffix:expr) => {
            buffers!(dir, $suffix)
        };
        ($root:expr,$suffix:expr) => {{
            if let Some(tree) = &lbvh {
                tree.write($root, $suffix);
            }
            if joint_scene || upload.is_some() {
                write($root, &format!("Joints{}.bin", $suffix), &joints);
                write($root, &format!("Builders{}.bin", $suffix), &builders);
                write(
                    $root,
                    &format!("JointConstraints{}.bin", $suffix),
                    &joint_constraints,
                );
                write($root, &format!("JointColors{}.bin", $suffix), &joint_colors);
                write($root, &format!("JointBias{}.bin", $suffix), &[1u32, guard]);
                write(
                    $root,
                    &format!("JointUnbiased{}.bin", $suffix),
                    &[0u32, guard],
                );
                write($root, &format!("MotorParams{}.bin", $suffix), &motor_params);
            }
            if traced {
                write($root, &format!("Trajectory{}.bin", $suffix), &trajectory);
                write($root, &format!("FrameIndex{}.bin", $suffix), &frame_index);
            }
            if general_scene {
                write(
                    $root,
                    &format!("GeometryIndices{}.bin", $suffix),
                    &geometry_indices,
                );
                write($root, &format!("PfmPairs{}.bin", $suffix), &pfm_pairs);
                write($root, &format!("PfmIdentity{}.bin", $suffix), &pfm_identity);
                write($root, &format!("PfmOrder{}.bin", $suffix), &pfm_order);
                write($root, &format!("PfmKeys{}.bin", $suffix), &pfm_keys);
                write(
                    $root,
                    &format!("PfmSortedKeys{}.bin", $suffix),
                    &pfm_sorted_keys,
                );
            }
            write($root, &format!("Poses{}.bin", $suffix), &poses);
            write($root, &format!("LocalPoses{}.bin", $suffix), &local_poses);
            write(
                $root,
                &format!("ColliderPoses{}.bin", $suffix),
                &collider_poses,
            );
            write($root, &format!("SolverPoses{}.bin", $suffix), &solver_poses);
            write($root, &format!("Velocities{}.bin", $suffix), &vels);
            write(
                $root,
                &format!("SolverVelocities{}.bin", $suffix),
                &solver_vels,
            );
            write($root, &format!("Increments{}.bin", $suffix), &inc);
            write($root, &format!("LocalMass{}.bin", $suffix), &local);
            write($root, &format!("WorldMass{}.bin", $suffix), &world);
            write($root, &format!("Shapes{}.bin", $suffix), &shapes);
            write($root, &format!("Materials{}.bin", $suffix), &materials);
            write($root, &format!("Parents{}.bin", $suffix), &parents);
            write($root, &format!("Filters{}.bin", $suffix), &filters);
            write($root, &format!("Groups{}.bin", $suffix), &interaction);
            write($root, &format!("Vertices{}.bin", $suffix), &vertices);
            write($root, &format!("Aabbs{}.bin", $suffix), &aabbs);
            write($root, &format!("Pairs{}.bin", $suffix), &pairs);
            write($root, &format!("PairCount{}.bin", $suffix), &pair_count);
            write($root, &format!("PfmCount{}.bin", $suffix), &pfm_count);
            write($root, &format!("Plan{}.bin", $suffix), &plan);
            write($root, &format!("PfmSortLen{}.bin", $suffix), &pfm_sort_len);
            write(
                $root,
                &format!("ContactsIndirect{}.bin", $suffix),
                &contacts_grid,
            );
            write($root, &format!("PfmIndirect{}.bin", $suffix), &pfm_grid);
            write($root, &format!("MbIndirect{}.bin", $suffix), &mb_grid);
            write($root, &format!("RawContacts{}.bin", $suffix), &raw);
            write($root, &format!("Contacts{}.bin", $suffix), &contacts);
            write($root, &format!("Counts{}.bin", $suffix), &counts);
            write($root, &format!("Ids{}.bin", $suffix), &ids);
            write($root, &format!("SortedIds{}.bin", $suffix), &sorted_ids);
            write($root, &format!("IsMultibody{}.bin", $suffix), &is_multibody);
            write($root, &format!("BodyGroup{}.bin", $suffix), &body_group);
            write($root, &format!("FirstSlot{}.bin", $suffix), &first);
            write($root, &format!("Links{}.bin", $suffix), &links);
            write($root, &format!("OldLinks{}.bin", $suffix), &old_links);
            write($root, &format!("OldCounts{}.bin", $suffix), &old_counts);
            write($root, &format!("OldIds{}.bin", $suffix), &old_ids);
            write($root, &format!("OldIndices{}.bin", $suffix), &old_indices);
            write($root, &format!("OldColors{}.bin", $suffix), &old_colors);
            write($root, &format!("States{}.bin", $suffix), &states);
            write($root, &format!("Colors{}.bin", $suffix), &colors);
            write($root, &format!("Rands{}.bin", $suffix), &rands);
            write($root, &format!("Colored{}.bin", $suffix), &colored);
            write($root, &format!("Pending{}.bin", $suffix), &pending);
            write($root, &format!("NumColors{}.bin", $suffix), &num_colors);
            write(
                $root,
                &format!("ColoringIndirect{}.bin", $suffix),
                &coloring_grid,
            );
            write($root, &format!("Tiles{}.bin", $suffix), &tiles);
            write($root, &format!("OldTiles{}.bin", $suffix), &old_tiles);
            write($root, &format!("SortedLinks{}.bin", $suffix), &sorted_links);
            write($root, &format!("Indices{}.bin", $suffix), &indices);
            write($root, &format!("Buckets{}.bin", $suffix), &bucket_ends);
            write($root, &format!("Stats{}.bin", $suffix), &stats);
            write($root, &format!("Indirect{}.bin", $suffix), &indirect);
            write($root, &format!("InOrder{}.bin", $suffix), &order);
            write($root, &format!("Keys{}.bin", $suffix), &keys);
            write($root, &format!("Values{}.bin", $suffix), &values);
            write($root, &format!("SortCount{}.bin", $suffix), &count);
            write($root, &format!("BatchIds{}.bin", $suffix), &[batch]);
            write($root, &format!("RbdParams{}.bin", $suffix), &[p]);
            write($root, &format!("Gravity{}.bin", $suffix), &[gravity]);
            write($root, &format!("HubCounts{}.bin", $suffix), &[hubs]);
            if hub_pool > 0 {
                split.write($root, $suffix);
            }
            for (k, v) in &sorting {
                write(
                    $root,
                    &format!(
                        "{}{}.bin",
                        if k == "Counts" { "RadixCounts" } else { k },
                        $suffix
                    ),
                    v.unwrap_slice(),
                );
            }
            for i in 0..4 {
                write($root, &format!("AdjAux{i}{}.bin", $suffix), &aux[i]);
                write(
                    $root,
                    &format!("BucketAux{i}{}.bin", $suffix),
                    &bucket_aux[i],
                );
            }
        }};
    }
    buffers!("");
    write(dir, "Params.bin", &[guard; 6]);
    write(dir, "Selector.bin", &[0u32, guard]);
    write(dir, "CurrentColor.bin", &[0u32, guard]);
    for i in 0..color_stride.max(joint_color_count) {
        write(dir, &format!("Color{i}.bin"), &[i as u32, guard]);
    }
    write(dir, "Mode.bin", &[mode, guard]);
    for i in 0..4 {
        write(dir, &format!("AdjStride{i}.bin"), &[strides[i], guard]);
        write(
            dir,
            &format!("BucketStride{i}.bin"),
            &[bucket_strides[i], guard],
        );
    }
    for frame in 0..if execute { frames } else { 0 } {
        let snapshot_dir = dir.join(format!("restart-{frame}"));
        if snapshots.contains(&frame) {
            fs::create_dir_all(&snapshot_dir).unwrap();
            buffers!(&snapshot_dir, "");
        }
        pair_count[0] = 0;
        for i in 0..bg * 64 {
            gpu_update_mprops(
                UVec3::new(i, 0, 0),
                UVec3::new(bg, 1, 1),
                &mut world,
                &local,
                &poses,
                &batch,
            );
        }
        for i in 0..bg * 64 {
            gpu_solver_cleanup(
                UVec3::new(i, 0, 0),
                UVec3::new(bg, 1, 1),
                &mut counts,
                &mut solver_vels,
                &vels,
                &world,
                &batch,
                &mut first,
                &mut hubs,
            );
            gpu_init_solver_bodies(
                UVec3::new(i, 0, 0),
                &poses,
                &local,
                &mut solver_poses,
                &batch,
            );
        }
        for i in 0..(collider_count as u32).div_ceil(64).max(1) * 64 {
            gpu_sync_collider_poses(
                UVec3::new(i, 0, 0),
                UVec3::new(bg, 1, 1),
                &poses,
                &local_poses,
                &mut collider_poses,
                &parents,
                &batch,
            );
        }
        gpu_reset_narrow_phase(UVec3::ZERO, &mut pfm_count[..1]);
        for i in 0..(collider_count as u32).div_ceil(64).max(1) * 64 {
            gpu_bf_compute_aabbs(
                UVec3::new(i, 0, 0),
                &collider_poses,
                &shapes,
                &mut aabbs,
                &batch,
                &vertices,
            );
        }
        if let Some(tree) = &mut lbvh {
            tree.run(
                &collider_poses,
                &shapes,
                &vertices,
                &interaction,
                &filters,
                &mut pairs,
                &mut pair_count,
            );
        } else {
            for i in 0..pair_capacity.div_ceil(64).max(1) as u32 * 64 {
                gpu_bf_find_pairs(
                    UVec3::new(i, 0, 0),
                    &aabbs,
                    &mut pairs,
                    &mut pair_count,
                    &interaction,
                    &batch,
                    &filters,
                    &p,
                );
            }
        }
        if general_scene {
            for i in 0..groups * 64 {
                gpu_narrow_phase_shape_shape_deferred(
                    UVec3::new(i, 0, 0),
                    UVec3::new(groups, 1, 1),
                    &pairs,
                    &mut pair_count,
                    &collider_poses,
                    &shapes,
                    &mut pfm_pairs,
                    &mut pfm_count,
                    &vertices,
                    &geometry_indices,
                    &batch,
                    &p,
                );
            }
            assert!(
                pair_count[0] <= pair_capacity as u32 && pfm_count[0] <= pair_capacity as u32,
                "Resize and rerun collision before integrating an overflowing scene"
            );
        }
        gpu_contact_plan(
            &mut pair_count,
            &mut pfm_count,
            &mut plan[0],
            &mut pfm_sort_len,
            &mut contacts_grid[0],
            &mut pfm_grid[0],
            &mut mb_grid[0],
            &batch,
        );
        for i in 0..groups * 64 {
            gpu_narrow_phase_shape_shape(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &pairs,
                &plan[0],
                &collider_poses,
                &shapes,
                &mut raw,
                &parents,
                &materials,
                &p,
            );
        }
        if general_scene {
            if contact_reduction {
                for pass_kind in 0..2 {
                    for i in 0..groups * 64 {
                        if pass_kind == 0 {
                            gpu_pfm_sub_shape_sort_keys(
                                UVec3::new(i, 0, 0),
                                UVec3::new(groups, 1, 1),
                                &pfm_pairs,
                                &plan[0],
                                &mut pfm_keys,
                            );
                        } else {
                            gpu_pfm_pair_sort_keys_permuted(
                                UVec3::new(i, 0, 0),
                                UVec3::new(groups, 1, 1),
                                &pfm_pairs,
                                &plan[0],
                                &pfm_order,
                                &mut pfm_keys,
                            );
                        }
                    }
                    sorting.insert("PfmKeys".into(), GpuBuffer::Cpu(pfm_keys.clone()));
                    sorting.insert("PfmIdentity".into(), GpuBuffer::Cpu(pfm_identity.clone()));
                    sorting.insert("PfmOrder".into(), GpuBuffer::Cpu(pfm_order.clone()));
                    sorting.insert(
                        "PfmSortedKeys".into(),
                        GpuBuffer::Cpu(pfm_sorted_keys.clone()),
                    );
                    sorting.insert("PfmSortLen".into(), GpuBuffer::Cpu(pfm_sort_len.to_vec()));
                    for pass in 0..8 {
                        let (ik, iv) = if pass == 0 {
                            (
                                "PfmKeys",
                                if pass_kind == 0 {
                                    "PfmIdentity"
                                } else {
                                    "PfmOrder"
                                },
                            )
                        } else if pass % 2 == 1 {
                            ("PongKeys", "PongValues")
                        } else {
                            ("PfmSortedKeys", "PfmOrder")
                        };
                        let (ok, ov) = if pass % 2 == 0 {
                            ("PongKeys", "PongValues")
                        } else {
                            ("PfmSortedKeys", "PfmOrder")
                        };
                        let mapping = [
                            ("Config", "Config"),
                            ("NumKeys", "PfmSortLen"),
                            ("Keys", ik),
                            ("Values", iv),
                            ("OutKeys", ok),
                            ("OutValues", ov),
                            ("Aux", "RadixAux"),
                            ("OutAux", "RadixOutAux"),
                            ("BatchIds", "RadixBatchIds"),
                            ("Counts", "Counts"),
                            ("Reduced", "Reduced"),
                            ("SortGrid", "SortGrid"),
                            ("ReduceGrid", "ReduceGrid"),
                        ];
                        let mut a = radix::Arrays::new();
                        for (local, physical) in mapping {
                            a.insert(
                                local.into(),
                                GpuBuffer::Cpu(sorting[physical].unwrap_slice().to_vec()),
                            );
                        }
                        for i in 0..8 {
                            let key = format!("Config{i}");
                            a.insert(
                                key.clone(),
                                GpuBuffer::Cpu(sorting[&key].unwrap_slice().to_vec()),
                            );
                        }
                        if pass == 0 {
                            radix::exec(&mut a, "radix_dispatch", cs as u32, 1, None);
                        }
                        for stage in radix::STAGES {
                            radix::exec(&mut a, stage, cs as u32, 1, Some(pass));
                        }
                        for (local, physical) in mapping {
                            sorting.insert(physical.into(), a.remove(local).unwrap());
                        }
                    }
                    pfm_order.copy_from_slice(sorting["PfmOrder"].unwrap_slice());
                    pfm_sorted_keys.copy_from_slice(sorting["PfmSortedKeys"].unwrap_slice());
                    for key in [
                        "PfmKeys",
                        "PfmOrder",
                        "PfmIdentity",
                        "PfmSortedKeys",
                        "PfmSortLen",
                    ] {
                        sorting.remove(key);
                    }
                }
            }
            for i in 0..groups * 64 {
                gpu_narrow_phase_pfm_pfm(
                    UVec3::new(i, 0, 0),
                    UVec3::new(groups, 1, 1),
                    &mut raw,
                    &pfm_pairs,
                    if contact_reduction {
                        &pfm_order
                    } else {
                        &pfm_identity
                    },
                    &plan[0],
                    &vertices,
                    &geometry_indices,
                    &parents,
                    &materials,
                    &p,
                );
            }
            if contact_reduction {
                for i in 0..groups * 64 {
                    gpu_reduce_contacts(
                        UVec3::new(i, 0, 0),
                        UVec3::new(groups, 1, 1),
                        &mut raw,
                        &pfm_sorted_keys,
                        &plan[0],
                        &p,
                    );
                }
            }
        }
        {
            let src = &raw;
            let dst = &mut contacts;
            let op = "canonical_contacts";
            let passes = vec![4, 1, 2];
            let g = groups;
            let storage = cs;
            if op != "permute_contacts" {
                for sel in &passes {
                    for gid in 0..g * 64 {
                        gpu_contact_sort_keys(
                            UVec3::new(gid, 0, 0),
                            UVec3::new(g, 1, 1),
                            src,
                            &plan[0],
                            &order,
                            &mut keys,
                            &mut values,
                            sel,
                            &mut count,
                            &batch,
                        );
                    }
                    if op == "canonical_contacts" {
                        sorting.insert("Keys".into(), GpuBuffer::Cpu(keys.clone()));
                        sorting.insert("Values".into(), GpuBuffer::Cpu(values.clone()));
                        sorting.insert("InOrder".into(), GpuBuffer::Cpu(order.clone()));
                        sorting.insert("SortCount".into(), GpuBuffer::Cpu(count.to_vec()));
                        for pass in 0..8 {
                            let (input_k, input_v) = if pass == 0 {
                                ("Keys", "Values")
                            } else if pass % 2 == 1 {
                                ("PongKeys", "PongValues")
                            } else {
                                ("SortKeys", "InOrder")
                            };
                            let (output_k, output_v) = if pass % 2 == 0 {
                                ("PongKeys", "PongValues")
                            } else {
                                ("SortKeys", "InOrder")
                            };
                            let mapping = [
                                ("Config", "Config"),
                                ("NumKeys", "SortCount"),
                                ("Keys", input_k),
                                ("Values", input_v),
                                ("OutKeys", output_k),
                                ("OutValues", output_v),
                                ("Aux", "RadixAux"),
                                ("OutAux", "RadixOutAux"),
                                ("BatchIds", "RadixBatchIds"),
                                ("Counts", "Counts"),
                                ("Reduced", "Reduced"),
                                ("SortGrid", "SortGrid"),
                                ("ReduceGrid", "ReduceGrid"),
                            ];
                            let mut local = radix::Arrays::new();
                            for (name, physical) in mapping {
                                local.insert(
                                    name.into(),
                                    GpuBuffer::Cpu(sorting[physical].unwrap_slice().to_vec()),
                                );
                            }
                            for i in 0..8 {
                                let k = format!("Config{i}");
                                local.insert(
                                    k.clone(),
                                    GpuBuffer::Cpu(sorting[&k].unwrap_slice().to_vec()),
                                );
                            }
                            if pass == 0 {
                                radix::exec(&mut local, "radix_dispatch", storage as u32, 1, None);
                            }
                            for stage in radix::STAGES {
                                radix::exec(&mut local, stage, storage as u32, 1, Some(pass));
                            }
                            for (name, physical) in mapping {
                                sorting.insert(physical.into(), local.remove(name).unwrap());
                            }
                        }
                        order.copy_from_slice(sorting["InOrder"].unwrap_slice());
                        for name in ["Keys", "Values", "InOrder", "SortCount"] {
                            sorting.remove(name);
                        }
                    }
                }
            }
            if op != "contact_keys" {
                for gid in 0..g * 64 {
                    gpu_permute_contacts(
                        UVec3::new(gid, 0, 0),
                        UVec3::new(g, 1, 1),
                        src,
                        dst,
                        &order,
                        &plan[0],
                    );
                }
            }
        }
        for i in 0..capacity as u32 {
            gpu_init_contact_links(
                UVec3::new(i, 0, 0),
                &contacts,
                &mut links,
                &is_multibody,
                &plan[0],
            );
        }
        for i in 0..groups * 64 {
            gpu_solver_count_constraints(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &contacts,
                &mut counts,
                &is_multibody,
                &world,
                &plan[0],
            );
        }
        prefix_program::scan(&mut counts, &mut aux, &strides, levels_scan);
        for i in 0..groups * 64 {
            gpu_solver_sort_constraints(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &mut counts,
                &world,
                &contacts,
                &plan[0],
                &mut ids,
                &is_multibody,
            );
        }
        if hub_pool > 0 {
            split.assign(
                &counts,
                &ids,
                &world,
                &body_group,
                &mut first,
                &mut hubs,
                &mut links,
                &batch,
            );
        }
        for body in 0..n as u32 {
            for lane in 0..64 {
                gpu_stabilize_body_constraint_ids(
                    UVec3::new(body, 0, 0),
                    UVec3::new(n as u32, 1, 1),
                    lane,
                    &counts,
                    &ids,
                    &mut sorted_ids,
                    &batch,
                );
            }
        }
        for i in 0..capacity as u32 {
            gpu_match_contact_links(
                UVec3::new(i, 0, 0),
                &contacts,
                &mut links,
                &old_counts,
                &old_ids,
                &old_links,
                &old_indices,
                &mut states,
                &collider_poses,
                &plan[0],
                &p,
                &recycle,
            );
        }
        for i in 0..capacity as u32 {
            gpu_reset_topo_gc(
                UVec3::new(i, 0, 0),
                &mut colors,
                &mut colored,
                &links,
                &plan[0],
                &mut num_colors[0],
                &mut pending,
            );
            gpu_seed_colors_from_warmstart(
                UVec3::new(i, 0, 0),
                &links,
                &old_colors,
                &mut colors,
                &mut colored,
                &plan[0],
            );
        }
        coloring_program::bounded(
            &counts,
            &sorted_ids,
            &links,
            &mut colors,
            &mut colored,
            &mut pending,
            &mut num_colors[0],
            &plan[0],
            &body_group,
            &batch,
            &contacts_grid,
            &mut coloring_grid,
            groups,
            rounds as u32,
            true,
        );
        let mut bucket_state = bucket_program::State {
            colors: GpuBuffer::Cpu(colors.clone()),
            links: GpuBuffer::Cpu(links.clone()),
            plan: GpuBuffer::Cpu(vec![plan[0]]),
            buckets: GpuBuffer::Cpu(bucket_ends.clone()),
            batch: GpuBuffer::Cpu(vec![batch]),
            sorted: GpuBuffer::Cpu(sorted_links.clone()),
            indices: GpuBuffer::Cpu(indices.clone()),
            stats: GpuBuffer::Cpu(stats.clone()),
            indirect: GpuBuffer::Cpu(indirect.clone()),
            aux: bucket_aux.clone(),
            strides: bucket_strides,
            levels: bucket_levels,
            groups,
        };
        bucket_state.reset();
        bucket_state.count();
        bucket_state.scan();
        bucket_state.scatter();
        bucket_state.dispatch();
        bucket_ends.copy_from_slice(bucket_state.buckets.unwrap_slice());
        sorted_links.copy_from_slice(bucket_state.sorted.unwrap_slice());
        indices.copy_from_slice(bucket_state.indices.unwrap_slice());
        stats.copy_from_slice(bucket_state.stats.unwrap_slice());
        indirect.copy_from_slice(bucket_state.indirect.unwrap_slice());
        bucket_aux = bucket_state.aux;
        for i in 0..capacity as u32 {
            M::prepare(
                UVec3::new(i, 0, 0),
                &sorted_links,
                &contacts,
                &world,
                &solver_poses,
                &vels,
                &states,
                &old_tiles,
                &mut tiles,
                &plan[0],
                &recycle,
            );
        }
        if joint_scene {
            for i in 0..joint_groups * 64 {
                gpu_init_joint_constraints(
                    UVec3::new(i, 0, 0),
                    UVec3::new(joint_groups, 1, 1),
                    &joints,
                    &mut builders,
                    &mut joint_constraints,
                    &local,
                    &batch,
                );
            }
        }
        for i in 0..bg * 64 {
            gpu_init_solver_vels_inc(UVec3::new(i, 0, 0), &mut inc, &world, &p, &batch, &gravity);
        }
        for _ in 0..substeps {
            for i in 0..bg * 64 {
                gpu_apply_solver_vels_inc(UVec3::new(i, 0, 0), &mut solver_vels, &inc, &batch);
            }
            if joint_scene {
                for i in 0..joint_groups * 64 {
                    gpu_update_joint_constraints(
                        UVec3::new(i, 0, 0),
                        UVec3::new(joint_groups, 1, 1),
                        &builders,
                        &mut joint_constraints,
                        &solver_poses,
                        &world,
                        &p,
                        &batch,
                    );
                }
            }
            if p.warmstart_coefficient != 1. {
                for i in 0..groups * 64 {
                    M::scale(UVec3::new(i, 0, 0), &bucket_ends, &mut tiles, &batch, &p);
                }
            }
            if hub_pool > 0 {
                split.scatter(&mut solver_vels, &hubs);
            }
            for color in 1..color_stride as u32 - 1 {
                for i in 0..groups * 64 {
                    M::warmstart(
                        UVec3::new(i, 0, 0),
                        UVec3::new(groups, 1, 1),
                        &tiles,
                        &mut solver_vels,
                        &bucket_ends,
                        &color,
                        &batch,
                    );
                }
            }
            if hub_pool > 0 {
                split.average(&mut solver_vels, &first, &counts, &hubs);
            }
            for _ in 0..pgs.max(1) {
                if joint_scene {
                    for color in 0..joint_color_count as u32 {
                        for i in 0..joint_groups * 64 {
                            gpu_solve_joint_constraints(
                                UVec3::new(i, 0, 0),
                                UVec3::new(joint_groups, 1, 1),
                                &mut joint_constraints,
                                &mut solver_vels,
                                &joint_colors,
                                &color,
                                &batch,
                                &1u32,
                            );
                        }
                    }
                }

                if hub_pool > 0 {
                    split.scatter(&mut solver_vels, &hubs);
                }
                for color in 1..color_stride as u32 - 1 {
                    for i in 0..groups * 64 {
                        M::solve(
                            UVec3::new(i, 0, 0),
                            UVec3::new(groups, 1, 1),
                            &mut tiles,
                            &mut solver_vels,
                            &bucket_ends,
                            &solver_poses,
                            &color,
                            &batch,
                            &mode,
                            &p,
                        );
                    }
                }
                if hub_pool > 0 {
                    split.average(&mut solver_vels, &first, &counts, &hubs);
                }
            }
            for i in 0..bg * 64 {
                gpu_integrate_linearized(
                    UVec3::new(i, 0, 0),
                    &mut solver_poses,
                    &mut solver_vels,
                    &p,
                    &batch,
                );
            }
            if joint_scene {
                for color in 0..joint_color_count as u32 {
                    for i in 0..joint_groups * 64 {
                        gpu_solve_joint_constraints(
                            UVec3::new(i, 0, 0),
                            UVec3::new(joint_groups, 1, 1),
                            &mut joint_constraints,
                            &mut solver_vels,
                            &joint_colors,
                            &color,
                            &batch,
                            &0u32,
                        );
                    }
                }
            }
            if hub_pool > 0 {
                split.scatter(&mut solver_vels, &hubs);
            }
            for color in 1..color_stride as u32 - 1 {
                for i in 0..groups * 64 {
                    M::solve(
                        UVec3::new(i, 0, 0),
                        UVec3::new(groups, 1, 1),
                        &mut tiles,
                        &mut solver_vels,
                        &bucket_ends,
                        &solver_poses,
                        &color,
                        &batch,
                        &0,
                        &p,
                    );
                }
            }
            if hub_pool > 0 {
                split.average(&mut solver_vels, &first, &counts, &hubs);
            }
        }
        for i in 0..bg * 64 {
            gpu_solver_finalize(
                UVec3::new(i, 0, 0),
                &mut vels,
                &solver_vels,
                &mut poses,
                &solver_poses,
                &local,
                &batch,
            );
        }
        if traced {
            let off = frame as usize * trace_stride;
            let pose_words: &[u32] = bytemuck::cast_slice(&poses[..n]);
            let vel_words: &[u32] = bytemuck::cast_slice(&vels[..n]);
            trajectory[off..off + pose_words.len()].copy_from_slice(pose_words);
            trajectory[off + pose_words.len()..off + trace_stride].copy_from_slice(vel_words);
            frame_index[0] += 1;
        }
        old_links.copy_from_slice(&links);
        old_counts.copy_from_slice(&counts);
        old_ids.copy_from_slice(&sorted_ids);
        old_indices.copy_from_slice(&indices);
        old_colors.copy_from_slice(&colors);
        old_tiles.copy_from_slice(&tiles);
        states.copy_within(cs..2 * cs, 0);
        if snapshots.contains(&frame) {
            buffers!(&snapshot_dir, ".expected");
        }
        if (frame + 1) % 30 == 0 {
            println!(
                "{label}: frame={} contacts={} positions={:?}",
                frame + 1,
                plan[0].bound,
                poses
                    .iter()
                    .take(n)
                    .map(|p| p.translation.y)
                    .collect::<Vec<_>>()
            );
        }
    }
    if execute {
        buffers!(".expected");
        write(
            dir,
            "Params.expected.bin",
            &[
                p.dt,
                p.max_linear_velocity(),
                p.max_angular_velocity(),
                gravity.x,
                gravity.y,
                gravity.z,
            ],
        );
    }
    let mut inputs = vec![
        "Poses".to_owned(),
        "LocalPoses".to_owned(),
        "ColliderPoses".to_owned(),
        "SolverPoses".to_owned(),
        "Velocities".to_owned(),
        "SolverVelocities".to_owned(),
        "Increments".to_owned(),
        "LocalMass".to_owned(),
        "WorldMass".to_owned(),
        "Shapes".to_owned(),
        "Materials".to_owned(),
        "Parents".to_owned(),
        "Filters".to_owned(),
        "Groups".to_owned(),
        "Vertices".to_owned(),
        "Aabbs".to_owned(),
        "Pairs".to_owned(),
        "PairCount".to_owned(),
        "PfmCount".to_owned(),
        "Plan".to_owned(),
        "PfmSortLen".to_owned(),
        "ContactsIndirect".to_owned(),
        "PfmIndirect".to_owned(),
        "MbIndirect".to_owned(),
        "RawContacts".to_owned(),
        "Contacts".to_owned(),
        "Counts".to_owned(),
        "Ids".to_owned(),
        "SortedIds".to_owned(),
        "IsMultibody".to_owned(),
        "BodyGroup".to_owned(),
        "FirstSlot".to_owned(),
        "Links".to_owned(),
        "OldLinks".to_owned(),
        "OldCounts".to_owned(),
        "OldIds".to_owned(),
        "OldIndices".to_owned(),
        "OldColors".to_owned(),
        "States".to_owned(),
        "Colors".to_owned(),
        "Rands".to_owned(),
        "Colored".to_owned(),
        "Pending".to_owned(),
        "NumColors".to_owned(),
        "ColoringIndirect".to_owned(),
        "Tiles".to_owned(),
        "OldTiles".to_owned(),
        "SortedLinks".to_owned(),
        "Indices".to_owned(),
        "Buckets".to_owned(),
        "Stats".to_owned(),
        "Indirect".to_owned(),
        "InOrder".to_owned(),
        "Keys".to_owned(),
        "Values".to_owned(),
        "SortCount".to_owned(),
        "BatchIds".to_owned(),
        "RbdParams".to_owned(),
        "Gravity".to_owned(),
        "HubCounts".to_owned(),
        "Params".to_owned(),
        "Selector".to_owned(),
        "CurrentColor".to_owned(),
        "Mode".to_owned(),
    ];
    inputs.extend(sorting.keys().map(|k| {
        if k == "Counts" {
            "RadixCounts".to_owned()
        } else {
            k.clone()
        }
    }));
    for i in 0..4 {
        inputs.extend([
            format!("AdjAux{i}"),
            format!("BucketAux{i}"),
            format!("AdjStride{i}"),
            format!("BucketStride{i}"),
        ]);
    }
    for i in 0..color_stride.max(joint_color_count) {
        inputs.push(format!("Color{i}"));
    }
    if joint_scene || upload.is_some() {
        inputs.extend(
            [
                "Joints",
                "Builders",
                "JointConstraints",
                "JointColors",
                "MotorParams",
                "JointBias",
                "JointUnbiased",
            ]
            .map(str::to_owned),
        );
    }
    if traced {
        inputs.extend(["Trajectory", "FrameIndex"].map(str::to_owned));
    }
    if general_scene {
        inputs.extend(
            [
                "GeometryIndices",
                "PfmPairs",
                "PfmIdentity",
                "PfmOrder",
                "PfmKeys",
                "PfmSortedKeys",
            ]
            .map(str::to_owned),
        );
    }
    if hub_pool > 0 {
        inputs.extend(scene_hubs::BUFFERS.map(str::to_owned));
    }
    let sizes = inputs
        .into_iter()
        .map(|name| {
            let size = fs::metadata(dir.join(format!("{name}.bin"))).unwrap().len() / 4;
            (name, json!(size))
        })
        .collect::<serde_json::Map<_, _>>();
    let mut case = json!({"name":dir.file_name().unwrap().to_str().unwrap(),"kind":if joint_scene {"joint_scene"}else if general_scene {"general_scene"}else{"analytic_scene"},"joint_count":nj,"joints_per_batch":joint_per,"joint_storage":js,"joint_color_count":joint_color_count,"trace_stride":trace_stride,"count":capacity,"scene":label,"columns":columns,"levels_scene":levels,"batches":nb,"colliders_per_batch":per,"body_count":n,"bodies_per_batch":body_per,"body_capacity_per_batch":body_capacity,"collider_count":collider_count,"body_storage":storage,"contact_capacity":capacity,"contact_storage":cs,"pair_capacity":pair_capacity,"vertex_count":vertices.len(),"geometry_indices":geometry_indices.len(),"pair_storage":pair_capacity+3,"groups":groups,"substeps":substeps,"pgs":pgs,"color_stride":color_stride,"rounds":rounds,"contact_reduction":contact_reduction,"frames":frames,"iterations":frames,"old_base":0,"new_base":cs,"strides":&strides[..4],"levels":levels_scan,"bucket_strides":&bucket_strides[..4],"bucket_levels":bucket_levels,"buffer_words":sizes});
    if upload.is_some() {
        case["native_body_slots"] = json!(native_body_slots);
        case["hub_pool"] = json!(hub_pool);
        case["solver_storage"] = json!(solver_vels.len());
        case["kind"] = json!("runtime_scene");
        case["general_collision"] = json!(true);
        case["scale_warmstart"] = json!(p.warmstart_coefficient != 1.);
    }
    if !execute {
        case["initial_upload_only"] = json!(true);
    }
    if use_lbvh {
        case["broadphase"] = json!("lbvh");
        for entry in fs::read_dir(dir).unwrap().map(Result::unwrap) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("Lbvh")
                && name.ends_with(".bin")
                && !name.ends_with(".expected.bin")
            {
                case["buffer_words"][name.trim_end_matches(".bin")] =
                    json!(entry.metadata().unwrap().len() / 4);
            }
        }
    }
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&case).unwrap(),
    )
    .unwrap();
    println!(
        "{label}: contacts={}, live={}, colors={:?}",
        plan[0].bound,
        bucket_ends[color_stride * nb - 1],
        &colors[..plan[0].bound as usize]
    );
    case
}
