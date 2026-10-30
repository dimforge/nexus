//! Portable native inputs for offline frame specialization. No physics executes here.
use nexus_rbd3d::pipeline::RbdSceneUpload;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

/// Validate and write a cold-start upload to a new directory.
///
/// These native words require a matching Rust-derived layout at specialization.
/// They are not a continuation checkpoint or a compiled execution bundle.
pub fn write_inputs(upload: &RbdSceneUpload, directory: &Path) -> Result<Value, String> {
    const {
        assert!(cfg!(target_endian = "little"));
    }
    let ids = upload.batch_indices;
    let batches = ids.num_batches as usize;
    let storage = batches
        .checked_mul(ids.colliders_batch_capacity as usize)
        .ok_or("Body storage overflow")?;
    if batches == 0
        || ids.bodies_len > ids.colliders_batch_capacity
        || ids.colliders_len > ids.colliders_batch_capacity
        || ids.multibodies_len != 0
    {
        return Err("Invalid ordinary-body upload counts".into());
    }
    for length in [
        upload.body_poses.len(),
        upload.velocities.len(),
        upload.local_mass.len(),
        upload.collider_local_poses.len(),
        upload.collider_parents.len(),
        upload.shapes.len(),
        upload.materials.len(),
        upload.groups.len(),
        upload.pair_filters.len(),
    ] {
        if length != storage {
            return Err("Native body/collider upload length differs from its capacity".into());
        }
    }
    let active = ids.bodies_len as usize * batches;
    for (slot, parent) in upload
        .collider_parents
        .iter()
        .take(ids.colliders_len as usize * batches)
        .enumerate()
    {
        if *parent as usize >= active || *parent as usize % batches != slot % batches {
            return Err("Collider parent is outside its active batch".into());
        }
    }
    let joint_storage = (upload.joints_per_batch as usize)
        .checked_mul(batches)
        .ok_or("Joint storage overflow")?;
    if upload.joints.len() != joint_storage
        || ids.impulse_joints_len > upload.joints_per_batch
        || upload.joint_color_ends.last().copied().unwrap_or(0) != ids.impulse_joints_len
        || upload
            .joint_color_ends
            .windows(2)
            .any(|ends| ends[0] >= ends[1])
    {
        return Err("Invalid native joint coloring/storage".into());
    }
    for joint in upload
        .joints
        .iter()
        .take(ids.impulse_joints_len as usize * batches)
    {
        if joint.body_a >= ids.bodies_len
            || joint.body_b >= ids.bodies_len
            || joint.body_a == joint.body_b
        {
            return Err("Joint endpoint is outside the ordinary body set".into());
        }
    }
    let mut start = 0;
    for &end in &upload.joint_color_ends {
        if end <= start {
            return Err("Empty native joint color".into());
        }
        let mut used = BTreeSet::new();
        for i in start as usize * batches..end as usize * batches {
            for local in [upload.joints[i].body_a, upload.joints[i].body_b] {
                if !used.insert(local as usize * batches + i % batches) {
                    return Err("Native joint color aliases a velocity writer".into());
                }
            }
        }
        start = end;
    }
    if !upload.gravity.is_finite()
        || !upload.params.dt.is_finite()
        || upload.params.dt <= 0.0
        || upload.params.num_solver_iterations == 0
    {
        return Err("Invalid timestep or gravity".into());
    }
    let mut buffers = BTreeMap::<&str, &[u8]>::new();
    macro_rules! input {
        ($name:literal, $data:expr) => {
            buffers.insert($name, bytemuck::cast_slice($data));
        };
    }
    input!("BatchIds", std::slice::from_ref(&upload.batch_indices));
    input!("RbdParams", std::slice::from_ref(&upload.params));
    input!("Gravity", std::slice::from_ref(&upload.gravity));
    input!("Poses", &upload.body_poses);
    input!("Velocities", &upload.velocities);
    input!("LocalMass", &upload.local_mass);
    input!("LocalPoses", &upload.collider_local_poses);
    input!("Parents", &upload.collider_parents);
    input!("Shapes", &upload.shapes);
    input!("Materials", &upload.materials);
    input!("Groups", &upload.groups);
    input!("Filters", &upload.pair_filters);
    input!("Vertices", &upload.vertices);
    input!("GeometryIndices", &upload.indices);
    input!("Joints", &upload.joints);
    input!("JointColors", &upload.joint_color_ends);
    let metadata = json!({
        "schema": 1, "kind": "native_scene_upload", "endianness": "little", "word_bytes": 4,
        "batches": batches, "bodies_per_batch": ids.bodies_len, "colliders_per_batch": ids.colliders_len,
        "capacity_per_batch": ids.colliders_batch_capacity,
        "joints_per_batch": upload.joints_per_batch, "active_joints_per_batch": ids.impulse_joints_len,
        "joint_color_count": upload.joint_color_ends.len(),
        "substeps": upload.params.num_solver_iterations, "pgs": upload.params.num_internal_pgs_iterations,
        "buffer_bytes": buffers.iter().map(|(n, b)| ((*n).to_owned(), json!(b.len()))).collect::<serde_json::Map<_, _>>()
    });
    fs::create_dir(directory).map_err(|e| e.to_string())?;
    for (name, bytes) in buffers {
        fs::write(directory.join(format!("{name}.bin")), bytes).map_err(|e| e.to_string())?;
    }
    fs::write(
        directory.join("scene.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(metadata)
}
