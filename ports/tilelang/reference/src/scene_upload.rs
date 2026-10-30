//! Checked reader for native RbdState cold-start uploads, without a GPU dependency.
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    dynamics::{ImpulseJoint, LocalMassProperties, RbdSimParams, Velocity},
    glamx::Vec4,
    queries::ColliderMaterial,
    rapier::geometry::{Group, InteractionGroups, InteractionTestMode},
    shapes::Shape,
    utils::BatchIndices,
};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

pub(crate) struct SceneUpload {
    pub contact_reduction: bool,
    pub batch: BatchIndices,
    pub params: RbdSimParams,
    pub gravity: Vec4,
    pub poses: Vec<Pose>,
    pub velocities: Vec<Velocity>,
    pub local: Vec<LocalMassProperties>,
    pub local_poses: Vec<Pose>,
    pub parents: Vec<u32>,
    pub shapes: Vec<Shape>,
    pub materials: Vec<ColliderMaterial>,
    pub groups: Vec<InteractionGroups>,
    pub filters: Vec<[u32; 2]>,
    pub vertices: Vec<PaddedVector>,
    pub geometry_indices: Vec<u32>,
    pub joints: Vec<ImpulseJoint>,
    pub joint_colors: Vec<u32>,
}
impl SceneUpload {
    pub fn read(directory: &Path) -> Result<Self, String> {
        const {
            assert!(cfg!(target_endian = "little"));
            assert!(size_of::<InteractionGroups>() == 12);
            assert!(std::mem::offset_of!(InteractionGroups, memberships) == 0);
            assert!(std::mem::offset_of!(InteractionGroups, filter) == 4);
            assert!(std::mem::offset_of!(InteractionGroups, test_mode) == 8);
        }
        let meta: Value = serde_json::from_slice(
            &fs::read(directory.join("scene.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if meta["schema"] != 1
            || meta["kind"] != "native_scene_upload"
            || meta["endianness"] != "little"
            || meta["word_bytes"] != 4
        {
            return Err("Unsupported native scene upload schema".into());
        }
        let names = [
            "BatchIds",
            "RbdParams",
            "Gravity",
            "Poses",
            "Velocities",
            "LocalMass",
            "LocalPoses",
            "Parents",
            "Shapes",
            "Materials",
            "Groups",
            "Filters",
            "Vertices",
            "GeometryIndices",
            "Joints",
            "JointColors",
        ];
        let sizes = meta["buffer_bytes"]
            .as_object()
            .ok_or("Missing native buffer sizes")?;
        if sizes.keys().map(String::as_str).collect::<BTreeSet<_>>() != names.into_iter().collect()
        {
            return Err("Native scene buffer inventory differs".into());
        }
        // Generic size_of::<T>() cannot be an as_chunks const argument on stable Rust.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        fn read<T: bytemuck::Pod>(dir: &Path, name: &str, meta: &Value) -> Result<Vec<T>, String> {
            let data = fs::read(dir.join(format!("{name}.bin"))).map_err(|e| e.to_string())?;
            if meta["buffer_bytes"][name].as_u64() != Some(data.len() as u64)
                || data.len() % size_of::<T>() != 0
            {
                return Err(format!("Native upload byte size differs: {name}"));
            }
            Ok(data
                .chunks_exact(size_of::<T>())
                .map(bytemuck::pod_read_unaligned)
                .collect())
        }
        fn scalar<T: bytemuck::Pod>(dir: &Path, name: &str, meta: &Value) -> Result<T, String> {
            let data = read::<T>(dir, name, meta)?;
            if data.len() != 1 {
                return Err(format!("Expected one native uniform: {name}"));
            }
            Ok(data[0])
        }
        let words = read::<[u32; 3]>(directory, "Groups", &meta)?;
        let mut groups = Vec::with_capacity(words.len());
        for [memberships, filter, mode] in words {
            let mode = match mode {
                0 => InteractionTestMode::And,
                1 => InteractionTestMode::Or,
                _ => return Err("Invalid native interaction mode".into()),
            };
            groups.push(InteractionGroups::new(
                Group::from_bits_retain(memberships),
                Group::from_bits_retain(filter),
                mode,
            ));
        }
        let result = Self {
            contact_reduction: false,
            batch: scalar(directory, "BatchIds", &meta)?,
            params: scalar(directory, "RbdParams", &meta)?,
            gravity: scalar(directory, "Gravity", &meta)?,
            poses: read(directory, "Poses", &meta)?,
            velocities: read(directory, "Velocities", &meta)?,
            local: read(directory, "LocalMass", &meta)?,
            local_poses: read(directory, "LocalPoses", &meta)?,
            parents: read(directory, "Parents", &meta)?,
            shapes: read(directory, "Shapes", &meta)?,
            materials: read(directory, "Materials", &meta)?,
            groups,
            filters: read(directory, "Filters", &meta)?,
            vertices: read(directory, "Vertices", &meta)?,
            geometry_indices: read(directory, "GeometryIndices", &meta)?,
            joints: read(directory, "Joints", &meta)?,
            joint_colors: read(directory, "JointColors", &meta)?,
        };
        let b = result.batch;
        let nb = b.num_batches as usize;
        for (name, count) in [
            ("batches", b.num_batches),
            ("bodies_per_batch", b.bodies_len),
            ("colliders_per_batch", b.colliders_len),
            ("capacity_per_batch", b.colliders_batch_capacity),
            ("active_joints_per_batch", b.impulse_joints_len),
            ("joint_color_count", result.joint_colors.len() as u32),
            ("substeps", result.params.num_solver_iterations),
            ("pgs", result.params.num_internal_pgs_iterations),
        ] {
            if meta[name].as_u64() != Some(count as u64) {
                return Err(format!("Native metadata differs from typed input: {name}"));
            }
        }
        let storage = nb
            .checked_mul(b.colliders_batch_capacity as usize)
            .ok_or("Native storage overflow")?;
        if nb == 0
            || b.bodies_len > b.colliders_batch_capacity
            || b.colliders_len > b.colliders_batch_capacity
            || b.multibodies_len != 0
        {
            return Err("Invalid ordinary scene counts".into());
        }
        for len in [
            result.poses.len(),
            result.velocities.len(),
            result.local.len(),
            result.local_poses.len(),
            result.parents.len(),
            result.shapes.len(),
            result.materials.len(),
            result.groups.len(),
            result.filters.len(),
        ] {
            if len != storage {
                return Err("Native capacity differs from input arrays".into());
            }
        }
        for (slot, parent) in result
            .parents
            .iter()
            .take(b.colliders_len as usize * nb)
            .enumerate()
        {
            if *parent as usize >= b.bodies_len as usize * nb || *parent as usize % nb != slot % nb
            {
                return Err("Invalid native collider parent/batch".into());
            }
        }
        let jp = meta["joints_per_batch"]
            .as_u64()
            .ok_or("Missing native joint capacity")?;
        if jp < b.impulse_joints_len as u64
            || jp.checked_mul(nb as u64) != Some(result.joints.len() as u64)
            || result.joint_colors.last().copied().unwrap_or(0) != b.impulse_joints_len
            || result
                .joint_colors
                .windows(2)
                .any(|ends| ends[0] >= ends[1])
        {
            return Err("Invalid native joint capacity/coloring".into());
        }
        for joint in result
            .joints
            .iter()
            .take(b.impulse_joints_len as usize * nb)
        {
            if joint.body_a >= b.bodies_len
                || joint.body_b >= b.bodies_len
                || joint.body_a == joint.body_b
            {
                return Err("Invalid native joint endpoint".into());
            }
        }
        let mut start = 0;
        for &end in &result.joint_colors {
            if end <= start {
                return Err("Empty native joint color".into());
            }
            let mut used = BTreeSet::new();
            for i in start as usize * nb..end as usize * nb {
                for local in [result.joints[i].body_a, result.joints[i].body_b] {
                    if !used.insert(local as usize * nb + i % nb) {
                        return Err("Native joint color aliases a velocity writer".into());
                    }
                }
            }
            start = end;
        }

        if !result.gravity.is_finite()
            || !result.params.dt.is_finite()
            || result.params.dt <= 0.0
            || result.params.num_solver_iterations == 0
        {
            return Err("Invalid native timestep/gravity".into());
        }
        Ok(result)
    }
}
