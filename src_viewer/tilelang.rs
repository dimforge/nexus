//! Testbed adapter for the independently maintained TileLang runtime.
//!
//! The CPU Nexus state supplies typed uploads and live controls. In comparison
//! mode it is an independent Rust simulation; otherwise it only carries the
//! published TileLang body state. Rendering never substitutes reference output
//! for a failed TileLang step. Kernel/runtime parity lives in `ports/tilelang`.
use crate::BackendType;
use khal::backend::{Backend, GpuBackend};
use nexus::prelude::{NexusPipeline, NexusState};
use nexus::rbd::{Velocity, math::Pose, pipeline::RbdSceneUpload};
use nexus_tilelang_native::runtime::{
    CacheStats, RuntimeOptions, TileLangBackend, TileLangSimulation,
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

type BodySlots = BTreeMap<(usize, u32, u32), u32>;

#[derive(Clone, Debug, Default)]
pub struct Comparison {
    pub bodies: usize,
    pub rms_position: f64,
    pub max_position: f32,
    pub max_angle: f32,
    pub max_linear_velocity: f32,
    pub max_angular_velocity: f32,
}

impl Comparison {
    fn measure(
        slots: &BodySlots,
        poses: &[Pose],
        reference: &[Pose],
        velocities: &[Velocity],
        reference_velocities: &[Velocity],
    ) -> Result<Self, String> {
        let mut result = Self::default();
        let mut squared = 0.0;
        for &slot in slots.values() {
            let i = slot as usize;
            let (Some(a), Some(b), Some(av), Some(bv)) = (
                poses.get(i),
                reference.get(i),
                velocities.get(i),
                reference_velocities.get(i),
            ) else {
                return Err("Comparison body slot is outside the returned state".into());
            };
            if !a.translation.is_finite()
                || !b.translation.is_finite()
                || !a.rotation.is_finite()
                || !b.rotation.is_finite()
                || !av.linear.is_finite()
                || !bv.linear.is_finite()
                || !av.angular.is_finite()
                || !bv.angular.is_finite()
            {
                return Err(format!("Non-finite state at body slot {slot}"));
            }
            if a.rotation.length_squared() < 1.0e-12 || b.rotation.length_squared() < 1.0e-12 {
                return Err(format!("Invalid zero quaternion at body slot {slot}"));
            }
            let distance = a.translation.distance(b.translation);
            // q and -q represent the same orientation; atan2 also retains small
            // rotation differences that acos(abs(dot)) rounds to zero.
            let q = a.rotation.conjugate() * b.rotation;
            let angle = 2.0 * glamx::Vec3::new(q.x, q.y, q.z).length().atan2(q.w.abs());
            result.max_position = result.max_position.max(distance);
            result.max_angle = result.max_angle.max(angle);
            result.max_linear_velocity = result
                .max_linear_velocity
                .max(av.linear.distance(bv.linear));
            result.max_angular_velocity = result
                .max_angular_velocity
                .max(av.angular.distance(bv.angular));
            squared += f64::from(distance).powi(2);
            result.bodies += 1;
        }
        result.rms_position = (squared / result.bodies.max(1) as f64).sqrt();
        Ok(result)
    }
}

/// Viewer controls and the last committed comparison. Changing `compare` must
/// restart the demo: a reference cannot be attached halfway through a trajectory.
#[derive(Default, Debug)]
pub struct TileLangUi {
    pub compare: bool,
    pub show_reference: bool,
    pub error: Option<String>,
    pub comparison: Option<Comparison>,
    pub frames: u64,
    pub step_ms: f64,
    pub rust_ms: Option<f64>,
    pub retries: usize,
    pub reconstructions: usize,
    pub workspace: Option<PathBuf>,
    pub preparation_ms: f64,
    pub reused_runtime: bool,
    pub cache: Option<CacheStats>,
    pub preparation_timings: std::collections::BTreeMap<String, f64>,
}

impl TileLangUi {
    pub fn clear_run(&mut self) {
        *self = Self {
            compare: self.compare,
            show_reference: self.show_reference,
            ..Self::default()
        };
    }
}

/// Native runtime options selected by the testbed. Environment paths are trusted
/// executable code, exactly as for `TileLangSimulation::from_upload`.
pub fn options(backend: BackendType, native_pair_capacity: u32) -> Result<RuntimeOptions, String> {
    let target = match backend {
        BackendType::TileLangC => TileLangBackend::C,
        #[cfg(feature = "metal")]
        BackendType::TileLangMetal => TileLangBackend::Metal,
        _ => return Err("Expected a TileLang backend".into()),
    };
    let port = std::env::var_os("NEXUS_TILELANG_PORT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ports/tilelang"));
    let python = std::env::var_os("NEXUS_TILELANG_PYTHON").map(PathBuf::from)
        .ok_or("Set NEXUS_TILELANG_PYTHON to the pinned TileLang Python environment (see src_viewer/TILELANG.md)")?;
    let parent = std::env::var_os("NEXUS_TILELANG_WORKDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("nexus-tilelang-testbed"));
    std::fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let max_pair_capacity = match std::env::var("NEXUS_TILELANG_MAX_PAIRS") {
        Ok(value) => value
            .parse::<u32>()
            .map_err(|_| "NEXUS_TILELANG_MAX_PAIRS must be a positive u32")?,
        Err(_) => native_pair_capacity.max(1_048_576),
    };
    let kernel_cache = match std::env::var_os("NEXUS_TILELANG_KERNEL_CACHE") {
        Some(value) if value == "off" => None,
        Some(value) => Some(PathBuf::from(value)),
        None => Some(port.join("../../target/tilelang-kernel-cache")),
    };
    Ok(RuntimeOptions {
        kernel_cache,
        backend: target,
        python,
        port_directory: port,
        workspace: parent.join(format!("{:?}-{}-{stamp}", backend, std::process::id())),
        initial_pair_capacity: native_pair_capacity.max(1),
        max_pair_capacity,
        trace_steps: None,
        contact_reduction: false,
    })
}

pub struct TileLangDriver {
    simulation: TileLangSimulation,
    original_options: RuntimeOptions,
    initial_scene: Vec<u8>,
    restartable: bool,
    topology: Vec<u8>,
    slots: BodySlots,
    previous_poses: Vec<Pose>,
    previous_velocities: Vec<Velocity>,
    pub poses: Vec<Pose>,
    pub reference_poses: Vec<Pose>,
    compare: bool,
}

fn slots(state: &NexusState) -> BodySlots {
    let mut result = BodySlots::new();
    for env in 0..state.num_environments() {
        for (handle, _) in state.rbd_world(env).bodies.iter() {
            if let Some(slot) = state.rigid_body_gpu_index(env, handle) {
                let (index, generation) = handle.into_raw_parts();
                result.insert((env, index, generation), slot);
            }
        }
    }
    result
}

fn topology(upload: &RbdSceneUpload) -> Vec<u8> {
    fn append<T: bytemuck::NoUninit>(out: &mut Vec<u8>, values: &[T]) {
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(values));
    }
    // Exact bytes, including length boundaries, avoid a hash collision silently
    // discarding a shape/material/topology edit. Dynamic state is handled below.
    let mut out = Vec::new();
    append(&mut out, std::slice::from_ref(&upload.batch_indices));
    append(&mut out, &upload.local_mass);
    append(&mut out, &upload.collider_local_poses);
    append(&mut out, &upload.collider_parents);
    append(&mut out, &upload.shapes);
    append(&mut out, &upload.materials);
    append(&mut out, &upload.groups);
    append(&mut out, &upload.pair_filters);
    append(&mut out, &upload.vertices);
    append(&mut out, &upload.indices);
    append(&mut out, &upload.joints);
    append(&mut out, &[upload.joints_per_batch]);
    append(&mut out, &upload.joint_color_ends);
    out
}

fn initial_scene(upload: &RbdSceneUpload) -> Vec<u8> {
    let mut bytes = topology(upload);
    bytes.extend_from_slice(bytemuck::bytes_of(&upload.params));
    bytes.extend_from_slice(bytemuck::bytes_of(&upload.gravity));
    for values in [
        bytemuck::cast_slice(&upload.body_poses),
        bytemuck::cast_slice(&upload.velocities),
    ] {
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        bytes.extend_from_slice(values);
    }
    bytes
}

fn changed_ranges<T: bytemuck::NoUninit>(before: &[T], after: &[T]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (i, value) in after.iter().enumerate() {
        let changed = before
            .get(i)
            .is_none_or(|old| bytemuck::bytes_of(old) != bytemuck::bytes_of(value));
        if changed {
            start.get_or_insert(i);
        } else if let Some(first) = start.take() {
            ranges.push(first..i);
        }
    }
    if let Some(first) = start {
        ranges.push(first..after.len());
    }
    ranges
}

fn validate_scene(state: &NexusState) -> Result<(), String> {
    if state.has_mpm() {
        return Err(
            "TileLang supports rigid-body scenes; select a Rust backend for MPM/coupled scenes"
                .into(),
        );
    }
    if state.counts().multibodies != 0 {
        return Err(
            "Articulated multibodies are deferred; select a Rust backend for this scene".into(),
        );
    }
    if state.rbd.is_none() {
        return Err("The scene has no rigid-body state".into());
    }
    Ok(())
}

impl TileLangDriver {
    /// The options identify trusted compiler code and a new artifact directory.
    /// No Rust solver is run here; both simulations begin from the same upload.
    ///
    /// # Safety
    /// Python, compiler sources and generated artifacts must be trusted executable
    /// code, as required by `TileLangSimulation::from_upload`.
    pub async unsafe fn new(
        state: &mut NexusState,
        options: RuntimeOptions,
        compare: bool,
    ) -> Result<Self, String> {
        let backend = GpuBackend::Cpu;
        state.finalize(&backend).await.map_err(|e| e.to_string())?;
        validate_scene(state)?;
        let upload = state
            .rbd
            .as_ref()
            .unwrap()
            .read_scene_upload(&backend)
            .await
            .map_err(|e| e.to_string())?;
        let topology = topology(&upload);
        let original_options = options.clone();
        let initial_scene = initial_scene(&upload);
        // SAFETY: compiler paths are selected explicitly by the local testbed
        // user, and its newly allocated artifact directory is caller-owned.
        let simulation = unsafe { TileLangSimulation::from_upload(&upload, options)? };
        Ok(Self {
            simulation,
            original_options,
            initial_scene,
            restartable: true,
            topology,
            slots: slots(state),
            previous_poses: upload.body_poses.clone(),
            previous_velocities: upload.velocities.clone(),
            poses: upload.body_poses.clone(),
            reference_poses: upload.body_poses,
            compare,
        })
    }

    /// Reuse only an unchanged initial scene and compiler selection. The fresh
    /// carrier remains the independent Rust reference; no prior trajectory survives.
    async fn restart(
        &mut self,
        state: &mut NexusState,
        options: &RuntimeOptions,
        compare: bool,
    ) -> Result<bool, String> {
        let mut expected = self.original_options.clone();
        expected.workspace = options.workspace.clone();
        if !self.restartable || expected != *options {
            return Ok(false);
        }
        let backend = GpuBackend::Cpu;
        state.finalize(&backend).await.map_err(|e| e.to_string())?;
        validate_scene(state)?;
        let upload = state
            .rbd
            .as_ref()
            .unwrap()
            .read_scene_upload(&backend)
            .await
            .map_err(|e| e.to_string())?;
        if self.initial_scene != initial_scene(&upload) || self.slots != slots(state) {
            return Ok(false);
        }
        self.simulation.reset()?;
        self.previous_poses = upload.body_poses.clone();
        self.previous_velocities = upload.velocities;
        self.poses = upload.body_poses.clone();
        self.reference_poses = upload.body_poses;
        self.compare = compare;
        Ok(true)
    }

    /// Select a loaded restart or prepare a fresh scene using the disk cache.
    ///
    /// # Safety
    /// The same trusted compiler and artifact requirements as `new` apply.
    pub async unsafe fn start(
        state: &mut NexusState,
        options: RuntimeOptions,
        previous: Option<Self>,
        ui: &mut TileLangUi,
    ) -> Result<Self, String> {
        let time = Instant::now();
        if let Some(mut previous) = previous
            && previous.restart(state, &options, ui.compare).await?
        {
            ui.reused_runtime = true;
            ui.preparation_ms = time.elapsed().as_secs_f64() * 1000.0;
            ui.workspace = Some(previous.original_options.workspace.clone());
            ui.cache = None;
            ui.preparation_timings.clear();
            return Ok(previous);
        }
        // SAFETY: forwarded caller contract; generated artifacts remain private.
        let driver = unsafe { Self::new(state, options, ui.compare).await? };
        ui.cache = driver.simulation.preparation_cache()?;
        ui.preparation_timings = driver.simulation.preparation_timings().clone();
        ui.reused_runtime = false;
        ui.preparation_ms = time.elapsed().as_secs_f64() * 1000.0;
        ui.workspace = Some(driver.original_options.workspace.clone());
        Ok(driver)
    }

    pub async fn step(
        &mut self,
        state: &mut NexusState,
        pipeline: &mut NexusPipeline,
        ui: &mut TileLangUi,
    ) -> Result<(), String> {
        let backend = GpuBackend::Cpu;
        state.finalize(&backend).await.map_err(|e| e.to_string())?;
        validate_scene(state)?;
        let upload = state
            .rbd
            .as_ref()
            .unwrap()
            .read_scene_upload(&backend)
            .await
            .map_err(|e| e.to_string())?;
        let new_slots = slots(state);
        let new_topology = topology(&upload);
        if self.topology != new_topology || self.slots != new_slots {
            self.restartable = false;
            // A native rebuild may reset its bodies from the host Rapier world.
            // Start a new comparison segment from that exact upload instead of
            // comparing preserved TileLang bodies with reset reference bodies.
            // This explicit reset is counted in the UI; steady-state frames do
            // not copy reference results into the independent TileLang solver.
            self.simulation.replace_scene(&upload, &[])?;
            self.topology = new_topology;
            ui.reconstructions += 1;
        } else {
            // Only external edits are forwarded. In comparison mode copying the
            // complete Rust state here would erase the very divergence measured.
            for range in changed_ranges(&self.previous_poses, &upload.body_poses) {
                self.simulation
                    .set_body_poses(range.start, &upload.body_poses[range])?;
            }
            for range in changed_ranges(&self.previous_velocities, &upload.velocities) {
                self.simulation
                    .set_velocities(range.start, &upload.velocities[range])?;
            }
        }
        self.slots = new_slots;
        // Reconfiguration replaces the runtime's original upload. A later demo
        // restart must then create fresh state, while reusing compatible code.
        if upload.params != self.simulation.solver_parameters() {
            self.restartable = false;
        }
        self.simulation.set_solver_parameters(upload.params)?;
        self.simulation
            .set_gravity(upload.gravity.truncate().to_array())?;
        let time = Instant::now();
        let mut retries = 0;
        for _ in 0..state.rbd_steps_per_frame().max(1) {
            retries += self.simulation.step().map_err(|e| e.to_string())?.retries;
        }
        let poses = self.simulation.body_poses()?;
        let velocities = self.simulation.velocities()?;
        ui.step_ms = time.elapsed().as_secs_f64() * 1000.0;
        if poses.len() != upload.body_poses.len() || velocities.len() != upload.velocities.len() {
            return Err("TileLang returned a different native body-storage layout".into());
        }
        Comparison::measure(&self.slots, &poses, &poses, &velocities, &velocities)?;
        if self.compare {
            let time = Instant::now();
            pipeline
                .simulate(&backend, state, None)
                .await
                .map_err(|e| e.to_string())?;
            ui.rust_ms = Some(time.elapsed().as_secs_f64() * 1000.0);
            let rbd = state.rbd.as_ref().unwrap();
            self.previous_poses = backend
                .slow_read_vec(rbd.body_poses().buffer())
                .await
                .map_err(|e| e.to_string())?;
            self.previous_velocities = backend
                .slow_read_vec(rbd.vels().buffer())
                .await
                .map_err(|e| e.to_string())?;
            ui.comparison = Some(Comparison::measure(
                &self.slots,
                &poses,
                &self.previous_poses,
                &velocities,
                &self.previous_velocities,
            )?);
            self.reference_poses = self.previous_poses.clone();
        } else {
            // Publish the actual TileLang state for ordinary viewer readback and
            // subsequent live pose/velocity edits. This never runs Rust physics.
            let rbd = state.rbd.as_mut().unwrap();
            backend
                .write_buffer(rbd.body_poses_mut().buffer_mut(), 0, &poses)
                .map_err(|e| e.to_string())?;
            backend
                .write_buffer(rbd.vels_mut().buffer_mut(), 0, &velocities)
                .map_err(|e| e.to_string())?;
            self.previous_poses = poses.clone();
            self.previous_velocities = velocities;
        }
        self.poses = poses;
        ui.frames += 1;
        ui.retries += retries;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tilelang_tests.rs"]
mod tests;
