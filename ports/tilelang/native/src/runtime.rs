//! Rust-owned TileLang simulation with explicit backend selection and lazy growth.
//!
//! The compiler and generated kernels are trusted executable code. No reference
//! physics runs during specialization or stepping. Ascend execution is not yet
//! available; its source bundles are produced by the separate compiler entrypoint.
use nexus_rbd3d::{
    Velocity,
    math::Pose,
    pipeline::RbdSceneUpload,
    shaders::dynamics::{FrictionModel, RbdSimParams},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Component, Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

#[derive(Clone, Copy, Debug)]
pub enum TileLangBackend {
    C,
    #[cfg(feature = "metal")]
    Metal,
}
impl TileLangBackend {
    fn target(self) -> &'static str {
        match self {
            Self::C => "c",
            #[cfg(feature = "metal")]
            Self::Metal => "metal",
        }
    }
}

/// Paths and allocation policy chosen by the caller. The workspace must not exist.
#[derive(Clone)]
pub struct RuntimeOptions {
    pub backend: TileLangBackend,
    pub python: PathBuf,
    pub port_directory: PathBuf,
    pub workspace: PathBuf,
    pub initial_pair_capacity: u32,
    /// An application resource limit, not a device or solver limit.
    pub max_pair_capacity: u32,
    /// None avoids trajectory allocation and permits ongoing frame stepping.
    pub trace_steps: Option<u32>,
    /// Merge per-subshape manifolds before solving. Native Nexus defaults to false.
    pub contact_reduction: bool,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct FrameProgress {
    pub step: u64,
    pub retries: usize,
    pub variant: usize,
    pub counts: BTreeMap<String, u64>,
}
#[derive(Debug)]
pub enum RuntimeError {
    Capacity {
        gate: String,
        required: u64,
        limit: u32,
    },
    Runtime(String),
}
impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity {
                gate,
                required,
                limit,
            } => write!(
                f,
                "{gate}: required capacity {required} exceeds application limit {limit}"
            ),
            Self::Runtime(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for RuntimeError {}
impl From<String> for RuntimeError {
    fn from(value: String) -> Self {
        Self::Runtime(value)
    }
}
enum StepFailure {
    Capacity { gate: String, required: u64 },
    Runtime(String),
}
struct CProcess {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl CProcess {
    fn new(options: &RuntimeOptions, variant: &Path) -> Result<Self, String> {
        let mut child = Command::new(&options.python)
            .arg("-u")
            .arg(options.port_directory.join("runtime_service.py"))
            .arg(variant)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("Missing runtime stdin")?;
        let output = BufReader::new(child.stdout.take().ok_or("Missing runtime stdout")?);
        let mut result = Self {
            child,
            input,
            output,
        };
        if result.receive()?["ready"] != true {
            return Err("C runtime did not initialize".into());
        }
        Ok(result)
    }
    fn receive(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        if self
            .output
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Err("Generated-C runtime exited before replying".into());
        }
        serde_json::from_str(&line).map_err(|e| format!("Invalid runtime response: {e}"))
    }
    fn request(&mut self, value: Value) -> Result<Value, StepFailure> {
        serde_json::to_writer(&mut self.input, &value)
            .map_err(|e| StepFailure::Runtime(e.to_string()))?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| StepFailure::Runtime(e.to_string()))?;
        let reply = self.receive().map_err(StepFailure::Runtime)?;
        if reply["kind"] == "capacity" {
            return Err(StepFailure::Capacity {
                gate: reply["gate"]
                    .as_str()
                    .ok_or_else(|| StepFailure::Runtime("Missing capacity gate".into()))?
                    .to_owned(),
                required: reply["required"]
                    .as_u64()
                    .ok_or_else(|| StepFailure::Runtime("Missing capacity requirement".into()))?,
            });
        }
        if let Some(error) = reply.get("error") {
            return Err(StepFailure::Runtime(error.to_string()));
        }
        reply
            .get("ok")
            .cloned()
            .ok_or_else(|| StepFailure::Runtime("Missing runtime result".into()))
    }
    fn call(&mut self, value: Value) -> Result<Value, String> {
        self.request(value).map_err(|e| match e {
            StepFailure::Runtime(message) => message,
            StepFailure::Capacity { .. } => "Unexpected capacity error outside step".into(),
        })
    }
}
impl Drop for CProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
enum Session {
    C(CProcess),
    #[cfg(feature = "metal")]
    Metal {
        backend: khal::backend::Metal,
        frame: crate::frame::FrameSession,
    },
}

/// A source/archive-bound continuation, including contact caches and step count.
/// Serialize to retain it between processes. Restoring serialized words requires trusted provenance.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCheckpoint {
    signature: Value,
    frame: crate::checkpoint::FrameCheckpoint,
}
impl RuntimeCheckpoint {
    pub fn step(&self) -> u64 {
        self.frame.steps
    }
}

pub struct TileLangSimulation {
    options: RuntimeOptions,
    archive: PathBuf,
    uploaded: RbdSceneUpload,
    workspace_root: PathBuf,
    next_configuration: usize,
    signature: Value,
    body_count: usize,
    capacity: u32,
    next_variant: usize,
    published_variant: usize,
    session: Session,
}
fn validate_parameters(p: &RbdSimParams) -> Result<(), String> {
    let floats = [
        p.dt,
        p.contact_damping_ratio,
        p.contact_natural_frequency,
        p.static_contact_damping_ratio,
        p.static_contact_natural_frequency,
        p.joint_natural_frequency,
        p.joint_damping_ratio,
        p.warmstart_coefficient,
        p.length_unit,
        p.normalized_allowed_linear_error,
        p.normalized_max_corrective_velocity,
        p.normalized_prediction_distance,
        p.normalized_max_linear_velocity,
        p.contact_merge_cos,
        p.normalized_contact_recycle_distance,
    ];
    if floats.iter().any(|v| !v.is_finite())
        || p.dt <= 0.
        || p.length_unit <= 0.
        || p.num_solver_iterations == 0
        || !(0. ..=1.).contains(&p.warmstart_coefficient)
    {
        return Err("Invalid solver parameters".into());
    }
    if p.friction_model != FrictionModel::Coulomb {
        return Err("Simplified friction is not yet supported by the runtime".into());
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn verify_artifacts(directory: &Path) -> Result<Value, String> {
    let metadata = read_json(&directory.join("specialization.json"))?;
    let files = metadata["files"]
        .as_object()
        .ok_or("Missing specialization inventory")?;
    for (name, expected) in files {
        let mut components = Path::new(name).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err("Invalid specialization artifact path".into());
        }
        let digest = format!(
            "{:x}",
            Sha256::digest(fs::read(directory.join(name)).map_err(|e| e.to_string())?)
        );
        if expected.as_str() != Some(digest.as_str()) {
            return Err(format!("Specialization artifact changed: {name}"));
        }
    }
    let frame = read_json(&directory.join("frame.json"))?;
    let mut required = vec![
        "frame.json".to_owned(),
        "case.json".to_owned(),
        "layout.json".to_owned(),
    ];
    for stage in frame["program"]["stages"]
        .as_array()
        .ok_or("Missing program stages")?
    {
        required.push(
            stage["source"]
                .as_str()
                .ok_or("Missing stage source")?
                .to_owned(),
        );
    }
    for name in frame["program"]["storage_buffers"]
        .as_array()
        .ok_or("Missing storage")?
    {
        required.push(format!(
            "{}.bin",
            name.as_str().ok_or("Invalid storage name")?
        ));
    }
    if required.iter().any(|name| !files.contains_key(name)) {
        return Err("Specialization omits required artifacts".into());
    }
    Ok(metadata)
}
fn compile(
    options: &RuntimeOptions,
    archive: &Path,
    directory: &Path,
    capacity: u32,
) -> Result<Value, String> {
    let mut command = Command::new(&options.python);
    command
        .arg(options.port_directory.join("runtime_compile.py"))
        .arg("--upload")
        .arg(archive)
        .arg("--output")
        .arg(directory)
        .arg("--target")
        .arg(options.backend.target())
        .arg("--pair-capacity")
        .arg(capacity.to_string());
    if options.contact_reduction {
        command.arg("--contact-reduction");
    }
    if let Some(steps) = options.trace_steps {
        command.arg("--trace-steps").arg(steps.to_string());
    }
    let output = command.output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "TileLang specialization failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    let metadata = verify_artifacts(directory)?;
    if metadata["target"] != options.backend.target()
        || metadata["pair_capacity"] != capacity
        || metadata["max_steps"] != json!(options.trace_steps)
        || metadata["contact_reduction"] != json!(options.contact_reduction)
    {
        return Err("Compiler specialization differs from requested configuration".into());
    }
    Ok(
        json!({"source": metadata["inputs"], "archive": metadata["archive_sha256"], "contact_reduction": metadata["contact_reduction"],"max_steps":metadata["max_steps"]}),
    )
}
#[cfg(feature = "metal")]
fn load_metal(directory: &Path) -> Result<crate::frame::FrameVariant, String> {
    let manifest: crate::frame::FrameManifest =
        serde_json::from_value(read_json(&directory.join("frame.json"))?)
            .map_err(|e| e.to_string())?;
    manifest.validate()?;
    let sources = manifest
        .program
        .stages
        .iter()
        .map(|s| {
            Ok((
                s.source.clone(),
                fs::read_to_string(directory.join(&s.source)).map_err(|e| e.to_string())?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let initial = manifest
        .program
        .storage_buffers
        .iter()
        .map(|n| {
            let bytes = fs::read(directory.join(format!("{n}.bin"))).map_err(|e| e.to_string())?;
            if !bytes.len().is_multiple_of(4) {
                return Err("Misaligned native input".into());
            }
            Ok((
                n.clone(),
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word))
                    .collect(),
            ))
        })
        .collect::<Result<_, String>>()?;
    Ok(crate::frame::FrameVariant {
        manifest,
        sources,
        initial,
    })
}
impl TileLangSimulation {
    /// Start a separate TileLang simulation from the native cold-start upload.
    ///
    /// # Safety
    /// The Python executable, port directory, generated kernels and workspace must
    /// be trusted and remain under the caller's control. They must satisfy the
    /// native dispatch and capacity-gate requirements documented by FrameSession.
    pub unsafe fn from_upload(
        upload: &RbdSceneUpload,
        mut options: RuntimeOptions,
    ) -> Result<Self, String> {
        validate_parameters(&upload.params)?;
        if options.initial_pair_capacity == 0
            || options.initial_pair_capacity > options.max_pair_capacity
            || options.max_pair_capacity > 0x1fffffff
            || options.trace_steps == Some(0)
        {
            return Err("Invalid runtime capacity or trace options".into());
        }
        options.port_directory = options
            .port_directory
            .canonicalize()
            .map_err(|e| e.to_string())?;
        fs::create_dir(&options.workspace).map_err(|e| e.to_string())?;
        options.workspace = options
            .workspace
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let archive = options.workspace.join("upload");
        crate::upload::write_inputs(upload, &archive)?;
        let directory = options.workspace.join("variant-0");
        let capacity = options.initial_pair_capacity;
        let signature = compile(&options, &archive, &directory, capacity)?;
        let layout = read_json(&directory.join("layout.json"))?;
        let case = read_json(&directory.join("case.json"))?;
        if layout["pose_words"] != size_of::<Pose>() / 4
            || layout["velocity_words"] != size_of::<Velocity>() / 4
        {
            return Err("Native readback ABI differs from Rust types".into());
        }
        let body_count = usize::try_from(case["body_count"].as_u64().ok_or("Missing body count")?)
            .map_err(|e| e.to_string())?;
        let session = match options.backend {
            TileLangBackend::C => Session::C(CProcess::new(&options, &directory)?),
            #[cfg(feature = "metal")]
            TileLangBackend::Metal => {
                let backend = khal::backend::Metal::new().map_err(|e| e.to_string())?;
                let frame = unsafe {
                    crate::frame::FrameSession::new(&backend, vec![load_metal(&directory)?])
                }?;
                Session::Metal { backend, frame }
            }
        };
        let workspace_root = options.workspace.clone();
        Ok(Self {
            uploaded: upload.clone(),
            workspace_root,
            next_configuration: 0,
            options,
            archive,
            signature,
            body_count,
            capacity,
            next_variant: 1,
            published_variant: 0,
            session,
        })
    }
    /// Advance once, compiling and retrying larger capacity only before commit.
    pub fn step(&mut self) -> Result<FrameProgress, RuntimeError> {
        loop {
            let outcome: Result<FrameProgress, StepFailure> = match &mut self.session {
                Session::C(process) => {
                    process
                        .request(json!({"command":"step"}))
                        .and_then(|value| {
                            serde_json::from_value(value)
                                .map_err(|e| StepFailure::Runtime(e.to_string()))
                        })
                }
                #[cfg(feature = "metal")]
                Session::Metal { backend, frame } => frame
                    .step(backend)
                    .map(|s| FrameProgress {
                        step: s.step,
                        retries: s.retries,
                        variant: s.variant,
                        counts: s.counts,
                    })
                    .map_err(|e| match e {
                        crate::frame::StepError::Capacity { gate, required, .. } => {
                            StepFailure::Capacity { gate, required }
                        }
                        other => StepFailure::Runtime(format!("{other:?}")),
                    }),
            };
            match outcome {
                Ok(mut progress) => {
                    progress.retries = progress
                        .variant
                        .checked_sub(self.published_variant)
                        .ok_or_else(|| {
                            RuntimeError::Runtime("Runtime variant moved backwards".into())
                        })?;
                    self.published_variant = progress.variant;
                    return Ok(progress);
                }
                Err(StepFailure::Runtime(message)) => return Err(RuntimeError::Runtime(message)),
                Err(StepFailure::Capacity { gate, required }) => {
                    if required > self.options.max_pair_capacity as u64 {
                        return Err(RuntimeError::Capacity {
                            gate,
                            required,
                            limit: self.options.max_pair_capacity,
                        });
                    }
                    let capacity = (self.capacity as u64 * 2)
                        .max(required)
                        .min(self.options.max_pair_capacity as u64)
                        as u32;
                    if capacity <= self.capacity {
                        return Err(RuntimeError::Capacity {
                            gate,
                            required,
                            limit: self.options.max_pair_capacity,
                        });
                    }
                    self.compile_capacity(capacity)?;
                }
            }
        }
    }
    fn compile_capacity(&mut self, capacity: u32) -> Result<(), String> {
        let directory = self
            .options
            .workspace
            .join(format!("variant-{}", self.next_variant));
        self.next_variant += 1;
        let signature = compile(&self.options, &self.archive, &directory, capacity)?;
        if signature != self.signature {
            return Err("Sources or scene archive changed during the session".into());
        }
        match &mut self.session {
            Session::C(process) => {
                process.call(json!({"command":"extend","directory":directory}))?;
            }
            #[cfg(feature = "metal")]
            Session::Metal { frame, .. } => unsafe { frame.extend(load_metal(&directory)?) }?,
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Capture persistent state and contact caches at a frame boundary.
    pub fn checkpoint(&mut self) -> Result<RuntimeCheckpoint, String> {
        let frame = match &mut self.session {
            Session::C(process) => {
                serde_json::from_value(process.call(json!({"command":"checkpoint"}))?)
                    .map_err(|e| e.to_string())?
            }
            #[cfg(feature = "metal")]
            Session::Metal { backend, frame } => frame.checkpoint(backend)?,
        };
        Ok(RuntimeCheckpoint {
            signature: self.signature.clone(),
            frame,
        })
    }
    /// Restore a continuation from the same source and original scene upload.
    /// This may compile larger buffers. Failed validation leaves the published state unchanged.
    ///
    /// # Safety
    /// The checkpoint must be unmodified output of a trusted simulation with these
    /// exact source/archive semantics. Deserializing arbitrary words does not prove
    /// cached indices or physical values meet the generated shaders' preconditions.
    pub unsafe fn restore_checkpoint(
        &mut self,
        checkpoint: &RuntimeCheckpoint,
    ) -> Result<(), String> {
        if checkpoint.signature != self.signature {
            return Err("Checkpoint source, scene or configuration differs".into());
        }
        checkpoint.frame.validate()?;
        let capacity = *checkpoint
            .frame
            .capacities
            .values()
            .max()
            .ok_or("Missing checkpoint capacities")?;
        if capacity > self.options.max_pair_capacity {
            return Err("Checkpoint exceeds application capacity limit".into());
        }
        if capacity > self.capacity {
            self.compile_capacity(capacity)?;
        }
        let index = match &mut self.session {
            Session::C(process) => serde_json::from_value(
                process.call(json!({"command":"restore","checkpoint":checkpoint.frame}))?,
            )
            .map_err(|e| e.to_string())?,
            #[cfg(feature = "metal")]
            Session::Metal { backend, frame } => {
                unsafe { frame.restore(backend, &checkpoint.frame) }?
            }
        };
        self.published_variant = index;
        Ok(())
    }
    /// Reset physical state, contact caches and the step counter to the original upload.
    pub fn reset(&mut self) -> Result<(), String> {
        match &mut self.session {
            Session::C(process) => {
                process.call(json!({"command":"reset"}))?;
            }
            #[cfg(feature = "metal")]
            Session::Metal { backend, frame } => frame.reset(backend)?,
        }
        self.published_variant = 0;
        Ok(())
    }

    /// Current native parameter block. `dt` is a substep duration.
    pub fn solver_parameters(&self) -> RbdSimParams {
        self.uploaded.params
    }

    /// Replace native solver parameters while preserving physical state, caches and step count.
    /// `dt` is the substep duration, as in RbdSceneUpload. Changed schedules are compiled
    /// into a separate configuration before it is published; this is not a cheap per-frame setter.
    pub fn set_solver_parameters(&mut self, params: RbdSimParams) -> Result<(), String> {
        validate_parameters(&params)?;
        if params == self.uploaded.params {
            return Ok(());
        }
        let checkpoint = self.checkpoint()?;
        let mut upload = self.uploaded.clone();
        upload.params = params;
        let mut options = self.options.clone();
        options.initial_pair_capacity = self.capacity;
        options.workspace = self
            .workspace_root
            .join(format!("configuration-{}", self.next_configuration));
        self.next_configuration = self
            .next_configuration
            .checked_add(1)
            .ok_or("Configuration counter overflow")?;
        // Preserve the constructor's trusted compiler/workspace contract. All scene
        // arrays are unchanged; only a checked native parameter block differs.
        let mut replacement = unsafe { Self::from_upload(&upload, options) }?;
        if replacement.signature["source"] != self.signature["source"] {
            return Err("Sources changed during reconfiguration".into());
        }
        let old_archive = self.signature["archive"]
            .as_object()
            .ok_or("Missing scene identity")?;
        let new_archive = replacement.signature["archive"]
            .as_object()
            .ok_or("Missing new scene identity")?;
        if old_archive.keys().ne(new_archive.keys())
            || old_archive.iter().any(|(name, value)| {
                !matches!(name.as_str(), "RbdParams.bin" | "scene.json")
                    && new_archive.get(name) != Some(value)
            })
        {
            return Err("Reconfiguration changed non-parameter scene inputs".into());
        }
        let mut old_metadata = read_json(&self.archive.join("scene.json"))?;
        let mut new_metadata = read_json(&replacement.archive.join("scene.json"))?;
        for name in ["substeps", "pgs"] {
            old_metadata
                .as_object_mut()
                .ok_or("Invalid scene metadata")?
                .remove(name);
            new_metadata
                .as_object_mut()
                .ok_or("Invalid new scene metadata")?
                .remove(name);
        }
        if old_metadata != new_metadata {
            return Err("Reconfiguration changed non-parameter scene metadata".into());
        }
        let continuation = RuntimeCheckpoint {
            signature: replacement.signature.clone(),
            frame: checkpoint.frame,
        };
        // Same scene, Coulomb layout and semantic regions: changing iteration counts,
        // timestep or coefficients does not invalidate native cache identities.
        unsafe { replacement.restore_checkpoint(&continuation) }?;
        replacement.workspace_root = self.workspace_root.clone();
        replacement.next_configuration = self.next_configuration;
        *self = replacement;
        Ok(())
    }
    /// Change the full-frame duration and TGS substep count, matching RbdState.
    pub fn set_timestep(&mut self, dt: f32, substeps: u32) -> Result<(), String> {
        let mut params = self.solver_parameters();
        params.num_solver_iterations = substeps.max(1);
        params.dt = dt / params.num_solver_iterations as f32;
        self.set_solver_parameters(params)
    }
    pub fn set_num_internal_pgs_iterations(&mut self, iterations: u32) -> Result<(), String> {
        let mut params = self.solver_parameters();
        params.num_internal_pgs_iterations = iterations.max(1);
        self.set_solver_parameters(params)
    }
    pub fn set_contact_merge_cos(&mut self, cos: f32) -> Result<(), String> {
        let mut params = self.solver_parameters();
        params.contact_merge_cos = cos;
        self.set_solver_parameters(params)
    }
    pub fn set_contact_recycle_distance(&mut self, distance: f32) -> Result<(), String> {
        let mut params = self.solver_parameters();
        params.normalized_contact_recycle_distance = distance;
        self.set_solver_parameters(params)
    }

    /// Read current typed inputs for a cold start, using the original native geometry.
    /// Contact caches and the frame counter are intentionally absent; use checkpoint
    /// for continuation without changing scene membership.
    pub fn read_scene_upload(&mut self) -> Result<RbdSceneUpload, String> {
        let mut upload = self.uploaded.clone();
        upload.body_poses[..self.body_count].copy_from_slice(&self.body_poses()?);
        upload.velocities[..self.body_count].copy_from_slice(&self.velocities()?);
        let gravity = self.read_words("Gravity")?;
        let values: [f32; 4] = gravity
            .get(..4)
            .ok_or("Incomplete gravity state")?
            .iter()
            .map(|v| f32::from_bits(*v))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Invalid gravity state")?;
        upload.gravity = values.into();
        Ok(upload)
    }

    /// Replace scene membership/geometry and optionally retain surviving body state.
    /// Each pair is (old global body slot, new global body slot), in native
    /// batch-interleaved order. Unmapped bodies use the new upload's state.
    /// This is a cold start: contacts, colors and step/trace counters reset. The
    /// new upload supplies gravity and solver parameters. No old collider IDs or
    /// cached impulses are carried across changed scene topology.
    /// Compilation completes before the live scene is replaced.
    pub fn replace_scene(
        &mut self,
        upload: &RbdSceneUpload,
        survivors: &[(usize, usize)],
    ) -> Result<(), String> {
        let mut upload = upload.clone();
        crate::scene_lifecycle::transfer_bodies(
            &mut upload,
            &self.body_poses()?,
            &self.velocities()?,
            survivors,
        )?;
        let mut options = self.options.clone();
        options.initial_pair_capacity = self.capacity;
        options.workspace = self
            .workspace_root
            .join(format!("scene-{}", self.next_configuration));
        self.next_configuration = self
            .next_configuration
            .checked_add(1)
            .ok_or("Configuration counter overflow")?;
        // The constructor's executable-code trust contract still holds. Scene
        // arrays are native typed inputs, checked by the same upload validator.
        let mut replacement = unsafe { Self::from_upload(&upload, options) }?;
        if replacement.signature["source"] != self.signature["source"] {
            return Err("Sources changed during scene replacement".into());
        }
        replacement.workspace_root = self.workspace_root.clone();
        replacement.next_configuration = self.next_configuration;
        *self = replacement;
        Ok(())
    }

    /// Adjust the application allocation limit after a capacity error.
    pub fn set_pair_capacity_limit(&mut self, limit: u32) -> Result<(), String> {
        if limit < self.capacity || limit > 0x1fffffff {
            return Err("Invalid runtime capacity limit".into());
        }
        self.options.max_pair_capacity = limit;
        Ok(())
    }
    pub fn body_count(&self) -> usize {
        self.body_count
    }

    fn write_persistent(
        &mut self,
        region: &str,
        word_offset: usize,
        words: &[u32],
    ) -> Result<(), String> {
        match &mut self.session {
            Session::C(process) => {
                process.call(json!({"command":"write", "region":region,"word_offset":word_offset,"words":words}))?;
                Ok(())
            }
            #[cfg(feature = "metal")]
            Session::Metal { backend, frame } => unsafe {
                frame.write_persistent(backend, region, word_offset, words)
            },
        }
    }
    fn body_range<T>(&self, first: usize, count: usize) -> Result<usize, String> {
        if first
            .checked_add(count)
            .is_none_or(|end| end > self.body_count)
        {
            return Err("Body update outside active body range".into());
        }
        first
            .checked_mul(size_of::<T>() / 4)
            .ok_or("Body offset overflow".into())
    }
    /// Set gravity for subsequent steps, including retries with larger buffers.
    pub fn set_gravity(&mut self, gravity: [f32; 3]) -> Result<(), String> {
        if gravity.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite gravity".into());
        }
        self.write_persistent(
            "Gravity",
            0,
            &[
                gravity[0].to_bits(),
                gravity[1].to_bits(),
                gravity[2].to_bits(),
                0,
            ],
        )
    }
    /// Replace a contiguous range in native interleaved body order.
    /// Contact caches are retained and revalidated by the next native matching pass.
    pub fn set_body_poses(&mut self, first: usize, poses: &[Pose]) -> Result<(), String> {
        let offset = self.body_range::<Pose>(first, poses.len())?;
        if poses.iter().any(|p| {
            !p.translation.is_finite()
                || !p.rotation.is_finite()
                || (p.rotation.length_squared() - 1.0).abs() > 2e-5
        }) {
            return Err("Body pose requires finite translation and a unit rotation".into());
        }
        self.write_persistent("Poses", offset, bytemuck::cast_slice(poses))
    }
    /// Replace finite linear/angular velocities in native interleaved body order.
    pub fn set_velocities(&mut self, first: usize, velocities: &[Velocity]) -> Result<(), String> {
        let offset = self.body_range::<Velocity>(first, velocities.len())?;
        if velocities
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
        {
            return Err("Nonfinite body velocity".into());
        }
        self.write_persistent("Velocities", offset, bytemuck::cast_slice(velocities))
    }

    /// Explicit readback for diagnostics or application state publication.
    pub fn read_words(&mut self, name: &str) -> Result<Vec<u32>, String> {
        match &mut self.session {
            Session::C(process) => {
                serde_json::from_value(process.call(json!({"command":"read","buffer":name}))?)
                    .map_err(|e| e.to_string())
            }
            #[cfg(feature = "metal")]
            Session::Metal { backend, frame } => frame.read(backend, name),
        }
    }
    fn read_typed<T: bytemuck::Pod>(&mut self, name: &str) -> Result<Vec<T>, String> {
        let words = self.read_words(name)?;
        let bytes = bytemuck::cast_slice::<u32, u8>(&words);
        let length = self
            .body_count
            .checked_mul(size_of::<T>())
            .ok_or("Readback size overflow")?;
        let data = bytes.get(..length).ok_or("Truncated native readback")?;
        // Generic size_of::<T>() is not a stable as_chunks const argument.
        #[allow(clippy::chunks_exact_to_as_chunks)]
        Ok(data
            .chunks_exact(size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect())
    }
    pub fn body_poses(&mut self) -> Result<Vec<Pose>, String> {
        self.read_typed("Poses")
    }
    pub fn velocities(&mut self) -> Result<Vec<Velocity>, String> {
        self.read_typed("Velocities")
    }
}
