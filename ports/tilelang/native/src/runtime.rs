//! Rust-owned TileLang simulation with explicit backend selection and lazy growth.
//!
//! The compiler and generated kernels are trusted executable code. No reference
//! physics runs during specialization or stepping. Ascend execution is not yet
//! available; its source bundles are produced by the separate compiler entrypoint.
use nexus_rbd3d::{Velocity, math::Pose, pipeline::RbdSceneUpload};
use serde::Deserialize;
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

pub struct TileLangSimulation {
    options: RuntimeOptions,
    archive: PathBuf,
    signature: Value,
    body_count: usize,
    capacity: u32,
    next_variant: usize,
    published_variant: usize,
    session: Session,
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
    {
        return Err("Compiler specialization differs from requested configuration".into());
    }
    Ok(json!({"source": metadata["inputs"], "archive": metadata["archive_sha256"]}))
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
        Ok(Self {
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
                    let directory = self
                        .options
                        .workspace
                        .join(format!("variant-{}", self.next_variant));
                    self.next_variant += 1;
                    let signature = compile(&self.options, &self.archive, &directory, capacity)?;
                    if signature != self.signature {
                        return Err(RuntimeError::Runtime(
                            "Sources or scene archive changed during the session".into(),
                        ));
                    }
                    match &mut self.session {
                        Session::C(process) => {
                            process.call(json!({"command":"extend","directory":directory}))?;
                        }
                        #[cfg(feature = "metal")]
                        Session::Metal { frame, .. } => {
                            unsafe { frame.extend(load_metal(&directory)?) }?
                        }
                    }
                    self.capacity = capacity;
                }
            }
        }
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
