//! Native execution of exported TileLang MSL with an existing khal device/queue.
//! This synchronous boundary flushes khal writes and preserves command ordering.
//! It is intended for correctness integration; asynchronous pass integration is
//! separate work. No Python, TVM, or repacking is used at execution time.
use khal::backend::{Backend, Metal};
use metal::{BufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub buffer_bytes: BTreeMap<String, u64>,
    pub entry: String,
    pub buffers: Vec<String>,
    pub storage_buffers: Vec<String>,
    pub blocks: u64,
    pub threads: u64,
    pub iterations: u32,
}

impl Manifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 1 {
            return Err("Unsupported native manifest schema".into());
        }
        if self.entry.is_empty() || self.blocks == 0 || self.threads == 0 || self.iterations == 0 {
            return Err("Empty entry or dispatch dimension".into());
        }
        let storage: BTreeSet<_> = self.storage_buffers.iter().collect();
        let bindings: BTreeSet<_> = self.buffers.iter().collect();
        if storage.len() != self.storage_buffers.len()
            || bindings.len() != self.buffers.len()
            || !bindings.is_subset(&storage)
            || storage.iter().any(|name| {
                name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
        {
            return Err("Invalid or duplicate buffer names".into());
        }
        if self.buffer_bytes.keys().collect::<BTreeSet<_>>() != storage
            || self
                .buffer_bytes
                .values()
                .any(|size| *size == 0 || size % 4 != 0)
        {
            return Err("Invalid specialization buffer sizes".into());
        }
        self.blocks
            .checked_mul(self.threads)
            .ok_or("Dispatch size overflow")?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct NativeKernel {
    pipeline: ComputePipelineState,
    device_id: u64,
    manifest: Manifest,
}

impl NativeKernel {
    pub fn new(backend: &Metal, source: &str, manifest: Manifest) -> Result<Self, String> {
        manifest.validate()?;
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = backend.device().new_library_with_source(source, &options)?;
        let function = library.get_function(&manifest.entry, None)?;
        let pipeline = backend
            .device()
            .new_compute_pipeline_state_with_function(&function)?;
        Self::with_pipeline(backend, pipeline, manifest)
    }

    fn with_pipeline(
        backend: &Metal,
        pipeline: ComputePipelineState,
        manifest: Manifest,
    ) -> Result<Self, String> {
        manifest.validate()?;
        if manifest.threads > pipeline.max_total_threads_per_threadgroup() {
            return Err("Thread count exceeds device/kernel limit".into());
        }
        Ok(Self {
            pipeline,
            device_id: backend.device().registry_id(),
            manifest,
        })
    }

    /// Execute on caller-owned native buffers and wait for completion.
    ///
    /// # Safety
    /// The caller must ensure every shader read/write is in bounds, buffer
    /// element layouts match the compiled specialization, writes are race-free,
    /// and no CPU/other queue accesses overlap the dispatch. The source is trusted
    /// executable code; a name/binding manifest does not prove memory safety.
    /// Buffers and the backend must belong to the device used to build this kernel.
    pub unsafe fn dispatch(
        &self,
        backend: &Metal,
        buffers: &BTreeMap<String, &BufferRef>,
    ) -> Result<(), String> {
        if backend.device().registry_id() != self.device_id {
            return Err("Kernel and backend devices differ".into());
        }
        for name in &self.manifest.storage_buffers {
            let buffer = buffers
                .get(name)
                .ok_or_else(|| format!("Missing buffer {name}"))?;
            if buffer.length() != self.manifest.buffer_bytes[name]
                || buffer.device().registry_id() != self.device_id
            {
                return Err(format!("Invalid buffer {name}"));
            }
        }
        // Flush khal's pending staged writes before submitting to its raw queue.
        backend.synchronize().map_err(|e| e.to_string())?;
        for _ in 0..self.manifest.iterations {
            let command = backend.queue().new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            for (slot, name) in self.manifest.buffers.iter().enumerate() {
                encoder.set_buffer(slot as u64, Some(buffers[name]), 0);
            }
            encoder.dispatch_thread_groups(
                MTLSize::new(self.manifest.blocks, 1, 1),
                MTLSize::new(self.manifest.threads, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("Metal dispatch failed: {:?}", command.status()));
            }
        }
        Ok(())
    }
}

/// Ordered kernels over persistent caller-owned storage. Each stage has its own
/// compiler-selected ABI; buffer_map maps those local names onto physical buffers.
#[derive(Debug, Clone, Deserialize)]
pub struct ProgramStage {
    pub name: String,
    pub source: String,
    pub bindings: Manifest,
    pub buffer_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProgramManifest {
    pub schema: u32,
    pub storage_buffers: Vec<String>,
    pub buffer_bytes: BTreeMap<String, u64>,
    pub iterations: u32,
    pub stages: Vec<ProgramStage>,
}

impl ProgramManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 2 || self.stages.is_empty() {
            return Err("Unsupported or empty ordered program".into());
        }
        // Reuse the storage-name, size and iteration checks of a kernel manifest.
        Manifest {
            schema: 1,
            storage_buffers: self.storage_buffers.clone(),
            buffer_bytes: self.buffer_bytes.clone(),
            buffers: vec![],
            entry: "program".into(),
            blocks: 1,
            threads: 1,
            iterations: self.iterations,
        }
        .validate()?;
        let mut names = BTreeSet::new();
        for stage in &self.stages {
            stage.bindings.validate()?;
            if stage.name.is_empty()
                || !names.insert(&stage.name)
                || stage.source.is_empty()
                || !stage
                    .source
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
                || stage.source == "."
                || stage.source == ".."
            {
                return Err("Invalid stage name or source filename".into());
            }
            if stage.buffer_map.keys().collect::<BTreeSet<_>>()
                != stage
                    .bindings
                    .storage_buffers
                    .iter()
                    .collect::<BTreeSet<_>>()
                || stage.buffer_map.values().collect::<BTreeSet<_>>().len()
                    != stage.buffer_map.len()
            {
                return Err("Missing or aliased stage buffer mapping".into());
            }
            for (local, physical) in &stage.buffer_map {
                if self.buffer_bytes.get(physical) != stage.bindings.buffer_bytes.get(local) {
                    return Err(format!(
                        "Invalid stage buffer mapping {local} -> {physical}"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// A correctness-first native schedule: no CPU physics, readback or repacking
/// between stages. Completed dispatches establish dependencies for later kernels.
#[derive(Clone)]
pub struct NativeProgram {
    manifest: ProgramManifest,
    kernels: Vec<NativeKernel>,
}
impl NativeProgram {
    pub fn new(
        backend: &Metal,
        manifest: ProgramManifest,
        sources: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        Self::with_libraries(backend, manifest, sources, &BTreeMap::new())
    }

    /// Load verified, trusted Metal libraries when supplied; otherwise compile MSL.
    /// Library entries must be compiled from the corresponding source and ABI.
    pub fn with_libraries(
        backend: &Metal,
        manifest: ProgramManifest,
        sources: &BTreeMap<String, String>,
        libraries: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, String> {
        manifest.validate()?;
        if libraries.keys().any(|name| !sources.contains_key(name)) {
            return Err("Library has no corresponding source".into());
        }
        let mut pipelines =
            BTreeMap::<(String, String, Option<&[u8]>), ComputePipelineState>::new();
        let kernels = manifest
            .stages
            .iter()
            .map(|stage| {
                let source = sources
                    .get(&stage.source)
                    .ok_or_else(|| format!("Missing source {}", stage.source))?;
                let binary = libraries.get(&stage.source).map(Vec::as_slice);
                let key = (source.clone(), stage.bindings.entry.clone(), binary);
                if let Some(pipeline) = pipelines.get(&key) {
                    NativeKernel::with_pipeline(backend, pipeline.clone(), stage.bindings.clone())
                } else {
                    let kernel = if let Some(binary) = binary {
                        let library = backend.device().new_library_with_data(binary)?;
                        let function = library.get_function(&stage.bindings.entry, None)?;
                        let pipeline = backend
                            .device()
                            .new_compute_pipeline_state_with_function(&function)?;
                        NativeKernel::with_pipeline(backend, pipeline, stage.bindings.clone())?
                    } else {
                        NativeKernel::new(backend, source, stage.bindings.clone())?
                    };
                    pipelines.insert(key, kernel.pipeline.clone());
                    Ok(kernel)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { manifest, kernels })
    }

    /// Execute the full schedule on persistent native buffers.
    ///
    /// # Safety
    /// The caller must meet NativeKernel::dispatch's requirements for every stage,
    /// including scratch initialization and dependencies between stages. Distinct
    /// physical names must refer to distinct allocations, not aliased buffers.
    pub unsafe fn dispatch(
        &self,
        backend: &Metal,
        buffers: &BTreeMap<String, &BufferRef>,
    ) -> Result<(), String> {
        for _ in 0..self.manifest.iterations {
            unsafe {
                self.dispatch_stages(backend, buffers, 0..self.manifest.stages.len())?;
            }
        }
        Ok(())
    }

    /// Execute one consecutive stage range without applying outer iterations.
    ///
    /// # Safety
    /// The caller must satisfy the same storage and dependency guarantees as
    /// dispatch, including dependencies on prior ranges in the same frame.
    pub unsafe fn dispatch_stages(
        &self,
        backend: &Metal,
        buffers: &BTreeMap<String, &BufferRef>,
        range: std::ops::Range<usize>,
    ) -> Result<(), String> {
        if range.start > range.end || range.end > self.kernels.len() {
            return Err("Invalid native stage range".into());
        }
        // Validate all physical storage before the first stage can mutate anything.
        for name in &self.manifest.storage_buffers {
            let buffer = buffers
                .get(name)
                .ok_or_else(|| format!("Missing buffer {name}"))?;
            if buffer.length() != self.manifest.buffer_bytes[name]
                || buffer.device().registry_id() != backend.device().registry_id()
            {
                return Err(format!("Invalid buffer {name}"));
            }
        }
        for index in range {
            let stage = &self.manifest.stages[index];
            let local = stage
                .buffer_map
                .iter()
                .map(|(name, physical)| (name.clone(), buffers[physical]))
                .collect();
            unsafe {
                self.kernels[index].dispatch(backend, &local)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::foreign_types::ForeignType;
    fn valid() -> Manifest {
        Manifest {
            schema: 1,
            buffer_bytes: BTreeMap::from([("A".into(), 4)]),
            entry: "main".into(),
            buffers: vec!["A".into()],
            storage_buffers: vec!["A".into()],
            blocks: 1,
            threads: 64,
            iterations: 1,
        }
    }
    #[test]
    fn rejects_invalid_binding_and_dispatch_contracts() {
        let mut m = valid();
        m.buffers.push("A".into());
        assert!(m.validate().is_err());
        m = valid();
        m.buffers[0] = "missing".into();
        assert!(m.validate().is_err());
        m = valid();
        m.storage_buffers[0] = "../outside".into();
        assert!(m.validate().is_err());
        m = valid();
        m.blocks = u64::MAX;
        assert!(m.validate().is_err());
        m = valid();
        m.iterations = 0;
        assert!(m.validate().is_err());
    }
    #[test]
    fn permits_compiler_eliminated_bindings() {
        let mut m = valid();
        m.buffers.clear();
        assert!(m.validate().is_ok());
    }
    #[test]
    fn rejects_missing_misaligned_or_unknown_buffer_sizes() {
        let mut m = valid();
        m.buffer_bytes.clear();
        assert!(m.validate().is_err());
        m = valid();
        m.buffer_bytes.insert("A".into(), 5);
        assert!(m.validate().is_err());
        m = valid();
        m.buffer_bytes.insert("unused".into(), 4);
        assert!(m.validate().is_err());
        m = valid();
        m.schema = 9;
        assert!(m.validate().is_err());
    }
    fn program() -> ProgramManifest {
        let mut local = valid();
        local.threads = 1;
        local.entry = "stage".into();
        ProgramManifest {
            schema: 2,
            storage_buffers: vec!["X".into(), "Y".into()],
            buffer_bytes: BTreeMap::from([("X".into(), 4), ("Y".into(), 4)]),
            iterations: 2,
            stages: vec![ProgramStage {
                name: "first".into(),
                source: "first.metal".into(),
                bindings: local,
                buffer_map: BTreeMap::from([("A".into(), "X".into())]),
            }],
        }
    }
    #[test]
    fn rejects_invalid_program_dependencies() {
        assert!(program().validate().is_ok());
        let mut m = program();
        m.stages.clear();
        assert!(m.validate().is_err());
        let mut m = program();
        m.stages[0].source = "../bad".into();
        assert!(m.validate().is_err());
        let mut m = program();
        m.stages[0].buffer_map.clear();
        assert!(m.validate().is_err());
        let mut m = program();
        m.stages[0].buffer_map.insert("A".into(), "missing".into());
        assert!(m.validate().is_err());
        let mut m = program();
        m.buffer_bytes.insert("X".into(), 8);
        assert!(m.validate().is_err());
        let mut m = program();
        m.stages.push(m.stages[0].clone());
        assert!(m.validate().is_err());
        let mut m = program();
        m.stages[0].bindings.storage_buffers.push("B".into());
        m.stages[0].bindings.buffer_bytes.insert("B".into(), 4);
        m.stages[0].buffer_map.insert("B".into(), "X".into());
        assert!(m.validate().is_err());
    }
    #[test]
    fn reused_metal_pipeline_keeps_bindings_iterations_and_launch_checks() {
        let backend = Metal::new().unwrap();
        let mut m = program();
        let mut second = m.stages[0].clone();
        second.name = "same_code_other_storage".into();
        second.source = "alias.metal".into();
        second.bindings.iterations = 3;
        second.buffer_map.insert("A".into(), "Y".into());
        m.stages.push(second);
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* a [[buffer(0)]]) { a[0] += 1; }".to_owned();
        let sources = BTreeMap::from([
            ("first.metal".into(), source.clone()),
            ("alias.metal".into(), source),
        ]);
        let compiled = NativeProgram::new(&backend, m.clone(), &sources).unwrap();
        assert_eq!(
            compiled.kernels[0].pipeline.as_ptr(),
            compiled.kernels[1].pipeline.as_ptr()
        );
        let usage = khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC;
        let x = backend.init_buffer(&[0u32], usage).unwrap();
        let y = backend.init_buffer(&[10u32], usage).unwrap();
        let buffers = BTreeMap::from([
            ("X".into(), x.raw().as_ref()),
            ("Y".into(), y.raw().as_ref()),
        ]);
        unsafe {
            compiled.dispatch(&backend, &buffers).unwrap();
            assert_eq!(*x.raw().contents().cast::<u32>(), 2);
            assert_eq!(*y.raw().contents().cast::<u32>(), 16);
        }
        m.stages[1].bindings.threads = compiled.kernels[0]
            .pipeline
            .max_total_threads_per_threadgroup()
            + 1;
        assert!(NativeProgram::new(&backend, m, &sources).is_err());
    }

    #[test]
    fn ordered_native_stages_share_storage_across_iterations() {
        let backend = Metal::new().unwrap();
        let mut m = program();
        let mut second = m.stages[0].clone();
        second.name = "second".into();
        second.source = "second.metal".into();
        second.bindings.storage_buffers.push("B".into());
        second.bindings.buffers.push("B".into());
        second.bindings.buffer_bytes.insert("B".into(), 4);
        second.buffer_map.insert("B".into(), "Y".into());
        m.stages.push(second);
        let sources = BTreeMap::from([
            ("first.metal".into(), "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* a [[buffer(0)]]) { a[0] += 1; }".into()),
            ("second.metal".into(), "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* a [[buffer(0)]], device uint* b [[buffer(1)]]) { b[0] += a[0]; }".into()),
        ]);
        let program = NativeProgram::new(&backend, m, &sources).unwrap();
        let usage = khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC;
        let x = backend.init_buffer(&[0u32], usage).unwrap();
        let y = backend.init_buffer(&[0u32], usage).unwrap();
        let buffers = BTreeMap::from([
            ("X".into(), x.raw().as_ref()),
            ("Y".into(), y.raw().as_ref()),
        ]);
        unsafe {
            program.dispatch(&backend, &buffers).unwrap();
            assert_eq!(*x.raw().contents().cast::<u32>(), 2);
            assert_eq!(*y.raw().contents().cast::<u32>(), 3);
        }
    }
}
