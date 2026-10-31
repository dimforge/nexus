//! Native execution of exported TileLang MSL with an existing khal device/queue.
//! Consecutive stages share one command buffer, with Metal resource tracking and GPU fences for untracked storage.
//! CPU completion boundaries remain at capacity checks and frame publication.
//! No Python, TVM, or repacking is used at execution time.
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
/// A trusted GPU producer writes a bounded (x, 1, 1) grid here before use.
/// Zero x is legal. x must not exceed the compiled specialization's block bound.
#[derive(Debug, Clone, Deserialize)]
pub struct IndirectDispatch {
    pub buffer: String,
    pub word: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProgramStage {
    pub name: String,
    pub source: String,
    pub bindings: Manifest,
    pub buffer_map: BTreeMap<String, String>,
    #[serde(default)]
    pub indirect: Option<IndirectDispatch>,
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
            if let Some(indirect) = &stage.indirect {
                let end = indirect.word.checked_add(3).and_then(|n| n.checked_mul(4));
                if end.is_none()
                    || self
                        .buffer_bytes
                        .get(&indirect.buffer)
                        .is_none_or(|size| end.unwrap() > *size)
                {
                    return Err("Invalid indirect dispatch buffer/offset".into());
                }
            }
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

/// Ordered native stages with explicit GPU dependencies and no intermediate
/// CPU physics, readback or repacking.
#[derive(Clone)]
pub struct NativeProgram {
    manifest: ProgramManifest,
    kernels: Vec<NativeKernel>,
    binding_slots: Vec<Vec<usize>>,
    device_id: u64,
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
        let binding_slots = manifest
            .stages
            .iter()
            .map(|stage| {
                stage
                    .bindings
                    .buffers
                    .iter()
                    .map(|local| {
                        manifest
                            .storage_buffers
                            .iter()
                            .position(|physical| physical == &stage.buffer_map[local])
                            .unwrap()
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            manifest,
            kernels,
            binding_slots,
            device_id: backend.device().registry_id(),
        })
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
        if backend.device().registry_id() != self.device_id {
            return Err("Program and backend devices differ".into());
        }
        // Validate every physical allocation before encoding any writes. Mappings
        // and the immutable ABI were checked when constructing the program.
        let physical = self
            .manifest
            .storage_buffers
            .iter()
            .map(|name| {
                let buffer = *buffers
                    .get(name)
                    .ok_or_else(|| format!("Missing buffer {name}"))?;
                if buffer.length() != self.manifest.buffer_bytes[name]
                    || buffer.device().registry_id() != self.device_id
                {
                    return Err(format!("Invalid buffer {name}"));
                }
                Ok(buffer)
            })
            .collect::<Result<Vec<_>, String>>()?;
        if range.is_empty() {
            return Ok(());
        }
        // Flush staged khal uploads once at this host boundary, never per shader.
        backend.synchronize().map_err(|e| e.to_string())?;
        let command = backend.queue().new_command_buffer();
        // Metal tracks resource hazards across compute passes. Untracked caller
        // storage additionally needs explicit GPU fences; retain support for it.
        let fence = physical
            .iter()
            .any(|buffer| buffer.hazard_tracking_mode() == metal::MTLHazardTrackingMode::Untracked)
            .then(|| backend.device().new_fence());
        let mut previous_pass = false;
        for index in range {
            let kernel = &self.kernels[index];
            for _ in 0..kernel.manifest.iterations {
                let encoder = command.new_compute_command_encoder();
                if previous_pass && let Some(fence) = &fence {
                    encoder.wait_for_fence(fence);
                }
                encoder.set_compute_pipeline_state(&kernel.pipeline);
                for (slot, &physical_slot) in self.binding_slots[index].iter().enumerate() {
                    encoder.set_buffer(slot as u64, Some(physical[physical_slot]), 0);
                }
                if let Some(indirect) = &self.manifest.stages[index].indirect {
                    encoder.dispatch_thread_groups_indirect(
                        buffers[&indirect.buffer],
                        indirect.word * 4,
                        MTLSize::new(kernel.manifest.threads, 1, 1),
                    );
                } else {
                    encoder.dispatch_thread_groups(
                        MTLSize::new(kernel.manifest.blocks, 1, 1),
                        MTLSize::new(kernel.manifest.threads, 1, 1),
                    );
                }
                if let Some(fence) = &fence {
                    encoder.update_fence(fence);
                }
                encoder.end_encoding();
                previous_pass = true;
            }
        }
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal program dispatch failed: {:?}",
                command.status()
            ));
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
                indirect: None,
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
    fn batched_dependencies_cover_untracked_buffers_and_workgroups() {
        let backend = Metal::new().unwrap();
        let mut m = program();
        m.iterations = 4;
        m.buffer_bytes.values_mut().for_each(|n| *n = 257 * 4);
        let first = &mut m.stages[0];
        first.bindings.buffer_bytes.insert("A".into(), 257 * 4);
        first.bindings.blocks = 5;
        first.bindings.threads = 64;
        first.bindings.iterations = 3;
        let mut second = first.clone();
        second.name = "cross_workgroup_consumer".into();
        second.source = "second.metal".into();
        second.bindings.iterations = 2;
        second.bindings.buffers.push("B".into());
        second.bindings.storage_buffers.push("B".into());
        second.bindings.buffer_bytes.insert("B".into(), 257 * 4);
        second.buffer_map.insert("B".into(), "Y".into());
        m.stages.push(second);
        let sources = BTreeMap::from([
            ("first.metal".into(), "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { if(i<257) a[i] += 1; }".into()),
            ("second.metal".into(), "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* a [[buffer(0)]], device uint* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { if(i<257) b[i] += a[(i+67)%257]; }".into()),
        ]);
        let program = NativeProgram::new(&backend, m, &sources).unwrap();
        let initial: Vec<u32> = (0..257).map(|i| i * 3).collect();
        let options = metal::MTLResourceOptions::StorageModeShared
            | metal::MTLResourceOptions::HazardTrackingModeUntracked;
        let x = backend
            .device()
            .new_buffer_with_data(initial.as_ptr().cast(), 257 * 4, options);
        let y = backend
            .device()
            .new_buffer_with_data(initial.as_ptr().cast(), 257 * 4, options);
        let bad = backend.device().new_buffer(4, options);
        let mut buffers = BTreeMap::from([("X".into(), x.as_ref()), ("Y".into(), bad.as_ref())]);
        unsafe {
            assert!(program.dispatch(&backend, &buffers).is_err());
            assert_eq!(
                std::slice::from_raw_parts(x.contents().cast::<u32>(), 257),
                initial
            );
            buffers.insert("Y".into(), y.as_ref());
            program.dispatch(&backend, &buffers).unwrap();
            let actual_x = std::slice::from_raw_parts(x.contents().cast::<u32>(), 257);
            let actual_y = std::slice::from_raw_parts(y.contents().cast::<u32>(), 257);
            for i in 0..257 {
                assert_eq!(actual_x[i], initial[i] + 12);
                assert_eq!(actual_y[i], initial[i] + 8 * initial[(i + 67) % 257] + 60);
            }
        }
    }

    #[test]
    fn gpu_indirect_dispatch_tracks_zero_and_changing_counts() {
        let backend = Metal::new().unwrap();
        let make = |name: &str, buffer: &str, bytes: u64, blocks: u64| {
            serde_json::json!({
                "name": name, "source": format!("{name}.metal"),
                "bindings": {"schema":1,"entry":"stage","buffers":[buffer],"storage_buffers":[buffer],
                    "buffer_bytes":{buffer:bytes},"blocks":blocks,"threads":64,"iterations":1},
                "buffer_map":{buffer:buffer}
            })
        };
        let producer = make("producer", "Grid", 16, 1);
        let mut consumer = make("consumer", "Output", 193 * 4, 4);
        consumer["indirect"] = serde_json::json!({"buffer":"Grid","word":1});
        let manifest: ProgramManifest = serde_json::from_value(serde_json::json!({
            "schema":2,"storage_buffers":["Grid","Output"],
            "buffer_bytes":{"Grid":16,"Output":193*4},"iterations":6,"stages":[producer,consumer]
        }))
        .unwrap();
        for (name, word) in [("Missing", 0), ("Grid", 2), ("Grid", u64::MAX)] {
            let mut bad = manifest.clone();
            bad.stages[1].indirect = Some(IndirectDispatch {
                buffer: name.into(),
                word,
            });
            assert!(bad.validate().is_err());
        }
        let sources = BTreeMap::from([
            ("producer.metal".into(), "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* g [[buffer(0)]], uint i [[thread_position_in_grid]]) {if(i==0){g[1]=g[0]%4;g[2]=1;g[3]=1;g[0]++;}}".into()),
            ("consumer.metal".into(), "#include <metal_stdlib>\nusing namespace metal;\nkernel void stage(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {if(i<193)o[i]++;}".into()),
        ]);
        let program = NativeProgram::new(&backend, manifest, &sources).unwrap();
        for hazard in [
            metal::MTLResourceOptions::HazardTrackingModeTracked,
            metal::MTLResourceOptions::HazardTrackingModeUntracked,
        ] {
            let options = metal::MTLResourceOptions::StorageModeShared | hazard;
            let g = [0u32; 4];
            let o = [0u32; 193];
            let grid = backend
                .device()
                .new_buffer_with_data(g.as_ptr().cast(), 16, options);
            let output = backend
                .device()
                .new_buffer_with_data(o.as_ptr().cast(), 193 * 4, options);
            let buffers = BTreeMap::from([
                ("Grid".into(), grid.as_ref()),
                ("Output".into(), output.as_ref()),
            ]);
            unsafe {
                program.dispatch(&backend, &buffers).unwrap();
                let words = std::slice::from_raw_parts(output.contents().cast::<u32>(), 193);
                for (i, &word) in words.iter().enumerate() {
                    assert_eq!(
                        word,
                        if i < 64 {
                            4
                        } else if i < 128 {
                            2
                        } else if i < 192 {
                            1
                        } else {
                            0
                        }
                    );
                }
            }
        }
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
