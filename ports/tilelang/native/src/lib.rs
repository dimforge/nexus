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

#[cfg(test)]
mod tests {
    use super::*;
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
}
