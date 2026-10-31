//! Transaction boundaries for resident frames and capacity growth before solving.
//!
//! Kernel source remains trusted. Every capacity gate must precede writes to the
//! persistent regions; callers certify this while exporting their frame schedule.
use crate::{NativeProgram, ProgramManifest};
use khal::backend::{Backend, Metal};
use metal::{Buffer, BufferRef};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Counter {
    pub buffer: String,
    pub word: usize,
}
#[derive(Clone, Debug, Deserialize)]
pub struct CapacityGate {
    /// Stable semantic identity, such as `pairs` or `deferred_pairs`.
    pub id: String,
    /// The gate follows this stage and precedes the next stage.
    pub after_stage: String,
    pub counters: Vec<Counter>,
    pub capacity: u32,
}
#[derive(Clone, Debug, Deserialize)]
pub struct PersistentRegion {
    pub buffer: String,
    pub word_start: usize,
    pub word_len: usize,
}
#[derive(Clone, Debug, Deserialize)]
pub struct FrameCondition {
    pub gate: String,
    pub minimum: u64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct FrameManifest {
    pub schema: u32,
    /// Logical ownership exchanges, applied only after all commit stages succeed.
    #[serde(default)]
    pub swaps: Vec<[String; 2]>,
    /// Omit empty commit stages using counters already read by capacity gates.
    #[serde(default)]
    pub conditions: BTreeMap<String, FrameCondition>,
    pub program: ProgramManifest,
    pub gates: Vec<CapacityGate>,
    /// Stable region identities. Larger variants retain each prior region's
    /// bytes at the new offset; new capacity starts from that variant's upload.
    pub persistent: BTreeMap<String, PersistentRegion>,
    /// Includes trace capacity, when tracing is enabled. No implicit wraparound.
    #[serde(deserialize_with = "deserialize_max_steps")]
    pub max_steps: Option<u32>,
}
// An explicit null means no trace limit; a missing field is an error.
fn deserialize_max_steps<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u32>, D::Error> {
    Option::<u32>::deserialize(d)
}
impl FrameManifest {
    pub fn validate(&self) -> Result<Vec<usize>, String> {
        self.program.validate()?;
        if !matches!(self.schema, 1 | 2)
            || (self.schema == 1 && (!self.swaps.is_empty() || !self.conditions.is_empty()))
            || self.program.iterations != 1
            || self.max_steps == Some(0)
            || (self.max_steps.is_none()
                && self
                    .program
                    .storage_buffers
                    .iter()
                    .any(|n| n == "Trajectory" || n == "FrameIndex"))
        {
            return Err("Invalid resident frame schema, iteration count or step limit".into());
        }
        let mut swapped = BTreeSet::new();
        for [a, b] in &self.swaps {
            if !swapped.insert(a)
                || !swapped.insert(b)
                || !self.program.buffer_bytes.contains_key(a)
                || !self.program.buffer_bytes.contains_key(b)
                || self.program.buffer_bytes[a] != self.program.buffer_bytes[b]
            {
                return Err("Swapped buffers must be distinct, disjoint and equally sized".into());
            }
        }
        let mut ends = Vec::new();
        let mut names = BTreeSet::new();
        for gate in &self.gates {
            if gate.id.is_empty() || !names.insert(&gate.id) || gate.counters.is_empty() {
                return Err("Invalid capacity gate identity or counters".into());
            }
            let end = self
                .program
                .stages
                .iter()
                .position(|s| s.name == gate.after_stage)
                .ok_or("Capacity gate stage is missing")?
                + 1;
            if end >= self.program.stages.len() || ends.last().is_some_and(|last| *last > end) {
                return Err(
                    "Capacity gates must be ordered before the frame's commit stages".into(),
                );
            }
            let mut counters = BTreeSet::new();
            for counter in &gate.counters {
                let len = self
                    .program
                    .buffer_bytes
                    .get(&counter.buffer)
                    .ok_or("Missing counter buffer")?
                    / 4;
                if counter.word as u64 >= len || !counters.insert((&counter.buffer, counter.word)) {
                    return Err("Invalid or duplicated capacity counter".into());
                }
            }
            ends.push(end);
        }
        for (name, condition) in &self.conditions {
            let index = self.program.stages.iter().position(|s| &s.name == name);
            if condition.minimum == 0
                || !names.contains(&condition.gate)
                || index.is_none_or(|i| i < ends.last().copied().unwrap_or(0))
            {
                return Err("Condition must use a completed gate before commit".into());
            }
        }
        let mut spans = BTreeMap::<&str, Vec<(usize, usize)>>::new();
        if self.persistent.is_empty() {
            return Err("A frame must identify its persistent state".into());
        }
        for (id, region) in &self.persistent {
            let bytes = self
                .program
                .buffer_bytes
                .get(&region.buffer)
                .ok_or("Missing persistent buffer")?;
            let end = region
                .word_start
                .checked_add(region.word_len)
                .ok_or("Persistent region overflow")?;
            if id.is_empty() || region.word_len == 0 || end as u64 > bytes / 4 {
                return Err("Invalid persistent region".into());
            }
            spans
                .entry(&region.buffer)
                .or_default()
                .push((region.word_start, end));
        }
        for ranges in spans.values_mut() {
            ranges.sort_unstable();
            if ranges.windows(2).any(|r| r[0].1 > r[1].0) {
                return Err("Overlapping persistent regions".into());
            }
        }
        Ok(ends)
    }
}

/// One complete, already exported capacity specialization and its initial storage.
/// Geometry and physics values are uploaded here, never computed by the executor.
pub struct FrameVariant {
    pub manifest: FrameManifest,
    pub sources: BTreeMap<String, String>,
    pub libraries: BTreeMap<String, Vec<u8>>,
    pub initial: BTreeMap<String, Vec<u32>>,
}
impl FrameVariant {
    fn validate(&self) -> Result<Vec<usize>, String> {
        let ends = self.manifest.validate()?;
        let sizes = &self.manifest.program.buffer_bytes;
        if self.initial.len() != sizes.len()
            || self
                .initial
                .iter()
                .any(|(name, words)| sizes.get(name).copied() != Some(words.len() as u64 * 4))
        {
            return Err("Frame upload does not match specialization storage".into());
        }
        Ok(ends)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum StepError {
    Capacity {
        gate: String,
        required: u64,
        capacity: u32,
    },
    StepLimit,
    Poisoned,
    Backend(String),
}
#[derive(Debug, PartialEq, Eq)]
pub struct FrameStats {
    pub step: u64,
    pub retries: usize,
    pub variant: usize,
    pub counts: BTreeMap<String, u64>,
}

pub struct FrameSession {
    variants: Vec<FrameVariant>,
    gate_ends: Vec<Vec<usize>>,
    current: usize,
    program: NativeProgram,
    initial_program: NativeProgram,
    storage: BTreeMap<String, Buffer>,
    steps: u64,
    poisoned: bool,
    device_id: u64,
}
fn upload(backend: &Metal, variant: &FrameVariant) -> Result<BTreeMap<String, Buffer>, String> {
    variant
        .initial
        .iter()
        .map(|(name, data)| {
            let buffer = backend
                .init_buffer(
                    data,
                    khal::BufferUsages::STORAGE
                        | khal::BufferUsages::COPY_SRC
                        | khal::BufferUsages::COPY_DST,
                )
                .map_err(|e| e.to_string())?;
            Ok((name.clone(), buffer.raw().to_owned()))
        })
        .collect()
}
fn validate_growth(old: &FrameManifest, new: &FrameManifest) -> Result<(), String> {
    if old.swaps != new.swaps
        || old.gates.len() != new.gates.len()
        || old.max_steps != new.max_steps
        || old.persistent.keys().ne(new.persistent.keys())
    {
        return Err("Incompatible frame capacity variants".into());
    }
    let mut grows = false;
    for (a, b) in old.gates.iter().zip(&new.gates) {
        if a.id != b.id || a.counters != b.counters || b.capacity < a.capacity {
            return Err("Capacity variants change gate meaning or shrink".into());
        }
        grows |= b.capacity > a.capacity;
    }
    if !grows
        || old
            .persistent
            .iter()
            .any(|(id, r)| new.persistent[id].word_len < r.word_len)
    {
        return Err("Capacity variants must grow without truncating state".into());
    }
    Ok(())
}
impl FrameSession {
    /// Load trusted kernels and initial storage. Variants increase capacities
    /// while preserving the same physical scene and semantic persistent regions.
    ///
    /// # Safety
    /// Every specialization must satisfy NativeProgram::dispatch's requirements.
    /// Gates must run before any persistent region is changed. Each variant must
    /// describe the same scene, parameters and stage semantics, with only larger
    /// scratch capacity and explicitly relocated persistent storage.
    pub unsafe fn new(backend: &Metal, variants: Vec<FrameVariant>) -> Result<Self, String> {
        if variants.is_empty() {
            return Err("No frame specialization".into());
        }
        let gate_ends = variants
            .iter()
            .map(FrameVariant::validate)
            .collect::<Result<Vec<_>, _>>()?;
        for pair in variants.windows(2) {
            validate_growth(&pair[0].manifest, &pair[1].manifest)?;
        }
        let first = &variants[0];
        let program = NativeProgram::with_libraries(
            backend,
            first.manifest.program.clone(),
            &first.sources,
            &first.libraries,
        )?;
        let storage = upload(backend, first)?;
        Ok(Self {
            variants,
            gate_ends,
            current: 0,
            initial_program: program.clone(),
            program,
            storage,
            steps: 0,
            poisoned: false,
            device_id: backend.device().registry_id(),
        })
    }
    /// Append a trusted specialization without changing live state.
    ///
    /// # Safety
    /// The same scene, gate-placement and dispatch requirements as `new` apply.
    /// New capacity must preserve every semantic persistent region.
    pub unsafe fn extend(&mut self, variant: FrameVariant) -> Result<(), String> {
        if self.poisoned {
            return Err("Poisoned frame".into());
        }
        let ends = variant.validate()?;
        validate_growth(&self.variants.last().unwrap().manifest, &variant.manifest)?;
        self.variants.push(variant);
        self.gate_ends.push(ends);
        Ok(())
    }
    /// Copy the semantic state required to continue at the same frame boundary.
    pub fn checkpoint(
        &self,
        backend: &Metal,
    ) -> Result<crate::checkpoint::FrameCheckpoint, String> {
        if self.poisoned {
            return Err("Poisoned frame".into());
        }
        let m = &self.variants[self.current].manifest;
        let mut regions = BTreeMap::new();
        for (name, r) in &m.persistent {
            let words = self.read(backend, &r.buffer)?;
            regions.insert(
                name.clone(),
                words[r.word_start..r.word_start + r.word_len].to_vec(),
            );
        }
        Ok(crate::checkpoint::FrameCheckpoint {
            schema: 1,
            steps: self.steps,
            capacities: m.gates.iter().map(|g| (g.id.clone(), g.capacity)).collect(),
            regions,
        })
    }
    /// Restore a checkpoint into compatible storage, resetting nonpersistent scratch.
    ///
    /// # Safety
    /// Words must be a valid checkpoint of the same scene and kernel semantics.
    /// Counters, cached indices, layouts and physical inputs must satisfy every shader precondition.
    pub unsafe fn restore(
        &mut self,
        backend: &Metal,
        state: &crate::checkpoint::FrameCheckpoint,
    ) -> Result<usize, String> {
        if self.poisoned || backend.device().registry_id() != self.device_id {
            return Err("Poisoned frame or wrong device".into());
        }
        state.validate()?;
        let index = self
            .variants
            .iter()
            .position(|v| {
                let m = &v.manifest;
                let caps: BTreeMap<_, _> = m.gates.iter().map(|g| (&g.id, g.capacity)).collect();
                caps.len() == state.capacities.len()
                    && state
                        .capacities
                        .iter()
                        .all(|(k, n)| caps.get(k).is_some_and(|cap| cap >= n))
                    && m.persistent.keys().eq(state.regions.keys())
                    && state
                        .regions
                        .iter()
                        .all(|(k, v)| v.len() <= m.persistent[k].word_len)
                    && m.max_steps.is_none_or(|limit| state.steps <= limit as u64)
            })
            .ok_or("Checkpoint has no compatible capacity specialization")?;
        let v = &self.variants[index];
        let program = NativeProgram::with_libraries(
            backend,
            v.manifest.program.clone(),
            &v.sources,
            &v.libraries,
        )?;
        let storage = upload(backend, v)?;
        backend.synchronize().map_err(|e| e.to_string())?;
        for (name, words) in &state.regions {
            let r = &v.manifest.persistent[name];
            let b = &storage[&r.buffer];
            unsafe {
                std::slice::from_raw_parts_mut(
                    b.contents().cast::<u32>().add(r.word_start),
                    words.len(),
                )
            }
            .copy_from_slice(words);
        }
        self.program = program;
        self.storage = storage;
        self.current = index;
        self.steps = state.steps;
        Ok(index)
    }
    /// Reset all state and scratch to the original upload.
    pub fn reset(&mut self, backend: &Metal) -> Result<(), String> {
        if self.poisoned || backend.device().registry_id() != self.device_id {
            return Err("Poisoned frame or wrong device".into());
        }
        let v = &self.variants[0];
        let program = self.initial_program.clone();
        let storage = upload(backend, v)?;
        backend.synchronize().map_err(|e| e.to_string())?;
        self.program = program;
        self.storage = storage;
        self.current = 0;
        self.steps = 0;
        Ok(())
    }
    fn refs(&self) -> BTreeMap<String, &BufferRef> {
        self.storage
            .iter()
            .map(|(name, buffer)| (name.clone(), buffer.as_ref()))
            .collect()
    }
    fn grow(&mut self, backend: &Metal, next: usize) -> Result<(), String> {
        let new = &self.variants[next];
        // Compile and allocate before touching the current state. Allocation or
        // compilation failure cannot publish a half-migrated specialization.
        let program = NativeProgram::with_libraries(
            backend,
            new.manifest.program.clone(),
            &new.sources,
            &new.libraries,
        )?;
        let storage = upload(backend, new)?;
        backend.synchronize().map_err(|e| e.to_string())?;
        let command = backend.queue().new_command_buffer();
        let encoder = command.new_blit_command_encoder();
        for (id, old) in &self.variants[self.current].manifest.persistent {
            let new = &new.manifest.persistent[id];
            encoder.copy_from_buffer(
                &self.storage[&old.buffer],
                old.word_start as u64 * 4,
                &storage[&new.buffer],
                new.word_start as u64 * 4,
                old.word_len as u64 * 4,
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != metal::MTLCommandBufferStatus::Completed {
            return Err("Persistent state migration failed".into());
        }
        self.program = program;
        self.storage = storage;
        self.current = next;
        Ok(())
    }
    /// Advance one frame. Capacity failures occur before commit stages and never
    /// advance the step counter. Successful retries preserve prior-frame state.
    pub fn step(&mut self, backend: &Metal) -> Result<FrameStats, StepError> {
        if self.poisoned {
            return Err(StepError::Poisoned);
        }
        if backend.device().registry_id() != self.device_id {
            return Err(StepError::Backend(
                "Frame and backend devices differ".into(),
            ));
        }
        if self.steps == u64::MAX
            || self.variants[self.current]
                .manifest
                .max_steps
                .is_some_and(|limit| self.steps >= limit as u64)
        {
            return Err(StepError::StepLimit);
        }
        let mut retries = 0;
        loop {
            let m = &self.variants[self.current].manifest;
            // Variants are immutable after new/extend validated the complete ABI.
            let ends = &self.gate_ends[self.current];
            let mut start = 0;
            let mut overflow = None;
            let mut counts = BTreeMap::new();
            for (gate, end) in m.gates.iter().zip(ends) {
                if let Err(error) = unsafe {
                    self.program
                        .dispatch_stages(backend, &self.refs(), start..*end)
                } {
                    self.poisoned = true;
                    return Err(StepError::Backend(error));
                }
                let required = gate
                    .counters
                    .iter()
                    .map(|counter| unsafe {
                        *self.storage[&counter.buffer]
                            .contents()
                            .cast::<u32>()
                            .add(counter.word) as u64
                    })
                    .sum::<u64>();
                counts.insert(gate.id.clone(), required);
                if required > gate.capacity as u64 {
                    overflow = Some((gate.id.clone(), required, gate.capacity));
                    break;
                }
                start = *end;
            }
            if let Some((gate, required, capacity)) = overflow {
                let next = (self.current + 1..self.variants.len()).find(|i| {
                    self.variants[*i]
                        .manifest
                        .gates
                        .iter()
                        .find(|g| g.id == gate)
                        .is_some_and(|g| g.capacity as u64 >= required)
                });
                let Some(next) = next else {
                    return Err(StepError::Capacity {
                        gate,
                        required,
                        capacity,
                    });
                };
                self.grow(backend, next).map_err(StepError::Backend)?;
                retries += 1;
                continue;
            }
            if let Err(error) = unsafe {
                self.program.dispatch_stages_if(
                    backend,
                    &self.refs(),
                    start..m.program.stages.len(),
                    |i| {
                        m.conditions
                            .get(&m.program.stages[i].name)
                            .is_none_or(|condition| counts[&condition.gate] >= condition.minimum)
                    },
                )
            } {
                self.poisoned = true;
                return Err(StepError::Backend(error));
            }
            for [a, b] in &m.swaps {
                let left = self.storage.remove(a).expect("validated swap buffer");
                let right = self
                    .storage
                    .insert(b.clone(), left)
                    .expect("validated swap buffer");
                self.storage.insert(a.clone(), right);
            }
            self.steps += 1;
            return Ok(FrameStats {
                step: self.steps,
                retries,
                variant: self.current,
                counts,
            });
        }
    }
    /// Update part of a semantic region between completed frames.
    ///
    /// # Safety
    /// The caller must preserve every kernel input invariant (including indices,
    /// floating-point constraints and valid native enum representations).
    pub unsafe fn write_persistent(
        &mut self,
        backend: &Metal,
        id: &str,
        word_offset: usize,
        words: &[u32],
    ) -> Result<(), String> {
        if self.poisoned {
            return Err("Poisoned frame".into());
        }
        if backend.device().registry_id() != self.device_id {
            return Err("Frame and backend devices differ".into());
        }
        let region = self.variants[self.current]
            .manifest
            .persistent
            .get(id)
            .ok_or("Missing persistent region")?;
        let end = word_offset
            .checked_add(words.len())
            .ok_or("Persistent write overflow")?;
        if end > region.word_len {
            return Err("Persistent write outside region".into());
        }
        backend.synchronize().map_err(|e| e.to_string())?;
        let buffer = &self.storage[&region.buffer];
        let start = region.word_start + word_offset;
        unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().cast::<u32>().add(start), words.len())
        }
        .copy_from_slice(words);
        Ok(())
    }
    /// Read explicit output only; no intermediate physics readback is performed.
    pub fn read(&self, backend: &Metal, name: &str) -> Result<Vec<u32>, String> {
        if backend.device().registry_id() != self.device_id {
            return Err("Frame and backend devices differ".into());
        }
        backend.synchronize().map_err(|e| e.to_string())?;
        let buffer = self.storage.get(name).ok_or("Missing frame buffer")?;
        Ok(unsafe {
            std::slice::from_raw_parts(
                buffer.contents().cast::<u32>(),
                buffer.length() as usize / 4,
            )
        }
        .to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Manifest, ProgramStage};
    fn variant(capacity: u32, offset: usize) -> FrameVariant {
        let sizes = BTreeMap::from([
            ("State".into(), ((offset + 4) * 4) as u64),
            ("Count".into(), 4),
            ("Input".into(), 4),
        ]);
        let local = |names: Vec<String>, entry: &str| Manifest {
            schema: 1,
            buffer_bytes: names.iter().map(|n| (n.clone(), sizes[n])).collect(),
            entry: entry.into(),
            buffers: names.clone(),
            storage_buffers: names,
            blocks: 1,
            threads: 1,
            iterations: 1,
        };
        let count = ProgramStage {
            indirect: None,
            name: "count".into(),
            source: "count.metal".into(),
            bindings: local(vec!["Input".into(), "Count".into()], "count"),
            buffer_map: BTreeMap::from([
                ("Input".into(), "Input".into()),
                ("Count".into(), "Count".into()),
            ]),
        };
        let commit = ProgramStage {
            indirect: None,
            name: "commit".into(),
            source: "commit.metal".into(),
            bindings: local(vec!["State".into()], "commit"),
            buffer_map: BTreeMap::from([("State".into(), "State".into())]),
        };
        let mut state = vec![0xdeadbeef; offset + 4];
        state[offset..offset + 2].copy_from_slice(&[42, 17]);
        FrameVariant{
            libraries:BTreeMap::new(),
            manifest:FrameManifest{schema:1,swaps:vec![],conditions:BTreeMap::new(),program:ProgramManifest{schema:2,storage_buffers:sizes.keys().cloned().collect(),buffer_bytes:sizes,iterations:1,stages:vec![count,commit]},
                gates:vec![CapacityGate{id:"pairs".into(),after_stage:"count".into(),counters:vec![Counter{buffer:"Count".into(),word:0}],capacity}],
                persistent:BTreeMap::from([("pose_and_velocity".into(),PersistentRegion{buffer:"State".into(),word_start:offset,word_len:2})]),max_steps:Some(2)},
            sources:BTreeMap::from([
                ("count.metal".into(),"#include <metal_stdlib>\nusing namespace metal;\nkernel void count(device uint* input [[buffer(0)]], device uint* count [[buffer(1)]]) {count[0]=input[0];}".into()),
                ("commit.metal".into(),format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void commit(device uint* state [[buffer(0)]]) {{state[{offset}]+=1;}}"))]),
            initial:BTreeMap::from([("State".into(),state),("Input".into(),vec![5]),("Count".into(),vec![0])]),
        }
    }
    fn swap_variant(capacity: u32, offset: usize) -> FrameVariant {
        let mut v = variant(capacity, offset);
        v.manifest.schema = 2;
        v.manifest.max_steps = None;
        v.manifest.swaps = vec![["Scratch".into(), "State".into()]];
        let bytes = v.manifest.program.buffer_bytes["State"];
        v.manifest.program.storage_buffers.push("Scratch".into());
        v.manifest
            .program
            .buffer_bytes
            .insert("Scratch".into(), bytes);
        v.initial
            .insert("Scratch".into(), vec![0xdeadbeef; bytes as usize / 4]);
        let stage = &mut v.manifest.program.stages[1];
        stage.bindings.buffers.push("Scratch".into());
        stage.bindings.storage_buffers.push("Scratch".into());
        stage.bindings.buffer_bytes.insert("Scratch".into(), bytes);
        stage.buffer_map.insert("Scratch".into(), "Scratch".into());
        v.sources.insert("commit.metal".into(), format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void commit(device uint* state [[buffer(0)]], device uint* scratch [[buffer(1)]]) {{scratch[{offset}]=state[{offset}]+1; scratch[{}]=state[{}];}}", offset+1, offset+1));
        v
    }
    #[test]
    fn swaps_preserve_odd_even_checkpoint_growth_and_failed_steps() {
        let backend = Metal::new().unwrap();
        let mut s = unsafe { FrameSession::new(&backend, vec![swap_variant(8, 0)]) }.unwrap();
        s.step(&backend).unwrap();
        assert_eq!(&s.read(&backend, "State").unwrap()[..2], &[43, 17]);
        let checkpoint = s.checkpoint(&backend).unwrap();
        // Fail after an odd number of swaps. Neither logical owner nor state changes.
        unsafe {
            *s.storage["Input"].contents().cast::<u32>() = 9;
        }
        assert!(matches!(s.step(&backend), Err(StepError::Capacity { .. })));
        assert_eq!(&s.read(&backend, "State").unwrap()[..2], &[43, 17]);
        unsafe { s.extend(swap_variant(16, 3)) }.unwrap();
        assert_eq!(s.step(&backend).unwrap().retries, 1);
        assert_eq!(&s.read(&backend, "State").unwrap()[3..5], &[44, 17]);
        unsafe { s.restore(&backend, &checkpoint) }.unwrap();
        s.step(&backend).unwrap();
        assert_eq!(&s.read(&backend, "State").unwrap()[..2], &[44, 17]);
        s.reset(&backend).unwrap();
        assert_eq!(&s.read(&backend, "State").unwrap()[..2], &[42, 17]);
        s.step(&backend).unwrap();
        assert_eq!(&s.read(&backend, "State").unwrap()[..2], &[43, 17]);
        for pair in [["State", "State"], ["State", "Input"], ["State", "missing"]] {
            let mut v = swap_variant(8, 0);
            v.manifest.swaps = vec![pair.map(String::from)];
            assert!(v.manifest.validate().is_err());
        }
        let mut v = swap_variant(8, 0);
        v.manifest.swaps.push(v.manifest.swaps[0].clone());
        assert!(v.manifest.validate().is_err());
        v.manifest.schema = 1;
        assert!(v.manifest.validate().is_err());
    }
    #[test]
    fn completed_gate_can_skip_and_reenable_a_commit_stage() {
        let backend = Metal::new().unwrap();
        let mut v = variant(8, 0);
        v.manifest.schema = 2;
        v.manifest.conditions.insert(
            "commit".into(),
            FrameCondition {
                gate: "pairs".into(),
                minimum: 6,
            },
        );
        let mut s = unsafe { FrameSession::new(&backend, vec![v]) }.unwrap();
        s.step(&backend).unwrap();
        assert_eq!(s.read(&backend, "State").unwrap()[0], 42);
        unsafe {
            *s.storage["Input"].contents().cast::<u32>() = 6;
        }
        s.step(&backend).unwrap();
        assert_eq!(s.read(&backend, "State").unwrap()[0], 43);
        for (stage, gate, minimum) in [
            ("count", "pairs", 1),
            ("commit", "missing", 1),
            ("commit", "pairs", 0),
            ("missing", "pairs", 1),
        ] {
            let mut v = variant(8, 0);
            v.manifest.schema = 2;
            v.manifest.conditions.insert(
                stage.into(),
                FrameCondition {
                    gate: gate.into(),
                    minimum,
                },
            );
            assert!(v.manifest.validate().is_err());
        }
    }
    #[test]
    fn same_stage_limits_retry_before_commit() {
        let backend = Metal::new().unwrap();
        let mut small = variant(8, 0);
        let mut large = variant(8, 3);
        for (v, capacity) in [(&mut small, 1), (&mut large, 8)] {
            let mut gate = v.manifest.gates[0].clone();
            gate.id = "colors".into();
            gate.capacity = capacity;
            v.manifest.gates.push(gate);
        }
        let mut session = unsafe { FrameSession::new(&backend, vec![small]) }.unwrap();
        let before = serde_json::to_value(session.checkpoint(&backend).unwrap()).unwrap();
        assert!(
            matches!(session.step(&backend), Err(StepError::Capacity {gate, ..}) if gate == "colors")
        );
        assert_eq!(
            serde_json::to_value(session.checkpoint(&backend).unwrap()).unwrap(),
            before
        );
        unsafe { session.extend(large) }.unwrap();
        let stats = session.step(&backend).unwrap();
        assert_eq!(
            stats.counts,
            BTreeMap::from([("pairs".into(), 5), ("colors".into(), 5)])
        );
        assert_eq!((stats.step, stats.retries), (1, 1));
        assert_eq!(
            session.read(&backend, "State").unwrap(),
            vec![
                0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 43, 17, 0xDEADBEEF, 0xDEADBEEF
            ]
        );
    }
    #[test]
    fn checkpoint_restore_reset_and_rejections_are_transactional() {
        let backend = Metal::new().unwrap();
        let mut session =
            unsafe { FrameSession::new(&backend, vec![variant(1, 0), variant(8, 3)]) }.unwrap();
        let initial = serde_json::to_value(session.checkpoint(&backend).unwrap()).unwrap();
        session.step(&backend).unwrap();
        let checkpoint = session.checkpoint(&backend).unwrap();
        session.step(&backend).unwrap();
        let expected = session.read(&backend, "State").unwrap();
        for mode in 0..5 {
            let mut bad = checkpoint.clone();
            match mode {
                0 => bad.schema = 0,
                1 => bad.steps = 3,
                2 => {
                    bad.capacities.insert("pairs".into(), 9);
                }
                3 => {
                    bad.regions.insert("missing".into(), vec![1]);
                }
                _ => bad.regions.get_mut("pose_and_velocity").unwrap().push(0),
            }
            assert!(unsafe { session.restore(&backend, &bad) }.is_err());
            assert_eq!(session.steps, 2);
            assert_eq!(session.read(&backend, "State").unwrap(), expected);
        }
        assert_eq!(
            unsafe { session.restore(&backend, &checkpoint) }.unwrap(),
            1
        );
        assert_eq!(session.steps, 1);
        session.step(&backend).unwrap();
        assert_eq!(session.read(&backend, "State").unwrap(), expected);
        let mut fresh = unsafe { FrameSession::new(&backend, vec![variant(16, 7)]) }.unwrap();
        unsafe { fresh.restore(&backend, &checkpoint) }.unwrap();
        assert_eq!(fresh.step(&backend).unwrap().step, 2);
        assert_eq!(
            &fresh.read(&backend, "State").unwrap()[7..9],
            &expected[3..5]
        );
        // Reset must retain the initial loaded pipelines, even after growth.
        // A broken source would make an accidental recompilation fail here.
        for source in session.variants[0].sources.values_mut() {
            *source = "intentionally invalid MSL".into();
        }
        session.reset(&backend).unwrap();
        assert_eq!(
            serde_json::to_value(session.checkpoint(&backend).unwrap()).unwrap(),
            initial
        );
        session.step(&backend).unwrap();
        assert_eq!(
            serde_json::to_value(session.checkpoint(&backend).unwrap()).unwrap(),
            serde_json::to_value(checkpoint).unwrap()
        );
    }
    #[test]
    fn growth_retries_before_commit_and_relocates_state_on_device() {
        let backend = Metal::new().unwrap();
        let mut large = variant(8, 3);
        large.initial.get_mut("State").unwrap()[3..5].fill(0);
        let mut session = unsafe {
            FrameSession::new(&backend, vec![variant(1, 0), variant(3, 1), large]).unwrap()
        };
        let result = session.step(&backend).unwrap();
        assert_eq!(
            result,
            FrameStats {
                step: 1,
                retries: 1,
                variant: 2,
                counts: BTreeMap::from([("pairs".into(), 5)])
            }
        );
        assert_eq!(
            session.read(&backend, "State").unwrap(),
            vec![
                0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 43, 17, 0xdeadbeef, 0xdeadbeef
            ]
        );
        assert_eq!(session.step(&backend).unwrap().retries, 0);
        assert_eq!(session.read(&backend, "State").unwrap()[3], 44);
        assert_eq!(session.step(&backend), Err(StepError::StepLimit));
        assert_eq!(session.read(&backend, "State").unwrap()[3], 44);
    }
    #[test]
    fn unbounded_session_accepts_capacity_compiled_after_exhaustion() {
        let backend = Metal::new().unwrap();
        let mut small = variant(1, 0);
        small.manifest.max_steps = None;
        let mut frame = unsafe { FrameSession::new(&backend, vec![small]) }.unwrap();
        assert!(matches!(
            frame.step(&backend),
            Err(StepError::Capacity { .. })
        ));
        let mut large = variant(8, 3);
        large.manifest.max_steps = None;
        unsafe { frame.extend(large) }.unwrap();
        assert_eq!(frame.steps, 0);
        assert_eq!(frame.read(&backend, "State").unwrap()[0], 42);
        assert_eq!(frame.step(&backend).unwrap().retries, 1);
        for _ in 0..9 {
            frame.step(&backend).unwrap();
        }
        assert_eq!(frame.steps, 10);
        assert_eq!(&frame.read(&backend, "State").unwrap()[3..5], &[52, 17]);
        frame.steps = u64::MAX;
        assert_eq!(frame.step(&backend), Err(StepError::StepLimit));
        assert_eq!(frame.read(&backend, "State").unwrap()[3], 52);
    }
    #[test]
    fn rejected_extension_preserves_the_current_session() {
        let backend = Metal::new().unwrap();
        let mut frame = unsafe { FrameSession::new(&backend, vec![variant(8, 0)]) }.unwrap();
        assert!(unsafe { frame.extend(variant(3, 0)) }.is_err());
        assert_eq!(frame.variants.len(), 1);
        assert_eq!(frame.read(&backend, "State").unwrap()[0], 42);
        assert_eq!(frame.step(&backend).unwrap().step, 1);
    }
    #[test]
    fn unbounded_manifest_cannot_silently_overrun_a_trace() {
        let mut v = variant(8, 0);
        v.manifest.max_steps = None;
        v.manifest.validate().unwrap();
        v.manifest.program.storage_buffers.push("Trajectory".into());
        v.manifest
            .program
            .buffer_bytes
            .insert("Trajectory".into(), 4);
        assert!(v.manifest.validate().is_err());
        // Option fields normally default to None; our deserializer requires the field.
        let missing = r#"{"schema":1,"program":{"schema":2,"stages":[],"storage_buffers":[],"buffer_bytes":{},"iterations":1},"gates":[],"persistent":{}}"#;
        assert!(serde_json::from_str::<FrameManifest>(missing).is_err());
    }
    #[test]
    fn input_updates_survive_relocation_and_reject_invalid_ranges() {
        let backend = Metal::new().unwrap();
        let mut frame =
            unsafe { FrameSession::new(&backend, vec![variant(1, 0), variant(8, 3)]) }.unwrap();
        unsafe { frame.write_persistent(&backend, "pose_and_velocity", 0, &[100, 200]) }.unwrap();
        for (name, offset, words) in [
            ("missing", 0, &[9][..]),
            ("pose_and_velocity", usize::MAX, &[9][..]),
            ("pose_and_velocity", 1, &[9, 10][..]),
        ] {
            assert!(unsafe { frame.write_persistent(&backend, name, offset, words) }.is_err());
            assert_eq!(
                frame.read(&backend, "State").unwrap(),
                [100, 200, 0xdeadbeef, 0xdeadbeef]
            );
        }
        assert_eq!(frame.step(&backend).unwrap().retries, 1);
        assert_eq!(
            frame.read(&backend, "State").unwrap(),
            [
                0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 101, 200, 0xdeadbeef, 0xdeadbeef
            ]
        );
        unsafe { frame.write_persistent(&backend, "pose_and_velocity", 1, &[300]) }.unwrap();
        frame.step(&backend).unwrap();
        assert_eq!(frame.read(&backend, "State").unwrap()[3..5], [102, 300]);
        frame.poisoned = true;
        assert!(unsafe { frame.write_persistent(&backend, "pose_and_velocity", 0, &[0]) }.is_err());
        assert_eq!(frame.read(&backend, "State").unwrap()[3], 102);
    }
    #[test]
    fn exhausted_capacity_never_advances_physics_or_frame_count() {
        let backend = Metal::new().unwrap();
        let mut session = unsafe { FrameSession::new(&backend, vec![variant(1, 0)]).unwrap() };
        for _ in 0..2 {
            assert_eq!(
                session.step(&backend),
                Err(StepError::Capacity {
                    gate: "pairs".into(),
                    required: 5,
                    capacity: 1
                })
            );
            assert_eq!(session.steps, 0);
            assert_eq!(
                session.read(&backend, "State").unwrap(),
                vec![42, 17, 0xdeadbeef, 0xdeadbeef]
            );
        }
    }
    #[test]
    fn invalid_gates_regions_and_variant_meanings_are_rejected() {
        let mut v = variant(1, 0);
        v.manifest.gates[0].after_stage = "commit".into();
        assert!(v.validate().is_err());
        let mut v = variant(1, 0);
        v.manifest.gates[0].counters[0].word = 1;
        assert!(v.validate().is_err());
        let mut v = variant(1, 0);
        let duplicate = v.manifest.gates[0].counters[0].clone();
        v.manifest.gates[0].counters.push(duplicate);
        assert!(v.validate().is_err());
        let mut v = variant(1, 0);
        let r = v.manifest.persistent["pose_and_velocity"].clone();
        v.manifest.persistent.insert("alias".into(), r);
        assert!(v.validate().is_err());
        let mut v = variant(1, 0);
        v.manifest
            .persistent
            .get_mut("pose_and_velocity")
            .unwrap()
            .word_start = usize::MAX;
        assert!(v.validate().is_err());
        let backend = Metal::new().unwrap();
        assert!(
            unsafe { FrameSession::new(&backend, vec![variant(8, 0), variant(3, 1)]) }.is_err()
        );
        let mut v = variant(8, 2);
        v.manifest
            .persistent
            .get_mut("pose_and_velocity")
            .unwrap()
            .word_len = 1;
        assert!(unsafe { FrameSession::new(&backend, vec![variant(1, 0), v]) }.is_err());
    }
    #[test]
    fn failed_growth_keeps_the_previous_frame_available() {
        let backend = Metal::new().unwrap();
        let mut larger = variant(8, 3);
        let correct = larger.sources["commit.metal"].clone();
        larger
            .sources
            .insert("commit.metal".into(), "not valid metal".into());
        let mut frame =
            unsafe { FrameSession::new(&backend, vec![variant(1, 0), larger]) }.unwrap();
        assert!(matches!(frame.step(&backend), Err(StepError::Backend(_))));
        assert_eq!(frame.current, 0);
        assert_eq!(frame.steps, 0);
        assert!(!frame.poisoned);
        assert_eq!(
            frame.read(&backend, "State").unwrap(),
            [42, 17, 0xdeadbeef, 0xdeadbeef]
        );
        frame.variants[1]
            .sources
            .insert("commit.metal".into(), correct);
        assert_eq!(frame.step(&backend).unwrap().retries, 1);
        assert_eq!(frame.read(&backend, "State").unwrap()[3..5], [43, 17]);
    }

    #[test]
    fn stage_ranges_run_once_and_reject_invalid_ranges_before_writes() {
        let backend = Metal::new().unwrap();
        let v = variant(8, 0);
        let program = NativeProgram::new(&backend, v.manifest.program.clone(), &v.sources).unwrap();
        let storage = upload(&backend, &v).unwrap();
        let refs = storage
            .iter()
            .map(|(n, b)| (n.clone(), b.as_ref()))
            .collect();
        unsafe {
            assert!(program.dispatch_stages(&backend, &refs, 0..3).is_err());
            assert!(
                program
                    .dispatch_stages(&backend, &refs, std::ops::Range { start: 2, end: 1 })
                    .is_err()
            );
            assert_eq!(*storage["State"].contents().cast::<u32>(), 42);
            program.dispatch_stages(&backend, &refs, 1..2).unwrap();
            assert_eq!(*storage["State"].contents().cast::<u32>(), 43);
            assert_eq!(*storage["Count"].contents().cast::<u32>(), 0);
        }
    }

    #[test]
    fn failed_dispatch_requires_explicit_recovery() {
        let backend = Metal::new().unwrap();
        let mut session = unsafe { FrameSession::new(&backend, vec![variant(8, 0)]).unwrap() };
        session.storage.remove("Input");
        assert!(matches!(session.step(&backend), Err(StepError::Backend(_))));
        assert_eq!(session.step(&backend), Err(StepError::Poisoned));
        assert_eq!(session.steps, 0);
    }
}
