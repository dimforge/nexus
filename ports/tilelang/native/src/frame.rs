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
pub struct FrameManifest {
    pub schema: u32,
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
        if self.schema != 1
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
            if end >= self.program.stages.len() || ends.last().is_some_and(|last| *last >= end) {
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
    current: usize,
    program: NativeProgram,
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
    if old.gates.len() != new.gates.len()
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
        for v in &variants {
            v.validate()?;
        }
        for pair in variants.windows(2) {
            validate_growth(&pair[0].manifest, &pair[1].manifest)?;
        }
        let first = &variants[0];
        let program = NativeProgram::new(backend, first.manifest.program.clone(), &first.sources)?;
        let storage = upload(backend, first)?;
        Ok(Self {
            variants,
            current: 0,
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
        variant.validate()?;
        validate_growth(&self.variants.last().unwrap().manifest, &variant.manifest)?;
        self.variants.push(variant);
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
        let program = NativeProgram::new(backend, new.manifest.program.clone(), &new.sources)?;
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
            let ends = m.validate().map_err(StepError::Backend)?;
            let mut start = 0;
            let mut overflow = None;
            let mut counts = BTreeMap::new();
            for (gate, end) in m.gates.iter().zip(&ends) {
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
                self.program
                    .dispatch_stages(backend, &self.refs(), start..m.program.stages.len())
            } {
                self.poisoned = true;
                return Err(StepError::Backend(error));
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
            name: "count".into(),
            source: "count.metal".into(),
            bindings: local(vec!["Input".into(), "Count".into()], "count"),
            buffer_map: BTreeMap::from([
                ("Input".into(), "Input".into()),
                ("Count".into(), "Count".into()),
            ]),
        };
        let commit = ProgramStage {
            name: "commit".into(),
            source: "commit.metal".into(),
            bindings: local(vec!["State".into()], "commit"),
            buffer_map: BTreeMap::from([("State".into(), "State".into())]),
        };
        let mut state = vec![0xdeadbeef; offset + 4];
        state[offset..offset + 2].copy_from_slice(&[42, 17]);
        FrameVariant{
            manifest:FrameManifest{schema:1,program:ProgramManifest{schema:2,storage_buffers:sizes.keys().cloned().collect(),buffer_bytes:sizes,iterations:1,stages:vec![count,commit]},
                gates:vec![CapacityGate{id:"pairs".into(),after_stage:"count".into(),counters:vec![Counter{buffer:"Count".into(),word:0}],capacity}],
                persistent:BTreeMap::from([("pose_and_velocity".into(),PersistentRegion{buffer:"State".into(),word_start:offset,word_len:2})]),max_steps:Some(2)},
            sources:BTreeMap::from([
                ("count.metal".into(),"#include <metal_stdlib>\nusing namespace metal;\nkernel void count(device uint* input [[buffer(0)]], device uint* count [[buffer(1)]]) {count[0]=input[0];}".into()),
                ("commit.metal".into(),format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void commit(device uint* state [[buffer(0)]]) {{state[{offset}]+=1;}}"))]),
            initial:BTreeMap::from([("State".into(),state),("Input".into(),vec![5]),("Count".into(),vec![0])]),
        }
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
