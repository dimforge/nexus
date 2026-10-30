//! Composition of native hub shaders; no replacement physics.
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::{dynamics::*, glamx::UVec3, utils::BatchIndices};
use std::{fs, path::Path};
pub const BUFFERS: [&str; 5] = [
    "HubList",
    "SlotBody",
    "SlotConstraint",
    "SlotsIndirect",
    "AverageIndirect",
];
pub struct Storage {
    list: Vec<u32>,
    body: Vec<u32>,
    constraint: Vec<u32>,
    slots: [[u32; 3]; 3],
    average: [[u32; 3]; 3],
}
impl Storage {
    pub fn new(pool: usize) -> Self {
        Self {
            list: vec![0xdeadbeef; pool / HUB_MIN_CONSTRAINTS as usize + 4],
            body: vec![0; pool + 3],
            constraint: vec![0; pool + 3],
            slots: [[0xdeadbeef; 3]; 3],
            average: [[0xdeadbeef; 3]; 3],
        }
    }
    pub fn write(&self, dir: &Path, suffix: &str) {
        for (name, words) in BUFFERS.into_iter().zip([
            &self.list[..],
            &self.body,
            &self.constraint,
            bytemuck::cast_slice(&self.slots),
            bytemuck::cast_slice(&self.average),
        ]) {
            fs::write(
                dir.join(format!("{name}{suffix}.bin")),
                bytemuck::cast_slice(words),
            )
            .unwrap();
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn assign(
        &mut self,
        counts: &[u32],
        ids: &[u32],
        mass: &[WorldMassProperties],
        groups: &[u32],
        first: &mut [u32],
        hubs: &mut HubCounts,
        links: &mut [ContactLink],
        batch: &BatchIndices,
    ) {
        let workgroups = (counts.len() as u32).div_ceil(64).max(1);
        for i in 0..workgroups * 64 {
            gpu_hub_assign(
                UVec3::new(i, 0, 0),
                UVec3::new(workgroups, 1, 1),
                counts,
                mass,
                groups,
                first,
                &mut self.list,
                hubs,
                batch,
            );
        }
        gpu_hub_dispatch(hubs, &mut self.slots, &mut self.average);
        let n = self.average[0][0];
        for g in 0..n {
            for lane in 0..64 {
                gpu_hub_split_constraints(
                    UVec3::new(lane, 0, 0),
                    UVec3::new(g, 0, 0),
                    UVec3::new(n, 1, 1),
                    counts,
                    ids,
                    first,
                    links,
                    &mut self.body,
                    &mut self.constraint,
                    hubs,
                    &self.list,
                );
            }
        }
    }
    pub fn scatter(&self, velocities: &mut [Velocity], hubs: &HubCounts) {
        for i in 0..self.slots[0][0] * 64 {
            gpu_hub_scatter(UVec3::new(i, 0, 0), velocities, &self.body, hubs);
        }
    }
    pub fn average(
        &self,
        velocities: &mut Vec<Velocity>,
        first: &[u32],
        counts: &[u32],
        hubs: &HubCounts,
    ) {
        if self.average[0][0] == 0 {
            return;
        }
        let kernel =
            GpuHubAverage::from_bytes(&GpuBackend::Cpu, &[], GpuHubAverage::ENTRY_POINT).unwrap();
        let mut v = GpuBuffer::Cpu(std::mem::take(velocities));
        kernel
            .call(
                &mut GpuPass::Cpu(None),
                crate::dispatch::workgroups(self.average[0]),
                &mut v,
                &GpuBuffer::Cpu(self.list.clone()),
                &GpuBuffer::Cpu(first.to_vec()),
                &GpuBuffer::Cpu(counts.to_vec()),
                &GpuBuffer::Cpu(vec![*hubs]),
            )
            .unwrap();
        let GpuBuffer::Cpu(v) = v else { unreachable!() };
        *velocities = v;
    }
}
