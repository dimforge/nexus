//! Native cooperative color bucket controller shared by fixture exporters.
use crate::{dispatch, prefix_program};
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan,
    dynamics::{
        ColorStats, ContactLink, GpuColorBucketsCount, GpuColorBucketsReset,
        GpuColorBucketsScatter, GpuColorDispatchGrid,
    },
    utils::BatchIndices,
};
pub(crate) struct State {
    pub(crate) colors: GpuBuffer<u32>,
    pub(crate) links: GpuBuffer<ContactLink>,
    pub(crate) plan: GpuBuffer<ContactPlan>,
    pub(crate) buckets: GpuBuffer<u32>,
    pub(crate) batch: GpuBuffer<BatchIndices>,
    pub(crate) sorted: GpuBuffer<ContactLink>,
    pub(crate) indices: GpuBuffer<u32>,
    pub(crate) stats: GpuBuffer<ColorStats>,
    pub(crate) indirect: GpuBuffer<[u32; 3]>,
    pub(crate) aux: Vec<Vec<u32>>,
    pub(crate) strides: [u32; 5],
    pub(crate) levels: usize,
    pub(crate) groups: u32,
}
impl State {
    pub(crate) fn reset(&mut self) {
        let k = GpuColorBucketsReset::from_bytes(
            &GpuBackend::Cpu,
            &[],
            GpuColorBucketsReset::ENTRY_POINT,
        )
        .unwrap();
        k.call(
            &mut GpuPass::Cpu(None),
            dispatch::workgroups([
                (self.buckets.unwrap_slice().len() as u32).div_ceil(64),
                1,
                1,
            ]),
            &mut self.buckets,
        )
        .unwrap();
    }
    pub(crate) fn count(&mut self) {
        let k = GpuColorBucketsCount::from_bytes(
            &GpuBackend::Cpu,
            &[],
            GpuColorBucketsCount::ENTRY_POINT,
        )
        .unwrap();
        k.call(
            &mut GpuPass::Cpu(None),
            dispatch::workgroups([self.groups, 1, 1]),
            &self.colors,
            &self.links,
            &self.plan,
            &mut self.buckets,
            &self.batch,
            &mut self.sorted,
        )
        .unwrap();
    }
    pub(crate) fn scan(&mut self) {
        let GpuBuffer::Cpu(ref mut data) = self.buckets else {
            unreachable!()
        };
        prefix_program::scan(data, &mut self.aux, &self.strides, self.levels);
    }
    pub(crate) fn scatter(&mut self) {
        let k = GpuColorBucketsScatter::from_bytes(
            &GpuBackend::Cpu,
            &[],
            GpuColorBucketsScatter::ENTRY_POINT,
        )
        .unwrap();
        k.call(
            &mut GpuPass::Cpu(None),
            dispatch::workgroups([self.groups, 1, 1]),
            &self.colors,
            &self.links,
            &self.plan,
            &mut self.buckets,
            &mut self.sorted,
            &self.batch,
            &mut self.indices,
        )
        .unwrap();
    }
    pub(crate) fn dispatch(&mut self) {
        let k = GpuColorDispatchGrid::from_bytes(
            &GpuBackend::Cpu,
            &[],
            GpuColorDispatchGrid::ENTRY_POINT,
        )
        .unwrap();
        k.call(
            &mut GpuPass::Cpu(None),
            dispatch::workgroups([1, 1, 1]),
            &self.buckets,
            &mut self.stats,
            &mut self.indirect,
            &self.batch,
        )
        .unwrap();
    }
}
