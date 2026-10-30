//! Shared live-Rust multi-dispatch prefix oracle for composed programs.
use crate::dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::utils::prefix_sum::{GpuAddDataGrp, GpuPrefixSumSweep};

/// The same 256-element hierarchy used by the native prefix controller.
/// Some isolated operation exporters use only `scan` with explicit fixture strides.
#[allow(dead_code)]
pub(crate) fn hierarchy(count: usize) -> ([u32; 5], usize) {
    let mut strides = [0u32; 5];
    strides[0] = count.try_into().expect("Prefix length exceeds u32");
    for i in 1..5 {
        strides[i] = strides[i - 1].div_ceil(256);
    }
    let levels = (1..5).find(|&i| strides[i] <= 1).unwrap();
    (strides, levels)
}

pub(crate) fn scan(counts: &mut Vec<u32>, aux: &mut [Vec<u32>], strides: &[u32; 5], levels: usize) {
    if strides[0] == 0 {
        return;
    }
    let sweep =
        GpuPrefixSumSweep::from_bytes(&GpuBackend::Cpu, &[], GpuPrefixSumSweep::ENTRY_POINT)
            .unwrap();
    let add = GpuAddDataGrp::from_bytes(&GpuBackend::Cpu, &[], GpuAddDataGrp::ENTRY_POINT).unwrap();
    let mut all: Vec<_> = std::iter::once(std::mem::take(counts))
        .chain(aux.iter_mut().map(std::mem::take))
        .map(GpuBuffer::Cpu)
        .collect();
    for level in 0..levels {
        let (left, right) = all.split_at_mut(level + 1);
        sweep
            .call(
                &mut GpuPass::Cpu(None),
                dispatch::workgroups([strides[level + 1], 1, 1]),
                &mut left[level],
                &mut right[0],
                &GpuBuffer::Cpu(vec![strides[level]]),
            )
            .unwrap();
    }
    for level in (0..levels - 1).rev() {
        let (left, right) = all.split_at_mut(level + 1);
        add.call(
            &mut GpuPass::Cpu(None),
            dispatch::workgroups([strides[level + 1], 1, 1]),
            &mut left[level],
            &right[0],
            &GpuBuffer::Cpu(vec![strides[level]]),
        )
        .unwrap();
    }
    for (dst, src) in std::iter::once(counts).chain(aux.iter_mut()).zip(all) {
        let GpuBuffer::Cpu(v) = src else {
            unreachable!()
        };
        *dst = v;
    }
}
