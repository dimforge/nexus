//! Cooperative native radix shader oracle, preserving every intermediate buffer.
use crate::dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::utils::radix_sort::{
    GpuInitSortBatched, GpuSortCount, GpuSortReduce, GpuSortScan, GpuSortScanAdd, GpuSortScatter,
    SortUniforms, gpu_init_sort_dispatch,
};
use std::collections::BTreeMap;
pub(crate) type Arrays = BTreeMap<String, GpuBuffer<u32>>;
pub(crate) fn exec(a: &mut Arrays, stage: &str, stride: u32, batches: u32, pass: Option<u32>) {
    let cfg = if stage == "radix_init_batched" {
        "InitConfig".to_string()
    } else {
        pass.map(|p| format!("Config{p}"))
            .unwrap_or("Config".to_string())
    };
    let v = a[&cfg].unwrap_slice();
    let config = GpuBuffer::Cpu(vec![SortUniforms {
        shift: v[0],
        max_keys_per_batch: v[1],
        has_aux: v[2],
    }]);
    let num = GpuBuffer::Cpu(a["NumKeys"].unwrap_slice().to_vec());
    let keys = GpuBuffer::Cpu(a["Keys"].unwrap_slice().to_vec());
    let vals = GpuBuffer::Cpu(a["Values"].unwrap_slice().to_vec());
    let aux = GpuBuffer::Cpu(a["Aux"].unwrap_slice().to_vec());
    let counts = GpuBuffer::Cpu(a["Counts"].unwrap_slice().to_vec());
    let reduced = GpuBuffer::Cpu(a["Reduced"].unwrap_slice().to_vec());
    let grid = dispatch::workgroups([stride.div_ceil(1024).max(1) + 1, batches, 1]);
    let reduce = dispatch::workgroups([16 * stride.div_ceil(1024).div_ceil(1024) + 1, batches, 1]);
    let mut gp = GpuPass::Cpu(None);
    match stage {
        "radix_dispatch" => {
            let mut sort = [0; 3];
            let mut red = [0; 3];
            gpu_init_sort_dispatch(num.unwrap_slice(), &mut sort, &mut red);
            a.get_mut("SortGrid").unwrap().unwrap_slice_mut()[..3].copy_from_slice(&sort);
            a.get_mut("ReduceGrid").unwrap().unwrap_slice_mut()[..3].copy_from_slice(&red);
        }
        "radix_init_batched" => {
            let mut out = a.remove("OutKeys").unwrap();
            let mut ov = a.remove("OutValues").unwrap();
            let mut ids = a.remove("BatchIds").unwrap();
            GpuInitSortBatched::from_bytes(&GpuBackend::Cpu, &[], GpuInitSortBatched::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    dispatch::workgroups([(stride * batches).div_ceil(256).max(1) + 1, 1, 1]),
                    &config,
                    &num,
                    &keys,
                    &vals,
                    &mut out,
                    &mut ov,
                    &mut ids,
                )
                .unwrap();
            a.insert("OutKeys".into(), out);
            a.insert("OutValues".into(), ov);
            a.insert("BatchIds".into(), ids);
        }
        "radix_count" => {
            GpuSortCount::from_bytes(&GpuBackend::Cpu, &[], GpuSortCount::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    grid,
                    &config,
                    &num,
                    &keys,
                    a.get_mut("Counts").unwrap(),
                )
                .unwrap();
        }
        "radix_reduce" => {
            GpuSortReduce::from_bytes(&GpuBackend::Cpu, &[], GpuSortReduce::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    reduce,
                    &config,
                    &num,
                    &counts,
                    a.get_mut("Reduced").unwrap(),
                )
                .unwrap();
        }
        "radix_scan" => {
            GpuSortScan::from_bytes(&GpuBackend::Cpu, &[], GpuSortScan::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    dispatch::workgroups([1, batches, 1]),
                    &num,
                    a.get_mut("Reduced").unwrap(),
                )
                .unwrap();
        }
        "radix_scan_add" => {
            GpuSortScanAdd::from_bytes(&GpuBackend::Cpu, &[], GpuSortScanAdd::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp,
                    reduce,
                    &config,
                    &num,
                    &reduced,
                    a.get_mut("Counts").unwrap(),
                )
                .unwrap();
        }
        "radix_scatter" => {
            let mut out = a.remove("OutKeys").unwrap();
            let mut ov = a.remove("OutValues").unwrap();
            let mut oa = a.remove("OutAux").unwrap();
            GpuSortScatter::from_bytes(&GpuBackend::Cpu, &[], GpuSortScatter::ENTRY_POINT)
                .unwrap()
                .call(
                    &mut gp, grid, &config, &num, &keys, &vals, &counts, &mut out, &mut ov, &aux,
                    &mut oa,
                )
                .unwrap();
            a.insert("OutKeys".into(), out);
            a.insert("OutValues".into(), ov);
            a.insert("OutAux".into(), oa);
        }
        _ => panic!("unknown stage {stage}"),
    }
}
pub(crate) fn swap_io(a: &mut Arrays) {
    for (x, y) in [
        ("Keys", "OutKeys"),
        ("Values", "OutValues"),
        ("Aux", "OutAux"),
    ] {
        let left = a.remove(x).unwrap();
        let right = a.remove(y).unwrap();
        a.insert(x.into(), right);
        a.insert(y.into(), left);
    }
}
pub(crate) const STAGES: [&str; 5] = [
    "radix_count",
    "radix_reduce",
    "radix_scan",
    "radix_scan_add",
    "radix_scatter",
];
