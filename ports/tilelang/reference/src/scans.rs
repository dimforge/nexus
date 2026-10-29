//! Original cooperative Rust GPU scan, including its multi-dispatch hierarchy.
mod dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::utils::prefix_sum::{GpuAddDataGrp, GpuPrefixSumSweep, WORKGROUP_SIZE};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

fn write(root: &Path, name: &str, words: &[u32]) {
    fs::write(
        root.join(name),
        words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
fn fixture(root: &Path, stride: u32, batches: u32, op: &str, pattern: &str) -> Value {
    let name = format!("n{stride}-b{batches}-{pattern}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let mut strides = [0u32; 5];
    strides[0] = stride;
    for i in 1..5 {
        strides[i] = strides[i - 1].div_ceil(256);
    }
    let levels = (1..5).find(|&i| strides[i] <= 1).unwrap();
    let mut arrays = BTreeMap::new();
    let guard = 0xDEADBEEFu32;
    let mut data = vec![guard; (stride * batches) as usize + 3];
    for b in 0..batches {
        for i in 0..stride {
            data[(b * stride + i) as usize] = match pattern {
                "zero" => 0,
                "spikes" => {
                    if i % 256 == 0 || i % 256 == 255 {
                        b + 7
                    } else {
                        0
                    }
                }
                _ => (i * 17 + b * 71 + 3) % 23,
            };
        }
    }
    arrays.insert("Data".to_string(), GpuBuffer::Cpu(data));
    for i in 0..4 {
        let mut aux = vec![guard; (strides[i + 1] * batches) as usize + 3];
        if op == "prefix_add" && i == 0 {
            for b in 0..batches {
                for j in 0..strides[1] {
                    aux[(b * strides[1] + j) as usize] = b * 111 + j * 13;
                }
            }
        }
        arrays.insert(format!("Aux{i}"), GpuBuffer::Cpu(aux));
        arrays.insert(
            format!("Stride{i}"),
            GpuBuffer::Cpu(vec![strides[i], guard]),
        );
    }
    let words: BTreeMap<_, _> = arrays
        .iter()
        .map(|(n, v)| (n.clone(), v.unwrap_slice().len()))
        .collect();
    for (n, v) in &arrays {
        write(&dir, &format!("{n}.bin"), v.unwrap_slice());
    }
    let sweep =
        GpuPrefixSumSweep::from_bytes(&GpuBackend::Cpu, &[], GpuPrefixSumSweep::ENTRY_POINT)
            .unwrap();
    let add = GpuAddDataGrp::from_bytes(&GpuBackend::Cpu, &[], GpuAddDataGrp::ENTRY_POINT).unwrap();
    let mut execute = |level: usize, is_add: bool| {
        if stride == 0 {
            return;
        }
        let data_name = if level == 0 {
            "Data".to_string()
        } else {
            format!("Aux{}", level - 1)
        };
        let aux_name = format!("Aux{level}");
        let mut data = arrays.remove(&data_name).unwrap();
        let mut aux = arrays.remove(&aux_name).unwrap();
        let grid = dispatch::workgroups([strides[level + 1], batches, 1]);
        let uniform = &arrays[&format!("Stride{level}")];
        if is_add {
            add.call(&mut GpuPass::Cpu(None), grid, &mut data, &aux, uniform)
                .unwrap();
        } else {
            sweep
                .call(&mut GpuPass::Cpu(None), grid, &mut data, &mut aux, uniform)
                .unwrap();
        }
        arrays.insert(data_name, data);
        arrays.insert(aux_name, aux);
    };
    match op {
        "prefix_sweep" => execute(0, false),
        "prefix_add" => execute(0, true),
        "prefix_scan" => {
            for level in 0..levels {
                execute(level, false);
            }
            for level in (0..levels - 1).rev() {
                execute(level, true);
            }
        }
        _ => unreachable!(),
    }
    for (n, v) in &arrays {
        if !n.starts_with("Stride") {
            write(&dir, &format!("{n}.expected.bin"), v.unwrap_slice());
        }
    }
    json!({"name":name,"kind":op,"count":stride*batches,"stride":stride,"batches":batches,
        "strides":&strides[..4],"levels":levels,"iterations":1,"buffer_words":words})
}
fn main() {
    let path = std::env::args().nth(1).expect("Expected output directory");
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&json!({"word_bytes":4,"workgroup_size":WORKGROUP_SIZE}))
            .unwrap(),
    )
    .unwrap();
    for op in ["prefix_sweep", "prefix_add", "prefix_scan"] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let mut cases = Vec::new();
        for (stride, batches, pattern) in [
            (0, 1, "mixed"),
            (0, 3, "mixed"),
            (1, 1, "mixed"),
            (31, 2, "mixed"),
            (255, 3, "mixed"),
            (256, 1, "mixed"),
            (257, 2, "mixed"),
            (513, 3, "spikes"),
            (1025, 2, "zero"),
            (65537, 2, "mixed"),
        ] {
            cases.push(fixture(&dir, stride, batches, op, pattern));
        }
        fs::write(
            dir.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
