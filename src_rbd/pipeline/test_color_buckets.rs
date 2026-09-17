//! GPU checks for bucket counts, membership, and the generic scatter fallback.
use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::*;
use crate::shaders::utils::BatchIndices;
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use vortx::tensor::Tensor;

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn color_bucket_histograms_preserve_membership_and_bounds() {
    let backend = GpuBackend::WebGpu(
        WebGpu::new(Default::default(), Default::default())
            .await
            .unwrap(),
    );
    let count = GpuColorBucketsCount::from_dir(&backend, &crate::SPIRV_DIR).unwrap();
    let scatter = GpuColorBucketsScatter::from_dir(&backend, &crate::SPIRV_DIR).unwrap();
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    for n in [0u32, 1, 63, 64, 65, 193] {
        for nb in [1u32, 3] {
            for stride in [5u32, 67, 129, 133] {
                for mixed in [false, true] {
                    let mut colors = vec![0u32; n.max(1) as usize];
                    let mut input = vec![ContactLink::default(); n.max(1) as usize];
                    let mut expected = vec![Vec::<u32>::new(); (stride * nb) as usize];
                    for i in 0..n as usize {
                        let color = if mixed {
                            [0, 1, 2, 63, 64, 127, stride - 1, u32::MAX][i % 8]
                        } else {
                            1
                        };
                        colors[i] = color;
                        input[i].solver_body_a = (i as u32 % 17) * nb + (i as u32 / 7) % nb;
                        input[i].contact = i as u32;
                        // Some inactive slots, left out of the tiles.
                        input[i].len = u32::from(!mixed || i % 5 != 4);
                        if input[i].len == 0 {
                            continue;
                        }
                        // Active constraints left uncolored go to the last bucket.
                        let bucket = if color != 0 && color < stride - 1 {
                            color
                        } else {
                            stride - 1
                        };
                        expected[(bucket * nb + input[i].solver_body_a % nb) as usize]
                            .push(i as u32);
                    }
                    let links = Tensor::vector(&backend, input, storage).unwrap();
                    let colors = Tensor::vector(&backend, colors, storage).unwrap();
                    let plan = Tensor::scalar(
                        &backend,
                        ContactPlan {
                            bound: n,
                            ..Default::default()
                        },
                        BufferUsages::UNIFORM,
                    )
                    .unwrap();
                    let ids = Tensor::scalar(
                        &backend,
                        BatchIndices {
                            num_batches: nb,
                            solver_color_buckets_stride: stride,
                            ..Default::default()
                        },
                        BufferUsages::UNIFORM,
                    )
                    .unwrap();
                    let sizes = expected.iter().map(|x| x.len() as u32).collect::<Vec<_>>();
                    let mut starts = Vec::new();
                    let mut ends = Vec::new();
                    let mut total = 0u32;
                    for &size in &sizes {
                        starts.push(total);
                        total += size;
                        ends.push(total);
                    }
                    let mut grids = vec![64u32, n.max(1).div_ceil(64) * 64];
                    grids.dedup();
                    for grid in grids {
                        let mut counts =
                            Tensor::vector(&backend, vec![0u32; sizes.len()], storage).unwrap();
                        let mut cursors =
                            Tensor::vector(&backend, starts.clone(), storage).unwrap();
                        // Stale links, which the count deactivates past the constraints.
                        let stale = ContactLink {
                            len: 7,
                            ..Default::default()
                        };
                        let mut sorted_links =
                            Tensor::vector(&backend, vec![stale; n.max(1) as usize], storage)
                                .unwrap();
                        let mut constraint_indices =
                            Tensor::vector(&backend, vec![0u32; n.max(1) as usize], storage)
                                .unwrap();
                        let mut encoder = backend.begin_encoding();
                        let mut pass = encoder.begin_pass("color-bucket-test", None);
                        count
                            .call(
                                &mut pass,
                                grid,
                                &colors,
                                &links,
                                &plan,
                                &mut counts,
                                &ids,
                                &mut sorted_links,
                            )
                            .unwrap();
                        scatter
                            .call(
                                &mut pass,
                                grid,
                                &colors,
                                &links,
                                &plan,
                                &mut cursors,
                                &mut sorted_links,
                                &ids,
                                &mut constraint_indices,
                            )
                            .unwrap();
                        drop(pass);
                        backend.submit(encoder).unwrap();
                        assert_eq!(backend.slow_read_vec(counts.buffer()).await.unwrap(), sizes);
                        assert_eq!(backend.slow_read_vec(cursors.buffer()).await.unwrap(), ends);
                        let actual: Vec<ContactLink> =
                            backend.slow_read_vec(sorted_links.buffer()).await.unwrap();
                        assert!(
                            actual[total as usize..n as usize]
                                .iter()
                                .all(|l| l.len == 0)
                        );
                        let constraint_indices = backend
                            .slow_read_vec(constraint_indices.buffer())
                            .await
                            .unwrap();
                        for i in (0..n).filter(|&i| !mixed || i % 5 != 4) {
                            assert_eq!(actual[constraint_indices[i as usize] as usize].contact, i);
                        }
                        for (bucket, expected) in expected.iter().enumerate() {
                            let mut members: Vec<u32> = actual
                                [starts[bucket] as usize..ends[bucket] as usize]
                                .iter()
                                .map(|l| l.contact)
                                .collect();
                            members.sort_unstable();
                            assert_eq!(
                                &members, expected,
                                "n={n}, nb={nb}, stride={stride}, grid={grid}, bucket={bucket}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Exercise the exact SPIR-V -> WGSL path used by wgpu's browser backend.
/// Set NEXUS_WEB_TEST_DIR to export the browser regression page and shaders.
/// Native shader validation does not catch Dawn's stricter uniformity errors.
#[test]
fn color_bucket_shaders_translate_for_web() {
    use khal::re_exports::wgpu::naga;
    use std::mem::{offset_of, size_of};

    let out = std::env::var_os("NEXUS_WEB_TEST_DIR").map(std::path::PathBuf::from);
    if let Some(out) = &out {
        std::fs::create_dir_all(out).unwrap();
        std::fs::write(
            out.join("index.html"),
            include_str!("test_color_buckets.html"),
        )
        .unwrap();
    }
    let dim = if cfg!(feature = "dim3") { "3d" } else { "2d" };
    let mut descriptions = Vec::new();
    for model in ["links"] {
        for phase in ["count", "scatter"] {
            let file = format!("dynamics-color_buckets-gpu_color_buckets_{phase}.spv");
            let spirv = crate::SPIRV_DIR.get_file(&file).unwrap().contents();
            let module = naga::front::spv::parse_u8_slice(
                spirv,
                &naga::front::spv::Options {
                    adjust_coordinate_space: false,
                    strict_capabilities: true,
                    block_ctx_dump_prefix: None,
                },
            )
            .unwrap();
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let wgsl = naga::back::wgsl::write_string(
                &module,
                &info,
                naga::back::wgsl::WriterFlags::empty(),
            )
            .unwrap();
            if let Some(out) = &out {
                std::fs::write(out.join(format!("{dim}-{model}-{phase}.wgsl")), wgsl).unwrap();
            }
        }
        descriptions.push(format!(
            r#"{{"model":"{model}","linkSize":{},"bodyOffset":{},"lenOffset":{},"contactOffset":{}}}"#,
            size_of::<ContactLink>(),
            offset_of!(ContactLink, solver_body_a),
            offset_of!(ContactLink, len),
            offset_of!(ContactLink, contact),
        ));
    }
    if let Some(out) = &out {
        let metadata = format!(
            r#"{{"planSize":{},"boundOffset":{},"batchSize":{},"numBatchesOffset":{},"strideOffset":{},"models":[{}]}}"#,
            size_of::<ContactPlan>(),
            offset_of!(ContactPlan, bound),
            size_of::<BatchIndices>(),
            offset_of!(BatchIndices, num_batches),
            offset_of!(BatchIndices, solver_color_buckets_stride),
            descriptions.join(","),
        );
        std::fs::write(out.join(format!("{dim}.json")), metadata).unwrap();
    }
}
