use crate::math::Vec3;
use crate::rapier::geometry::{Group, InteractionGroups};
use crate::shaders::bounding_volumes::Aabb;
use crate::shaders::broad_phase::{CollisionPair, GpuBfFindPairs, GpuBfFindPairsSerial};
use crate::shaders::dynamics::RbdSimParams;
use crate::shaders::utils::BatchIndices;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use std::collections::BTreeSet;
use vortx::tensor::Tensor;
#[derive(Shader)]
struct Kernels {
    pairs: GpuBfFindPairs,
    serial: GpuBfFindPairsSerial,
}
#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a GPU"]
async fn test_batched_pair_packing_equivalence() {
    #[cfg(feature = "metal")]
    let backend = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
    #[cfg(not(feature = "metal"))]
    let backend = GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap());
    let kernels = Kernels::from_backend(&backend).unwrap();
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let params = Tensor::scalar(&backend, RbdSimParams::tgs_soft(), BufferUsages::UNIFORM).unwrap();
    for n in [1u32, 7, 32, 64] {
        for batches in [1u32, 5, 129] {
            let mut aabbs = vec![];
            let mut groups = vec![InteractionGroups::all(); (n * batches) as usize];
            let mut filters = vec![[0u32; 2]; (n * batches) as usize];
            for b in 0..batches {
                for i in 0..n {
                    let center = Vec3::new(
                        ((i * 7 + b) % 13) as f32 * 0.17,
                        ((i * 3 + b) % 7) as f32 * 0.1,
                        0.0,
                    );
                    aabbs.push(Aabb::new(
                        center - Vec3::splat(0.31),
                        center + Vec3::splat(0.31),
                    ));
                    let at = (i * batches + b) as usize;
                    groups[at].memberships = Group::from_bits_truncate(1 << (i % 3));
                    groups[at].filter = Group::from_bits_truncate(if b % 2 == 0 { 7 } else { 3 });
                    filters[at] = [i / 2, if i % 4 < 2 { 0 } else { 1 + i / 8 }];
                }
            }
            let boxes = Tensor::vector(&backend, aabbs, storage).unwrap();
            let groups = Tensor::vector(&backend, groups, storage).unwrap();
            let filters = Tensor::vector(&backend, filters, storage).unwrap();
            let mut reference = BTreeSet::new();
            let mut expected_count = 0;
            for capacity in [(n * n * batches).max(1), 3] {
                let batch = Tensor::scalar(
                    &backend,
                    BatchIndices {
                        num_batches: batches,
                        colliders_batch_capacity: n,
                        colliders_len: n,
                        collision_pairs_capacity: capacity,
                        ..Default::default()
                    },
                    BufferUsages::UNIFORM,
                )
                .unwrap();
                for serial in [false, true] {
                    let mut pairs = Tensor::vector(
                        &backend,
                        vec![CollisionPair::default(); capacity as usize],
                        storage,
                    )
                    .unwrap();
                    let mut count = Tensor::scalar(&backend, 0u32, storage).unwrap();
                    let mut enc = backend.begin_encoding();
                    {
                        let mut pass = enc.begin_pass("pair-packing", None);
                        if serial {
                            kernels
                                .serial
                                .call(
                                    &mut pass,
                                    n * batches,
                                    &boxes,
                                    &mut pairs,
                                    &mut count,
                                    &groups,
                                    &batch,
                                    &filters,
                                    &params,
                                )
                                .unwrap();
                        } else {
                            kernels
                                .pairs
                                .call(
                                    &mut pass,
                                    n * n * batches,
                                    &boxes,
                                    &mut pairs,
                                    &mut count,
                                    &groups,
                                    &batch,
                                    &filters,
                                    &params,
                                )
                                .unwrap();
                        }
                    }
                    backend.submit(enc).unwrap();
                    let count: Vec<u32> = backend.slow_read_vec(count.buffer()).await.unwrap();
                    let output: Vec<CollisionPair> =
                        backend.slow_read_vec(pairs.buffer()).await.unwrap();
                    let output: BTreeSet<_> = output[..count[0].min(capacity) as usize]
                        .iter()
                        .map(|p| (p.colliders.x, p.colliders.y))
                        .collect();
                    assert_eq!(
                        output.len(),
                        count[0].min(capacity) as usize,
                        "duplicate pairs"
                    );
                    if capacity == n * n * batches && !serial {
                        reference = output.clone();
                        expected_count = count[0];
                    }
                    assert_eq!(count[0], expected_count, "n={n} batches={batches}");
                    assert!(output.is_subset(&reference));
                    if capacity >= count[0] {
                        assert_eq!(output, reference);
                    }
                }
            }
        }
    }
}
