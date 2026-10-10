//! The same checks run natively and in the browser regression harness.
use crate::shaders::Vector;
use crate::shaders::bounding_volumes::Aabb;
use crate::shaders::broad_phase::{
    GpuLbvhRefitChunks, GpuLbvhRefitFrontier, GpuLbvhRefitPlan, LbvhNode,
};
use crate::shaders::utils::BatchIndices;
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend};
use vortx::tensor::Tensor;

fn tree_fixture(n: usize, capacity: usize, batch: usize) -> Vec<LbvhNode> {
    fn build(
        nodes: &mut [LbvhNode],
        next: &mut usize,
        n: usize,
        first: usize,
        end: usize,
        parent: u32,
        batch: usize,
        depth: usize,
    ) -> usize {
        if end - first == 1 {
            let id = n - 1 + first;
            let center = Vector::X * ((first * 13 % 97) as f32 + batch as f32 * 128.0)
                + Vector::Y * (first % 31) as f32;
            let half = Vector::splat((first % 5 + 1) as f32 * 0.125);
            nodes[id] = LbvhNode {
                aabb: Aabb::new(center - half, center + half),
                first_leaf: first as u32,
                last_leaf: first as u32,
                parent,
                max_extent: half.x * 2.0,
                ..Default::default()
            };
            return id;
        }
        let id = *next;
        *next += 1;
        // Distinct frontier sizes per batch, including a non-balanced tree.
        let split = if batch % 2 == 1 && depth < 12 {
            1
        } else {
            (end - first) / 2
        };
        let a = build(
            nodes,
            next,
            n,
            first,
            first + split,
            id as u32,
            batch,
            depth + 1,
        );
        let b = build(
            nodes,
            next,
            n,
            first + split,
            end,
            id as u32,
            batch,
            depth + 1,
        );
        nodes[id] = LbvhNode {
            aabb: nodes[a].aabb.merged(&nodes[b].aabb),
            left: a as u32,
            right: b as u32,
            parent,
            first_leaf: first as u32,
            last_leaf: (end - 1) as u32,
            max_extent: nodes[a].max_extent.max(nodes[b].max_extent),
            ..Default::default()
        };
        id
    }
    let mut nodes = vec![LbvhNode::default(); 2 * capacity];
    if n > 0 {
        build(&mut nodes, &mut 0, n, 0, n, 0, batch, 0);
    }
    nodes
}

/// Validate every node against CPU subtree bounds, including empty and unequal frontiers.
pub async fn check_refit(backend: &GpuBackend) -> usize {
    let chunks = GpuLbvhRefitChunks::from_dir(backend, &crate::SPIRV_DIR).unwrap();
    let plan = GpuLbvhRefitPlan::from_dir(backend, &crate::SPIRV_DIR).unwrap();
    let frontier = GpuLbvhRefitFrontier::from_dir(backend, &crate::SPIRV_DIR).unwrap();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let mut cases = 0;
    for n in [0usize, 1, 2, 255, 256, 257, 513, 1025, 4097] {
        for nb in [1usize, 3] {
            let capacity = n + 7;
            let expected: Vec<_> = (0..nb).flat_map(|b| tree_fixture(n, capacity, b)).collect();
            let mut initial = expected.clone();
            let mut expected_counts = vec![0u32; nb];
            for (batch, expected_count) in expected_counts.iter_mut().enumerate() {
                let start = batch * capacity * 2;
                for id in 1..n.saturating_mul(2).saturating_sub(1) {
                    let node = &expected[start + id];
                    let parent = &expected[start + node.parent as usize];
                    if node.first_leaf / 256 == node.last_leaf / 256
                        && parent.first_leaf / 256 != parent.last_leaf / 256
                    {
                        *expected_count += 1;
                    }
                }
                for node in &mut initial[start..start + n.saturating_sub(1)] {
                    node.aabb = Aabb::default();
                    node.max_extent = 0.0;
                }
            }
            let mut tree = Tensor::vector(backend, initial, usage).unwrap();
            let mut entries =
                Tensor::vector(backend, vec![u32::MAX; capacity * nb], usage).unwrap();
            let mut counts = Tensor::vector(backend, vec![0u32; nb], usage).unwrap();
            let mut rounds =
                Tensor::scalar(backend, 777u32, usage | BufferUsages::UNIFORM).unwrap();
            let ids = Tensor::scalar(
                backend,
                BatchIndices {
                    num_batches: nb as u32,
                    colliders_len: n as u32,
                    colliders_batch_capacity: capacity as u32,
                    ..Default::default()
                },
                BufferUsages::UNIFORM,
            )
            .unwrap();
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("web-refit-regression", None);
            // An extra chunk exercises inactive workgroups and partial final chunks.
            chunks
                .call(
                    &mut pass,
                    [n as u32 + 256, nb as u32, 1],
                    &mut tree,
                    &mut entries,
                    &mut counts,
                    &ids,
                )
                .unwrap();
            plan.call(&mut pass, 1u32, &counts, &mut rounds, &ids)
                .unwrap();
            frontier
                .call(
                    &mut pass,
                    [1u32, nb as u32, 1],
                    &mut tree,
                    &entries,
                    &counts,
                    &ids,
                    &rounds,
                )
                .unwrap();
            drop(pass);
            backend.submit(encoder).unwrap();
            let actual = backend.slow_read_vec(tree.buffer()).await.unwrap();
            for (id, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.aabb.mins, b.aabb.mins, "n={n}, batches={nb}, node={id}");
                assert_eq!(a.aabb.maxs, b.aabb.maxs, "n={n}, batches={nb}, node={id}");
                assert_eq!(a.max_extent, b.max_extent, "n={n}, batches={nb}, node={id}");
            }
            assert_eq!(
                backend.slow_read_vec(counts.buffer()).await.unwrap(),
                expected_counts
            );
            assert_eq!(
                backend.slow_read_vec(rounds.buffer()).await.unwrap(),
                [expected_counts.iter().copied().max().unwrap().div_ceil(256)]
            );
            cases += 1;
        }
    }
    cases
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn web_preparation_refit_preserves_every_subtree() {
    let backend = GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap());
    assert_eq!(check_refit(&backend).await, 18);
}
