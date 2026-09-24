//! Compare the GPU allocator with the original serial policy, including overflow.
use crate::shaders::dynamics::{GpuMbConsOffsetsScan, MB_CONS_SLOT_RESERVE, MultibodyInfo};
use crate::shaders::utils::BatchIndices;
use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct LayoutShader {
    scan: GpuMbConsOffsetsScan,
}

fn serial_layout(
    infos: &mut [MultibodyInfo],
    counts: &[u32],
    indices: &[u32],
    capacity: u32,
) -> u32 {
    let demand = counts.iter().sum();
    let reserved: u32 = counts.iter().map(|c| (*c).min(MB_CONS_SLOT_RESERVE)).sum();
    let mut extra_budget = capacity.saturating_sub(reserved);
    let (mut acc, mut index_acc) = (0, 0);
    for ((mb, &count), &index_count) in infos.iter_mut().zip(counts).zip(indices) {
        let reserve = count.min(MB_CONS_SLOT_RESERVE);
        let extra = (count - reserve).min(extra_budget);
        extra_budget -= extra;
        let start = acc.min(capacity);
        let avail = (reserve + extra).min(capacity - start);
        mb.contact_constraint_start = start;
        mb.contact_constraint_count = avail;
        acc = start + avail;
        mb.contact_index_start = index_acc;
        mb.contact_index_len = index_count;
        index_acc += index_count;
    }
    demand
}

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a GPU device"]
async fn test_multibody_layout_parallel_scan() {
    #[cfg(feature = "metal")]
    let backend = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
    #[cfg(not(feature = "metal"))]
    let backend = GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap());
    let shader = LayoutShader::from_backend(&backend).unwrap();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let mut cases = 0;
    for (bodies, batches) in [
        (0, 1),
        (1, 1),
        (1, 31),
        (1, 255),
        (1, 256),
        (1, 257),
        (3, 257),
        (1, 4096),
        (1, 4097),
        (2, 8193),
    ] {
        let n = (bodies * batches) as usize;
        let counts: Vec<u32> = (0..n)
            .map(|i| {
                [
                    0,
                    1,
                    MB_CONS_SLOT_RESERVE - 1,
                    MB_CONS_SLOT_RESERVE,
                    MB_CONS_SLOT_RESERVE + 1,
                    63,
                    192,
                ][i % 7]
            })
            .collect();
        let indices: Vec<u32> = (0..n).map(|i| (i % 11) as u32).collect();
        let demand: u32 = counts.iter().sum();
        let reserved: u32 = counts.iter().map(|c| (*c).min(MB_CONS_SLOT_RESERVE)).sum();
        for capacity in [
            0,
            1,
            reserved / 2,
            reserved,
            reserved + 1,
            demand.saturating_sub(1),
            demand,
            demand + 97,
        ] {
            // Padding must stay untouched, even for partial chunks and empty input.
            let mut infos = vec![MultibodyInfo::default(); n + 3];
            for (i, info) in infos.iter_mut().enumerate() {
                info.first_link = i as u32;
                info.contact_constraint_start = 999;
                info.contact_constraint_count = 777;
                info.contact_index_start = 555;
                info.contact_index_len = 333;
            }
            let mut expected = infos.clone();
            let expected_demand = serial_layout(&mut expected[..n], &counts, &indices, capacity);
            let mut padded_counts = counts.clone();
            padded_counts.extend([111, 222, 333]);
            let mut padded_indices = indices.clone();
            padded_indices.extend([444, 555, 666]);
            let mut gpu_infos = Tensor::vector(&backend, infos, usage).unwrap();
            let mut gpu_counts = Tensor::vector(&backend, padded_counts, usage).unwrap();
            let mut gpu_indices = Tensor::vector(&backend, padded_indices.clone(), usage).unwrap();
            let mut gpu_demand = Tensor::vector(&backend, [u32::MAX], usage).unwrap();
            let params = Tensor::scalar(
                &backend,
                BatchIndices {
                    multibodies_len: bodies,
                    num_batches: batches,
                    mb_contact_constraints_capacity: capacity,
                    ..Default::default()
                },
                BufferUsages::UNIFORM,
            )
            .unwrap();
            let mut encoder = backend.begin_encoding();
            {
                let mut pass = encoder.begin_pass("layout-equivalence", None);
                shader
                    .scan
                    .call(
                        &mut pass,
                        1u32,
                        &mut gpu_infos,
                        &mut gpu_counts,
                        &mut gpu_indices,
                        &mut gpu_demand,
                        &params,
                    )
                    .unwrap();
            }
            backend.submit(encoder).unwrap();
            let actual: Vec<MultibodyInfo> =
                backend.slow_read_vec(gpu_infos.buffer()).await.unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&actual),
                bytemuck::cast_slice::<_, u8>(&expected),
                "infos: n={n} capacity={capacity}"
            );
            let actual_counts: Vec<u32> = backend.slow_read_vec(gpu_counts.buffer()).await.unwrap();
            assert_eq!(&actual_counts[..n], vec![0; n]);
            assert_eq!(&actual_counts[n..], &[111, 222, 333]);
            let actual_indices: Vec<u32> =
                backend.slow_read_vec(gpu_indices.buffer()).await.unwrap();
            assert_eq!(actual_indices, padded_indices);
            let actual_demand: Vec<u32> = backend.slow_read_vec(gpu_demand.buffer()).await.unwrap();
            assert_eq!(actual_demand, [expected_demand]);
            cases += 1;
        }
    }
    println!("parallel contact layout matches serial allocation in {cases} cases");
}
