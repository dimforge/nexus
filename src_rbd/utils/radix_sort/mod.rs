//! Radix sort implementation, ported from `brush-sort`: <https://github.com/ArthurBrussee/brush/tree/main/crates/brush-sort>

use crate::shaders::utils::radix_sort::{
    GpuInitSortBatched, GpuInitSortDispatch, GpuSortCount, GpuSortReduce, GpuSortScan,
    GpuSortScanAdd, GpuSortScatter, SortUniforms,
};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

// NOTE: must match the values from the shaders.
const WG: u32 = 256;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;
const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1 << BITS_PER_PASS;

/// GPU-accelerated radix sort for sorting large arrays of u32 keys with associated values.
///
/// Sorts up to 32-bit integer keys. When `num_batches > 1`, all batches are
/// flattened into a single buffer with extra stable passes for batch_id bits,
/// keeping elements grouped by batch (avoids dispatching many underutilized
/// workgroups for small batches).
#[derive(Shader)]
pub struct RadixSort {
    init: GpuInitSortDispatch,
    init_batched: GpuInitSortBatched,
    count: GpuSortCount,
    reduce: GpuSortReduce,
    scan: GpuSortScan,
    scan_add: GpuSortScanAdd,
    scatter: GpuSortScatter,
}

/// Workspace buffers for radix sort operations.
///
/// Reusable across multiple sort operations; automatically resizes its
/// intermediate buffers as needed.
pub struct RadixSortWorkspace {
    pass_uniforms: Vec<Tensor<SortUniforms>>,
    /// Cache key for [`Self::pass_uniforms`] (and `n_sort_flat`):
    /// `(max_keys, passes, num_batches, per_batch)`. Rebuilding the uniform
    /// tensors on every sort would allocate during CUDA-graph capture
    /// (`cuMemAlloc` invalidates a capture) and churn memory for no reason.
    uniforms_key: Option<(u32, u32, u32, u32)>,
    reduced_buf: Tensor<u32>, // Tensor of size BLOCK_SIZE
    count_buf: Tensor<u32>,
    num_wgs: Tensor<[u32; 3]>,
    num_reduce_wgs: Tensor<[u32; 3]>,
    output_keys_pong: Tensor<u32>,   // dual-buffering for output keys.
    output_values_pong: Tensor<u32>, // dual-buffering for output values.
    // Buffers for flattened batched sort (batch_ids tracking).
    batch_ids: Tensor<u32>,
    batch_ids_pong: Tensor<u32>,
    n_sort_flat: Tensor<u32>,
}

impl RadixSortWorkspace {
    /// Creates a new radix sort workspace with default buffer sizes. Buffers are
    /// automatically resized on first use to match input data size.
    pub fn new(backend: &GpuBackend) -> Self {
        let zeros = vec![0u32; BLOCK_SIZE as usize];
        Self {
            pass_uniforms: vec![],
            uniforms_key: None,
            reduced_buf: Tensor::vector(backend, &zeros, BufferUsages::STORAGE).unwrap(),
            count_buf: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            num_wgs: Tensor::scalar(
                backend,
                [1, 1, 1],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            )
            .unwrap(),
            num_reduce_wgs: Tensor::scalar(
                backend,
                [1, 1, 1],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            )
            .unwrap(),
            output_keys_pong: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            output_values_pong: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            batch_ids: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            batch_ids_pong: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            n_sort_flat: Tensor::scalar(backend, 0u32, BufferUsages::STORAGE).unwrap(),
        }
    }
}

impl RadixSort {
    /// Dispatches the radix sort operation to sort keys with associated values.
    ///
    /// The sort is stable: elements with equal keys maintain their relative order.
    /// Both keys and values are sorted together, making this useful for indirect sorting
    /// (where values are indices into another array).
    ///
    /// When `num_batches > 1`, all batches are flattened into a single sort with extra
    /// passes for batch_id bits. All batches must have the same allocated size
    /// (`input_keys.len() / num_batches`), though active counts per batch may differ
    /// (given by `n_sort`). Inactive elements get sentinel keys and sort to the end
    /// of their batch.
    pub fn dispatch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        workspace: &mut RadixSortWorkspace,
        input_keys: &Tensor<u32>,
        input_values: &Tensor<u32>,
        n_sort: &Tensor<u32>,
        sorting_bits: u32,
        num_batches: u32,
        output_keys: &mut Tensor<u32>,
        output_values: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        assert_eq!(
            input_keys.len(),
            input_values.len(),
            "Input keys and values must have the same number of elements"
        );
        assert!(sorting_bits <= 32, "Can only sort up to 32 bits");

        #[cfg(feature = "cpu")]
        if pass.is_cpu() {
            return Self::dispatch_cpu(
                input_keys,
                input_values,
                n_sort,
                num_batches,
                output_keys,
                output_values,
            );
        }

        if num_batches <= 1 {
            self.dispatch_single_batch(
                backend,
                pass,
                workspace,
                input_keys,
                input_values,
                n_sort,
                sorting_bits,
                output_keys,
                output_values,
            )
        } else {
            self.dispatch_flattened(
                backend,
                pass,
                workspace,
                input_keys,
                input_values,
                n_sort,
                sorting_bits,
                num_batches,
                output_keys,
                output_values,
            )
        }
    }

    /// CPU fast path: sort keys and values using the standard library sort.
    #[cfg(feature = "cpu")]
    fn dispatch_cpu(
        input_keys: &Tensor<u32>,
        input_values: &Tensor<u32>,
        n_sort: &Tensor<u32>,
        num_batches: u32,
        output_keys: &mut Tensor<u32>,
        output_values: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        let in_keys = input_keys.buffer().unwrap_slice();
        let in_values = input_values.buffer().unwrap_slice();
        let n_sort_slice = n_sort.buffer().unwrap_slice();
        let out_keys = output_keys.buffer_mut().unwrap_slice_mut();
        let out_values = output_values.buffer_mut().unwrap_slice_mut();

        let num_batches = num_batches.max(1) as usize;
        let per_batch = in_keys.len() / num_batches;

        // Reusable index buffer to avoid allocating per batch.
        let mut indices: Vec<u32> = Vec::new();

        for batch in 0..num_batches {
            let n = n_sort_slice[batch] as usize;
            let offset = batch * per_batch;
            let keys = &in_keys[offset..offset + n];

            // Sort by permutation: sort indices by their corresponding key.
            indices.clear();
            indices.extend(0..n as u32);
            indices.sort_unstable_by_key(|&i| keys[i as usize]);

            // Scatter using the sorted permutation.
            for (dst, &src) in indices.iter().enumerate() {
                out_keys[offset + dst] = in_keys[offset + src as usize];
                out_values[offset + dst] = in_values[offset + src as usize];
            }
            // Fill remaining slots with sentinel keys.
            for i in n..per_batch {
                out_keys[offset + i] = u32::MAX;
                out_values[offset + i] = 0;
            }
        }

        Ok(())
    }

    /// Single-batch dispatch (num_batches <= 1).
    fn dispatch_single_batch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        workspace: &mut RadixSortWorkspace,
        input_keys: &Tensor<u32>,
        input_values: &Tensor<u32>,
        n_sort: &Tensor<u32>,
        sorting_bits: u32,
        output_keys: &mut Tensor<u32>,
        output_values: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        let max_n = input_keys.len() as u32;
        let per_batch_max = max_n;

        // Resize workspace buffers
        let max_needed_wgs = per_batch_max.div_ceil(BLOCK_SIZE);
        let needed_count = max_needed_wgs as u64 * BIN_COUNT as u64;
        if workspace.count_buf.len() < needed_count {
            workspace.count_buf =
                Tensor::vector_uninit(backend, needed_count as u32, BufferUsages::STORAGE)?;
        }

        let needed_reduced = BLOCK_SIZE as u64;
        if workspace.reduced_buf.len() < needed_reduced {
            let zeros = vec![0u32; needed_reduced as usize];
            workspace.reduced_buf = Tensor::vector(backend, &zeros, BufferUsages::STORAGE)?;
        }

        self.init.call(
            pass,
            1u32,
            n_sort,
            &mut workspace.num_wgs,
            &mut workspace.num_reduce_wgs,
        )?;

        // Capture-safe fixed grids matching `gpu_init_sort_dispatch` from
        // host-side capacities (`max_n`): no host count readback, so the CUDA
        // backend neither drains the stream per dispatch nor breaks CUDA-graph
        // capture. MUST be `DispatchGrid::Grid` (raw workgroup counts).
        let nwg = per_batch_max.div_ceil(BLOCK_SIZE).max(1);
        let sort_grid = [nwg, 1u32, 1u32];
        let reduce_grid = [BIN_COUNT * nwg.div_ceil(BLOCK_SIZE), 1u32, 1u32];

        if workspace.output_keys_pong.len() < input_keys.len() {
            workspace.output_keys_pong =
                Tensor::vector_uninit(backend, input_keys.len() as u32, BufferUsages::STORAGE)?;
            workspace.output_values_pong =
                Tensor::vector_uninit(backend, input_values.len() as u32, BufferUsages::STORAGE)?;
        }

        // Ensure batch_ids buffers exist for scatter aux binding (not accessed with has_aux=0).
        if workspace.batch_ids.len() < input_keys.len() {
            workspace.batch_ids =
                Tensor::vector_uninit(backend, input_keys.len() as u32, BufferUsages::STORAGE)?;
            workspace.batch_ids_pong =
                Tensor::vector_uninit(backend, input_keys.len() as u32, BufferUsages::STORAGE)?;
        }

        let num_passes = sorting_bits.div_ceil(4);

        // Create uniforms (has_aux=0 for single batch), cached across calls
        // (see `RadixSortWorkspace::uniforms_key`).
        let uniforms_key = (per_batch_max, num_passes, 0, 0);
        if workspace.uniforms_key != Some(uniforms_key) {
            workspace.pass_uniforms.clear();
            for pass_id in 0..num_passes {
                workspace.pass_uniforms.push(Tensor::scalar(
                    backend,
                    SortUniforms {
                        shift: pass_id * 4,
                        max_keys_per_batch: per_batch_max,
                        has_aux: 0,
                    },
                    BufferUsages::STORAGE | BufferUsages::UNIFORM,
                )?);
            }
            workspace.uniforms_key = Some(uniforms_key);
        }

        let mut output_keys = output_keys;
        let mut output_values = output_values;
        let mut output_keys_pong = &mut workspace.output_keys_pong;
        let mut output_values_pong = &mut workspace.output_values_pong;

        if num_passes.is_multiple_of(2) {
            // Make sure the last pass has the user provided `output_keys`
            // set as the output buffer so that the final results doesn't end
            // up stored in the workspace's pong buffers instead.
            std::mem::swap(&mut output_keys, &mut output_keys_pong);
            std::mem::swap(&mut output_values, &mut output_values_pong);
        }

        macro_rules! run_pass {
            ($pass_id: expr, $src: expr, $values: expr) => {
                let uniforms_buffer = &workspace.pass_uniforms[$pass_id as usize];

                self.count.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(sort_grid),
                    uniforms_buffer,
                    n_sort,
                    $src,
                    &mut workspace.count_buf,
                )?;

                self.reduce.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(reduce_grid),
                    uniforms_buffer,
                    n_sort,
                    &workspace.count_buf,
                    &mut workspace.reduced_buf,
                )?;

                self.scan
                    .call(pass, [1u32, 1, 1], n_sort, &mut workspace.reduced_buf)?;

                self.scan_add.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(reduce_grid),
                    uniforms_buffer,
                    n_sort,
                    &workspace.reduced_buf,
                    &mut workspace.count_buf,
                )?;

                self.scatter.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(sort_grid),
                    uniforms_buffer,
                    n_sort,
                    $src,
                    $values,
                    &workspace.count_buf,
                    output_keys,
                    output_values,
                    &workspace.batch_ids,
                    &mut workspace.batch_ids_pong,
                )?;
            };
        }

        if num_passes > 0 {
            run_pass!(0, input_keys, input_values);

            let mut cur_keys = output_keys;
            let mut cur_vals = output_values;
            output_keys = output_keys_pong;
            output_values = output_values_pong;

            for pass_id in 1..num_passes {
                run_pass!(pass_id, cur_keys, cur_vals);
                std::mem::swap(&mut cur_keys, &mut output_keys);
                std::mem::swap(&mut cur_vals, &mut output_values);
            }
        }
        Ok(())
    }

    /// Flattened batched dispatch (num_batches > 1). All batches are treated as one big
    /// buffer. Extra stable radix passes for batch_id bits keep elements grouped by batch.
    fn dispatch_flattened(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        workspace: &mut RadixSortWorkspace,
        input_keys: &Tensor<u32>,
        input_values: &Tensor<u32>,
        n_sort: &Tensor<u32>,
        sorting_bits: u32,
        num_batches: u32,
        output_keys: &mut Tensor<u32>,
        output_values: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        let total_n = input_keys.len() as u32;
        let per_batch = total_n / num_batches;
        let key_passes = sorting_bits.div_ceil(4);
        let batch_id_bits = 32 - (num_batches - 1).leading_zeros();
        let batch_id_passes = batch_id_bits.div_ceil(4);
        // Ensure even total_passes so the init buffer and final buffer align without
        // pre-swapping references. Extra batch_id passes are no-ops (sorting by
        // upper bits that are zero for small batch counts).
        let total_passes = {
            let raw = key_passes + batch_id_passes;
            raw + (raw % 2)
        };
        // Resize workspace for flattened single-batch view.
        let max_needed_wgs = total_n.div_ceil(BLOCK_SIZE);
        let needed_count = max_needed_wgs as u64 * BIN_COUNT as u64;
        if workspace.count_buf.len() < needed_count {
            workspace.count_buf =
                Tensor::vector_uninit(backend, needed_count as u32, BufferUsages::STORAGE)?;
        }

        let needed_reduced = BLOCK_SIZE as u64;
        if workspace.reduced_buf.len() < needed_reduced {
            let zeros = vec![0u32; needed_reduced as usize];
            workspace.reduced_buf = Tensor::vector(backend, &zeros, BufferUsages::STORAGE)?;
        }

        // Resize ping-pong buffers.
        if workspace.output_keys_pong.len() < total_n as u64 {
            workspace.output_keys_pong =
                Tensor::vector_uninit(backend, total_n, BufferUsages::STORAGE)?;
            workspace.output_values_pong =
                Tensor::vector_uninit(backend, total_n, BufferUsages::STORAGE)?;
        }
        if workspace.batch_ids.len() < total_n as u64 {
            workspace.batch_ids = Tensor::vector_uninit(backend, total_n, BufferUsages::STORAGE)?;
            workspace.batch_ids_pong =
                Tensor::vector_uninit(backend, total_n, BufferUsages::STORAGE)?;
        }

        // n_sort_flat and the per-pass uniforms are cached across calls
        // (see `RadixSortWorkspace::uniforms_key`): rebuilding them every sort
        // would allocate during CUDA-graph capture.
        let init_uniform_idx = total_passes as usize;
        let uniforms_key = (total_n, total_passes, num_batches, per_batch);
        if workspace.uniforms_key != Some(uniforms_key) {
            // n_sort_flat = [total_n] for the flattened single-batch view.
            workspace.n_sort_flat = Tensor::scalar(backend, total_n, BufferUsages::STORAGE)?;

            // Create uniforms for all passes.
            workspace.pass_uniforms.clear();
            for pass_id in 0..total_passes {
                let shift = if pass_id < key_passes {
                    pass_id * 4
                } else {
                    (pass_id - key_passes) * 4
                };
                workspace.pass_uniforms.push(Tensor::scalar(
                    backend,
                    SortUniforms {
                        shift,
                        max_keys_per_batch: total_n,
                        has_aux: 1,
                    },
                    BufferUsages::STORAGE | BufferUsages::UNIFORM,
                )?);
            }

            // Extra uniform for init_batched (max_keys_per_batch = per_batch, not total_n).
            // shift is repurposed to carry num_batches for this kernel.
            workspace.pass_uniforms.push(Tensor::scalar(
                backend,
                SortUniforms {
                    shift: num_batches,
                    max_keys_per_batch: per_batch,
                    has_aux: 0,
                },
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )?);
            workspace.uniforms_key = Some(uniforms_key);
        }

        // Init writes to output buffers. After even total_passes, data stays in output.
        // NOTE: call() takes a thread count, not workgroup count (khal resolves internally).
        self.init_batched.call(
            pass,
            total_n,
            &workspace.pass_uniforms[init_uniform_idx],
            n_sort,
            input_keys,
            input_values,
            output_keys,
            output_values,
            &mut workspace.batch_ids,
        )?;
        let mut cur_keys = output_keys;
        let mut next_keys = &mut workspace.output_keys_pong;
        let mut cur_vals = output_values;
        let mut next_vals = &mut workspace.output_values_pong;

        // Init indirect dispatch for flattened single-batch.
        self.init.call(
            pass,
            1u32,
            &workspace.n_sort_flat,
            &mut workspace.num_wgs,
            &mut workspace.num_reduce_wgs,
        )?;

        // Capture-safe fixed grids for the flattened sort passes — host-exact
        // match to `gpu_init_sort_dispatch` (no host count readback ⇒ CUDA-graph
        // capturable, and no per-dispatch stream drain on the CUDA backend).
        // MUST be `DispatchGrid::Grid` (raw workgroup counts), NOT a bare
        // `[u32;3]` (that converts to ThreadCount and divides by block size).
        //   num_wgs    = ceil(total_n / BLOCK_SIZE)
        //   reduce_wgs = BIN_COUNT * ceil(num_wgs / BLOCK_SIZE)
        let nwg = total_n.div_ceil(BLOCK_SIZE);
        let sort_grid = [nwg, 1u32, 1u32];
        let reduce_grid = [BIN_COUNT * nwg.div_ceil(BLOCK_SIZE), 1u32, 1u32];

        // Run all sort passes with 3-stream ping-pong.
        // Stream mapping changes between key passes and batch_id passes:
        //   Key passes:      scatter(src=keys, values=vals, aux=bids)
        //   Batch_id passes: scatter(src=bids, values=keys, aux=vals)
        let n_sort_flat = &workspace.n_sort_flat;
        let mut cur_aux = &mut workspace.batch_ids;
        let mut next_aux = &mut workspace.batch_ids_pong;

        for pass_id in 0..total_passes {
            let is_batch_pass = pass_id >= key_passes;
            let uniforms = &workspace.pass_uniforms[pass_id as usize];

            if !is_batch_pass {
                // Key pass: digit extraction from keys.
                self.count.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(sort_grid),
                    uniforms,
                    n_sort_flat,
                    cur_keys,
                    &mut workspace.count_buf,
                )?;
                self.reduce.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(reduce_grid),
                    uniforms,
                    n_sort_flat,
                    &workspace.count_buf,
                    &mut workspace.reduced_buf,
                )?;
                self.scan
                    .call(pass, [1u32, 1, 1], n_sort_flat, &mut workspace.reduced_buf)?;
                self.scan_add.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(reduce_grid),
                    uniforms,
                    n_sort_flat,
                    &workspace.reduced_buf,
                    &mut workspace.count_buf,
                )?;
                self.scatter.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(sort_grid),
                    uniforms,
                    n_sort_flat,
                    cur_keys,
                    cur_vals,
                    &workspace.count_buf,
                    next_keys,
                    next_vals,
                    cur_aux,
                    next_aux,
                )?;
            } else {
                // Batch_id pass: digit extraction from batch_ids.
                // Remap: src=batch_ids, values=keys, aux=vals.
                self.count.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(sort_grid),
                    uniforms,
                    n_sort_flat,
                    cur_aux,
                    &mut workspace.count_buf,
                )?;
                self.reduce.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(reduce_grid),
                    uniforms,
                    n_sort_flat,
                    &workspace.count_buf,
                    &mut workspace.reduced_buf,
                )?;
                self.scan
                    .call(pass, [1u32, 1, 1], n_sort_flat, &mut workspace.reduced_buf)?;
                self.scan_add.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(reduce_grid),
                    uniforms,
                    n_sort_flat,
                    &workspace.reduced_buf,
                    &mut workspace.count_buf,
                )?;
                // scatter(src=bids, values=keys, aux=vals,
                //         out=next_bids, out_values=next_keys, out_aux=next_vals)
                self.scatter.call(
                    pass,
                    khal::backend::DispatchGrid::Grid(sort_grid),
                    uniforms,
                    n_sort_flat,
                    cur_aux,
                    cur_keys,
                    &workspace.count_buf,
                    next_aux,
                    next_keys,
                    cur_vals,
                    next_vals,
                )?;
            }

            std::mem::swap(&mut cur_keys, &mut next_keys);
            std::mem::swap(&mut cur_vals, &mut next_vals);
            std::mem::swap(&mut cur_aux, &mut next_aux);
        }

        // After all passes, cur_keys and cur_vals hold the final sorted data.
        // Thanks to the pre-swap, they point to the user's output buffers.
        Ok(())
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::utils::RadixSort;
    use crate::utils::radix_sort::RadixSortWorkspace;
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use khal::{BufferUsages, Shader};
    use vortx::tensor::Tensor;

    pub fn cpu_argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    async fn test_sorting_generic(gpu: &GpuBackend, num_iterations: u32) {
        let sort = RadixSort::from_backend(gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu);

        for i in 0u32..num_iterations {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2u32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            let input_usages = BufferUsages::STORAGE;
            let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

            let keys = Tensor::vector(gpu, keys_inp, input_usages).unwrap();
            let values = Tensor::vector(gpu, &values_inp, input_usages).unwrap();
            let mut out_keys = Tensor::vector(gpu, keys_inp, output_usages).unwrap();
            let mut out_values = Tensor::vector(gpu, &values_inp, output_usages).unwrap();
            let num_points =
                Tensor::scalar(gpu, keys_inp.len() as u32, BufferUsages::STORAGE).unwrap();

            let mut encoder = gpu.begin_encoding();
            let mut pass = encoder.begin_pass("test", None);
            sort.dispatch(
                gpu,
                &mut pass,
                &mut workspace,
                &keys,
                &values,
                &num_points,
                32,
                1,
                &mut out_keys,
                &mut out_values,
            )
            .unwrap();
            drop(pass);
            gpu.submit(encoder).unwrap();

            let result_keys = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();
            let result_values = gpu.slow_read_vec(out_values.buffer()).await.unwrap();

            let inds = cpu_argsort(&keys_inp);
            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

            assert_eq!(ref_keys, result_keys);
            assert_eq!(ref_values, result_values);
        }
    }

    async fn test_sorting_big_generic(gpu: &GpuBackend, num_ranges: u32) {
        use rand::Rng;
        let sort = RadixSort::from_backend(gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu);

        // Simulate some data as one might find for a bunch of gaussians.
        let mut rng = rand::rng();
        let mut keys_inp = Vec::new();
        for i in 0..num_ranges {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    keys_inp.push(j);
                }
            }
        }
        let values_inp: Vec<_> = keys_inp.iter().map(|&x| x * 2 + 5).collect();

        let input_usages = BufferUsages::STORAGE;
        let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        let keys = Tensor::vector(gpu, &keys_inp, input_usages).unwrap();
        let values = Tensor::vector(gpu, &values_inp, input_usages).unwrap();
        let mut out_keys = Tensor::vector(gpu, &keys_inp, output_usages).unwrap();
        let mut out_values = Tensor::vector(gpu, &values_inp, output_usages).unwrap();
        let num_points = Tensor::scalar(gpu, keys_inp.len() as u32, BufferUsages::STORAGE).unwrap();

        let mut encoder = gpu.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        sort.dispatch(
            gpu,
            &mut pass,
            &mut workspace,
            &keys,
            &values,
            &num_points,
            32,
            1,
            &mut out_keys,
            &mut out_values,
        )
        .unwrap();
        drop(pass);
        gpu.submit(encoder).unwrap();

        let result_keys = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();
        let result_values = gpu.slow_read_vec(out_values.buffer()).await.unwrap();

        let inds = cpu_argsort(&keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        assert_eq!(ref_keys, result_keys);
        assert_eq!(ref_values, result_values);
    }

    async fn test_sorting_batched_generic(gpu: &GpuBackend, num_batches: u32, per_batch: u32) {
        let sort = RadixSort::from_backend(gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu);

        // Create per-batch keys: each batch has keys in a different range.
        let mut keys_inp = Vec::new();
        let mut values_inp = Vec::new();
        for batch_id in 0..num_batches {
            for j in 0..per_batch {
                // Keys are descending within each batch so sorting actually changes order.
                keys_inp.push((per_batch - 1 - j) + batch_id * 1000);
                values_inp.push(batch_id * per_batch + j);
            }
        }

        let input_usages = BufferUsages::STORAGE;
        let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        let keys = Tensor::vector(gpu, &keys_inp, input_usages).unwrap();
        let values = Tensor::vector(gpu, &values_inp, input_usages).unwrap();
        let mut out_keys = Tensor::vector(gpu, &keys_inp, output_usages).unwrap();
        let mut out_values = Tensor::vector(gpu, &values_inp, output_usages).unwrap();
        let n_sort_data = vec![per_batch; num_batches as usize];
        let n_sort = Tensor::vector(gpu, &n_sort_data, BufferUsages::STORAGE).unwrap();

        let mut encoder = gpu.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        sort.dispatch(
            gpu,
            &mut pass,
            &mut workspace,
            &keys,
            &values,
            &n_sort,
            32,
            num_batches,
            &mut out_keys,
            &mut out_values,
        )
        .unwrap();
        drop(pass);
        gpu.submit(encoder).unwrap();

        let result_keys: Vec<u32> = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();
        let result_values: Vec<u32> = gpu.slow_read_vec(out_values.buffer()).await.unwrap();

        // Verify each batch is independently sorted.
        for batch_id in 0..num_batches {
            let start = (batch_id * per_batch) as usize;
            let end = start + per_batch as usize;
            let batch_keys = &result_keys[start..end];
            let batch_values = &result_values[start..end];

            // Keys should be sorted ascending within the batch.
            for i in 1..batch_keys.len() {
                assert!(
                    batch_keys[i - 1] <= batch_keys[i],
                    "batch {batch_id}: keys not sorted at index {i}: {} > {}",
                    batch_keys[i - 1],
                    batch_keys[i]
                );
            }

            // All keys should be in the expected range for this batch.
            for &k in batch_keys {
                assert!(
                    k >= batch_id * 1000 && k < batch_id * 1000 + per_batch,
                    "batch {batch_id}: unexpected key {k}"
                );
            }

            // Values should correspond to the original key-value pairs.
            for (i, (&k, &v)) in batch_keys.iter().zip(batch_values.iter()).enumerate() {
                let orig_j = k - batch_id * 1000;
                let expected_v = batch_id * per_batch + (per_batch - 1 - orig_j);
                assert_eq!(
                    v, expected_v,
                    "batch {batch_id} index {i}: key={k}, value={v}, expected value={expected_v}"
                );
            }
        }
    }

    async fn test_sorting_batched_multi_wg_generic(
        gpu: &GpuBackend,
        num_batches: u32,
        per_batch: u32,
    ) {
        use rand::Rng;
        let sort = RadixSort::from_backend(gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu);

        let total = num_batches * per_batch;
        let mut rng = rand::rng();

        let mut keys_inp = Vec::new();
        let mut values_inp = Vec::new();
        for _batch in 0..num_batches {
            for j in 0..per_batch {
                keys_inp.push(rng.random_range(0..10000u32));
                values_inp.push(j);
            }
        }

        let input_usages = BufferUsages::STORAGE;
        let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        let keys = Tensor::vector(gpu, &keys_inp, input_usages).unwrap();
        let values = Tensor::vector(gpu, &values_inp, input_usages).unwrap();
        let mut out_keys = Tensor::vector_uninit(gpu, total, output_usages).unwrap();
        let mut out_values = Tensor::vector_uninit(gpu, total, output_usages).unwrap();
        let n_sort = Tensor::vector(
            gpu,
            vec![per_batch; num_batches as usize],
            BufferUsages::STORAGE,
        )
        .unwrap();

        let mut encoder = gpu.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        sort.dispatch(
            gpu,
            &mut pass,
            &mut workspace,
            &keys,
            &values,
            &n_sort,
            32,
            num_batches,
            &mut out_keys,
            &mut out_values,
        )
        .unwrap();
        drop(pass);
        gpu.submit(encoder).unwrap();

        let result_keys: Vec<u32> = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();

        // Verify each batch is independently sorted.
        for batch in 0..num_batches {
            let start = (batch * per_batch) as usize;
            let end = start + per_batch as usize;
            let batch_keys = &result_keys[start..end];

            for i in 1..batch_keys.len() {
                assert!(
                    batch_keys[i - 1] <= batch_keys[i],
                    "batch {batch}: keys not sorted at index {i}: {} > {}",
                    batch_keys[i - 1],
                    batch_keys[i]
                );
            }

            let mut orig_keys = keys_inp[start..end].to_vec();
            orig_keys.sort();
            assert_eq!(
                batch_keys,
                &orig_keys[..],
                "batch {batch}: sorted keys don't match expected"
            );
        }
    }

    async fn test_sorting_many_small_batches_generic(
        gpu: &GpuBackend,
        num_batches: u32,
        per_batch: u32,
    ) {
        use rand::Rng;
        let sort = RadixSort::from_backend(gpu).unwrap();
        let mut workspace = RadixSortWorkspace::new(gpu);

        let total = num_batches * per_batch;
        let mut rng = rand::rng();

        let mut keys_inp = Vec::new();
        let mut values_inp = Vec::new();
        for _batch in 0..num_batches {
            for j in 0..per_batch {
                keys_inp.push(rng.random_range(0..10000u32));
                values_inp.push(j); // local index within batch
            }
        }

        let input_usages = BufferUsages::STORAGE;
        let output_usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        let keys = Tensor::vector(gpu, &keys_inp, input_usages).unwrap();
        let values = Tensor::vector(gpu, &values_inp, input_usages).unwrap();
        let mut out_keys = Tensor::vector_uninit(gpu, total, output_usages).unwrap();
        let mut out_values = Tensor::vector_uninit(gpu, total, output_usages).unwrap();
        let n_sort_data = vec![per_batch; num_batches as usize];
        let n_sort = Tensor::vector(gpu, &n_sort_data, BufferUsages::STORAGE).unwrap();

        let mut encoder = gpu.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        sort.dispatch(
            gpu,
            &mut pass,
            &mut workspace,
            &keys,
            &values,
            &n_sort,
            32,
            num_batches,
            &mut out_keys,
            &mut out_values,
        )
        .unwrap();
        drop(pass);
        gpu.submit(encoder).unwrap();

        let result_keys: Vec<u32> = gpu.slow_read_vec(out_keys.buffer()).await.unwrap();

        // Verify each batch is independently sorted.
        for batch in 0..num_batches {
            let start = (batch * per_batch) as usize;
            let end = start + per_batch as usize;
            let batch_keys = &result_keys[start..end];

            // Keys should be sorted ascending.
            for i in 1..batch_keys.len() {
                assert!(
                    batch_keys[i - 1] <= batch_keys[i],
                    "batch {batch}: keys not sorted at index {i}: {} > {}",
                    batch_keys[i - 1],
                    batch_keys[i]
                );
            }

            // The sorted keys should be the same set as the original batch keys.
            let orig_start = start;
            let orig_end = end;
            let mut orig_keys = keys_inp[orig_start..orig_end].to_vec();
            orig_keys.sort();
            assert_eq!(
                batch_keys,
                &orig_keys[..],
                "batch {batch}: sorted keys don't match expected"
            );
        }
    }

    // --- WebGpu test wrappers ---

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting() {
        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        test_sorting_generic(&gpu, 128).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting_big() {
        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        test_sorting_big_generic(&gpu, 10000).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting_batched() {
        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        test_sorting_batched_generic(&gpu, 4, 256).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting_batched_multi_wg() {
        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        test_sorting_batched_multi_wg_generic(&gpu, 4, 128).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn test_sorting_many_small_batches() {
        let gpu = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        test_sorting_many_small_batches_generic(&gpu, 1000, 8).await;
    }

    // --- CPU test wrappers (reduced sizes to keep tests fast) ---

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn test_sorting_cpu() {
        let gpu = GpuBackend::Cpu;
        test_sorting_generic(&gpu, 3).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn test_sorting_big_cpu() {
        let gpu = GpuBackend::Cpu;
        test_sorting_big_generic(&gpu, 2).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn test_sorting_batched_cpu() {
        let gpu = GpuBackend::Cpu;
        test_sorting_batched_generic(&gpu, 2, 64).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn test_sorting_batched_multi_wg_cpu() {
        let gpu = GpuBackend::Cpu;
        test_sorting_batched_multi_wg_generic(&gpu, 2, 64).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn test_sorting_many_small_batches_cpu() {
        let gpu = GpuBackend::Cpu;
        test_sorting_many_small_batches_generic(&gpu, 10, 8).await;
    }
}
