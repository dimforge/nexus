//! GPU parallel exclusive prefix sum (scan).
//!
//! Uses the variant where a 0 is prepended as the first element (useful for index/offset computation).

use crate::shaders::utils::prefix_sum::{GpuAddDataGrp, GpuPrefixSumSweep};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU shader for parallel prefix sum.
///
/// This shader implements a work-efficient parallel scan algorithm optimized for GPUs.
#[derive(Shader)]
pub struct GpuPrefixSum {
    /// Main prefix sum kernel (both up-sweep and down-sweep).
    prefix_sum_kernel: GpuPrefixSumSweep,
    /// Kernel for adding partial sums from coarser levels.
    add_data_grp_kernel: GpuAddDataGrp,
}

impl GpuPrefixSum {
    const THREADS: u32 = 256;

    /// Dispatches the prefix sum algorithm on GPU data.
    ///
    /// Supports batched operation: if `num_batches > 1`, each contiguous sub-array
    /// is independently prefix-summed.
    pub fn launch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        workspace: &mut PrefixSumWorkspace,
        data: &mut Tensor<u32>,
        num_batches: u32,
    ) -> Result<(), GpuBackendError> {
        #[cfg(feature = "cpu")]
        if pass.is_cpu() {
            return Self::dispatch_cpu(data, num_batches);
        }

        // If this assert fails, the kernel launches below must be changed because we are using
        // a fixed size for the shared memory currently.
        assert_eq!(
            Self::THREADS,
            256,
            "Internal error: prefix sum assumes a thread count equal to 256"
        );

        let batch_stride = data.len() as u32 / num_batches;
        workspace.reserve(backend, batch_stride, num_batches);

        let ngroups0 = workspace.per_batch_ngroups[0];

        self.prefix_sum_kernel.call(
            pass,
            [ngroups0 * Self::THREADS, num_batches, 1],
            data,
            &mut workspace.stages[0].buffer,
            &workspace.batch_stride_tensors[0],
        )?;

        for i in 0..workspace.num_stages - 1 {
            let ngroups = workspace.per_batch_ngroups[i + 1];
            let batch_stride_tensor = &workspace.batch_stride_tensors[i + 1];

            let (left, right) = workspace.stages.split_at_mut(i + 1);
            let data_stage = &mut left[i];
            let aux_stage = &mut right[0];

            self.prefix_sum_kernel.call(
                pass,
                [ngroups * Self::THREADS, num_batches, 1],
                &mut data_stage.buffer,
                &mut aux_stage.buffer,
                batch_stride_tensor,
            )?;
        }

        if workspace.num_stages > 2 {
            for i in (0..workspace.num_stages - 2).rev() {
                let ngroups = workspace.per_batch_ngroups[i + 1];
                let batch_stride_tensor = &workspace.batch_stride_tensors[i + 1];

                let (left, right) = workspace.stages.split_at_mut(i + 1);
                let data_stage = &mut left[i];
                let aux_stage = &right[0];

                self.add_data_grp_kernel.call(
                    pass,
                    [ngroups * Self::THREADS, num_batches, 1],
                    &mut data_stage.buffer,
                    &aux_stage.buffer,
                    batch_stride_tensor,
                )?;
            }
        }

        if workspace.num_stages > 1 {
            self.add_data_grp_kernel.call(
                pass,
                [ngroups0 * Self::THREADS, num_batches, 1],
                data,
                &workspace.stages[0].buffer,
                &workspace.batch_stride_tensors[0],
            )?;
        }

        Ok(())
    }

    #[cfg(feature = "cpu")]
    fn dispatch_cpu(data: &mut Tensor<u32>, num_batches: u32) -> Result<(), GpuBackendError> {
        let slice = data.buffer_mut().unwrap_slice_mut();
        let batch_stride = slice.len() / num_batches.max(1) as usize;

        for batch in 0..num_batches.max(1) as usize {
            let start = batch * batch_stride;
            let batch_data = &mut slice[start..start + batch_stride];

            // Inclusive prefix sum. Uses wrapping_add to match GPU u32
            // semantics: callers may leave tail entries uninitialized, and the
            // corresponding summed results are never read back downstream.
            for i in 0..batch_data.len() - 1 {
                batch_data[i + 1] = batch_data[i + 1].wrapping_add(batch_data[i]);
            }

            // Shift right to get exclusive prefix sum with leading 0.
            for i in (1..batch_data.len()).rev() {
                batch_data[i] = batch_data[i - 1];
            }
            batch_data[0] = 0;
        }

        Ok(())
    }

    /// CPU reference implementation for testing.
    pub fn eval_cpu(&self, v: &mut [u32]) {
        for i in 0..v.len() - 1 {
            v[i + 1] += v[i];
        }

        // NOTE: we actually have a special variant of the prefix-sum
        //       where the result is as if a 0 was appended to the input vector.
        for i in (1..v.len()).rev() {
            v[i] = v[i - 1];
        }

        v[0] = 0;
    }
}

/// One stage in the multi-level prefix sum hierarchy.
struct PrefixSumStage {
    /// GPU buffer for storing partial sums at this level.
    buffer: Tensor<u32>,
}

/// Workspace containing auxiliary buffers for hierarchical prefix sum.
///
/// The workspace maintains a hierarchy of buffers for the multi-level scan algorithm.
/// It automatically resizes when the input data size changes.
#[derive(Default)]
pub struct PrefixSumWorkspace {
    stages: Vec<PrefixSumStage>,
    num_stages: usize,
    /// Batch stride tensors for each level (data level + one per stage).
    batch_stride_tensors: Vec<Tensor<u32>>,
    /// Per-batch number of workgroups at each stage level.
    per_batch_ngroups: Vec<u32>,
    /// Cached batch parameters for resize detection.
    cached_batch_stride: u32,
    cached_num_batches: u32,
}

impl PrefixSumWorkspace {
    /// Creates a new empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a workspace pre-allocated for a specific buffer size.
    pub fn with_capacity(backend: &GpuBackend, buffer_len: u32) -> Self {
        let mut result = Self::default();
        result.reserve(backend, buffer_len, 1);
        result
    }

    /// Ensures the workspace has sufficient capacity for a given buffer size and batch count.
    ///
    /// Called automatically by [`GpuPrefixSum::launch`].
    pub fn reserve(&mut self, backend: &GpuBackend, batch_stride: u32, num_batches: u32) {
        if batch_stride == self.cached_batch_stride && num_batches == self.cached_num_batches {
            return;
        }

        self.cached_batch_stride = batch_stride;
        self.cached_num_batches = num_batches;

        let mut per_batch_len = batch_stride.div_ceil(GpuPrefixSum::THREADS);

        // Reinitialize the auxiliary buffers.
        self.stages.clear();
        self.batch_stride_tensors.clear();
        self.per_batch_ngroups.clear();

        // Batch stride tensor for the data level.
        self.batch_stride_tensors.push(
            Tensor::scalar(
                backend,
                batch_stride,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap(),
        );

        while per_batch_len != 1 {
            self.per_batch_ngroups.push(per_batch_len);

            let total = per_batch_len * num_batches;
            let buffer =
                Tensor::vector(backend, vec![0u32; total as usize], BufferUsages::STORAGE).unwrap();
            self.stages.push(PrefixSumStage { buffer });

            // Batch stride tensor for this stage level.
            self.batch_stride_tensors.push(
                Tensor::scalar(
                    backend,
                    per_batch_len,
                    BufferUsages::STORAGE | BufferUsages::UNIFORM,
                )
                .unwrap(),
            );

            per_batch_len = per_batch_len.div_ceil(GpuPrefixSum::THREADS);
        }

        // The last stage always has 1 element per batch.
        self.per_batch_ngroups.push(1);
        self.stages.push(PrefixSumStage {
            buffer: Tensor::vector(
                backend,
                vec![0u32; num_batches.max(1) as usize],
                BufferUsages::STORAGE,
            )
            .unwrap(),
        });
        self.num_stages = self.stages.len();
    }
}
