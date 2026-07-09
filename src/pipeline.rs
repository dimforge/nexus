use crate::rbd::pipeline::RbdPipeline;
use crate::state::NexusState;
use khal::backend::{GpuBackend, GpuBackendError, GpuTimestamps};

bitflags::bitflags! {
    /// A bit mask identifying nexus pipelines.
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct NexusPipelineMask: u8 {
        const RBD = 1 << 0;
    }
}

#[derive(Default)]
pub struct NexusPipeline {
    pub rbd_pipeline: Option<RbdPipeline>,
    /// CUDA-graph capture of one frame's rigid-body step sequence
    /// (`rbd_steps_per_frame × step`). Replaying it costs a single
    /// `cuGraphLaunch` instead of re-encoding every kernel dispatch from the
    /// host. See [`Self::capture_rbd_graph`].
    #[cfg(feature = "cuda")]
    rbd_graph: Option<khal::backend::cuda::CapturedGraph>,
}

impl NexusPipeline {
    pub fn preload_pipelines(
        &mut self,
        backend: &GpuBackend,
        pipelines: NexusPipelineMask,
    ) -> Result<(), GpuBackendError> {
        if pipelines.contains(NexusPipelineMask::RBD) && self.rbd_pipeline.is_none() {
            self.rbd_pipeline = Some(RbdPipeline::new(backend)?);
        }
        Ok(())
    }

    /// Advances the physics simulation by one GPU timestep.
    ///
    /// The compute pipelines are compiled lazily the first time their
    /// sub-state is stepped, so the initial call is more expensive (shader
    /// compilation) than the subsequent ones.
    ///
    /// In addition, resources are loaded lazily on the GPU, so the first step
    /// after inserting/removing entities can be slower too. Call `Self::finalize`
    /// to pay that cost upfront.
    /// Captures one frame's rigid-body dispatch sequence
    /// (`rbd_steps_per_frame × step`) into a CUDA graph and stores it on the
    /// pipeline. Returns `false` (and captures nothing) when the backend is
    /// not CUDA or there is no rigid-body state.
    ///
    /// Requirements: call after the scene is final (`finalize` runs here) and
    /// after a few warmup `simulate` calls so the buffer sizes are stable —
    /// the graph records raw buffer addresses, so any later reallocation
    /// (e.g. `auto_resize_buffers` growth) invalidates it. Fixed-grid dispatch
    /// must be active (it is the CUDA default): indirect dispatches read
    /// counts on the host and cannot be captured.
    ///
    /// Timestamps and buffer auto-resize are skipped during capture and
    /// replay; `run_stats` stops updating while replaying.
    #[cfg(feature = "cuda")]
    pub async fn capture_rbd_graph(
        &mut self,
        backend: &GpuBackend,
        state: &mut NexusState,
    ) -> Result<bool, GpuBackendError> {
        state.finalize(backend).await?;
        use khal::backend::Backend as _;
        let Some(cuda) = backend.as_cuda() else {
            return Ok(false);
        };
        let Some(rbd) = state.rbd.as_mut() else {
            return Ok(false);
        };
        self.preload_pipelines(backend, NexusPipelineMask::RBD)?;
        let pipeline = self.rbd_pipeline.as_mut().unwrap_or_else(|| unreachable!());
        let steps = state.rbd_steps_per_frame.max(1);
        cuda.begin_capture().map_err(khal::backend::GpuBackendError::Cuda)?;
        let mut step_result = Ok(state.run_stats.clone());
        for _ in 0..steps {
            step_result = pipeline.step(backend, rbd, None);
            if step_result.is_err() {
                break;
            }
        }
        // Always end the capture, even on error, so the stream isn't left in
        // capture mode.
        let graph = cuda.end_capture().map_err(khal::backend::GpuBackendError::Cuda)?;
        state.run_stats = step_result?;
        graph.upload().map_err(khal::backend::GpuBackendError::Cuda)?;
        // Capture only records; execute the captured sequence once so this
        // call has the same effect as a `simulate`.
        graph.launch().map_err(khal::backend::GpuBackendError::Cuda)?;
        self.rbd_graph = Some(graph);
        Ok(true)
    }

    /// Replays the captured rigid-body graph (one `cuGraphLaunch` for the whole
    /// `rbd_steps_per_frame × step` sequence). Returns `false` when no graph
    /// has been captured.
    #[cfg(feature = "cuda")]
    pub fn replay_rbd_graph(&self) -> Result<bool, GpuBackendError> {
        match &self.rbd_graph {
            Some(g) => {
                g.launch().map_err(khal::backend::GpuBackendError::Cuda)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub async fn simulate(
        &mut self,
        backend: &GpuBackend,
        state: &mut NexusState,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        state.finalize(backend).await?;

        let t0 = web_time::Instant::now();

        // Profiling timestamps use a non-blocking readback (harvested in the
        // viewer's `sync`). Only record a fresh frame while the previous
        // readback has been consumed — otherwise we'd resolve into a staging
        // buffer that's still mapped. This mirrors `auto_resize_buffers`'
        // "request only when idle" rule, so timings simply update a frame or two
        // apart instead of stalling the step on a blocking readback.
        let mut timestamps = timestamps.filter(|ts| ts.is_idle());

        // Rigid-bodies. `auto_resize_buffers` grows the collision-pair / coloring
        // buffers when the previous step overflowed them.
        if let Some(rbd) = state.rbd.as_mut() {
            self.preload_pipelines(backend, NexusPipelineMask::RBD)?;
            let pipeline = self.rbd_pipeline.as_mut().unwrap_or_else(|| unreachable!());
            let steps = state.rbd_steps_per_frame.max(1);
            for _ in 0..steps {
                state.run_stats = pipeline.step(backend, rbd, timestamps.as_deref_mut())?;
            }
            pipeline.auto_resize_buffers(backend, rbd)?;
        }

        state.run_stats.encoding_time = t0.elapsed();

        // If we recorded this frame, kick off the non-blocking readback of the
        // resolved timestamps (the passes were already resolved + submitted in
        // `step`). The viewer harvests it later with `try_take`, without ever
        // blocking on the GPU.
        if let Some(ts) = timestamps {
            ts.request_read(backend);
        }

        Ok(())
    }
}
