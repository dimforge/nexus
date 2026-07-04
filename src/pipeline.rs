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
