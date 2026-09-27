#[cfg(feature = "mpm")]
use crate::mpm::pipeline::MpmPipeline;
use crate::rbd::pipeline::RbdPipeline;
use crate::rbd::utils::ComputeGraphPlan;
use crate::state::NexusState;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuGraph, GpuTimestamps};

bitflags::bitflags! {
    /// A bit mask identifying nexus pipelines.
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct NexusPipelineMask: u8 {
        const RBD = 1 << 0;
        const MPM = 1 << 1;
    }
}

#[derive(Default)]
pub struct NexusPipeline {
    pub rbd_pipeline: Option<RbdPipeline>,
    #[cfg(feature = "mpm")]
    pub mpm_pipeline: Option<MpmPipeline>,
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
        #[cfg(feature = "mpm")]
        if pipelines.contains(NexusPipelineMask::MPM) && self.mpm_pipeline.is_none() {
            self.mpm_pipeline = Some(MpmPipeline::new(backend)?);
        }
        Ok(())
    }

    /// Launches a captured graph inside a timed pass so the profiler shows the
    /// replayed frame as one entry.
    fn launch_graph(
        backend: &GpuBackend,
        graph: &GpuGraph,
        label: &str,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        let mut encoder = backend.begin_encoding();
        {
            let _pass = encoder.begin_pass(label, timestamps);
            graph.launch(backend)?;
        }
        backend.submit(encoder)
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
    ///
    /// With [`NexusState::compute_graphs`] set, on a backend that supports
    /// graph capture, the rigid-body steps and the MPM substeps of a frame are
    /// recorded once into compute graphs and replayed on the following frames
    /// until the scene's structure changes.
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

        let use_graphs = state.compute_graphs && backend.supports_graphs();

        // Rigid-bodies. `auto_resize_buffers` grows the collision-pair / coloring
        // buffers when the previous step overflowed them.
        if let Some(rbd) = state.rbd.as_mut() {
            self.preload_pipelines(backend, NexusPipelineMask::RBD)?;
            let pipeline = self.rbd_pipeline.as_mut().unwrap_or_else(|| unreachable!());
            let steps = state.rbd_steps_per_frame.max(1);
            let plan = if use_graphs {
                let key = rbd.graph_key(steps);
                rbd.compute_graph.plan(key)
            } else {
                rbd.compute_graph.invalidate();
                ComputeGraphPlan::Direct
            };
            match plan {
                ComputeGraphPlan::Replay => {
                    let graph = rbd.compute_graph.graph().unwrap_or_else(|| unreachable!());
                    Self::launch_graph(
                        backend,
                        graph,
                        "[RBD] compute graph",
                        timestamps.as_deref_mut(),
                    )?;
                }
                ComputeGraphPlan::Capture => {
                    // Recording: nothing executes, and pass timestamps would
                    // record into the graph, so run without them.
                    backend.begin_capture()?;
                    let mut stats = Ok(state.run_stats.clone());
                    for _ in 0..steps {
                        stats = pipeline.step(backend, rbd, None);
                        if stats.is_err() {
                            break;
                        }
                    }
                    match (stats, backend.end_capture()) {
                        (Ok(stats), Ok(graph)) => {
                            state.run_stats = stats;
                            Self::launch_graph(
                                backend,
                                &graph,
                                "[RBD] compute graph",
                                timestamps.as_deref_mut(),
                            )?;
                            rbd.compute_graph.captured(graph);
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            eprintln!(
                                "[nexus] compute graph capture of the rigid-body step failed ({e}); \
                                 running it directly for this state"
                            );
                            rbd.compute_graph.capture_failed();
                            // The recorded (unexecuted) steps must still happen.
                            for _ in 0..steps {
                                state.run_stats =
                                    pipeline.step(backend, rbd, timestamps.as_deref_mut())?;
                            }
                        }
                    }
                }
                ComputeGraphPlan::Direct => {
                    for _ in 0..steps {
                        state.run_stats = pipeline.step(backend, rbd, timestamps.as_deref_mut())?;
                    }
                }
            }
            pipeline.auto_resize_buffers(backend, rbd)?;
        }

        // MPM pipeline
        #[cfg(feature = "mpm")]
        if let Some(mpm) = state.mpm.as_mut() {
            self.preload_pipelines(backend, NexusPipelineMask::MPM)?;
            let pipeline = self.mpm_pipeline.as_mut().unwrap_or_else(|| unreachable!());

            // MPM needs many small substeps per visible frame for stability.
            // Upload the per-substep dt once, then run the substep loop.
            let substeps = state.mpm_substeps.max(1);
            let _ = mpm.write_substep_params(backend, substeps);
            let parity = mpm.grid.parity() as usize;
            let plan = if use_graphs {
                let key = mpm.graph_key(substeps);
                mpm.compute_graphs[parity].plan(key)
            } else {
                for cache in &mut mpm.compute_graphs {
                    cache.invalidate();
                }
                ComputeGraphPlan::Direct
            };
            match plan {
                ComputeGraphPlan::Replay => {
                    let graph = mpm.compute_graphs[parity]
                        .graph()
                        .unwrap_or_else(|| unreachable!());
                    Self::launch_graph(
                        backend,
                        graph,
                        "[MPM] compute graph",
                        timestamps.as_deref_mut(),
                    )?;
                    // Mirror the host-side buffer swaps the recorded substeps did.
                    for _ in 0..substeps % 2 {
                        mpm.grid.swap_buffers();
                    }
                }
                ComputeGraphPlan::Capture => {
                    backend.begin_capture()?;
                    let mut result = Ok(());
                    for _ in 0..substeps {
                        result = pipeline.step(backend, mpm, None);
                        if result.is_err() {
                            break;
                        }
                    }
                    match (result, backend.end_capture()) {
                        (Ok(()), Ok(graph)) => {
                            Self::launch_graph(
                                backend,
                                &graph,
                                "[MPM] compute graph",
                                timestamps.as_deref_mut(),
                            )?;
                            mpm.compute_graphs[parity].captured(graph);
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            eprintln!(
                                "[nexus] compute graph capture of the MPM substeps failed ({e}); \
                                 running them directly for this state"
                            );
                            mpm.compute_graphs[parity].capture_failed();
                            // The recorded substeps did not execute: undo the
                            // host-side buffer swaps they performed, then run
                            // them for real.
                            for _ in 0..substeps % 2 {
                                mpm.grid.swap_buffers();
                            }
                            for _ in 0..substeps {
                                let _ = pipeline.step(backend, mpm, timestamps.as_deref_mut());
                            }
                        }
                    }
                }
                ComputeGraphPlan::Direct => {
                    for _ in 0..substeps {
                        let _ = pipeline.step(backend, mpm, timestamps.as_deref_mut());
                    }
                }
            }
        }

        // MPM owns the pose of every body it is coupled to: it integrates its
        // own copy each substep while the rigid-body pipeline treats those
        // bodies as static. Push that copy back so rendering and the next
        // step's broad phase see a boundary that actually moved.
        // FIXME: the RBD pipeline should remain in charge of moving the bodies.
        #[cfg(feature = "mpm")]
        if let (Some(rbd), Some(mpm)) = (state.rbd.as_mut(), state.mpm.as_ref()) {
            let pipeline = self.mpm_pipeline.as_ref().unwrap_or_else(|| unreachable!());
            pipeline.writeback_body_poses(backend, mpm, rbd.body_poses_mut())?;
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
