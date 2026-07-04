use super::SimulationBackend;
use crate::rbd::SimulationState;
use khal::backend::{Backend, GpuBackend as KhalGpuBackend, GpuTimestamps};
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::{RbdPipeline, RbdState, RunStats};
use rapier::prelude::JointAxis;

/// GPU-based physics backend using nexus
pub struct GpuBackend {
    gpu: KhalGpuBackend,
    pipeline: RbdPipeline,
    state: RbdState,
    poses_cache: Vec<Pose>,
    timestamps: GpuTimestamps,
}

impl GpuBackend {
    /// Reads poses from GPU buffer, handling dimension-specific conversion.
    async fn read_poses(
        gpu: &KhalGpuBackend,
        state: &RbdState,
    ) -> Result<Vec<Pose>, String> {
        gpu.slow_read_vec(state.poses().buffer())
            .await
            .map_err(|e| format!("Failed to read poses: {:?}", e))
    }

    /// Reads poses into an existing buffer, handling dimension-specific conversion.
    async fn read_poses_into(
        gpu: &KhalGpuBackend,
        state: &RbdState,
        poses_cache: &mut Vec<Pose>,
    ) {
        poses_cache.resize(state.poses().len() as usize, Pose::default());
        let _ = gpu
            .slow_read_buffer(state.poses().buffer(), poses_cache)
            .await;
    }

    /// Attempts to create a new GPU backend, returning an error if initialization fails.
    ///
    /// This method can fail if:
    /// - Shader compilation fails
    /// - GPU device doesn't support required features
    /// - Memory allocation fails
    pub async fn try_new(gpu: &KhalGpuBackend, phys: &SimulationState) -> Result<Self, String> {
        let pipeline = RbdPipeline::from_backend(gpu);
        let envs: Vec<_> = phys
            .environments
            .iter()
            .map(|e| {
                (
                    &e.bodies,
                    &e.colliders,
                    &e.impulse_joints,
                    &e.multibody_joints,
                    &e.sim_params,
                )
            })
            .collect();
        let state = RbdState::from_rapier(gpu, &envs);
        let poses_cache = Self::read_poses(gpu, &state).await?;
        let timestamps = GpuTimestamps::new(gpu, 2048);

        Ok(Self {
            gpu: gpu.clone(),
            pipeline,
            state,
            poses_cache,
            timestamps,
        })
    }

    /// Creates a new GPU backend with a pre-compiled pipeline.
    ///
    /// This is faster than [`try_new`](Self::try_new) when switching demos because
    /// it reuses the existing pipeline instead of recompiling shaders.
    pub async fn with_pipeline(
        gpu: &KhalGpuBackend,
        pipeline: RbdPipeline,
        phys: &SimulationState,
    ) -> Self {
        let envs: Vec<_> = phys
            .environments
            .iter()
            .map(|e| {
                (
                    &e.bodies,
                    &e.colliders,
                    &e.impulse_joints,
                    &e.multibody_joints,
                    &e.sim_params,
                )
            })
            .collect();
        let state = RbdState::from_rapier(gpu, &envs);
        let poses_cache = Self::read_poses(gpu, &state).await.unwrap_or_default();
        let timestamps = GpuTimestamps::new(gpu, 2048);

        Self {
            gpu: gpu.clone(),
            pipeline,
            state,
            poses_cache,
            timestamps,
        }
    }

    /// Extracts the pipeline from this backend, consuming it.
    ///
    /// Useful for reusing the pipeline when switching demos.
    pub fn into_pipeline(self) -> RbdPipeline {
        self.pipeline
    }

    /// Creates a new GPU backend, panicking if initialization fails.
    ///
    /// Use [`try_new`](Self::try_new) for error handling.
    pub async fn new(gpu: &KhalGpuBackend, phys: &SimulationState) -> Self {
        Self::try_new(gpu, phys).await.unwrap()
    }

    /// See [`super::PhysicsBackend::set_multibody_motor_velocity`].
    pub fn set_multibody_motor_velocity(
        &mut self,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) {
        #[cfg(feature = "dim3")]
        let _ = self
            .state
            .multibodies_mut()
            .set_motor_velocity(&self.gpu, batch, link_id, axis, target_vel);
        #[cfg(feature = "dim2")]
        {
            let _ = (batch, link_id, axis, target_vel);
            todo!()
        }
    }
}

impl SimulationBackend for GpuBackend {
    fn poses(&self) -> &[Pose] {
        &self.poses_cache
    }
    fn num_bodies(&self) -> usize {
        self.state.num_colliders_per_batch() as usize
    }
    fn num_joints(&self) -> usize {
        self.state.joints().len()
    }
    fn num_batches(&self) -> usize {
        self.state.num_batches() as usize
    }

    async fn step(&mut self, _gpu: Option<&KhalGpuBackend>) -> RunStats {
        let gpu = &self.gpu;

        self.timestamps.reset();

        let t0 = web_time::Instant::now();
        let mut run_stats = self
            .pipeline
            .step(gpu, &mut self.state, Some(&mut self.timestamps))
            .unwrap();

        // Read back poses (synchronizes with the GPU when using WebGPU backend).
        gpu.synchronize().unwrap();
        run_stats.total_simulation_time_without_readback = t0.elapsed();

        self.pipeline
            .auto_resize_buffers(gpu, &mut self.state)
            .await;
        Self::read_poses_into(gpu, &self.state, &mut self.poses_cache).await;

        // Read timestamp results.
        if let Ok(results) = self.timestamps.read(gpu).await {
            let mut aggregated: Vec<(String, f64)> = Vec::new();
            for r in &results {
                if let Some(existing) = aggregated.iter_mut().find(|(label, _)| label == &r.label) {
                    existing.1 += r.duration_ms;
                } else {
                    aggregated.push((r.label.clone(), r.duration_ms));
                }
            }
            run_stats.gpu_total_time = aggregated.iter().map(|e| e.1).sum();
            run_stats.gpu_pass_times = aggregated;
        }

        run_stats.total_simulation_time_with_readback = t0.elapsed();

        run_stats
    }
}
