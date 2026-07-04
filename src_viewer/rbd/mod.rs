pub mod backend;
pub mod graphics;

pub use backend::{BackendType, CpuBackend, GpuBackend, PhysicsBackend};
pub use graphics::{RenderContext};

use crate::RunState;
use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::rbd::dynamics::RbdSimParams;
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::{RbdPipeline, RunStats};
use rapier::geometry::{ColliderHandle, ColliderSet, SharedShape};
use rapier::prelude::{ImpulseJointSet, MultibodyJointSet, RigidBodySet};
use std::collections::HashMap;
use nexus::state::NexusState;

/// Custom visual shape that overrides a collider's default rendering. The shape is
/// drawn at the collider's world pose composed with [`Self::local_pose`] (handy when
/// the collider's shape is a proxy — e.g. an OBB — and the visual mesh sits in a
/// different local frame).
#[derive(Clone)]
pub struct VisualShape {
    pub shape: SharedShape,
    pub local_pose: Pose,
}

impl VisualShape {
    pub fn new(shape: SharedShape) -> Self {
        Self {
            shape,
            local_pose: Pose::IDENTITY,
        }
    }

    pub fn with_local_pose(shape: SharedShape, local_pose: Pose) -> Self {
        Self { shape, local_pose }
    }
}

pub struct BatchEnvironment {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub sim_params: RbdSimParams,
    /// Optional per-collider visual override. When a collider handle is present in
    /// this map its [`VisualShape`] is rendered instead of the collider's own shape.
    pub visuals: HashMap<ColliderHandle, VisualShape>,
}

pub struct SimulationState {
    pub environments: Vec<BatchEnvironment>,
    /// Number of physics steps run per render frame (default: `1`).
    pub num_steps_per_frame: u32,
}

impl SimulationState {
    pub fn from_environments(environments: Vec<BatchEnvironment>) -> Self {
        Self {
            environments,
            num_steps_per_frame: 1,
        }
    }

    pub fn single(
        bodies: RigidBodySet,
        colliders: ColliderSet,
        impulse_joints: ImpulseJointSet,
    ) -> Self {
        Self::single_with_multibody(bodies, colliders, impulse_joints, MultibodyJointSet::new())
    }

    pub fn single_with_multibody(
        bodies: RigidBodySet,
        colliders: ColliderSet,
        impulse_joints: ImpulseJointSet,
        multibody_joints: MultibodyJointSet,
    ) -> Self {
        Self::single_with_multibody_and_visuals(
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            HashMap::new(),
        )
    }

    pub fn single_with_multibody_and_visuals(
        bodies: RigidBodySet,
        colliders: ColliderSet,
        impulse_joints: ImpulseJointSet,
        multibody_joints: MultibodyJointSet,
        visuals: HashMap<ColliderHandle, VisualShape>,
    ) -> Self {
        Self::from_environments(vec![BatchEnvironment {
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            sim_params: RbdSimParams::default(),
            visuals,
        }])
    }

    /// Sets `sim_params.dt` on every environment to `dt`.
    pub fn with_dt(mut self, dt: f32) -> Self {
        for env in &mut self.environments {
            env.sim_params.dt = dt;
        }
        self
    }

    /// Sets the number of physics steps run between two renders.
    pub fn with_num_steps_per_frame(mut self, num_steps_per_frame: u32) -> Self {
        self.num_steps_per_frame = num_steps_per_frame;
        self
    }
}

pub struct PhysicsContext {
    pub backend: PhysicsBackend,
}

impl PhysicsContext {
    pub fn new(backend: PhysicsBackend) -> Self {
        Self { backend }
    }
}

/// A rigid-body scene: GPU/CPU physics state plus its rendering instances.
///
/// Built via [`crate::NexusViewer::set_rbd`]. The example owns this and drives the
/// loop with [`RbdScene::simulate`].
pub struct RbdScene {
    pub physics: PhysicsContext,
    pub render_ctx: RenderContext,
    /// Total elapsed simulated time, accumulated across non-paused steps.
    pub sim_time: f64,
    /// Per-step timestep length (copied from the scene's `sim_params.dt`).
    pub dt: f32,
    /// Number of physics steps run per render frame.
    pub num_steps_per_frame: u32,
    /// Backend the scene was created with, used to decide whether the GPU
    /// pipeline can be cached on teardown.
    pub(crate) created_backend: BackendType,
}

impl RbdScene {
    /// Mutable access to the physics backend (e.g. to drive joint motors).
    pub fn backend_mut(&mut self) -> &mut PhysicsBackend {
        &mut self.physics.backend
    }

    /// Runs a single physics step. Self-contained (uses the backend's own GPU
    /// device); does not render. This is the headless/Python entry point.
    pub async fn step(&mut self) -> RunStats {
        self.physics.backend.step(None).await
    }

    /// Pushes the latest poses into the kiss3d render instances.
    pub fn sync_graphics(&mut self, state: &NexusState) {
        self.render_ctx.update_instances(state, &self.physics.backend);
    }

    /// Advances the simulation for one render frame (honoring pause/step) and
    /// syncs graphics. Call this inside the example's loop body.
    pub async fn simulate(&mut self, viewer: &mut crate::NexusViewer) {
        todo!()
        // if viewer.ui.run_state != RunState::Paused {
        //     for _ in 0..self.num_steps_per_frame {
        //         viewer.ui.run_stats = self.step().await;
        //         self.sim_time += self.dt as f64;
        //     }
        // }
        // self.sync_graphics();
        // if viewer.ui.run_state == RunState::Step {
        //     viewer.ui.run_state = RunState::Paused;
        // }
    }

    /// Detaches the render nodes and, when the backend is unchanged, caches the
    /// compiled GPU pipeline in the viewer for reuse by the next RBD scene.
    pub fn detach(self, viewer: &mut crate::NexusViewer) {
        let RbdScene {
            mut render_ctx,
            physics,
            created_backend,
            ..
        } = self;
        render_ctx.clear();
        if created_backend == viewer.ui.backend_type {
            if let PhysicsBackend::Gpu(gpu_backend) = physics.backend {
                viewer.cache_pipeline(gpu_backend.into_pipeline());
            }
        }
    }
}

pub async fn setup_physics(
    gpu: Option<&KhalGpuBackend>,
    phys: &SimulationState,
    backend_type: BackendType,
    gpu_error: &mut Option<String>,
    cached_pipeline: &mut Option<RbdPipeline>,
) -> PhysicsContext {
    let backend = match backend_type {
        BackendType::Gpu => {
            let gpu = gpu.unwrap();

            if let Some(pipeline) = cached_pipeline.take() {
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "GPU backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
        }
        BackendType::Cpu => {
            #[cfg(feature = "cpu")]
            {
                let cpu_backend = KhalGpuBackend::Cpu;
                match GpuBackend::try_new(&cpu_backend, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "Nexus CPU backend initialization failed: {}. Using rapier CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
            #[cfg(not(feature = "cpu"))]
            {
                *gpu_error =
                    Some("CPU backend not available (compiled without 'cpu' feature).".to_string());
                PhysicsBackend::Cpu(CpuBackend::new(phys))
            }
        }
        #[cfg(feature = "cuda")]
        BackendType::Cuda => {
            let gpu = gpu.expect("Cuda device initialization failed");

            if let Some(pipeline) = cached_pipeline.take() {
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "CUDA backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        BackendType::Metal => {
            let gpu = gpu.expect("Metal device initialization failed");

            if let Some(pipeline) = cached_pipeline.take() {
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "Metal backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
        }
        BackendType::Rapier => PhysicsBackend::Cpu(CpuBackend::new(phys)),
    };

    PhysicsContext::new(backend)
}
