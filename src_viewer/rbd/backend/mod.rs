mod cpu;
mod gpu;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;

use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::RunStats;
use rapier::prelude::JointAxis;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendType {
    /// CPU physics using rapier.
    Rapier,
    /// GPU-accelerated physics using nexus + WebGPU.
    Gpu,
    /// CPU physics using nexus (same pipeline as GPU, executed on CPU).
    Cpu,
    /// GPU-accelerated physics using nexus + CUDA.
    #[cfg(feature = "cuda")]
    Cuda,
    /// GPU-accelerated physics using nexus + native Metal (macOS only).
    #[cfg(feature = "metal")]
    Metal,
}

/// Trait for physics simulation backends (CPU or GPU)
pub trait SimulationBackend {
    /// Get the current poses for rendering
    fn poses(&self) -> &[Pose];
    fn num_bodies(&self) -> usize;
    fn num_joints(&self) -> usize;
    fn num_batches(&self) -> usize;

    /// Step the simulation
    #[allow(async_fn_in_trait)]
    async fn step(&mut self, gpu: Option<&KhalGpuBackend>) -> RunStats;
}

#[allow(clippy::large_enum_variant)]
pub enum PhysicsBackend {
    Cpu(CpuBackend),
    Gpu(GpuBackend),
}

impl PhysicsBackend {
    pub async fn step(&mut self, gpu: Option<&KhalGpuBackend>) -> RunStats {
        match self {
            PhysicsBackend::Cpu(backend) => backend.step(gpu).await,
            PhysicsBackend::Gpu(backend) => backend.step(gpu).await,
        }
    }

    pub fn poses(&self) -> &[Pose] {
        match self {
            PhysicsBackend::Cpu(backend) => backend.poses(),
            PhysicsBackend::Gpu(backend) => backend.poses(),
        }
    }

    pub fn num_bodies(&self) -> usize {
        match self {
            PhysicsBackend::Cpu(backend) => backend.num_bodies(),
            PhysicsBackend::Gpu(backend) => backend.num_bodies(),
        }
    }

    pub fn num_joints(&self) -> usize {
        match self {
            PhysicsBackend::Cpu(backend) => backend.num_joints(),
            PhysicsBackend::Gpu(backend) => backend.num_joints(),
        }
    }

    pub fn num_batches(&self) -> usize {
        match self {
            PhysicsBackend::Cpu(backend) => backend.num_batches(),
            PhysicsBackend::Gpu(backend) => backend.num_batches(),
        }
    }

    /// Set a multibody joint motor's target velocity for the link `link_id`
    /// in batch `batch`, on joint axis `axis` (0..=2 = linear, 3..=5 = angular).
    /// The motor on that axis is enabled automatically.
    pub fn set_multibody_motor_velocity(
        &mut self,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) {
        match self {
            PhysicsBackend::Cpu(backend) => {
                backend.set_multibody_motor_velocity(batch, link_id, axis, target_vel)
            }
            PhysicsBackend::Gpu(backend) => {
                backend.set_multibody_motor_velocity(batch, link_id, axis, target_vel)
            }
        }
    }
}
