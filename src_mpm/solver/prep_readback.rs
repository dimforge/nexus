//! GPU readback preparation kernel and associated data structures.
//!
//! Computes per-particle render data on the GPU, reducing the amount of data
//! transferred back to the CPU compared to reading raw positions and dynamics.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::solver::prep_readback::{
    GpuMpmPrepRender, GpuPrepReadback, GpuPrepReadbackRigid,
};
pub use crate::mpm_shaders::solver::prep_readback::{ReadbackData, RenderConfig};
use crate::solver::{GpuParticles, GpuRigidParticles, GpuSimulationParams};
use glamx::Vec4;
use khal::backend::{
    Encoder, GpuBackend, GpuBackendError, GpuBufferSliceMut, GpuEncoder, GpuTimestamps,
};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU compute kernel for preparing per-particle readback data.
///
/// Runs a compute shader that transforms particle positions and dynamics
/// into compact `ReadbackData` suitable for rendering, then copies the
/// result to a staging buffer for CPU readback.
#[derive(Shader)]
pub struct WgPrepReadback {
    prep_readback: GpuPrepReadback,
    prep_readback_rigid: GpuPrepReadbackRigid,
    prep_render: GpuMpmPrepRender,
}

/// GPU-resident buffers for the readback preparation pipeline.
///
/// Contains the render configuration, group palette, and output buffers
/// for the readback shader.
pub struct GpuReadbackData {
    /// Render mode configuration (uniform, written by CPU).
    pub mode: Tensor<RenderConfig>,
    /// Color palette indexed by `ParticleProperties::group_id`.
    pub group_colors: Tensor<Vec4>,
    /// Number of entries in `group_colors`, mirrored into `RenderConfig`.
    pub num_groups: u32,
    /// Shader output buffer (written by GPU, source for staging copy).
    pub instances: Tensor<ReadbackData>,
    /// Staging buffer for CPU readback (MAP_READ).
    pub instances_staging: Tensor<ReadbackData>,
    /// Per-rigid-particle base colors.
    pub rigid_base_colors: Tensor<Vec4>,
    /// Rigid particle shader output buffer.
    pub rigid_instances: Tensor<ReadbackData>,
    /// Staging buffer for rigid particle CPU readback (MAP_READ).
    pub rigid_instances_staging: Tensor<ReadbackData>,
    /// Rigid particle count uniform for the shader.
    pub rigid_len: Tensor<u32>,
}

impl GpuReadbackData {
    /// Fallback palette for a scene that supplies none.
    pub const DEFAULT_GROUP_COLORS: [Vec4; 6] = [
        Vec4::new(124.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
        Vec4::new(8.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
        Vec4::new(124.0 / 255.0, 7.0 / 255.0, 1.0, 1.0),
        Vec4::new(124.0 / 255.0, 144.0 / 255.0, 7.0 / 255.0, 1.0),
        Vec4::new(200.0 / 255.0, 37.0 / 255.0, 1.0, 1.0),
        Vec4::new(124.0 / 255.0, 230.0 / 255.0, 25.0 / 255.0, 1.0),
    ];

    /// Creates new readback data buffers for the given number of particles.
    ///
    /// `group_colors` is the palette particle group ids index into; an empty
    /// slice falls back to [`Self::DEFAULT_GROUP_COLORS`].
    pub fn new(
        backend: &GpuBackend,
        num_particles: usize,
        num_rigid_particles: usize,
        mode: u32,
        group_colors: &[Vec4],
    ) -> Result<Self, GpuBackendError> {
        let group_colors: Vec<Vec4> = if group_colors.is_empty() {
            Self::DEFAULT_GROUP_COLORS.to_vec()
        } else {
            group_colors.to_vec()
        };
        let num_groups = group_colors.len() as u32;
        let rigid_base_colors: Vec<Vec4> = (0..num_rigid_particles)
            .map(|i| group_colors[i % group_colors.len()])
            .collect();

        // Use at least 1 element for GPU buffers to avoid zero-sized allocations.
        let rigid_buf_len = num_rigid_particles.max(1) as u32;

        Ok(Self {
            num_groups,
            mode: Tensor::scalar(
                backend,
                RenderConfig {
                    mode,
                    num_groups,
                    ..Default::default()
                },
                // STORAGE for the readback kernel, UNIFORM for the render kernel.
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            group_colors: Tensor::vector(backend, group_colors, BufferUsages::STORAGE)?,
            instances: Tensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            instances_staging: Tensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            rigid_base_colors: Tensor::vector(backend, rigid_base_colors, BufferUsages::STORAGE)?,
            rigid_instances: Tensor::vector_uninit(
                backend,
                rigid_buf_len,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            rigid_instances_staging: Tensor::vector_uninit(
                backend,
                rigid_buf_len,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            rigid_len: Tensor::scalar(
                backend,
                num_rigid_particles as u32,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )?,
        })
    }

    /// Recreates all buffers for a new particle count.
    pub fn resize(
        &mut self,
        backend: &GpuBackend,
        num_particles: usize,
        num_rigid_particles: usize,
        mode: u32,
        group_colors: &[Vec4],
    ) -> Result<(), GpuBackendError> {
        *self = Self::new(
            backend,
            num_particles,
            num_rigid_particles,
            mode,
            group_colors,
        )?;
        Ok(())
    }
}

impl WgPrepReadback {
    /// Launches the readback preparation shader and copies results to staging.
    ///
    /// This runs a compute pass that writes `ReadbackData` into `instances`,
    /// then copies `instances` → `instances_staging` for CPU readback.
    /// Also dispatches the rigid particle readback shader if there are rigid particles.
    pub fn launch(
        &self,
        encoder: &mut GpuEncoder,
        timestamps: Option<&mut GpuTimestamps>,
        readback: &mut GpuReadbackData,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &GpuParticles,
        rigid_particles: &GpuRigidParticles,
    ) -> Result<(), GpuBackendError> {
        let len = particles.len() as u32;
        let rigid_len = rigid_particles.len() as u32;
        {
            let mut pass = encoder.begin_pass("prep-readback", timestamps);
            self.prep_readback.call(
                &mut pass,
                [len, 1, 1],
                &mut readback.instances,
                &particles.positions,
                &particles.kinematics,
                &particles.def_grad,
                &particles.properties,
                &grid.meta,
                &sim_params.params,
                &readback.mode,
                &particles.gpu_len,
                &readback.group_colors,
            )?;

            if rigid_len > 0 {
                self.prep_readback_rigid.call(
                    &mut pass,
                    [rigid_len, 1, 1],
                    &mut readback.rigid_instances,
                    &rigid_particles.sample_points,
                    &readback.rigid_base_colors,
                    &grid.meta,
                    &readback.rigid_len,
                )?;
            }
        }
        readback
            .instances_staging
            .copy_from_view(encoder, &readback.instances)?;
        if rigid_len > 0 {
            readback
                .rigid_instances_staging
                .copy_from_view(encoder, &readback.rigid_instances)?;
        }
        Ok(())
    }

    /// Zero-readback variant of [`Self::launch`]: writes per-particle render data
    /// straight into a renderer's SoA instance buffers (`positions`,
    /// `deformations`, `colors`), reading particle state directly on the GPU. No
    /// staging copy and no CPU readback. `readback` is reused only for its group
    /// palette and render-mode uniform.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_render(
        &self,
        encoder: &mut GpuEncoder,
        positions: &mut GpuBufferSliceMut<f32>,
        deformations: &mut GpuBufferSliceMut<f32>,
        colors: &mut GpuBufferSliceMut<f32>,
        readback: &GpuReadbackData,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &GpuParticles,
    ) -> Result<(), GpuBackendError> {
        let len = particles.len() as u32;
        if len == 0 {
            return Ok(());
        }
        let mut pass = encoder.begin_pass("mpm-prep-render", None);
        self.prep_render.call(
            &mut pass,
            [len, 1, 1],
            positions,
            deformations,
            colors,
            &particles.positions,
            &particles.kinematics,
            &particles.def_grad,
            &particles.properties,
            &grid.meta,
            &sim_params.params,
            &readback.mode,
            &particles.gpu_len,
            &readback.group_colors,
        )?;
        Ok(())
    }
}
