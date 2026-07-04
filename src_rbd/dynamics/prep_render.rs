//! Host side of the zero-readback rigid-body render-prep kernel.
//!
//! Dispatches [`gpu_rbd_prep_render`](crate::shaders::dynamics::GpuRbdPrepRender)
//! to write per-instance render data (world position + deformation matrix +
//! color) straight into a renderer's GPU instance buffers, indexed by a
//! per-instance descriptor array. Used by the viewer when khal shares the
//! renderer's wgpu device, replacing the GPU→CPU pose readback.

use crate::math::Pose;
use crate::shaders::dynamics::GpuRbdPrepRender;
pub use crate::shaders::dynamics::RbdInstanceDesc;
use khal::Shader;
use khal::backend::{Encoder, GpuBackendError, GpuBufferSliceMut, GpuEncoder};
use vortx::tensor::Tensor;

/// GPU compute kernel that prepares per-instance render data directly in a
/// renderer's instance buffers.
#[derive(Shader)]
pub struct WgRbdPrepRender {
    prep_render: GpuRbdPrepRender,
}

impl WgRbdPrepRender {
    /// Dispatches the render-prep kernel for one instanced node.
    ///
    /// `positions`, `deformations` and `colors` are the renderer's tightly-packed
    /// SoA float buffers (foreign `wgpu::Buffer`s wrapped via
    /// [`GpuBufferSliceMut::from_wgpu`]); `descriptors` and `count` describe the
    /// instances to write, and `body_poses` is the live GPU pose buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        encoder: &mut GpuEncoder,
        positions: &mut GpuBufferSliceMut<f32>,
        deformations: &mut GpuBufferSliceMut<f32>,
        colors: &mut GpuBufferSliceMut<f32>,
        body_poses: &Tensor<Pose>,
        descriptors: &Tensor<RbdInstanceDesc>,
        count: &Tensor<u32>,
        instance_count: u32,
    ) -> Result<(), GpuBackendError> {
        if instance_count == 0 {
            return Ok(());
        }
        let mut pass = encoder.begin_pass("rbd-prep-render", None);
        self.prep_render.call(
            &mut pass,
            [instance_count, 1, 1],
            positions,
            deformations,
            colors,
            body_poses,
            descriptors,
            count,
        )?;
        Ok(())
    }
}
