//! Rigid-bodies world-space mass properties calculation.

use crate::math::Pose;
use crate::shaders::dynamics::{GpuSyncColliderPoses, GpuUpdateMprops};
use crate::shaders::dynamics::{LocalMassProperties, WorldMassProperties};
use crate::shaders::utils::BatchIndices;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader for updating the world-space mass properties of rigid-bodies.
#[derive(Shader)]
pub struct GpuMpropsUpdate {
    /// Compute pipeline for the world mass properties update kernel.
    pub update_mprops_kernel: GpuUpdateMprops,
}

impl GpuMpropsUpdate {
    /// Dispatches the mass properties update kernel.
    pub fn dispatch(
        &self,
        pass: &mut GpuPass,
        mprops: &mut Tensor<WorldMassProperties>,
        local_mprops: &Tensor<LocalMassProperties>,
        body_poses: &Tensor<Pose>,
        batch_indices: &Tensor<BatchIndices>,
        num_bodies: u32,
        num_batches: u32,
    ) -> Result<(), GpuBackendError> {
        self.update_mprops_kernel.call(
            pass,
            [num_bodies, num_batches, 1],
            mprops,
            local_mprops,
            body_poses,
            batch_indices,
        )?;
        Ok(())
    }
}

/// GPU shader that recomputes the world pose of every collider from the body
/// world pose and the collider's body-local offset.
///
/// Now that the per-substep solver work runs on COM-centered solver poses
/// (`solver_body_poses`), the collider world poses only need to be refreshed
/// once per step, before broad-phase / narrow-phase / contact-to-constraint
/// conversion. Inside the substep loop, only the solver poses are mutated.
#[derive(Shader)]
pub struct GpuSyncColliderPosesShader {
    /// Compute pipeline for the collider-pose sync kernel.
    pub sync_kernel: GpuSyncColliderPoses,
}

impl GpuSyncColliderPosesShader {
    /// Dispatches the collider-pose sync kernel.
    pub fn dispatch(
        &self,
        pass: &mut GpuPass,
        body_poses: &Tensor<Pose>,
        collider_local_poses: &Tensor<Pose>,
        collider_world_poses: &mut Tensor<Pose>,
        collider_parent: &Tensor<u32>,
        batch_indices: &Tensor<BatchIndices>,
        num_colliders: u32,
        num_batches: u32,
    ) -> Result<(), GpuBackendError> {
        self.sync_kernel.call(
            pass,
            [num_colliders, num_batches, 1],
            body_poses,
            collider_local_poses,
            collider_world_poses,
            collider_parent,
            batch_indices,
        )?;
        Ok(())
    }
}
