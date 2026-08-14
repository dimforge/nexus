//! Impulse accumulation and application for MPM-rigid body coupling.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::solver::p2g::IntegerImpulse;
use crate::mpm_shaders::solver::rigid_impulses::{
    GpuRigidImpulsesUpdate, GpuUpdateWorldMassProperties, GpuWritebackBodyPoses,
};
use crate::solver::GpuSimulationParams;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use nexus_rbd::dynamics::GpuBodySet;
use nexus_rbd::math::Pose;
use vortx::tensor::Tensor;

/// GPU kernels for computing and applying impulses to rigid bodies from MPM.
///
/// Accumulates forces from MPM particles and applies them as impulses to
/// coupled rigid bodies for two-way interaction.
#[derive(Shader)]
pub struct WgIntegrateBodies {
    /// Kernel for computing and applying impulses.
    update: GpuRigidImpulsesUpdate,
    /// Kernel for updating world-space mass properties.
    update_world_mass_properties: GpuUpdateWorldMassProperties,
    /// Kernel copying the coupled bodies' poses back to the rigid-body pipeline.
    writeback_body_poses: GpuWritebackBodyPoses,
}

/// GPU buffers for storing impulses from MPM to rigid bodies.
pub struct GpuImpulses {
    /// Per-timestep incremental impulses (uses atomic integer operations).
    pub incremental_impulses: Tensor<IntegerImpulse>,
}

impl GpuImpulses {
    /// Creates impulse buffers for rigid bodies.
    ///
    /// Allocates space for up to 16 bodies (CPIC limitation).
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        const MAX_BODY_COUNT: usize = 16; // CPIC doesn't support more.
        let impulses = [IntegerImpulse::default(); MAX_BODY_COUNT];
        Ok(Self {
            incremental_impulses: Tensor::vector(backend, impulses, BufferUsages::STORAGE)?,
        })
    }
}

impl WgIntegrateBodies {
    /// Computes and applies impulses to rigid bodies from MPM grid.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        grid: &GpuGrid,
        sim_params: &GpuSimulationParams,
        impulses: &mut GpuImpulses,
        bodies: &mut GpuBodySet,
    ) -> Result<(), GpuBackendError> {
        if bodies.is_empty() {
            return Ok(());
        }

        self.update.call(
            pass,
            1u32,
            &sim_params.params,
            &grid.meta,
            &bodies.local_mprops,
            &mut bodies.poses,
            &mut bodies.vels,
            &mut bodies.mprops,
            &mut impulses.incremental_impulses,
        )
    }

    /// Updates world-space mass properties for rigid bodies.
    ///
    /// Transforms local inertia tensors to world coordinates based on current poses.
    pub fn launch_update_world_mass_properties(
        &self,
        pass: &mut GpuPass,
        impulses: &mut GpuImpulses,
        bodies: &mut GpuBodySet,
    ) -> Result<(), GpuBackendError> {
        if bodies.is_empty() {
            return Ok(());
        }

        let len = bodies.len();
        self.update_world_mass_properties.call(
            pass,
            [len, 1, 1],
            &bodies.poses,
            &bodies.local_mprops,
            &mut bodies.mprops,
            &mut impulses.incremental_impulses,
        )
    }

    /// Copies the coupled bodies' MPM-integrated poses into `rbd_poses`, the
    /// rigid-body pipeline's body-pose buffer, at the slots given by
    /// `rbd_slots` (one per coupled body).
    ///
    /// See `gpu_writeback_body_poses`: MPM owns the pose of every body it is
    /// coupled to, so the rigid-body copy that rendering and the broad phase
    /// read has to be refreshed from it once per frame.
    pub fn launch_writeback_body_poses(
        &self,
        pass: &mut GpuPass,
        bodies: &GpuBodySet,
        rbd_slots: &Tensor<u32>,
        rbd_poses: &mut Tensor<Pose>,
    ) -> Result<(), GpuBackendError> {
        let len = rbd_slots.len() as u32;
        if bodies.is_empty() || len == 0 {
            return Ok(());
        }

        self.writeback_body_poses
            .call(pass, [len, 1, 1], &bodies.poses, rbd_slots, rbd_poses)
    }
}
