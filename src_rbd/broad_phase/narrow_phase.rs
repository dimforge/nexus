//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{
    CollisionPair, GpuInitPfmPfmDispatch, GpuNarrowPhaseInitContactsDispatch, GpuNarrowPhasePfmPfm,
    GpuNarrowPhaseShapeShape, GpuNarrowPhaseShapeShapeDeferred, GpuResetNarrowPhase,
    NarrowPhasePfmPair,
};
use crate::shaders::shapes::Shape;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader for narrow-phase collision detection.
#[derive(Shader)]
pub struct GpuNarrowPhase {
    reset_narrow_phase: GpuResetNarrowPhase,
    narrow_phase: GpuNarrowPhaseShapeShape,
    /// Pass 2: defers complex shape pairs (PFM / trimesh / polyline) into the
    /// `pfm_pairs` work-list. Split from `narrow_phase` to fit 8 storage buffers.
    narrow_phase_deferred: GpuNarrowPhaseShapeShapeDeferred,
    narrow_phase_pfm_pfm: GpuNarrowPhasePfmPfm,
    init_pfm_pfm_indirect_args: GpuInitPfmPfmDispatch,
    init_contacts_indirect_args: GpuNarrowPhaseInitContactsDispatch,
}

impl GpuNarrowPhase {
    /// Dispatches the narrow-phase collision detection pipeline.
    pub fn dispatch(
        &self,
        pass: &mut GpuPass,
        _num_colliders: u32,
        poses: &Tensor<Pose>,
        shapes: &Tensor<Shape>,
        vertices: &Tensor<PaddedVector>,
        indices: &Tensor<u32>,
        collision_pairs: &Tensor<CollisionPair>,
        collision_pairs_len: &Tensor<u32>,
        collision_pairs_indirect: &Tensor<[u32; 3]>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        // Capacity-based grids for fixed-grid dispatch (see `crate::dispatch_grid`).
        let nb = num_batches.max(1);
        let pairs_grid = [
            (collision_pairs.len() as u32 / nb).max(1).div_ceil(64),
            num_batches,
            1,
        ];
        let pfm_grid = [(pfm_pairs.len() as u32 / nb).max(1).div_ceil(64), num_batches, 1];
        self.reset_narrow_phase
            .call(pass, [1u32, num_batches, 1], contacts_len, pfm_pairs_len)?;

        self.narrow_phase.call(
            pass,
            crate::dispatch_grid(collision_pairs_indirect, pairs_grid),
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            contacts,
            contacts_len,
            batch_indices,
            collider_parent,
            collider_materials,
        )?;

        // Pass 2: defer the complex shape pairs into `pfm_pairs` (kept as a
        // separate dispatch so each pass fits 8 storage buffers).
        self.narrow_phase_deferred.call(
            pass,
            crate::dispatch_grid(collision_pairs_indirect, pairs_grid),
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
        )?;

        self.init_pfm_pfm_indirect_args
            .call(pass, 1u32, pfm_pairs_len, pfm_pairs_indirect)?;
        self.narrow_phase_pfm_pfm.call(
            pass,
            crate::dispatch_grid(&*pfm_pairs_indirect, pfm_grid),
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
            collider_parent,
            collider_materials,
        )?;
        self.init_contacts_indirect_args
            .call(pass, 1u32, contacts_len, contacts_indirect)?;

        Ok(())
    }
}
