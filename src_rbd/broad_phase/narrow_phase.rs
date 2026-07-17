//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{
    CollisionPair, GpuFlattenBatchesDispatch, GpuNarrowPhaseInitContactsDispatch,
    GpuNarrowPhasePfmPfm, GpuNarrowPhaseShapeShape, GpuNarrowPhaseShapeShapeDeferred,
    GpuResetNarrowPhase, NarrowPhasePfmPair,
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
    /// Builds the flat 1-D dispatch grid + prefix offsets for a per-batch
    /// work-list (used for both the collision pairs and the PFM pairs), so the
    /// kernels pack items from many batches into full warps instead of one
    /// mostly-idle workgroup per batch.
    flatten_batches: GpuFlattenBatchesDispatch,
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
        collision_pairs_len: &mut Tensor<u32>,
        collision_pairs_indirect: &mut Tensor<[u32; 3]>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
        pairs_offsets: &mut Tensor<u32>,
        pfm_offsets: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        self.reset_narrow_phase
            .call(pass, [1u32, num_batches, 1], contacts_len, pfm_pairs_len)?;

        // The broad phase wrote a `[max/64, num_batches, 1]` grid into
        // `collision_pairs_indirect`; rewrite it (and derive the offsets) for
        // the flat layout. Nothing else consumes the batched form.
        self.flatten_batches.call(
            pass,
            1u32,
            collision_pairs_len,
            pairs_offsets,
            collision_pairs_indirect,
            batch_indices,
        )?;

        self.narrow_phase.call(
            pass,
            &*collision_pairs_indirect,
            collision_pairs,
            pairs_offsets,
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
            &*collision_pairs_indirect,
            collision_pairs,
            pairs_offsets,
            poses,
            shapes,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
        )?;

        self.flatten_batches.call(
            pass,
            1u32,
            pfm_pairs_len,
            pfm_offsets,
            pfm_pairs_indirect,
            batch_indices,
        )?;
        self.narrow_phase_pfm_pfm.call(
            pass,
            &*pfm_pairs_indirect,
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_offsets,
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
