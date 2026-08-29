//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
#[cfg(feature = "dim3")]
use crate::shaders::broad_phase::GpuReduceContacts;
use crate::shaders::broad_phase::{
    CollisionPair, GpuContactOffsetsScan, GpuCountPairsPerBatch, GpuCountPfmPerBatch,
    GpuFlatListDispatch, GpuNarrowPhaseInitContactsDispatch, GpuNarrowPhasePfmPfm,
    GpuNarrowPhaseShapeShape, GpuNarrowPhaseShapeShapeDeferred, GpuResetNarrowPhase,
    GpuZeroContactLens, NarrowPhasePfmPair,
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
    #[cfg(feature = "dim3")]
    reduce_contacts: GpuReduceContacts,
    count_pairs_per_batch: GpuCountPairsPerBatch,
    count_pfm_per_batch: GpuCountPfmPerBatch,
    contact_offsets_scan: GpuContactOffsetsScan,
    zero_contact_lens: GpuZeroContactLens,
    flat_list_dispatch: GpuFlatListDispatch,
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
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        contact_offsets: &mut Tensor<u32>,
        pair_batch_counts: &mut Tensor<u32>,
        pfm_batch_counts: &mut Tensor<u32>,
        mb_sweep_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
        sim_params: &Tensor<crate::shaders::dynamics::RbdSimParams>,
        // Optional: merge each collider pair's manifolds into one before the
        // solvers see them. `false` skips the kernel entirely.
        reduce_contacts: bool,
        collision_pairs_indirect: &Tensor<[u32; 3]>,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        self.reset_narrow_phase.call(
            pass,
            [num_batches, 1, 1],
            contacts_len,
            pfm_pairs_len,
            pair_batch_counts,
            pfm_batch_counts,
        )?;

        // Pass 2: defer the complex shape pairs into `pfm_pairs` (kept as a
        // separate dispatch so each pass fits 8 storage buffers).
        self.narrow_phase_deferred.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            sim_params,
            vertices,
            indices,
        )?;

        self.count_pairs_per_batch.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            collision_pairs_len,
            pair_batch_counts,
            batch_indices,
        )?;
        self.flat_list_dispatch
            .call(pass, 1u32, pfm_pairs_len, pfm_pairs_indirect, batch_indices)?;
        self.count_pfm_per_batch.call(
            pass,
            &*pfm_pairs_indirect,
            pfm_pairs,
            pfm_pairs_len,
            pfm_batch_counts,
            batch_indices,
        )?;
        self.contact_offsets_scan.call(
            pass,
            1u32,
            pair_batch_counts,
            pfm_batch_counts,
            collision_pairs_len,
            pfm_pairs_len,
            contact_offsets,
            contacts_indirect,
            batch_indices,
        )?;
        self.zero_contact_lens.call(
            pass,
            &*contacts_indirect,
            contacts,
            contact_offsets,
            batch_indices,
        )?;

        self.narrow_phase.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            contact_offsets,
            poses,
            shapes,
            contacts,
            contacts_len,
            batch_indices,
            collider_parent,
            collider_materials,
            sim_params,
        )?;
        self.narrow_phase_pfm_pfm.call(
            pass,
            &*pfm_pairs_indirect,
            contacts,
            contacts_len,
            pfm_pairs,
            contact_offsets,
            batch_indices,
            vertices,
            indices,
            collider_parent,
            collider_materials,
            sim_params,
        )?;
        #[cfg(feature = "dim3")]
        if reduce_contacts {
            self.reduce_contacts.call(
                pass,
                [1u32, num_batches, 1],
                contacts,
                contacts_len,
                contact_offsets,
                batch_indices,
                sim_params,
            )?;
        }
        #[cfg(not(feature = "dim3"))]
        let _ = reduce_contacts;
        self.init_contacts_indirect_args.call(
            pass,
            256u32,
            contacts_len,
            mb_sweep_indirect,
            batch_indices,
        )?;

        Ok(())
    }
}
