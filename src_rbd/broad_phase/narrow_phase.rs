//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
#[cfg(feature = "dim3")]
use crate::shaders::broad_phase::GpuReduceContacts;
use crate::shaders::broad_phase::{
    CollisionPair, ContactPlan, GpuContactPlan, GpuNarrowPhasePfmPfm, GpuNarrowPhaseShapeShape,
    GpuNarrowPhaseShapeShapeDeferred, GpuPfmSortKeys, GpuResetNarrowPhase, NarrowPhasePfmPair,
};
use crate::shaders::shapes::Shape;
use crate::utils::{RadixSort, RadixSortWorkspace};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct GpuNarrowPhaseShaders {
    reset_narrow_phase: GpuResetNarrowPhase,
    narrow_phase: GpuNarrowPhaseShapeShape,
    /// `pfm_pairs` work-list. Split from `narrow_phase` to fit 8 storage buffers.
    narrow_phase_deferred: GpuNarrowPhaseShapeShapeDeferred,
    narrow_phase_pfm_pfm: GpuNarrowPhasePfmPfm,
    #[cfg(feature = "dim3")]
    reduce_contacts: GpuReduceContacts,
    contact_plan: GpuContactPlan,
    pfm_sort_keys: GpuPfmSortKeys,
}

pub struct GpuNarrowPhase {
    shaders: GpuNarrowPhaseShaders,
    sort: RadixSort,
}

impl GpuNarrowPhase {
    pub fn from_backend(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            shaders: GpuNarrowPhaseShaders::from_backend(backend)?,
            sort: RadixSort::from_backend(backend)?,
        })
    }
}

pub struct PfmSortState {
    keys: Tensor<u32>,
    identity: Tensor<u32>,
    sorted_keys: Tensor<u32>,
    sorted_values: Tensor<u32>,
    sort_len: Tensor<u32>,
    workspace: RadixSortWorkspace,
}

impl PfmSortState {
    pub fn new(backend: &GpuBackend, capacity: u32) -> Self {
        let storage = BufferUsages::STORAGE;
        let identity: Vec<u32> = (0..capacity).collect();
        Self {
            keys: Tensor::vector_uninit(backend, capacity.max(1), storage).unwrap(),
            identity: Tensor::vector(backend, &identity, storage).unwrap(),
            sorted_keys: Tensor::vector_uninit(backend, capacity.max(1), storage).unwrap(),
            sorted_values: Tensor::vector_uninit(backend, capacity.max(1), storage).unwrap(),
            sort_len: Tensor::vector(backend, &[0u32], storage).unwrap(),
            workspace: RadixSortWorkspace::new(backend),
        }
    }

    pub fn resize(&mut self, backend: &GpuBackend, capacity: u32) {
        *self = Self::new(backend, capacity);
    }

    #[cfg(feature = "dim3")]
    pub fn sorted_keys(&self) -> &Tensor<u32> {
        &self.sorted_keys
    }
}

impl GpuNarrowPhase {
    /// Dispatches the narrow-phase collision detection pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        poses: &Tensor<Pose>,
        shapes: &Tensor<Shape>,
        vertices: &Tensor<PaddedVector>,
        indices: &Tensor<u32>,
        collision_pairs: &Tensor<CollisionPair>,
        collision_pairs_len: &mut Tensor<u32>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        contact_plan: &mut Tensor<ContactPlan>,
        mb_sweep_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        pfm_sort: &mut PfmSortState,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
        sim_params: &Tensor<crate::shaders::dynamics::RbdSimParams>,
        // Optional: merge each collider pair's manifolds into one before the
        reduce_contacts: bool,
        collision_pairs_indirect: &Tensor<[u32; 3]>,
        collision_pairs_capacity: u32,
    ) -> Result<(), GpuBackendError> {
        let reduce_contacts = reduce_contacts && cfg!(feature = "dim3");

        self.shaders
            .reset_narrow_phase
            .call(pass, 1u32, pfm_pairs_len)?;

        self.shaders.narrow_phase_deferred.call(
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

        self.shaders.contact_plan.call(
            pass,
            1u32,
            collision_pairs_len,
            pfm_pairs_len,
            contact_plan,
            &mut pfm_sort.sort_len,
            contacts_indirect,
            pfm_pairs_indirect,
            mb_sweep_indirect,
            batch_indices,
        )?;

        self.shaders.narrow_phase.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            &*contact_plan,
            poses,
            shapes,
            contacts,
            collider_parent,
            collider_materials,
            sim_params,
        )?;

        if reduce_contacts {
            self.shaders.pfm_sort_keys.call(
                pass,
                &*pfm_pairs_indirect,
                pfm_pairs,
                &*contact_plan,
                &mut pfm_sort.keys,
            )?;
            let sorting_bits =
                (32 - collision_pairs_capacity.saturating_sub(1).leading_zeros()).max(1);
            let PfmSortState {
                keys,
                identity,
                sorted_keys,
                sorted_values,
                sort_len,
                workspace,
            } = pfm_sort;
            self.sort.dispatch(
                backend,
                pass,
                workspace,
                keys,
                identity,
                sort_len,
                sorting_bits,
                1,
                sorted_keys,
                sorted_values,
            )?;
        }

        let pfm_order = if reduce_contacts {
            &pfm_sort.sorted_values
        } else {
            &pfm_sort.identity
        };
        self.shaders.narrow_phase_pfm_pfm.call(
            pass,
            &*pfm_pairs_indirect,
            contacts,
            pfm_pairs,
            pfm_order,
            &*contact_plan,
            vertices,
            indices,
            collider_parent,
            collider_materials,
            sim_params,
        )?;

        #[cfg(feature = "dim3")]
        if reduce_contacts {
            self.shaders.reduce_contacts.call(
                pass,
                &*pfm_pairs_indirect,
                contacts,
                &pfm_sort.sorted_keys,
                &*contact_plan,
                sim_params,
            )?;
        }

        Ok(())
    }
}
