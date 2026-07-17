//! Linear Bounding Volume Hierarchy (LBVH) broad-phase collision detection.
//!
//! Implements the Karras 2012 parallel LBVH construction algorithm on the GPU,
//! providing O(n log n) collision detection suitable for large dynamic scenes.

use crate::math::Pose;
use crate::shaders::PaddedVector;
use crate::shaders::bounding_volumes::Aabb;
use crate::shaders::broad_phase::{
    CollisionPair, GpuLbvhBuild, GpuLbvhComputeDomain, GpuLbvhComputeMorton,
    GpuLbvhFindCollisionPairs, GpuLbvhInitDispatch, GpuLbvhRefitInternal, GpuLbvhRefitLeaves,
    GpuLbvhResetCollisionPairs, LbvhNode,
};
use crate::shaders::shapes::Shape;
use crate::utils::{RadixSort, RadixSortWorkspace};
use khal::backend::{
    Backend, Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuPass, GpuTimestamps,
};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU shader for Linear Bounding Volume Hierarchy (LBVH) construction and traversal.
///
/// Implements the Karras 2012 parallel LBVH construction algorithm on the GPU, providing
/// O(n log n) collision detection suitable for large dynamic scenes.
#[derive(Shader)]
pub struct GpuLbvh {
    compute_domain: GpuLbvhComputeDomain,
    compute_morton: GpuLbvhComputeMorton,
    build: GpuLbvhBuild,
    refit_leaves: GpuLbvhRefitLeaves,
    refit_internal: GpuLbvhRefitInternal,
    reset_collision_pairs: GpuLbvhResetCollisionPairs,
    find_collision_pairs: GpuLbvhFindCollisionPairs,
    lbvh_init_indirect_args: GpuLbvhInitDispatch,
}

/// GPU-resident state for LBVH construction and queries.
///
/// Buffers automatically resize when the number of colliders changes.
pub struct LbvhState {
    buffer_usages: BufferUsages,
    domain_aabb: Tensor<Aabb>,
    n_sort: Tensor<u32>,
    /// Per-batch active key count currently uploaded to `n_sort`, as
    /// `(active_per_batch, num_batches)`. `None` forces a re-upload (e.g. after
    /// a resize re-seeds `n_sort` with the capacity). Avoids rewriting `n_sort`
    /// every frame when the live collider count hasn't changed.
    n_sort_active: Option<(u32, u32)>,
    unsorted_morton_keys: Tensor<u32>,
    sorted_morton_keys: Tensor<u32>,
    unsorted_colliders: Tensor<u32>,
    sorted_colliders: Tensor<u32>,
    tree: Tensor<LbvhNode>,
    sort_workspace: RadixSortWorkspace,
}

/// High-level LBVH broad-phase interface.
pub struct Lbvh {
    shaders: GpuLbvh,
    sort: RadixSort,
}

impl LbvhState {
    /// Creates a new LBVH state with default buffer usage flags.
    pub fn new(backend: &GpuBackend) -> Self {
        Self::with_usages(backend, BufferUsages::STORAGE)
    }

    /// Creates a new LBVH state with custom buffer usage flags.
    pub fn with_usages(backend: &GpuBackend, usages: BufferUsages) -> Self {
        Self {
            n_sort: Tensor::scalar(backend, 0, usages).unwrap(),
            n_sort_active: None,
            domain_aabb: Tensor::scalar_uninit(backend, usages).unwrap(),
            unsorted_morton_keys: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            sorted_morton_keys: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            unsorted_colliders: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            sorted_colliders: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            tree: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            sort_workspace: RadixSortWorkspace::new(backend),
            buffer_usages: usages,
        }
    }

    pub(crate) fn tree(&self) -> &Tensor<LbvhNode> {
        &self.tree
    }

    pub(crate) fn sorted_colliders(&self) -> &Tensor<u32> {
        &self.sorted_colliders
    }

    fn resize_buffers(&mut self, backend: &GpuBackend, colliders_len: u32, num_batches: u32) {
        if (self.domain_aabb.len() as u32) < num_batches {
            self.domain_aabb =
                Tensor::vector_uninit(backend, num_batches, self.buffer_usages).unwrap();
        }

        // NOTE: colliders_len is the total colliders count, already taking all batches into account.
        if (self.tree.len() as u32) < 2 * colliders_len {
            self.unsorted_morton_keys =
                Tensor::vector_uninit(backend, colliders_len, self.buffer_usages).unwrap();
            self.sorted_morton_keys =
                Tensor::vector_uninit(backend, colliders_len, self.buffer_usages).unwrap();
            // Use per-batch LOCAL indices so that after sorting, each batch's
            // sorted_colliders slice contains local indices usable with per-batch
            // Slice offsets in the shaders.
            let colliders_per_batch = colliders_len / num_batches;
            let unsorted_colliders: Vec<_> = (0..num_batches)
                .flat_map(|_| 0..colliders_per_batch)
                .collect();
            self.unsorted_colliders =
                Tensor::vector(backend, &unsorted_colliders, self.buffer_usages).unwrap();
            self.sorted_colliders =
                Tensor::vector_uninit(backend, colliders_len, self.buffer_usages).unwrap();
            self.tree =
                Tensor::vector_uninit(backend, 2 * colliders_len, self.buffer_usages).unwrap();

            // FIXME: this doesn’t account for batches having mismatched numbers of colliders.
            // n_sort is a per-batch vector: each element is the per-batch *active*
            // key count, rewritten by `update_tree` (hence COPY_DST) when the live
            // collider count changes, so dynamic body insertion/removal is reflected
            // without a resize. Seeded here with the capacity; the next `update_tree`
            // narrows it to the live count (the invalidated cache below forces it).
            let n_sort_data = vec![colliders_per_batch; num_batches as usize];
            self.n_sort = Tensor::vector(
                backend,
                &n_sort_data,
                self.buffer_usages | BufferUsages::COPY_DST,
            )
            .unwrap();
            // The new buffer holds the capacity, not the active count — force the
            // next `update_tree` to upload the real per-batch live count.
            self.n_sort_active = None;
        }
    }
}

impl Lbvh {
    /// Creates a new LBVH instance by loading shaders on the given backend.
    pub fn from_backend(backend: &GpuBackend) -> Self {
        Self {
            shaders: GpuLbvh::from_backend(backend).unwrap(),
            sort: RadixSort::from_backend(backend).unwrap(),
        }
    }

    /// Rebuilds the LBVH tree from current collider poses and shapes.
    ///
    /// Should be called each frame before [`find_pairs`](Self::find_pairs) if colliders have moved.
    pub fn update_tree(
        &self,
        backend: &GpuBackend,
        encoder: &mut GpuEncoder,
        state: &mut LbvhState,
        colliders_len: u32,
        active_per_batch: u32,
        num_batches: u32,
        poses: &Tensor<Pose>,
        vertex_buffers: &Tensor<PaddedVector>,
        shapes: &Tensor<Shape>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        // `colliders_len` is the full per-batch *capacity* × batches (it sizes the
        // buffers and is the per-batch stride). The sort and tree build, however,
        // only need to touch the live colliders, so every per-collider dispatch
        // below — and the radix sort's `n_sort` — uses `active_per_batch`. The
        // padding slots `[active_per_batch, capacity)` are never sorted or built.
        state.resize_buffers(backend, colliders_len, num_batches);

        // Tell the radix sort how many keys are actually live per batch (it sizes
        // its indirect dispatch from `max(n_sort)` and sorts only that many,
        // leaving padding untouched). The live count only changes when bodies are
        // added/removed (or after a resize re-seeds the buffer), so skip the upload
        // when it's unchanged rather than rewriting `n_sort` every frame.
        if state.n_sort_active != Some((active_per_batch, num_batches)) {
            let n_sort_data = vec![active_per_batch; num_batches as usize];
            backend.write_buffer(state.n_sort.buffer_mut(), 0, &n_sort_data)?;
            state.n_sort_active = Some((active_per_batch, num_batches));
        }

        let colliders_per_batch = active_per_batch;

        let mut pass = encoder.begin_pass("[RBD] lbvh-compute-domain", timestamps.as_deref_mut());
        self.shaders.compute_domain.call(
            &mut pass,
            [1u32, num_batches, 1],
            poses,
            &mut state.domain_aabb,
            batch_indices,
        )?;
        drop(pass);

        let mut pass = encoder.begin_pass("[RBD] lbvh-compute-morton", timestamps.as_deref_mut());
        self.shaders.compute_morton.call(
            &mut pass,
            [colliders_per_batch, num_batches, 1],
            poses,
            &state.domain_aabb,
            &mut state.unsorted_morton_keys,
            batch_indices,
        )?;
        drop(pass);

        let mut pass = encoder.begin_pass("[RBD] lbvh-sort-dispatch", timestamps.as_deref_mut());
        self.sort.dispatch(
            backend,
            &mut pass,
            &mut state.sort_workspace,
            &state.unsorted_morton_keys,
            &state.unsorted_colliders,
            &state.n_sort,
            32,
            num_batches,
            &mut state.sorted_morton_keys,
            &mut state.sorted_colliders,
        )?;
        drop(pass);

        let mut pass = encoder.begin_pass("[RBD] lbvh-build", timestamps.as_deref_mut());
        self.shaders.build.call(
            &mut pass,
            [colliders_per_batch.saturating_sub(1), num_batches, 1],
            &state.sorted_morton_keys,
            &mut state.tree,
            batch_indices,
        )?;
        drop(pass);

        let mut pass = encoder.begin_pass("[RBD] lbvh-refit_leaves", timestamps.as_deref_mut());
        self.shaders.refit_leaves.call(
            &mut pass,
            // Flat over all batches' leaves (kernel recovers batch by division).
            [colliders_per_batch * num_batches, 1, 1],
            poses,
            shapes,
            &state.sorted_colliders,
            &mut state.tree,
            batch_indices,
            vertex_buffers,
        )?;
        drop(pass);

        let mut pass = encoder.begin_pass("[RBD] lbvh-refit-internal", timestamps);
        self.shaders.refit_internal.call(
            &mut pass,
            [1u32, num_batches, 1],
            &mut state.tree,
            batch_indices,
        )?;
        drop(pass);

        Ok(())
    }

    /// Traverses the LBVH tree to find potentially colliding pairs.
    ///
    /// After the tree has been built with [`update_tree`](Self::update_tree), this method
    /// traverses it to identify pairs of colliders whose AABBs overlap.
    pub fn find_pairs(
        &self,
        pass: &mut GpuPass,
        state: &mut LbvhState,
        active_per_batch: u32,
        num_batches: u32,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collision_pairs: &mut Tensor<CollisionPair>,
        collision_pairs_len: &mut Tensor<u32>,
        collision_pairs_indirect: &mut Tensor<[u32; 3]>,
        collision_groups: &Tensor<crate::rapier::geometry::InteractionGroups>,
    ) -> Result<(), GpuBackendError> {
        // One thread per live collider (leaf); padding slots aren't in the tree.
        let colliders_per_batch = active_per_batch;

        self.shaders.reset_collision_pairs.call(
            pass,
            [1u32, num_batches, 1],
            collision_pairs_len,
        )?;
        self.shaders.find_collision_pairs.call(
            pass,
            [colliders_per_batch, num_batches, 1],
            &state.tree,
            collision_pairs,
            collision_pairs_len,
            collision_groups,
            batch_indices,
        )?;
        self.shaders.lbvh_init_indirect_args.call(
            pass,
            1u32,
            collision_pairs_len,
            collision_pairs_indirect,
        )?;
        Ok(())
    }
}
