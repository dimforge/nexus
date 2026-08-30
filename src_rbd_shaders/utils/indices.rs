use crate::utils::{ISlice, ISliceMut};

/// Per-batch capacities and packed-buffer section offsets, shared by every
/// kernel that needs to slice a flat tensor into its batch's slot.
///
/// Combining 30+ scalar uniforms into a single struct keeps the WebGPU
/// uniform count under control and centralises the per-buffer slicing logic.
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BatchIndices {
    /// Total number of simulation batches (environments).
    pub num_batches: u32,

    /*
     * RBD / collision-detection capacities.
     */
    pub colliders_batch_capacity: u32,
    /// Number of *active* colliders per batch.
    pub colliders_len: u32,
    /// Number of *active* rigid bodies per batch.
    pub bodies_len: u32,
    /// Total capacity of the flat collision-pair buffer (all batches share it;
    /// also the capacity of the flat PFM work-list, which is sized identically).
    pub collision_pairs_capacity: u32,
    /// Total capacity of the flat contacts buffer (and of every contacts-keyed
    /// buffer: constraints, builders, colors, sorted ids). Contact slots are
    /// positional (pair `t` owns slot `t`, PFM entry `i` owns slot
    /// `pairs_total + i`), so this is always `2 * collision_pairs_capacity`.
    pub contacts_capacity: u32,
    /// Number of *active* free-body impulse joints per batch (the loop bound;
    /// the joint-keyed buffers are batch-interleaved).
    pub impulse_joints_len: u32,

    /*
     * Multibody core capacities.
     */
    pub multibodies_batch_capacity: u32,
    /// Number of *active* multibodies per batch (the loop bound for
    /// per-multibody kernels).
    pub multibodies_len: u32,
    pub links_batch_capacity: u32,
    pub coriolis_batch_capacity: u32,
    pub dof_batch_capacity: u32,

    /*
     * Multibody constraint slab capacities.
     */
    pub mb_joint_constraints_batch_capacity: u32,
    pub mb_joint_constraint_columns_batch_capacity: u32,
    /// Total capacity (in slots) of the flat multibody contact-constraint
    /// buffer; per-multibody segments within it are dynamic (see
    /// `MultibodyInfo::contact_constraint_start`). The paired jac/column arena
    /// holds `2 * dof_batch_capacity` floats per slot (`Jᵀ` row, then its
    /// `M⁻¹·Jᵀ` column).
    pub mb_contact_constraints_capacity: u32,
    /// Per-batch multibody-touching impulse-joint slot count (loop bound for
    /// the flat sweeps).
    pub mb_imp_joints_batch_capacity: u32,
    /// Actual max `ndofs` across every multibody in every batch (often smaller
    /// than the fixed `MAX_MB_DOFS` limit).
    pub mb_max_ndofs: u32,
    /// Actual max link count across every multibody in every batch.
    pub mb_max_links: u32,
    /// Lanes per multibody for the packed per-multibody workgroup kernels:
    /// `next_power_of_two(mb_max_ndofs).clamp(8, 64)`.
    pub mb_pack_lanes: u32,
    /// Max `max_constraints` across every multibody in every batch.
    pub mb_max_joint_constraints: u32,
    /// Per-batch stride of the contact-solver color-bucket buffers
    /// (`color_counts` / `color_starts` / `color_cursors`), = `max_colors + 3`
    /// so that `starts[c + 1]` is in bounds for every swept color.
    pub solver_color_buckets_stride: u32,

    /*
     * Intra-batch offsets for multi-purpose buffers.
     * These are buffers that were combined into a single storage
     * buffer to comply with the 10 storage buffers limit on the web.
     */
    /// Offset (in f32 entries, within a batch's `mass_matrices` view) of the
    /// section holding the coriolis-aware "acceleration" mass matrix
    /// (rapier's `acc_augmented_mass`).
    /// Non-zero = implicit coriolis on: split matrices, the acc section drives
    /// the acceleration solve while the plain matrix drives constraints.
    /// Zero = implicit coriolis off: a single plain (coriolis-free) matrix
    /// serves both; coriolis/gyroscopic forces are applied explicitly only.
    pub mass_matrix_acc_section_offset: u32,
    /// Per-batch stride (capacity) of the multibody DoF-coupling buffer.
    pub mb_dof_couplings_batch_capacity: u32,
}

impl BatchIndices {
    /*
     * Raw batch-start offsets (in element units, not bytes) for buffers
     * whose batch stride is one of the `*_batch_capacity` fields. Used to
     * compute base indices into flat f32 buffers (e.g. when constructing a
     * `MatSlice::dense(base, ...)`).
     */
    /// Global id of body/collider `local` of `batch_id`: bodies are stored
    /// batch-interleaved (`local * num_batches + batch`), so the same
    /// topological entity across environments sits on adjacent lanes.
    #[inline]
    pub fn body_global(&self, batch_id: u32, local: u32) -> usize {
        local as usize * self.num_batches as usize + batch_id as usize
    }

    /// Strided body indexer for `batch_id` (see [`BodyIx`]), for helpers that
    /// resolve env-local body ids without carrying a slice view.
    #[inline]
    pub fn body_ix(&self, batch_id: u32) -> BodyIx {
        BodyIx {
            stride: self.num_batches,
            shift: batch_id,
        }
    }

    /// Interleaved flat index for the multibody dynamics buffers.
    #[inline]
    pub fn mbi(&self, batch_id: u32, intra: usize) -> usize {
        intra * self.num_batches as usize + batch_id as usize
    }

    /// Interleaved view of a multibody dynamics buffer for batch `batch_id`
    /// (use `.offset(...)` for the intra-batch element offset).
    #[inline]
    pub fn ib<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> ISlice<'s, T> {
        ISlice {
            buf: slice,
            base: 0,
            stride: self.num_batches,
            shift: batch_id,
        }
    }

    /// Mutable interleaved view.
    #[inline]
    pub fn ib_mut<'s, T>(&self, batch_id: u32, slice: &'s mut [T]) -> ISliceMut<'s, T> {
        ISliceMut {
            buf: slice,
            base: 0,
            stride: self.num_batches,
            shift: batch_id,
        }
    }

    /// Base of batch `batch_id`'s dense region in a per-multibody-region
    /// buffer (mass matrices, LU pivots, body jacobians, coriolis blocks,
    /// generalized forces, impulse-joint jacobians): regions tile as
    /// `offset * num_batches + batch * len`, dense inside.
    #[inline]
    pub fn mb_region(&self, batch_id: u32, offset: u32, len: u32) -> usize {
        offset as usize * self.num_batches as usize + batch_id as usize * len as usize
    }

    /// Batch owning the collider with global id `collider_id` (the flat
    /// collision-pair and contact buffers store global collider/body ids;
    /// bodies are batch-interleaved so the batch is the id modulo the batch
    /// count).
    #[inline]
    pub fn collider_batch(&self, collider_id: u32) -> u32 {
        collider_id % self.num_batches
    }

    #[inline]
    pub fn mb_joint_constraints_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_joint_constraints_batch_capacity as usize
    }

    #[inline]
    pub fn mb_joint_constraint_columns_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_joint_constraint_columns_batch_capacity as usize
    }

    #[inline]
    pub fn mb_dof_couplings_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_dof_couplings_batch_capacity as usize
    }
}

/// Strided indexer mapping an env-local body/collider id to its global slot in
/// the batch-interleaved per-body buffers: `global = id * stride + shift`
/// (`stride = num_batches`, `shift = batch_id`).
#[derive(Copy, Clone)]
pub struct BodyIx {
    pub stride: u32,
    pub shift: u32,
}

impl BodyIx {
    #[inline(always)]
    pub fn at(self, id: u32) -> usize {
        id as usize * self.stride as usize + self.shift as usize
    }
}
