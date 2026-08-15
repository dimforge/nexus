use crate::utils::linalg::{MatSlice, VSlice};
use crate::utils::{ISlice, ISliceMut, Slice, SliceMut};

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
    pub collision_pairs_batch_capacity: u32,
    pub contacts_batch_capacity: u32,
    /// Free-body impulse joints — buffer stride (capacity) per batch.
    pub impulse_joints_batch_capacity: u32,
    /// Number of *active* free-body impulse joints per batch (the loop bound).
    pub impulse_joints_len: u32,

    /*
     * Multibody core capacities.
     */
    pub multibodies_batch_capacity: u32,
    /// Number of *active* multibodies per batch (the loop bound for
    /// per-multibody kernels).
    pub multibodies_len: u32,
    pub links_batch_capacity: u32,
    pub jacobians_batch_capacity: u32,
    pub mass_matrix_batch_capacity: u32,
    pub coriolis_batch_capacity: u32,
    pub i_coriolis_dt_batch_capacity: u32,
    pub dof_batch_capacity: u32,

    /*
     * Multibody constraint slab capacities.
     */
    pub mb_joint_constraints_batch_capacity: u32,
    pub mb_joint_constraint_columns_batch_capacity: u32,
    pub mb_contact_constraints_batch_capacity: u32,
    pub mb_contact_constraint_columns_batch_capacity: u32,
    pub mb_imp_joints_batch_capacity: u32,
    pub mb_imp_joint_constraints_batch_capacity: u32,
    pub mb_imp_joint_jacobians_batch_capacity: u32,
    /// Multibody-touching impulse-joint color-group slab (per-batch stride
    /// = number of colors). The free-body impulse-joint color groups, by
    /// contrast, are stored single-batch (identical coloring across batches)
    /// and read at offset 0.
    pub mb_imp_joint_color_groups_batch_capacity: u32,
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
    pub coriolis_w_section_offset: u32,
    pub i_coriolis_dt_section_offset: u32,
    pub dof_damping_section_offset: u32,
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
    #[inline]
    pub fn coll_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.colliders_batch_capacity as usize
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

    /// Interleaved dense matrix view at intra-batch element offset `offset`.
    #[inline]
    pub fn imat(&self, batch_id: u32, offset: usize, rows: u32, cols: u32) -> MatSlice {
        MatSlice::interleaved(offset, rows, cols, self.num_batches, batch_id)
    }

    /// Interleaved vector view at intra-batch element offset `offset`.
    #[inline]
    pub fn ivec(&self, batch_id: u32, offset: usize) -> VSlice {
        VSlice::interleaved(offset, self.num_batches, batch_id)
    }

    #[inline]
    pub fn collision_pairs_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.collision_pairs_batch_capacity as usize
    }

    #[inline]
    pub fn contacts_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.contacts_batch_capacity as usize
    }

    #[inline]
    pub fn impulse_joints_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.impulse_joints_batch_capacity as usize
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
    pub fn mb_contact_constraints_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_contact_constraints_batch_capacity as usize
    }

    #[inline]
    pub fn mb_contact_constraint_columns_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_contact_constraint_columns_batch_capacity as usize
    }

    #[inline]
    pub fn mb_dof_couplings_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_dof_couplings_batch_capacity as usize
    }

    #[inline]
    pub fn mb_imp_joints_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_imp_joints_batch_capacity as usize
    }

    #[inline]
    pub fn mb_imp_joint_constraints_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_imp_joint_constraints_batch_capacity as usize
    }

    #[inline]
    pub fn mb_imp_joint_jacobians_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_imp_joint_jacobians_batch_capacity as usize
    }

    #[inline]
    pub fn mb_imp_joint_color_groups_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mb_imp_joint_color_groups_batch_capacity as usize
    }

    /*
     * Typed batch slices — for buffers consumed via `Slice<T>` / `SliceMut<T>`
     * wrappers rather than as raw f32 arrays.
     */
    #[inline]
    pub fn coll_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.coll_start(batch_id))
    }

    #[inline]
    pub fn coll_batch_mut<'s, T>(&self, batch_id: u32, slice: &'s mut [T]) -> SliceMut<'s, T> {
        SliceMut(slice, self.coll_start(batch_id))
    }

    #[inline]
    pub fn collision_pairs_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.collision_pairs_start(batch_id))
    }

    #[inline]
    pub fn collision_pairs_batch_mut<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s mut [T],
    ) -> SliceMut<'s, T> {
        SliceMut(slice, self.collision_pairs_start(batch_id))
    }

    #[inline]
    pub fn contact_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.contacts_start(batch_id))
    }

    #[inline]
    pub fn contact_batch_mut<'s, T>(&self, batch_id: u32, slice: &'s mut [T]) -> SliceMut<'s, T> {
        SliceMut(slice, self.contacts_start(batch_id))
    }

    #[inline]
    pub fn impulse_joints_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.impulse_joints_start(batch_id))
    }

    #[inline]
    pub fn impulse_joints_batch_mut<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s mut [T],
    ) -> SliceMut<'s, T> {
        SliceMut(slice, self.impulse_joints_start(batch_id))
    }

    #[inline]
    pub fn mb_joint_constraints_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.mb_joint_constraints_start(batch_id))
    }

    #[inline]
    pub fn mb_joint_constraints_batch_mut<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s mut [T],
    ) -> SliceMut<'s, T> {
        SliceMut(slice, self.mb_joint_constraints_start(batch_id))
    }

    #[inline]
    pub fn mb_contact_constraints_batch<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s [T],
    ) -> Slice<'s, T> {
        Slice(slice, self.mb_contact_constraints_start(batch_id))
    }

    #[inline]
    pub fn mb_contact_constraints_batch_mut<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s mut [T],
    ) -> SliceMut<'s, T> {
        SliceMut(slice, self.mb_contact_constraints_start(batch_id))
    }

    #[inline]
    pub fn mb_imp_joints_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.mb_imp_joints_start(batch_id))
    }

    #[inline]
    pub fn mb_imp_joints_batch_mut<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s mut [T],
    ) -> SliceMut<'s, T> {
        SliceMut(slice, self.mb_imp_joints_start(batch_id))
    }

    #[inline]
    pub fn mb_imp_joint_constraints_batch<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s [T],
    ) -> Slice<'s, T> {
        Slice(slice, self.mb_imp_joint_constraints_start(batch_id))
    }

    #[inline]
    pub fn mb_imp_joint_constraints_batch_mut<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s mut [T],
    ) -> SliceMut<'s, T> {
        SliceMut(slice, self.mb_imp_joint_constraints_start(batch_id))
    }

    #[inline]
    pub fn mb_imp_joint_color_groups_batch<'s, T>(
        &self,
        batch_id: u32,
        slice: &'s [T],
    ) -> Slice<'s, T> {
        Slice(slice, self.mb_imp_joint_color_groups_start(batch_id))
    }
}
