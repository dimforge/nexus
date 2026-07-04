use crate::utils::{Slice, SliceMut};

/// Per-batch capacities and packed-buffer section offsets, shared by every
/// kernel that needs to slice a flat tensor into its batch's slot.
///
/// Combining 30+ scalar uniforms into a single struct keeps the WebGPU
/// uniform count under control and centralises the per-buffer slicing logic.
#[derive(Copy, Clone, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BatchIndices {
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

    /*
     * Intra-batch offsets for multi-purpose buffers.
     * These are buffers that were combined into a single storage
     * buffer to comply with the 10 storage buffers limit on the web.
     */
    pub coriolis_w_section_offset: u32,
    pub i_coriolis_dt_section_offset: u32,
    pub dof_damping_section_offset: u32,
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
    pub fn mb_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.multibodies_batch_capacity as usize
    }

    #[inline]
    pub fn links_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.links_batch_capacity as usize
    }

    #[inline]
    pub fn jac_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.jacobians_batch_capacity as usize
    }

    #[inline]
    pub fn mm_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.mass_matrix_batch_capacity as usize
    }

    #[inline]
    pub fn cor_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.coriolis_batch_capacity as usize
    }

    #[inline]
    pub fn icdt_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.i_coriolis_dt_batch_capacity as usize
    }

    #[inline]
    pub fn dof_start(&self, batch_id: u32) -> usize {
        batch_id as usize * self.dof_batch_capacity as usize
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
    pub fn mb_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.mb_start(batch_id))
    }

    #[inline]
    pub fn mb_batch_mut<'s, T>(&self, batch_id: u32, slice: &'s mut [T]) -> SliceMut<'s, T> {
        SliceMut(slice, self.mb_start(batch_id))
    }

    #[inline]
    pub fn mb_links_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.links_start(batch_id))
    }

    #[inline]
    pub fn mb_links_batch_mut<'s, T>(&self, batch_id: u32, slice: &'s mut [T]) -> SliceMut<'s, T> {
        SliceMut(slice, self.links_start(batch_id))
    }

    #[inline]
    pub fn dof_batch<'s, T>(&self, batch_id: u32, slice: &'s [T]) -> Slice<'s, T> {
        Slice(slice, self.dof_start(batch_id))
    }

    #[inline]
    pub fn dof_batch_mut<'s, T>(&self, batch_id: u32, slice: &'s mut [T]) -> SliceMut<'s, T> {
        SliceMut(slice, self.dof_start(batch_id))
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
