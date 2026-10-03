//! Host side of the passes that put the contacts and constraint lists in a fixed order.

use crate::queries::GpuIndexedContact;
use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::{
    CONTACT_KEY_COLLIDER_A, CONTACT_KEY_COLLIDER_B, CONTACT_KEY_IDENTITY, CONTACT_KEY_PAIR,
    CONTACT_KEY_SUB_SHAPE, GpuContactSortKeys, GpuPermuteContacts, GpuStabilizeBodyConstraintIds,
};
use crate::shaders::utils::BatchIndices;
use crate::utils::{RadixSort, RadixSortWorkspace};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

#[derive(Shader)]
struct CanonicalOrderShaders {
    contact_sort_keys: GpuContactSortKeys,
    permute_contacts: GpuPermuteContacts,
    stabilize_body_constraint_ids: GpuStabilizeBodyConstraintIds,
}

/// GPU kernels that put the per-step lists in a fixed order.
pub struct GpuCanonicalOrder {
    shaders: CanonicalOrderShaders,
    sort: RadixSort,
}

impl GpuCanonicalOrder {
    /// Loads the kernels.
    pub fn from_backend(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            shaders: CanonicalOrderShaders::from_backend(backend)?,
            sort: RadixSort::from_backend(backend)?,
        })
    }
}

/// The buffers of the contact sort, all sized like the contact buffer.
pub struct CanonicalContactsArgs<'a> {
    /// Contacts as written by the narrow phase.
    pub contacts: &'a Tensor<GpuIndexedContact>,
    /// Output of the permutation, swapped with `contacts` by the caller.
    pub contacts_scratch: &'a mut Tensor<GpuIndexedContact>,
    /// The list totals of this frame. `bound` is the number of contacts to sort.
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// The dispatch grid over `[0, bound)`.
    pub contacts_indirect: &'a Tensor<[u32; 3]>,
    /// The number of elements to sort, written by the key pass.
    pub sort_n: &'a mut Tensor<u32>,
    /// Sort input keys, rebuilt for each pass.
    pub sort_keys: &'a mut Tensor<u32>,
    /// Sort input values: the contact indices.
    pub sort_ids: &'a mut Tensor<u32>,
    /// Sorted keys (unused, but needed by the sort).
    pub sort_keys_out: &'a mut Tensor<u32>,
    /// Sorted values, i.e. the permutation read by the next pass.
    pub sort_ids_out: &'a mut Tensor<u32>,
    /// Workspace of the radix sort.
    pub sort_workspace: &'a mut RadixSortWorkspace,
    /// `key_selector_uniforms[s] == s`, built once so no buffer is rewritten between dispatches.
    pub key_selector_uniforms: &'a [Tensor<u32>],
    /// The uniform with `contact_sort_collider_shift`.
    pub batch_indices: &'a Tensor<BatchIndices>,
    /// Number of active colliders over all batches (bound of the global collider ids).
    pub num_colliders: u32,
    /// Whether any collider is a trimesh or polyline (several manifolds per pair).
    /// Without them, the sub-shape pass is skipped.
    pub has_composite_shapes: bool,
}

impl GpuCanonicalOrder {
    /// Sorts the contacts by `(collider_a, collider_b, subshape)`, empty slots last.
    /// The caller must then swap `contacts` and `contacts_scratch`.
    pub fn canonicalize_contacts(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        args: CanonicalContactsArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        // Least significant field first: the radix sort is stable.
        // Each pass costs many dispatches, so use a single pass when possible.
        let collider_bits = contact_sort_collider_shift(args.num_colliders);
        let mut passes = Vec::with_capacity(3);

        if !args.has_composite_shapes && collider_bits * 2 <= u32::BITS {
            // No composite shapes and the pair fits in one key: a single pass is enough.
            passes.push((CONTACT_KEY_PAIR, collider_bits * 2));
        } else {
            if args.has_composite_shapes {
                // A sub-shape index is a triangle or segment index, so it needs all 32 bits.
                passes.push((CONTACT_KEY_SUB_SHAPE, 32));
            }
            passes.push((CONTACT_KEY_COLLIDER_B, collider_bits));
            passes.push((CONTACT_KEY_COLLIDER_A, collider_bits));
        }

        for (pass_idx, (key, bits)) in passes.iter().enumerate() {
            // The first pass has no permutation yet: contact `i` is at index `i`.
            let selector = if pass_idx == 0 {
                key + CONTACT_KEY_IDENTITY
            } else {
                *key
            };

            self.shaders.contact_sort_keys.call(
                pass,
                args.contacts_indirect,
                args.contacts,
                args.contact_plan,
                &*args.sort_ids_out,
                args.sort_keys,
                args.sort_ids,
                &args.key_selector_uniforms[selector as usize],
                args.sort_n,
                args.batch_indices,
            )?;

            self.sort.dispatch(
                backend,
                pass,
                args.sort_workspace,
                args.sort_keys,
                args.sort_ids,
                args.sort_n,
                *bits,
                1,
                args.sort_keys_out,
                args.sort_ids_out,
            )?;
        }

        self.shaders.permute_contacts.call(
            pass,
            args.contacts_indirect,
            args.contacts,
            args.contacts_scratch,
            &*args.sort_ids_out,
            args.contact_plan,
        )?;

        Ok(())
    }

    /// Sorts the constraint list of each body by constraint index.
    /// The caller must then swap `body_constraint_ids` and `scratch`.
    pub fn stabilize_body_constraint_ids(
        &self,
        pass: &mut GpuPass,
        body_constraint_counts: &Tensor<u32>,
        body_constraint_ids: &Tensor<u32>,
        scratch: &mut Tensor<u32>,
        batch_indices: &Tensor<BatchIndices>,
        num_bodies: u32,
    ) -> Result<(), GpuBackendError> {
        // One workgroup per body, capped at 65535 workgroups.
        // The kernel loops over the remaining bodies.
        let workgroups = num_bodies.clamp(1, MAX_WORKGROUPS_PER_DIM);
        self.shaders.stabilize_body_constraint_ids.call(
            pass,
            [workgroups * WORKGROUP_SIZE, 1, 1],
            body_constraint_counts,
            body_constraint_ids,
            scratch,
            batch_indices,
        )?;
        Ok(())
    }
}

/// Maximum workgroups along one dispatch dimension (WebGPU's limit).
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Workgroup size of the `canonical_order` kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Number of bits needed to index `n` elements (at least one).
fn bits_for(n: u32) -> u32 {
    (u32::BITS - n.max(2).next_power_of_two().leading_zeros() - 1).clamp(1, 32)
}

/// Bits of a global collider id in the sort keys, also the shift of `collider_a` in a pair key.
/// One spare value keeps the empty-slot key above all valid keys.
pub fn contact_sort_collider_shift(num_colliders: u32) -> u32 {
    bits_for(num_colliders.saturating_add(1))
}
