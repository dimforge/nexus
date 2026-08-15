//! Particle sorting kernels for the sparse MPM grid.
//!
//! These kernels handle:
//! 1. Marking blocks as active based on particle positions.
//! 2. Counting particles per block.
//! 3. Building sorted particle arrays.
//!
//! The sorting pipeline runs in multiple passes:
//! 1. `touch_particle_blocks` / `touch_rigid_particle_blocks` - mark active blocks
//! 2. `mark_rigid_particles_needing_block` - flag rigid particles near block boundaries
//! 3. `update_block_particle_count` - count particles per active block
//! 4. `copy_particles_len_to_scan_value` - prepare prefix sum input
//! 5. prefix sum (external) - compute exclusive scan of particle counts
//! 6. `copy_scan_values_to_first_particles` - write back sorted offsets
//! 7. `finalize_particles_sort` - place particles in sorted order
//!
//! Rigid particles go through the same count / prefix-sum / finalize sequence
//! (`update_block_rigid_particle_count`, `copy_rigid_particles_len_to_scan_value`,
//! `copy_scan_values_to_first_rigid_particles`, `finalize_rigid_particles_sort`),
//! reusing the scan workspace after the regular particle sort completed.

// Indexed loops on purpose throughout this module: iterator-based loops over
// storage buffers are fragile under rust-gpu's SPIR-V codegen.
#![allow(clippy::needless_range_loop)]
use crate::grid::grid::*;
use crate::solver::particle::Position;
use crate::{IVector, UVector};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::atomic_add_u32;

#[cfg(feature = "dim2")]
const EXTRA_PARTICLE_MIN_SHIFT: u32 = 6;
#[cfg(feature = "dim3")]
const EXTRA_PARTICLE_MIN_SHIFT: u32 = 2;

/// Returns the within-block sort bucket for a particle counted/inserted into its
/// primary block: one bucket per associated-cell slab along the slowest-varying node
/// axis (y in 2D, z in 3D).
#[inline]
fn primary_sort_bucket(assoc: UVector) -> usize {
    #[cfg(feature = "dim2")]
    {
        assoc.y as usize
    }
    #[cfg(feature = "dim3")]
    {
        assoc.z as usize
    }
}

/// Returns the within-block sort bucket for a particle counted/inserted as an "extra"
/// into the neighbour block shifted by `bshift` from its primary block. Slabs below
/// -2 are clamped into the -2 bucket.
#[inline]
fn extra_sort_bucket(assoc: UVector, bshift: IVector) -> usize {
    #[cfg(feature = "dim2")]
    let local = assoc.y as i32 - bshift.y * 8;
    #[cfg(feature = "dim3")]
    let local = assoc.z as i32 - bshift.z * 4;
    NUM_PRIMARY_SORT_BUCKETS + (local.max(-2) + 2) as usize
}

/// Marks all blocks associated with each particle as active.
///
/// For each particle, computes the set of blocks whose stencil could overlap
/// the particle, and inserts them into the hashmap. This must be run before
/// any per-block operations.
// TODO HACK: enabling spirv-passthrough for this shader since naga panics
//            on the spv backend because of https://github.com/gfx-rs/wgpu/issues/7315
//            (in our case, it’s caused by the lines involving the atomic compare-exchange).
#[spirv_bindgen(spirv_passthrough)]
#[spirv(compute(threads(64)))]
pub fn gpu_touch_particle_blocks(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] particles_len: &u32,
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);
        for i in 0..NUM_ASSOC_BLOCKS {
            grid.mark_block_as_active(hmap_entries, active_blocks, &blocks[i]);
        }
    }
}

/// Marks only each particle's **primary** (base) block as active.
///
/// First half of the two-pass block activation (with `gpu_touch_neighbor_blocks`)
/// that replaces `gpu_touch_particle_blocks`; the union of activated blocks is
/// identical.
// TODO HACK: spirv_passthrough because naga panics on the atomic compare-exchange
//            in `mark_block_as_active` (see `gpu_touch_particle_blocks`).
#[spirv_bindgen(spirv_passthrough)]
#[spirv(compute(threads(64)))]
pub fn gpu_touch_primary_blocks(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] particles_len: &u32,
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let block = BlockVirtualId::block_associated_to_point(cell_width, particle.pt);
        grid.mark_block_as_active(hmap_entries, active_blocks, &block);
    }
}

/// Marks the +1 neighbour blocks of every already-active base block as active.
///
/// Second half of the two-pass block activation (see `gpu_touch_primary_blocks`).
/// `num_base_blocks` must be a snapshot of `grid.num_active_blocks` taken *before*
/// this pass runs, so that the neighbour blocks appended during the pass are not
/// themselves processed.
// TODO HACK: spirv_passthrough because naga panics on the atomic compare-exchange
//            in `mark_block_as_active` (see `gpu_touch_particle_blocks`).
#[spirv_bindgen(spirv_passthrough)]
#[spirv(compute(threads(64)))]
pub fn gpu_touch_neighbor_blocks(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_base_blocks: &[u32],
) {
    let id = invocation_id.x;
    if id < num_base_blocks.read(0) {
        let raw = active_blocks.at(id as usize).virtual_id;
        // Rematerialize the block id into a fresh, register-aligned vector. Reading
        // `IVec3` straight out of the storage buffer keeps naga's packed-`int3` type,
        // and the vector arithmetic in `blocks_associated_to_block` then emits illegal
        // packed↔aligned `as_type` casts under the spirv-passthrough path (Metal).
        #[cfg(feature = "dim2")]
        let vid = BlockVirtualId {
            id: IVector::new(raw.id.x, raw.id.y),
        };
        #[cfg(feature = "dim3")]
        let vid = BlockVirtualId {
            id: IVector::new(raw.id.x, raw.id.y, raw.id.z),
            padding: 0,
        };
        let blocks = BlockVirtualId::blocks_associated_to_block(&vid);
        for i in 1..NUM_ASSOC_BLOCKS {
            grid.mark_block_as_active(hmap_entries, active_blocks, &blocks[i]);
        }
    }
}

/// Marks all blocks associated with each rigid particle as active.
///
/// Similar to `gpu_touch_particle_blocks`, but operates on rigid body surface
/// particles. Only touches blocks for rigid particles that are flagged as needing
/// a block (via the `rigid_particle_needs_block` bitfield).
// TODO HACK: enabling spirv-passthrough for this shader since naga panics
//            on the spv backend because of https://github.com/gfx-rs/wgpu/issues/7315
//            (in our case, it’s caused by the lines involving the atomic compare-exchange).
#[spirv_bindgen(spirv_passthrough)]
#[spirv(compute(threads(64)))]
pub fn gpu_touch_rigid_particle_blocks(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rigid_particle_needs_block: &[u32],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let entry_id = (id / 32) as usize;
        let entry_bit = 1u32 << (id % 32);
        let needs_block = (rigid_particle_needs_block.read(entry_id) & entry_bit) != 0;

        if needs_block {
            let particle = rigid_particles_pos.read(id as usize);
            let block = BlockVirtualId::block_associated_to_point(cell_width, particle.pt);
            grid.mark_block_as_active(hmap_entries, active_blocks, &block);
        }
    }
}

/// Flags rigid particles that need their own block activated.
///
/// A rigid particle needs its own block if at least one (but not all) of its
/// associated blocks are already active. This means the particle is near a
/// block boundary and its contributions would be lost without an additional block.
///
/// The result is stored as a bitfield in `rigid_particle_needs_block`, where
/// each u32 holds flags for 32 rigid particles.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mark_rigid_particles_needing_block(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    rigid_particle_needs_block: &mut [u32],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let particle = rigid_particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);

        // Find the first block that already has a header in the hashmap.
        let mut i = 0u32;
        for _ in 0..NUM_ASSOC_BLOCKS {
            if grid
                .find_block_header_id(hmap_entries, &blocks[i as usize])
                .id
                != NONE
            {
                break;
            }
            i += 1;
        }

        let entry_id = (id / 32) as usize;
        let entry_bit = 1u32 << (id % 32);

        // If some but not all associated blocks are active, the rigid particle
        // needs its own block to ensure proper grid transfers.
        if i > 0 && i < NUM_ASSOC_BLOCKS as u32 {
            // Set the bit atomically.
            khal_std::sync::atomic_or_u32(rigid_particle_needs_block.at_mut(entry_id), entry_bit);
        } else {
            // Clear the bit atomically.
            khal_std::sync::atomic_and_u32(rigid_particle_needs_block.at_mut(entry_id), !entry_bit);
        }
    }
}

/// Precomputes, for each active block, the header IDs of its +1 neighbour blocks,
/// caching them in `ActiveBlockHeader::nbh_block_ids`. Inactive neighbours are
/// stored as `NONE`.
///
/// Must run after all blocks have been touched (so `num_active_blocks` is final and every
/// neighbour that exists is in the hashmap) and before the particle/rigid count and
/// finalize passes that read `nbh_block_ids`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_nbh_block_ids(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        let block0 = active_blocks.at_mut(id as usize);
        let vid = &block0.virtual_id;
        let assoc = BlockVirtualId::blocks_associated_to_block(vid);
        for nbh in 0..NUM_ASSOC_BLOCKS - 1 {
            let nbh_vid = assoc[nbh + 1];
            let nbh_hid = grid.find_block_header_id(hmap_entries, &nbh_vid);
            block0.nbh_block_ids.write(nbh, nbh_hid);
        }
    }
}

/// Counts the number of particles in each active block.
///
/// Each thread processes one particle, finds its associated block, and
/// atomically increments that block's `num_particles` counter.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_block_particle_count(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] particles_len: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);

        // The particle's primary (base) block gets it as a regular particle and
        // as an "extra". Only the per-slab-bucket counter is incremented here:
        // `num_particles` and `num_particles_with_extras` are derived per block in
        // the copy passes, avoiding two heavily contended atomics per particle.
        let assoc = particle.associated_cell_index_in_block_off_by_one(cell_width);
        let block0 = grid.find_block_header_id(hmap_entries, &blocks[0]);
        atomic_add_u32(
            active_blocks
                .at_mut(block0.id as usize)
                .sort_bucket_cursors
                .at_mut(primary_sort_bucket(assoc)),
            1,
        );

        // Each +1 neighbour block also receives the particle as an "extra" if the
        // quadratic stencil actually spills into it, i.e. the local base-cell index is
        // >= EXTRA_PARTICLE_MIN_SHIFT along every axis where that block is the +1 neighbour.
        let id0 = blocks[0].id;
        for i in 1..NUM_ASSOC_BLOCKS {
            let bshift = blocks[i].id - id0;
            #[cfg(feature = "dim2")]
            let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT);
            #[cfg(feature = "dim3")]
            let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT)
                && (bshift.z == 0 || assoc.z >= EXTRA_PARTICLE_MIN_SHIFT);
            if spills {
                // The header IDs of the +1 neighbour blocks were precomputed by
                // `gpu_update_nbh_block_ids`, so we read them from the primary block
                // instead of doing a hashmap lookup per particle. Only the extra slab
                // bucket is incremented; the neighbour's `num_particles_with_extras` is
                // recovered as the sum of its buckets in the copy passes.
                let block_i = active_blocks
                    .at(block0.id as usize)
                    .nbh_block_ids
                    .read(i - 1);
                atomic_add_u32(
                    active_blocks
                        .at_mut(block_i.id as usize)
                        .sort_bucket_cursors
                        .at_mut(extra_sort_bucket(assoc, bshift)),
                    1,
                );
            }
        }
    }
}

/// Copies each active block's particle count into the scan_values buffer.
///
/// This prepares the input for the prefix sum pass. After the prefix sum,
/// `scan_values[i]` will contain the global offset for block `i`'s particles.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_copy_particles_len_to_scan_value(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scan_values: &mut [u32],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        // The sorted array reserves room for every particle a block touches, extras
        // included. `num_particles_with_extras` is the sum of all slab buckets (the count
        // pass no longer maintains it as a separate atomic).
        let mut total = 0u32;
        for k in 0..NUM_SORT_BUCKETS {
            total += active_blocks.at(id as usize).sort_bucket_cursors.read(k);
        }
        scan_values.write(id as usize, total);
    }
}

/// Writes the prefix sum results back as `first_particle` offsets and resets particle counts.
///
/// After the prefix sum, `scan_values[i]` contains the exclusive scan result.
/// This kernel copies it into `active_blocks[i].first_particle`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_copy_scan_values_to_first_particles(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scan_values: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        let idx = id as usize;
        let first = scan_values.read(idx);
        active_blocks.at_mut(idx).first_particle = first;
        // Convert the per-bucket counts accumulated by the count pass into running
        // insertion cursors. The cursors are *absolute* offsets into the sorted array
        // (i.e. `first_particle` is baked in), so the finalize pass can scatter each
        // particle with a single atomic and no per-contribution `first_particle` read.
        // Primary buckets come first, so primaries land in
        // [first_particle, first_particle + num_particles) as G2P expects, with the
        // extras after them; both segments end up ordered by slab key.
        //
        // The running total advanced past the primary buckets is `num_particles` (primaries
        // only land in primary buckets), and the grand total is `num_particles_with_extras`.
        // Both fields are derived here rather than maintained as per-particle atomics in
        // the count pass.
        let mut running = first;
        for k in 0..NUM_SORT_BUCKETS {
            if k == NUM_PRIMARY_SORT_BUCKETS {
                active_blocks.at_mut(idx).num_particles = running - first;
            }
            let count = active_blocks.at(idx).sort_bucket_cursors.read(k);
            active_blocks
                .at_mut(idx)
                .sort_bucket_cursors
                .write(k, running);
            running += count;
        }
        active_blocks.at_mut(idx).num_particles_with_extras = running - first;
    }
}

/// Places particles into their sorted positions.
///
/// Each thread processes one particle:
/// 1. Finds the particle's active block via the hashmap.
/// 2. Atomically claims a slot in the sorted array (using `scan_values` as a counter).
/// 3. Writes the particle's original index into `sorted_particle_ids`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_finalize_particles_sort(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] particles_len: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sorted_particle_ids: &mut [u32],
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);

        // Place the particle in its primary block's range. The prepare pass turned the
        // bucket counts into absolute insertion cursors (first_particle baked in), so the
        // atomically-claimed slot is already the final sorted index.
        let assoc = particle.associated_cell_index_in_block_off_by_one(cell_width);
        let block0 = grid.find_block_header_id(hmap_entries, &blocks[0]);
        let slot0 = atomic_add_u32(
            active_blocks
                .at_mut(block0.id as usize)
                .sort_bucket_cursors
                .at_mut(primary_sort_bucket(assoc)),
            1,
        );
        sorted_particle_ids.write(slot0 as usize, id);

        // Place the particle as an "extra" into each +1 neighbour block whose stencil it
        // spills into, using the extra slab bucket cursors (extras land after the
        // primaries because their buckets come last).
        let id0 = blocks[0].id;
        for i in 1..NUM_ASSOC_BLOCKS {
            let bshift = blocks[i].id - id0;
            #[cfg(feature = "dim2")]
            let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT);
            #[cfg(feature = "dim3")]
            let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT)
                && (bshift.z == 0 || assoc.z >= EXTRA_PARTICLE_MIN_SHIFT);
            if spills {
                // Reuse the neighbour header IDs precomputed by `gpu_update_nbh_block_ids`
                // rather than re-querying the hashmap.
                let block_i = active_blocks
                    .at(block0.id as usize)
                    .nbh_block_ids
                    .read(i - 1);
                let slot_i = atomic_add_u32(
                    active_blocks
                        .at_mut(block_i.id as usize)
                        .sort_bucket_cursors
                        .at_mut(extra_sort_bucket(assoc, bshift)),
                    1,
                );
                sorted_particle_ids.write(slot_i as usize, id);
            }
        }
    }
}

/// Counts the number of rigid particles contributing to each active block.
///
/// Mirrors `gpu_update_block_particle_count`, with two differences: rigid particles
/// whose primary block is not active are silently skipped (they can't affect the
/// simulation), and neighbour blocks receiving an "extra" may be inactive too (only
/// the primary block's activation is guaranteed by the touch passes).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_block_rigid_particle_count(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let particle = rigid_particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);

        let block0 = grid.find_block_header_id(hmap_entries, &blocks[0]);
        if block0.id != NONE {
            atomic_add_u32(
                &mut active_blocks
                    .at_mut(block0.id as usize)
                    .num_rigid_particles_with_extras,
                1,
            );

            // Each +1 neighbour block also receives the particle as an "extra" if its
            // 3-cell influence range actually spills into it.
            let assoc = particle.associated_cell_index_in_block_off_by_one(cell_width);
            let id0 = blocks[0].id;
            for i in 1..NUM_ASSOC_BLOCKS {
                let bshift = blocks[i].id - id0;
                #[cfg(feature = "dim2")]
                let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                    && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT);
                #[cfg(feature = "dim3")]
                let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                    && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT)
                    && (bshift.z == 0 || assoc.z >= EXTRA_PARTICLE_MIN_SHIFT);
                if spills {
                    // The neighbour header IDs (or NONE for inactive neighbours) were
                    // precomputed by `gpu_update_nbh_block_ids`.
                    let block_i = active_blocks
                        .at(block0.id as usize)
                        .nbh_block_ids
                        .read(i - 1);
                    if block_i.id != NONE {
                        atomic_add_u32(
                            &mut active_blocks
                                .at_mut(block_i.id as usize)
                                .num_rigid_particles_with_extras,
                            1,
                        );
                    }
                }
            }
        }
    }
}

/// Copies each active block's rigid particle count into the scan_values buffer.
///
/// Prepares the prefix sum input for the rigid particle sort. Must run after the
/// regular particle sort no longer needs `scan_values`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_copy_rigid_particles_len_to_scan_value(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scan_values: &mut [u32],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        scan_values.write(
            id as usize,
            active_blocks
                .at(id as usize)
                .num_rigid_particles_with_extras,
        );
    }
}

/// Writes the prefix sum results back as `first_rigid_particle` offsets.
///
/// Also resets `num_rigid_particles_with_extras` to 0 so the finalize pass can
/// re-purpose it as the running insertion cursor (it ends up back at the count).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_copy_scan_values_to_first_rigid_particles(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scan_values: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        let idx = id as usize;
        active_blocks.at_mut(idx).first_rigid_particle = scan_values.read(idx);
        active_blocks.at_mut(idx).num_rigid_particles_with_extras = 0;
    }
}

/// Places rigid particles into their sorted positions.
///
/// Mirrors `gpu_finalize_particles_sort` with the rigid-specific inactive-block
/// skips; each particle must contribute to the exact same set of blocks as in the
/// count pass so every claimed slot stays within its block's range.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_finalize_rigid_particles_sort(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    sorted_rigid_particle_ids: &mut [u32],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let particle = rigid_particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);

        let block0 = grid.find_block_header_id(hmap_entries, &blocks[0]);
        if block0.id != NONE {
            let first0 = active_blocks.at(block0.id as usize).first_rigid_particle;
            let slot0 = atomic_add_u32(
                &mut active_blocks
                    .at_mut(block0.id as usize)
                    .num_rigid_particles_with_extras,
                1,
            );
            sorted_rigid_particle_ids.write((first0 + slot0) as usize, id);

            let assoc = particle.associated_cell_index_in_block_off_by_one(cell_width);
            let id0 = blocks[0].id;
            for i in 1..NUM_ASSOC_BLOCKS {
                let bshift = blocks[i].id - id0;
                #[cfg(feature = "dim2")]
                let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                    && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT);
                #[cfg(feature = "dim3")]
                let spills = (bshift.x == 0 || assoc.x >= EXTRA_PARTICLE_MIN_SHIFT)
                    && (bshift.y == 0 || assoc.y >= EXTRA_PARTICLE_MIN_SHIFT)
                    && (bshift.z == 0 || assoc.z >= EXTRA_PARTICLE_MIN_SHIFT);
                if spills {
                    // Reuse the neighbour header IDs precomputed by `gpu_update_nbh_block_ids`.
                    let block_i = active_blocks
                        .at(block0.id as usize)
                        .nbh_block_ids
                        .read(i - 1);
                    if block_i.id != NONE {
                        let first_i = active_blocks.at(block_i.id as usize).first_rigid_particle;
                        let slot_i = atomic_add_u32(
                            &mut active_blocks
                                .at_mut(block_i.id as usize)
                                .num_rigid_particles_with_extras,
                            1,
                        );
                        sorted_rigid_particle_ids.write((first_i + slot_i) as usize, id);
                    }
                }
            }
        }
    }
}
