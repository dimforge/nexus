//! Narrow phase contact generation kernels.
//!
//! Computes contact manifolds from collision pairs detected by the broad phase.

use crate::queries::{
    ColliderMaterial, ContactManifold, IndexedManifold, ball_ball, ball_convex, convex_ball,
    cuboid_cuboid, pfm_pfm,
};
use crate::shapes::{
    Capsule, Polyline, SHAPE_TYPE_BALL, SHAPE_TYPE_CAPSULE, SHAPE_TYPE_CONE, SHAPE_TYPE_CUBOID,
    SHAPE_TYPE_CYLINDER, SHAPE_TYPE_POLYLINE, SHAPE_TYPE_TRIMESH, Shape, TriMesh,
};
use crate::{PaddedVector, Pose, Vector};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{
    iter::StepRng,
    sync::{atomic_add_u32, atomic_load_u32},
};

use crate::broad_phase::CollisionPair;
use crate::utils::{BatchIndices, SliceMut};
use glamx::UVec2;

const WORKGROUP_SIZE: u32 = 64;

/// Resets the contacts counter.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_narrow_phase(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pfm_pairs_len: &mut [u32],
) {
    let batch_id = workgroup_id.y as usize;

    // NOTE: this `for` loop is silly. It doesn’t do anything
    //       more than a `*contacts_len = 0` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        contacts_len.write(batch_id, k);
        pfm_pairs_len.write(batch_id, k);
    }
}

/// Initializes indirect dispatch arguments for constraint solver.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_narrow_phase_init_contacts_dispatch(
    // NOTE: the `contacts_len` is mutable here even though we don’t modify it. That’s
    //       because we access it with an atomic load otherwise it would occasionally read
    //       stale data (on Windows+Nvidia+wgpu backend). This might be caused by:
    //       https://github.com/gfx-rs/wgpu/issues/9221
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
) {
    // For indirect dispatch, get the largest length along all batch dimensions.
    let num_batches = contacts_len.len();
    let mut highest_contacts_len = 0;
    for i in 0..num_batches {
        // NOTE: atomic_load is needed for correctness on some platforms (see comment above `contacts_len`).
        highest_contacts_len = highest_contacts_len.max(atomic_load_u32(contacts_len.at_mut(i)));
    }

    *indirect_args.at_mut(0) = highest_contacts_len.div_ceil(WORKGROUP_SIZE);
    *indirect_args.at_mut(1) = num_batches as u32;
    *indirect_args.at_mut(2) = 1;
}

const PREDICTION: f32 = 2.0e-3; // TODO: make the prediction configurable.

/// Narrow phase, pass 1 of 2: analytic shape-shape contacts for ball / cuboid
/// pairs, written straight into the `contacts` buffer.
///
/// The complex cases (generic convex via PFM, trimesh, polyline) are deferred
/// to `gpu_narrow_phase_shape_shape_deferred`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_narrow_phase_shape_shape(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs: &[CollisionPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] collision_pairs_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contacts: &mut [IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contacts_len: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
    // Per-collider parent body id, used to resolve `IndexedManifold::bodies` here,
    // at the last moment before the solver consumes it (instead of carrying the
    // body ids all the way through the broad-phase collision-pair buffer).
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] collider_parent: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)]
    collider_materials: &[ColliderMaterial],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let contacts_batch_capacity = batch_ids.contacts_batch_capacity as usize;

    let collision_pairs = batch_ids.contact_batch(batch_id, collision_pairs);
    let poses = batch_ids.coll_batch(batch_id, poses);
    let shapes = batch_ids.coll_batch(batch_id, shapes);
    let collider_materials = batch_ids.coll_batch(batch_id, collider_materials);
    let mut contacts = batch_ids.contact_batch_mut(batch_id, contacts);
    let contacts_len = contacts_len.at_mut(batch_id as usize);

    // NOTE: `collision_pairs_len` might be greater than `contacts_batch_apacity` if the
    //       narrow-phase found more pairs than the buffer can contain.
    let len = collision_pairs_len
        .read(batch_id as usize)
        .min(contacts_batch_capacity as u32);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let pair = collision_pairs[i as usize];
        // Resolve the parent rigid-bodies here (the broad phase no longer does)
        // and skip pairs whose colliders share the same body.
        let body1 = collider_parent.read(pair.colliders.x as usize);
        let body2 = collider_parent.read(pair.colliders.y as usize);
        if body1 == body2 {
            continue;
        }
        let pose1 = poses[pair.colliders.x as usize];
        let pose2 = poses[pair.colliders.y as usize];
        let shape1 = &shapes[pair.colliders.x as usize];
        let shape2 = &shapes[pair.colliders.y as usize];
        let shape_ty1 = shape1.shape_type();
        let shape_ty2 = shape2.shape_type();
        let mut manifold = ContactManifold::default();
        let pose12 = pose1.inverse() * pose2;

        // Ball - Convex
        if shape_ty1 == SHAPE_TYPE_BALL {
            if shape_ty2 == SHAPE_TYPE_BALL {
                let ball1 = shape1.to_ball();
                let ball2 = shape2.to_ball();
                manifold = ball_ball(pose12, &ball1, &ball2);
            } else if shape_ty2 == SHAPE_TYPE_CUBOID
                || shape_ty2 == SHAPE_TYPE_CAPSULE
                || shape_ty2 == SHAPE_TYPE_CONE
                || shape_ty2 == SHAPE_TYPE_CYLINDER
            {
                let ball1 = shape1.to_ball();
                manifold = ball_convex(pose12, &ball1, shape2);
            }
        }

        // Convex - Ball
        if shape_ty2 == SHAPE_TYPE_BALL
            && (shape_ty1 == SHAPE_TYPE_CUBOID
                || shape_ty1 == SHAPE_TYPE_CAPSULE
                || shape_ty1 == SHAPE_TYPE_CONE
                || shape_ty1 == SHAPE_TYPE_CYLINDER)
        {
            let ball2 = shape2.to_ball();
            manifold = convex_ball(pose12, shape1, &ball2);
        }

        // Cuboid - Cuboid
        if shape_ty1 == SHAPE_TYPE_CUBOID && shape_ty2 == SHAPE_TYPE_CUBOID {
            let cuboid1 = shape1.to_cuboid();
            let cuboid2 = shape2.to_cuboid();
            manifold = cuboid_cuboid(pose12, &cuboid1, &cuboid2, PREDICTION);
        }

        // Everything else (PFM / trimesh / polyline) is handled by the deferred
        // pass; `manifold.len` stays 0 here so nothing is written.
        if manifold.len > 0 && manifold.points_a.at(0).dist < PREDICTION {
            let target_contact_index = atomic_add_u32(contacts_len, 1) as usize;

            // NOTE: if we exceed the contacts allocation size, just skip
            //       the contact. It’s up to the caller to resize the buffer
            //       and re-run the narrow-phase.
            if target_contact_index < contacts_batch_capacity {
                let mat1 = collider_materials[pair.colliders.x as usize];
                let mat2 = collider_materials[pair.colliders.y as usize];
                contacts[target_contact_index] = IndexedManifold {
                    contact: manifold,
                    colliders: pair.colliders,
                    bodies: UVec2::new(body1, body2),
                    friction: mat1.combined_friction(&mat2),
                    restitution: mat1.combined_restitution(&mat2),
                    _padding: [0.0; 2],
                };
            }
        }
    }
}

/// Narrow phase, pass 2 of 2: defer the complex shape-shape pairs (generic
/// convex via PFM, trimesh, polyline) into the `pfm_pairs` work-list consumed by
/// `gpu_narrow_phase_pfm_pfm`. Ball / cuboid pairs were already resolved by
/// `gpu_narrow_phase_shape_shape`; this pass skips them via the same shape-type
/// predicate. See that kernel for why the work is split.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_narrow_phase_shape_shape_deferred(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs: &[CollisionPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] collision_pairs_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    pfm_pairs: &mut [NarrowPhasePfmPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] pfm_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] indices: &[u32],
    // NOTE: we assume that max_pfm_pairs == contacts_batch_capacity
    //       And we assume all batch dimensions are given the same buffer allocation sizes
    //       (i.e. the same `contacts_batch_capacity`).
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let contacts_batch_capacity = batch_ids.contacts_batch_capacity as usize;

    let collision_pairs = batch_ids.contact_batch(batch_id, collision_pairs);
    let poses = batch_ids.coll_batch(batch_id, poses);
    let shapes = batch_ids.coll_batch(batch_id, shapes);
    let mut pfm_pairs = batch_ids.contact_batch_mut(batch_id, pfm_pairs);
    let pfm_pairs_len = pfm_pairs_len.at_mut(batch_id as usize);

    let len = collision_pairs_len
        .read(batch_id as usize)
        .min(contacts_batch_capacity as u32);

    // NOTE: same-body collider pairs are *not* filtered in this pass — it is
    //       already at the 8-storage-buffer WebGPU limit and can't take the
    //       `collider_parent` binding. The complex pairs it emits are filtered
    //       downstream in `gpu_narrow_phase_pfm_pfm` (which has room) before any
    //       contact is written.
    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let pair = collision_pairs[i as usize];
        let pose1 = poses[pair.colliders.x as usize];
        let pose2 = poses[pair.colliders.y as usize];
        let shape1 = &shapes[pair.colliders.x as usize];
        let shape2 = &shapes[pair.colliders.y as usize];
        let shape_ty1 = shape1.shape_type();
        let shape_ty2 = shape2.shape_type();
        let pose12 = pose1.inverse() * pose2;

        // Mirror pass 1's analytic-pair predicate (ball/cuboid) so those pairs
        // are skipped here — they were already turned into contacts. Only the
        // complex cases fall through to the PFM / trimesh / polyline handling.
        let mut checked = false;
        if shape_ty1 == SHAPE_TYPE_BALL
            && (shape_ty2 == SHAPE_TYPE_BALL
                || shape_ty2 == SHAPE_TYPE_CUBOID
                || shape_ty2 == SHAPE_TYPE_CAPSULE
                || shape_ty2 == SHAPE_TYPE_CONE
                || shape_ty2 == SHAPE_TYPE_CYLINDER)
        {
            checked = true;
        }
        if !checked
            && shape_ty2 == SHAPE_TYPE_BALL
            && (shape_ty1 == SHAPE_TYPE_CUBOID
                || shape_ty1 == SHAPE_TYPE_CAPSULE
                || shape_ty1 == SHAPE_TYPE_CONE
                || shape_ty1 == SHAPE_TYPE_CYLINDER)
        {
            checked = true;
        }
        if !checked && shape_ty1 == SHAPE_TYPE_CUBOID && shape_ty2 == SHAPE_TYPE_CUBOID {
            checked = true;
        }

        // PFM - PFM (generic convex shapes via GJK/EPA)
        if !checked {
            let sub1 = shape1.pfm_subshape();
            let sub2 = shape2.pfm_subshape();

            if sub1.valid && sub2.valid {
                let pfm_pair = NarrowPhasePfmPair {
                    shape1: sub1.shape,
                    shape2: sub2.shape,
                    pose12,
                    thickness1: sub1.thickness,
                    thickness2: sub2.thickness,
                    colliders: pair.colliders,
                };
                let pfm_index = atomic_add_u32(pfm_pairs_len, 1);
                // NOTE: if we exceed the work-list allocation size, just skip
                //       the pair (same policy as the contacts buffer). It’s up
                //       to the caller to resize the buffer and re-run.
                if (pfm_index as usize) < contacts_batch_capacity {
                    pfm_pairs.write(pfm_index as usize, pfm_pair);
                }

                // The actual calculations are deferred to another kernel.
                continue;
            }
        }

        // TriMesh - Convex
        // Note: trimesh collision writes contacts directly to the buffer and early-exits.
        if !checked && shape_ty1 == SHAPE_TYPE_TRIMESH {
            let mesh = shape1.to_trimesh();
            let convex = shape2;
            trimesh_convex(
                pose12,
                &mesh,
                convex,
                pair.colliders,
                &mut pfm_pairs,
                pfm_pairs_len,
                contacts_batch_capacity,
                vertices,
                indices,
            );
            continue;
        }

        if !checked && shape_ty2 == SHAPE_TYPE_TRIMESH {
            let convex = shape1;
            let mesh = shape2.to_trimesh();
            // NOTE: pair indices are flipped.
            trimesh_convex(
                pose12.inverse(),
                &mesh,
                convex,
                UVec2::new(pair.colliders.y, pair.colliders.x),
                &mut pfm_pairs,
                pfm_pairs_len,
                contacts_batch_capacity,
                vertices,
                indices,
            );
            continue;
        }

        // Polyline - Convex
        // Note: polyline collision writes contacts directly to the buffer and early-exits.
        if !checked && shape_ty1 == SHAPE_TYPE_POLYLINE {
            let pline = shape1.to_polyline();
            let convex = shape2;
            polyline_convex(
                pose12,
                &pline,
                convex,
                pair.colliders,
                &mut pfm_pairs,
                pfm_pairs_len,
                contacts_batch_capacity,
                vertices,
                indices,
            );
            continue;
        }

        if !checked && shape_ty2 == SHAPE_TYPE_POLYLINE {
            let convex = shape1;
            let pline = shape2.to_polyline();
            // NOTE: pair indices are flipped.
            polyline_convex(
                pose12.inverse(),
                &pline,
                convex,
                UVec2::new(pair.colliders.y, pair.colliders.x),
                &mut pfm_pairs,
                pfm_pairs_len,
                contacts_batch_capacity,
                vertices,
                indices,
            );
            continue;
        }
    }
}

/// Collision detection between a triangle mesh and a convex shape.
fn trimesh_convex(
    pose12: Pose,
    mesh: &TriMesh,
    convex: &Shape,
    colliders: UVec2,
    pfm_pairs: &mut SliceMut<NarrowPhasePfmPair>,
    pfm_pairs_len: &mut u32,
    pfm_pairs_capacity: usize,
    vertices: &[PaddedVector],
    indices: &[u32],
) {
    let sub2 = convex.pfm_subshape();
    if !sub2.valid {
        // Collisions with non-PFM shapes is not supported.
        return;
    }

    // Get the convex shape's AABB in the trimesh's local space, and enlarge with the PREDICTION.
    let mut test_aabb = convex.compute_aabb(pose12, vertices);
    test_aabb.mins -= Vector::splat(PREDICTION);
    test_aabb.maxs += Vector::splat(PREDICTION);

    if !test_aabb.intersects(&mesh.root_aabb) {
        // No collision possible.
        return;
    }

    let mut curr = 0u32;

    // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
    for _ in 0..mesh.bvh_node_len {
        if curr >= mesh.bvh_node_len {
            break;
        }

        let idx = mesh.bvh_node_idx(indices, curr);
        if idx.entry_index == 0xffffffff {
            // This is a leaf.
            let tri = mesh.triangle(indices, vertices, idx.shape_index);
            let tri_shape = Shape::from_triangle(&tri);
            let sub1 = tri_shape.pfm_subshape();
            // TODO PERF: add special-cases for pairs that can be handled more efficiently than with GJK/EPA.
            let pfm_pair = NarrowPhasePfmPair {
                shape1: sub1.shape,
                shape2: sub2.shape,
                pose12,
                thickness1: sub1.thickness,
                thickness2: sub2.thickness,
                colliders,
            };
            let pfm_index = atomic_add_u32(pfm_pairs_len, 1);
            // Skip (don’t write) on overflow; the caller resizes and re-runs.
            if (pfm_index as usize) < pfm_pairs_capacity {
                pfm_pairs.write(pfm_index as usize, pfm_pair);
            }

            // Continue traversal.
            curr = idx.exit_index;
        } else {
            let node_aabb = mesh.bvh_node_aabb(vertices, curr);
            if test_aabb.intersects(&node_aabb) {
                curr = idx.entry_index;
            } else {
                curr = idx.exit_index;
            }
        }
    }
}

/// Collision detection between a polyline and a convex shape.
fn polyline_convex(
    pose12: Pose,
    mesh: &Polyline,
    convex: &Shape,
    colliders: UVec2,
    pfm_pairs: &mut SliceMut<NarrowPhasePfmPair>,
    pfm_pairs_len: &mut u32,
    pfm_pairs_capacity: usize,
    vertices: &[PaddedVector],
    indices: &[u32],
) {
    let sub2 = convex.pfm_subshape();
    if !sub2.valid {
        // Collisions with non-PFM shapes is not supported.
        return;
    }

    // Get the convex shape's AABB in the polyline's local space, and enlarge with the PREDICTION.
    let thickness = 0.4; // TODO: make thickness configurable or part of the polyline struct
    let mut test_aabb = convex.compute_aabb(pose12, vertices);
    test_aabb.mins -= Vector::splat(PREDICTION + thickness);
    test_aabb.maxs += Vector::splat(PREDICTION + thickness);

    if !test_aabb.intersects(&mesh.root_aabb) {
        // No collision possible.
        return;
    }

    let mut curr = 0u32;

    // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
    for _ in 0..mesh.bvh_node_len {
        if curr >= mesh.bvh_node_len {
            break;
        }

        let idx = mesh.bvh_node_idx(curr, indices);
        if idx.entry_index == 0xffffffff {
            // This is a leaf.
            let seg = mesh.segment(idx.shape_index, vertices, indices);
            // The segment is seen as a capsule with the given thickness.
            let capsule = Capsule::new(seg, thickness);
            let capsule_shape = Shape::from_capsule(&capsule);
            let sub1 = capsule_shape.pfm_subshape();
            // TODO PERF: add special-cases for pairs that can be handled more efficiently than with GJK/EPA.
            let pfm_pair = NarrowPhasePfmPair {
                shape1: sub1.shape,
                shape2: sub2.shape,
                pose12,
                thickness1: sub1.thickness,
                thickness2: sub2.thickness,
                colliders,
            };
            let pfm_index = atomic_add_u32(pfm_pairs_len, 1);
            // Skip (don’t write) on overflow; the caller resizes and re-runs.
            if (pfm_index as usize) < pfm_pairs_capacity {
                pfm_pairs.write(pfm_index as usize, pfm_pair);
            }

            // Continue traversal.
            curr = idx.exit_index;
        } else {
            let node_aabb = mesh.bvh_node_aabb(curr, vertices);
            if test_aabb.intersects(&node_aabb) {
                curr = idx.entry_index;
            } else {
                curr = idx.exit_index;
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NarrowPhasePfmPair {
    shape1: Shape,
    shape2: Shape,
    pose12: Pose,
    thickness1: f32,
    thickness2: f32,
    colliders: UVec2,
}

/// Initializes PFM-PFM dispatch arguments for constraint solver.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_init_pfm_pfm_dispatch(
    // NOTE: the `pfm_pairs_len` is mutable here even though we don’t modify it. That’s
    //       because we access it with an atomic load otherwise it would occasionally read
    //       stale data (on Windows+Nvidia+wgpu backend). This might be caused by:
    //       https://github.com/gfx-rs/wgpu/issues/9221
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] pfm_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
) {
    let num_batches = pfm_pairs_len.len();
    let mut highest_pfm_pairs_len = 0;
    for batch_id in 0..num_batches {
        // NOTE: atomic_load is needed for correctness on some platforms (see comment above `pfm_pairs_len`).
        highest_pfm_pairs_len =
            highest_pfm_pairs_len.max(atomic_load_u32(pfm_pairs_len.at_mut(batch_id)));
    }
    // TODO PERF: pfm_pfm is very divergent. Use a smaller workgroup size?
    *indirect_args.at_mut(0) = highest_pfm_pairs_len.div_ceil(WORKGROUP_SIZE);
    *indirect_args.at_mut(1) = num_batches as u32;
    *indirect_args.at_mut(2) = 1;
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))] // TODO PERF: pfm_pfm is very divergent. Use a smaller workgroup size?
pub fn gpu_narrow_phase_pfm_pfm(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &mut [IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pfm_pairs: &[NarrowPhasePfmPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pfm_pairs_len: &[u32],
    // NOTE: we assume that max_pfm_pairs == contacts_batch_capacity
    //       And we assume all batch dimensions are given the same buffer allocation sizes
    //       (i.e. the same `contacts_batch_capacity`).
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] vertices: &[PaddedVector],
    #[allow(unused_variables)]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    indices: &[u32],
    // Per-collider parent body id, used to resolve `IndexedManifold::bodies` here
    // (see the note on `gpu_narrow_phase_shape_shape`).
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] collider_parent: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)]
    collider_materials: &[ColliderMaterial],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let contacts_batch_capacity = batch_ids.contacts_batch_capacity as usize;

    let mut contacts = batch_ids.contact_batch_mut(batch_id, contacts);
    let collider_materials = batch_ids.coll_batch(batch_id, collider_materials);
    let pfm_pairs = batch_ids.contact_batch(batch_id, pfm_pairs);
    let contacts_len = contacts_len.at_mut(batch_id as usize);
    // The producer counter can exceed the allocation on overflow (writes are
    // skipped past capacity); clamp so we never read uninitialized slots.
    let pfm_pairs_len = pfm_pairs_len
        .read(batch_id as usize)
        .min(contacts_batch_capacity as u32);

    for i in StepRng::new(invocation_id.x..pfm_pairs_len, num_threads) {
        let pair = pfm_pairs[i as usize];
        // Resolve the parent rigid-bodies and skip same-body collider pairs. This
        // is where the deferred (PFM / trimesh / polyline) pairs get the same-body
        // filtering that the analytic pass does inline — the broad phase no longer
        // does it, and the deferred pass has no spare storage binding for it.
        let body1 = collider_parent.read(pair.colliders.x as usize);
        let body2 = collider_parent.read(pair.colliders.y as usize);
        if body1 == body2 {
            continue;
        }
        let manifold = pfm_pfm(
            pair.pose12,
            &pair.shape1,
            pair.thickness1,
            &pair.shape2,
            pair.thickness2,
            PREDICTION,
            vertices,
            #[cfg(feature = "dim3")]
            indices,
        );

        if manifold.len > 0 && manifold.points_a.at(0).dist < PREDICTION {
            let target_contact_index = atomic_add_u32(contacts_len, 1) as usize;

            // NOTE: if we exceed the contacts allocation size, just skip
            //       the contact. It’s up to the caller to resize the buffer
            //       and re-run the narrow-phase.
            if target_contact_index < contacts_batch_capacity {
                let mat1 = collider_materials[pair.colliders.x as usize];
                let mat2 = collider_materials[pair.colliders.y as usize];
                contacts[target_contact_index] = IndexedManifold {
                    contact: manifold,
                    colliders: pair.colliders,
                    bodies: UVec2::new(body1, body2),
                    friction: mat1.combined_friction(&mat2),
                    restitution: mat1.combined_restitution(&mat2),
                    _padding: [0.0; 2],
                };
            }
        }
    }
}
