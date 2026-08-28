//! Narrow phase contact generation kernels.
//!
//! Computes contact manifolds from collision pairs detected by the broad phase.

use crate::dynamics::RbdSimParams;
use crate::queries::{
    ColliderMaterial, ContactManifold, IndexedManifold, ball_ball, ball_convex, convex_ball,
    cuboid_cuboid, pfm_pfm,
};
#[cfg(feature = "dim3")]
use crate::queries::{ContactPoint, MAX_MANIFOLD_POINTS, manifold_reduction};
use crate::shapes::{
    Capsule, Polyline, SHAPE_TYPE_BALL, SHAPE_TYPE_CAPSULE, SHAPE_TYPE_CONE, SHAPE_TYPE_CUBOID,
    SHAPE_TYPE_CYLINDER, SHAPE_TYPE_POLYLINE, SHAPE_TYPE_TRIMESH, Shape, TriMesh,
};
use crate::{PaddedVector, Pose, Vector};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{iter::StepRng, sync::atomic_add_u32};

use super::lbvh::{MAX_REDUCE_LANES, max_len_indirect_args};
use crate::broad_phase::CollisionPair;
use crate::utils::{BatchIndices, SliceMut};
use glamx::UVec2;

const WORKGROUP_SIZE: u32 = 64;

/// Resets the contacts counter. One thread per batch.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset_narrow_phase(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pfm_pairs_len: &mut [u32],
) {
    let batch_id = invocation_id.x as usize;
    if batch_id < contacts_len.len() {
        contacts_len.write(batch_id, 0);
        pfm_pairs_len.write(batch_id, 0);
    }
}

/// Initializes indirect dispatch arguments for constraint solver.
///
/// Also inits `mb_sweep_indirect`, the workgroup grid for the per-multibody
/// contact-constraint dispatches (`[multibodies_batch_capacity, num_batches,
/// 1]`).
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_narrow_phase_init_contacts_dispatch(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mb_sweep_indirect: &mut [u32; 3],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] partial: &mut [u32; MAX_REDUCE_LANES as usize],
) {
    max_len_indirect_args(lid.x, contacts_len, indirect_args, partial);
    // `partial[0]` holds the max after the reduction (all lanes synced).
    if lid.x == 0 {
        let any_contacts = partial.read(0) > 0;
        *mb_sweep_indirect.at_mut(0) = if any_contacts {
            batch_ids.multibodies_batch_capacity
        } else {
            0
        };
        *mb_sweep_indirect.at_mut(1) = batch_ids.num_batches;
        *mb_sweep_indirect.at_mut(2) = 1;
    }
}

/// Default cluster threshold: normals must agree within ~5.1 degrees,
/// matching rapier's `contact_clustering::COS_MERGE_ANGLE`. Passed as a
/// uniform so it can be loosened (`-1` merges every manifold of a pair,
/// whatever its normal) to trade contact fidelity for solver cost.
pub const COS_MERGE_ANGLE: f32 = 0.996;

/// Pools `pt` into `cand`, deduplicating against points already there.
///
/// Composite shapes emit near-coincident points on both sides of a shared
/// triangle edge; rapier's clustering collapses those within a quarter of the
/// prediction distance, keeping the deeper one. Same rule here.
#[cfg(feature = "dim3")]
#[inline]
fn pool_dedup(cand: &mut [ContactPoint; 8], num: &mut usize, pt: ContactPoint, dedup_eps_sq: f32) {
    let mut hit = false;
    for k in 0..*num {
        let d = cand.read(k).pt - pt.pt;
        if !hit && d.dot(d) < dedup_eps_sq {
            if pt.dist < cand.read(k).dist {
                cand.write(k, pt);
            }
            hit = true;
        }
    }
    if !hit && *num < 8 {
        cand.write(*num, pt);
        *num += 1;
    }
}

/// Optional contact reduction: compacts each batch's contacts in place by
/// merging manifolds that share a collider pair AND a (nearly) parallel
/// normal into a single `MAX_MANIFOLD_POINTS` manifold. This mirrors rapier's
/// `cluster_manifolds_for_solver` + `reduce_manifold_naive`: cluster by
/// normal, deduplicate near-coincident points, then keep the deepest point,
/// the point furthest from it, and the two tangent extremes.
///
/// Per-triangle trimesh contacts share one `colliders` key and one collider-A
/// local frame, so a flat patch collapses to one manifold while a ridge keeps
/// one cluster per face. The first record of a cluster is kept verbatim, so
/// single-manifold pairs are bit-identical to the unreduced path.
///
/// Two deliberate divergences from rapier. Clusters are reduced incrementally
/// at each merge against an 8-point pool, where rapier accumulates every point
/// (up to 255) and reduces once, so the selection here depends on manifold
/// emission order. And the cluster's normal comes from the deepest point
/// rather than from the manifold that opened it: identical in effect at the
/// default threshold, where every member is within ~5.1 degrees, but it keeps
/// the choice sane when `merge_cos` is loosened.
///
/// Grid `[1, num_batches, 1]`, serial per batch.
#[cfg(feature = "dim3")]
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reduce_contacts(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &mut [IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] params: &RbdSimParams,
) {
    let prediction = params.prediction_distance();
    let merge_cos = params.contact_merge_cos;
    let batch_id = workgroup_id.y;
    let capacity = batch_ids.contacts_batch_capacity as usize;
    let mut contacts = batch_ids.contact_batch_mut(batch_id, contacts);
    let n = (contacts_len.read(batch_id as usize) as usize).min(capacity);

    let mut w = 0usize; // write cursor — always ≤ read cursor, in-place safe
    for i in 0..n {
        let im = contacts[i];
        let mut merged = false;
        for j in 0..w {
            let out = contacts[j];
            if out.colliders.x == im.colliders.x
                && out.colliders.y == im.colliders.y
                && out.contact.normal_a.dot(im.contact.normal_a) >= merge_cos
            {
                // Pool the two manifolds' points (same collider-A local frame),
                // dropping near-duplicates as rapier's clustering does.
                let na = (out.contact.len as usize).min(MAX_MANIFOLD_POINTS);
                let nb = (im.contact.len as usize).min(MAX_MANIFOLD_POINTS);
                let dedup_eps = prediction * 0.25;
                let dedup_eps_sq = dedup_eps * dedup_eps;
                let mut cand = [ContactPoint::default(); 8];
                let mut num = 0usize;
                for k in 0..na {
                    pool_dedup(
                        &mut cand,
                        &mut num,
                        out.contact.points_a.read(k),
                        dedup_eps_sq,
                    );
                }
                for k in 0..nb {
                    pool_dedup(
                        &mut cand,
                        &mut num,
                        im.contact.points_a.read(k),
                        dedup_eps_sq,
                    );
                }
                // Normal of whichever manifold holds the deepest point. rapier
                // keeps the opener's normal instead, which it can afford
                // because its ~5.1 degree cone makes every member equivalent;
                // this degrades gracefully when `merge_cos` is loosened, and
                // agrees with rapier's choice when it is not.
                let mut deep_out = out.contact.points_a.at(0).dist;
                for k in 1..na {
                    let d = out.contact.points_a.at(k).dist;
                    if d < deep_out {
                        deep_out = d;
                    }
                }
                let mut deep_in = im.contact.points_a.at(0).dist;
                for k in 1..nb {
                    let d = im.contact.points_a.at(k).dist;
                    if d < deep_in {
                        deep_in = d;
                    }
                }
                let normal = if deep_in < deep_out {
                    im.contact.normal_a
                } else {
                    out.contact.normal_a
                };
                let mut reduced = manifold_reduction(&cand, num as u32, normal, prediction);
                // `manifold_reduction` fills points/len only.
                reduced.normal_a = normal;
                let mut kept = out;
                kept.contact = reduced;
                contacts[j] = kept;
                merged = true;
                break;
            }
        }
        if !merged {
            contacts[w] = im;
            w += 1;
        }
    }
    // Compacted count; plain store, single writer per batch. (Loop shell per
    // the `gpu_reset_narrow_phase` rustgpu-triviality workaround.)
    for _ in 0..1 {
        contacts_len.write(batch_id as usize, w as u32);
    }
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
    #[spirv(uniform, descriptor_set = 0, binding = 9)] params: &RbdSimParams,
) {
    let prediction = params.prediction_distance();
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
        // and skip pairs whose colliders share the same body. Pair ids are
        // env-local, and `collider_parent` is batch-strided like the other
        // per-collider buffers — an unsliced read here silently returned
        // batch 0's parents for every batch (masked whenever all envs are
        // identical, wrong the moment they aren't).
        let coll_base = batch_ids.coll_start(batch_id);
        let body1 = collider_parent.read(coll_base + pair.colliders.x as usize);
        let body2 = collider_parent.read(coll_base + pair.colliders.y as usize);
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
            manifold = cuboid_cuboid(pose12, &cuboid1, &cuboid2, prediction);
        }

        // Everything else (PFM / trimesh / polyline) is handled by the deferred
        // pass; `manifold.len` stays 0 here so nothing is written.
        if manifold.len > 0 && manifold.points_a.at(0).dist < prediction {
            let target_contact_index = atomic_add_u32(contacts_len, 1) as usize;

            // NOTE: if we exceed the contacts allocation size, just skip
            //       the contact.
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
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &RbdSimParams,
) {
    let prediction = params.prediction_distance();
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
        let shape1 = &shapes[pair.colliders.x as usize];
        let shape2 = &shapes[pair.colliders.y as usize];
        let shape_ty1 = shape1.shape_type();
        let shape_ty2 = shape2.shape_type();

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
        if checked {
            continue;
        }

        let pose1 = poses[pair.colliders.x as usize];
        let pose2 = poses[pair.colliders.y as usize];
        let pose12 = pose1.inverse() * pose2;

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
                // NOTE: if we exceed capacity, just skip the pair.
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
                prediction,
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
                prediction,
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
                prediction,
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
                prediction,
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
    prediction: f32,
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

    // Get the convex shape's AABB in the trimesh's local space, and enlarge with the prediction distance.
    let mut test_aabb = convex.compute_aabb(pose12, vertices);
    test_aabb.mins -= Vector::splat(prediction);
    test_aabb.maxs += Vector::splat(prediction);

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
    prediction: f32,
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

    // Get the convex shape's AABB in the polyline's local space, and enlarge with the prediction distance.
    let thickness = 0.4; // TODO: make thickness configurable or part of the polyline struct
    let mut test_aabb = convex.compute_aabb(pose12, vertices);
    test_aabb.mins -= Vector::splat(prediction + thickness);
    test_aabb.maxs += Vector::splat(prediction + thickness);

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

/// Initializes PFM-PFM dispatch arguments for constraint solver. Dispatch one
/// [`MAX_REDUCE_LANES`]-thread workgroup.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_init_pfm_pfm_dispatch(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] pfm_pairs_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
    #[spirv(workgroup)] partial: &mut [u32; MAX_REDUCE_LANES as usize],
) {
    max_len_indirect_args(lid.x, pfm_pairs_len, indirect_args, partial);
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
    #[spirv(uniform, descriptor_set = 0, binding = 9)] params: &RbdSimParams,
) {
    let prediction = params.prediction_distance();
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
        let coll_base = batch_ids.coll_start(batch_id);
        let body1 = collider_parent.read(coll_base + pair.colliders.x as usize);
        let body2 = collider_parent.read(coll_base + pair.colliders.y as usize);
        if body1 == body2 {
            continue;
        }
        let manifold = pfm_pfm(
            pair.pose12,
            &pair.shape1,
            pair.thickness1,
            &pair.shape2,
            pair.thickness2,
            prediction,
            vertices,
            #[cfg(feature = "dim3")]
            indices,
        );

        if manifold.len > 0 && manifold.points_a.at(0).dist < prediction {
            let target_contact_index = atomic_add_u32(contacts_len, 1) as usize;

            // NOTE: if we exceed capacity, just skip the pair.
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
