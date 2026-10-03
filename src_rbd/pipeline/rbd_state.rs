//! GPU-resident rigid-body state ([`RbdState`]): buffer definitions, accessors,
//! run statistics and capacity/resize policies.
use crate::broad_phase::{BRUTE_FORCE_MAX_COLLIDERS, LbvhState, PfmSortState};
use crate::dynamics::GpuImpulseJointSet;
#[cfg(feature = "dim3")]
use crate::dynamics::GpuMultibodySet;
use crate::math::{Pose, Vector};
use crate::queries::{GpuColliderMaterial, GpuIndexedContact};
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{CollisionPair, ContactPlan, LbvhNode, NarrowPhasePfmPair};
#[cfg(feature = "dim3")]
use crate::shaders::dynamics::MbContactIndexEntry;
use crate::shaders::dynamics::{
    LocalMassProperties as GpuLocalMassProperties, RbdSimParams, TwoBodyConstraint,
    TwoBodyConstraintBuilder, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use crate::shaders::queries::MAX_MANIFOLD_POINTS;
use crate::shaders::shapes::Shape;
use crate::shaders::utils::BatchIndices;
use crate::utils::{ComputeGraphCache, PrefixSumWorkspace, RadixSortWorkspace};

use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuReadback};
use std::time::Duration;
use vortx::tensor::Tensor;

/// One world-space contact point, read back by [`RbdState::debug_contacts`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DebugContact {
    /// Contact point on the first collider, in world space.
    pub point: Vector,
    /// World-space contact normal, pointing away from the first collider.
    pub normal: Vector,
    /// Signed distance along the normal (negative if penetrating).
    pub dist: f32,
    /// Batch (environment) the contact belongs to.
    pub batch: u32,
    /// Env-local indices of the two colliders.
    pub colliders: [u32; 2],
    /// Env-local indices of the parent rigid-bodies of the two colliders.
    pub bodies: [u32; 2],
}

impl DebugContact {
    /// The contact point on the second collider, in world space.
    pub fn point_b(&self) -> Vector {
        self.point + self.normal * self.dist
    }

    /// The point the contact constraint acts on: the middle of the two contact points.
    pub fn solver_point(&self) -> Vector {
        self.point + self.normal * (self.dist * 0.5)
    }
}

/// One node of the broad-phase LBVH, read back by [`RbdState::debug_lbvh`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DebugLbvhNode {
    /// Lower corner of the world-space AABB of the node.
    pub mins: Vector,
    /// Upper corner of that AABB.
    pub maxs: Vector,
    /// Depth in the tree of its batch (the root is at depth 0).
    pub depth: u32,
    /// Whether the node is a leaf, i.e. the AABB of a single collider.
    pub leaf: bool,
    /// Batch (environment) of the tree of this node.
    pub batch: u32,
}

/// Performance statistics collected during a physics simulation step.
#[derive(Default, Clone, Debug)]
pub struct RunStats {
    /// Number of colors used in the graph coloring algorithm for parallel constraint solving.
    pub num_colors: u32,
    /// Number of iterations the coloring algorithm took to converge.
    pub coloring_iterations: u32,
    /// Total command encoding time.
    pub encoding_time: Duration,
    /// Per-pass GPU timestamp durations (label, milliseconds).
    pub gpu_pass_times: Vec<(String, f64)>,
    /// Total GPU time across all measured passes, in milliseconds.
    pub gpu_total_time_ms: f64,
}

impl RunStats {
    /// Returns the command encoding time in milliseconds.
    pub fn encoding_time_ms(&self) -> f32 {
        self.encoding_time.as_secs_f32() * 1000.0
    }
}

/// Minimal capacities used when allocating a rigid-body scene's GPU buffers.
///
/// Consumed by [`RbdState::empty`]; the higher-level `NexusState` stores one of
/// these and forwards it when the rigid-body sub-state is first created.
#[derive(Copy, Clone, Debug)]
pub struct RbdCapacities {
    /// Number of independent simulation batches (environments).
    pub batches: u32,
    /// Maximum number of rigid-bodies (and colliders) per batch.
    pub body_capacity: u32,
    /// Maximum number of collision pairs reserved per batch.
    ///
    /// This may or may not be automatically resized depending on [`Self::collisions_resize_policy`].
    pub collisions_capacity: u32,
    /// Maximum number of multibody contact-constraint slots reserved per batch
    /// (each contact point involving a multibody costs one normal slot plus
    /// `DIM - 1` friction slots). Resized according to
    /// [`Self::collisions_resize_policy`], like the collision buffers, and never
    /// shrunk below the per-multibody overflow reservation.
    pub mb_contact_constraints_capacity: u32,
    /// How internal collision buffers gets automatically resized (or not).
    ///
    /// Note that setting both [`Self::collisions_resize_policy`] and
    /// [`Self::solver_colors_resize_policy`] to [`RbdResizePolicy::Fixed`] eliminates a
    /// GPU->CPU buffer readback, resulting in a larger performance gain than just setting
    /// only one of them to `Fixed`.
    pub collisions_resize_policy: RbdResizePolicy,
    /// Maximum number of colors used by the solver for constraints coloring.
    pub solver_colors: u32,
    /// How internal constraints coloring gets automatically adjusted (or not).
    ///
    /// While this doesn’t change any buffer allocation, this affects the number of
    /// iterations the constraints coloring step applies, which has a computational cost.
    ///
    /// Note that `RbdResizePolicy::Fit` for solver colors will currently act like `::Grow`
    /// (i.e. the color won’t go back down yet).
    ///
    /// Note that setting both [`Self::collisions_resize_policy`] and
    /// [`Self::solver_colors_resize_policy`] to [`RbdResizePolicy::Fixed`] eliminates a
    /// GPU->CPU buffer readback, resulting in a larger performance gain than just setting
    /// only one of them to `Fixed`.
    pub solver_colors_resize_policy: RbdResizePolicy,
}

impl Default for RbdCapacities {
    fn default() -> Self {
        Self {
            batches: 1,
            body_capacity: 65536,
            collisions_capacity: 4096,
            mb_contact_constraints_capacity: 256,
            collisions_resize_policy: RbdResizePolicy::Grow,
            solver_colors: 8,
            solver_colors_resize_policy: RbdResizePolicy::Grow,
        }
    }
}

/// Governs the way the rigid-body dynamics pipeline automatically resizes internal buffers storing
/// data with unpredictable size (like collisions).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum RbdResizePolicy {
    /// If specified, the internal storage buffers are never resized.
    ///
    /// Overflowing the buffers may result is dropped collisions.
    Fixed,
    /// If specified, the internal storage buffers are grown automatically (but never shrunk).
    #[default]
    Grow,
    /// If specified, the internal storage buffers are grown and shrunk automatically.
    Fit,
}

/// GPU-resident physics simulation state containing all rigid bodies, shapes, and solver data.
///
/// Holds all the buffers needed for a complete physics simulation on the GPU
/// (poses, velocities, mass properties, shapes, contacts, constraints, solver
/// state, LBVH, etc.). Can be initialized from CPU-side Rapier data structures
/// and then updated entirely on the GPU each frame.
pub struct RbdState {
    pub(super) capacities: RbdCapacities,
    pub(super) num_batches: u32,
    pub(super) num_colliders_per_batch: u32,
    pub(super) num_solver_iterations: u32,
    pub(super) sim_params: Tensor<RbdSimParams>,
    /// CPU mirror of `sim_params`, so runtime setters can patch one field and
    /// re-upload without rebuilding the whole state.
    pub(super) sim_params_cpu: RbdSimParams,
    /// Per-body world-origin pose (matches rapier's `RigidBody::position`).
    pub(super) body_poses: Tensor<Pose>,
    /// Per-body COM-centered pose (rapier's `SolverPose`) used temporarily by
    /// the solver (and then written back to `body_poses` after un-centering).
    pub(super) solver_body_poses: Tensor<Pose>,
    pub(super) local_mprops: Tensor<GpuLocalMassProperties>,
    pub(super) mprops: Tensor<GpuWorldMassProperties>,
    pub(super) vels: Tensor<GpuVelocity>,
    /// GPU-resident rigid-body reset templates, published by
    /// `publish_reset_templates` and consumed by `reset_envs_from_templates`.
    #[cfg(feature = "dim3")]
    pub(super) reset_templates_bodies: Option<ResetTemplatesBodies>,
    pub(super) solver_vels: Tensor<GpuVelocity>,
    pub(super) solver_vels_inc: Tensor<GpuVelocity>,
    pub(super) vertex_buffers: Tensor<PaddedVector>,
    pub(super) index_buffers: Tensor<u32>,
    pub(super) shapes: Tensor<Shape>,
    /// Per-collider local pose, relative to its parent rigid-body.
    pub(super) collider_local_poses: Tensor<Pose>,
    /// Per-collider parent rigid-body index.
    ///
    /// Multiple colliders can be attached to the same rigid-body.
    pub(super) collider_parent: Tensor<u32>,
    /// World-pose of colliders, used by collision detection.
    pub(super) collider_world_poses: Tensor<Pose>,
    /// Per-collider [`crate::rapier::geometry::InteractionGroups`].
    pub(super) collision_groups: Tensor<crate::rapier::geometry::InteractionGroups>,
    /// Per-collider broad-phase pair-filter key:
    /// - `[0]`: to prevent colliders of the same body from colliding.
    /// - `[1]`: to prevent colliders of adjacent links from a multibody from coliding.
    ///
    /// Nonzero keys that are equal never collide.
    pub(super) pair_filter: Tensor<[u32; 2]>,
    /// Per-collider friction / restitution coefficients (+ combine rules),
    pub(super) collider_materials: Tensor<GpuColliderMaterial>,
    pub(super) collision_pairs: Tensor<CollisionPair>,
    /// Single global live collision-pair count (length 1): every batch appends
    /// to the same flat pair buffer.
    pub(super) collision_pairs_len: Tensor<u32>,
    /// Non-blocking readback of `[collision_pairs_len, pfm_pairs_len,
    /// uncolored]` used by
    /// [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers)
    /// to grow buffers without stalling.
    pub(super) resize_readback: GpuReadback<u32>,
    pub(super) collision_pairs_indirect: Tensor<[u32; 3]>,
    /// CPU-side mirrors of the dynamic capacities. The capacity values
    /// live in the [`BatchIndices`] uniform; these mirrors let
    /// [`Self::rebuild_batch_indices`] re-emit it whenever a buffer grows.
    /// Total capacity of the flat contacts buffer (and of every contacts-keyed
    /// buffer).
    pub(super) contacts_capacity_cpu: u32,
    /// Total capacity of the flat collision-pair (and PFM) buffer.
    pub(super) collision_pairs_capacity_cpu: u32,
    /// Most recently read live collision-pair count — the total across all
    /// batches, harvested by the non-blocking readback in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers).
    /// Surfaced in the viewer UI; lags the GPU by a frame or two like the resize.
    pub(super) collision_pairs_len_cpu: u32,
    /// Bumped on every GPU buffer (re)allocation or capacity change; part of
    /// [`Self::graph_key`].
    pub(super) graph_generation: u64,
    /// Cached compute graph of this state's step loop, keyed by
    /// [`Self::graph_key`]. Driven by `NexusPipeline::simulate`.
    pub compute_graph: ComputeGraphCache<RbdGraphKey>,
    /// CPU mirror of the multibody contact-constraint slot demand, refreshed
    /// by the same (asynchronous) readback as `collision_pairs_len_cpu`.
    #[cfg(feature = "dim3")]
    pub(super) mb_cons_demand_cpu: u32,
    /// Single uniform aggregating every per-batch capacity and packed-buffer
    /// section offset consumed by the compute kernels (multibody and RBD
    /// sides). Rebuilt by [`Self::rebuild_batch_indices`] whenever any of its
    /// constituent caps changes (e.g. when the contacts buffer grows).
    pub(super) batch_indices: Tensor<BatchIndices>,
    pub(super) pfm_pairs: Tensor<NarrowPhasePfmPair>,
    /// Single global live PFM work-list count (length 1).
    pub(super) pfm_pairs_len: Tensor<u32>,
    pub(super) pfm_pairs_indirect: Tensor<[u32; 3]>,
    pub(super) contacts: Tensor<GpuIndexedContact>,
    /// Flat dispatch grid over the whole contacts range, written by
    /// `gpu_contact_plan`.
    pub(super) contacts_indirect: Tensor<[u32; 3]>,
    /// Clamped per-frame list totals (see `gpu_contact_plan`): the sweep
    /// bound and the positional-slot bases of the flat contacts buffer.
    pub(super) contact_plan: Tensor<ContactPlan>,
    /// Buffers backing the per-pair PFM sort of the contact-reduction path.
    pub(super) pfm_sort: PfmSortState,
    /// Contact→multibody index: per-(multibody, batch) segments of contact
    /// slots (one entry per contact touching a multibody link), rebuilt each
    /// step. Sized like `contacts` (each contact owns at most one entry).
    #[cfg(feature = "dim3")]
    pub(super) mb_contact_index: Tensor<MbContactIndexEntry>,
    /// Workgroup grid for the per-multibody contact-constraint dispatches:
    /// `[multibodies_batch_capacity, num_batches, 1]`.
    pub(super) mb_sweep_indirect: Tensor<[u32; 3]>,
    pub(super) new_constraints: Tensor<TwoBodyConstraint>,
    pub(super) new_constraint_builders: Tensor<TwoBodyConstraintBuilder>,
    pub(super) new_constraints_counts: Tensor<u32>,
    pub(super) new_body_constraint_ids: Tensor<u32>,
    pub(super) old_constraints: Tensor<TwoBodyConstraint>,
    pub(super) old_constraint_builders: Tensor<TwoBodyConstraintBuilder>,
    pub(super) old_constraints_counts: Tensor<u32>,
    pub(super) old_body_constraint_ids: Tensor<u32>,
    pub(super) constraints_colors: Tensor<u32>,
    pub(super) old_constraints_colors: Tensor<u32>,
    pub(super) colored: Tensor<u32>,
    pub(super) constraints_rands: Tensor<u32>,
    /// The single `(color, batch)` bucket buffer, color-major, of length
    /// `(max_colors + 3) * num_batches`: counts, then scanned exclusive
    /// starts, then post-scatter exclusive ends (what the sweeps read).
    pub(super) color_buckets: Tensor<u32>,
    /// Constraint indices bucket-sorted by `(color, batch)`.
    pub(super) color_sorted_ids: Tensor<u32>,
    pub(super) curr_color: Tensor<u32>,
    /// Pre-built per-color-index uniforms: `color_uniforms[c] == c`.
    /// [`Self::ensure_color_uniforms`].
    pub(super) color_uniforms: Vec<Tensor<u32>>,
    pub(super) uncolored: Tensor<u32>,
    pub(super) uncolored_staging: Tensor<u32>,
    pub(super) lbvh: LbvhState,
    pub(super) joints: GpuImpulseJointSet,
    #[cfg(feature = "dim3")]
    pub(super) multibodies: GpuMultibodySet,
    /// The one gravity uniform every rigid-body and multibody kernel reads.
    pub(super) gravity: Tensor<glamx::Vec4>,
    /// Per-body "graph group" id, used by graph coloring to treat all bodies of
    /// the same multibody as a single node. For free bodies, `body_group[i] = i`;
    /// bodies of a multibody all share the group id of the root link, so two
    /// contacts touching different bodies of the same multibody can never be
    /// assigned the same color.
    pub(super) body_group: Tensor<u32>,
    /// Per-body flag, 1 for the links of a multibody. The rigid-body contact
    /// pipeline skips every manifold touching such a body: the multibody
    /// solver owns those contacts.
    pub(super) body_is_multibody: Tensor<u32>,
    pub(super) prefix_sum_workspace: PrefixSumWorkspace,
    /// Separate workspace for the color-bucket prefix scan (different length
    /// than the body-count scan, so sharing one workspace would thrash its
    /// cached sizing).
    pub(super) bucket_prefix_workspace: PrefixSumWorkspace,
    /// Maximum number of constraint colors the solver will iterate.
    pub(super) max_colors: u32,
    /// `true` when every body is either non-dynamic or multibody-controlled
    /// (its rb-side `inv_mass` is zero),i.e., we can skip the contact pipelines.
    pub(super) rb_contacts_inert: bool,
    /// CPU-side mirror of the number of *active* colliders per batch. Identical
    /// across all batches by the equal-topology invariant; slots in
    /// `[num_active_colliders .. num_colliders_per_batch)` are reserved padding.
    /// Mirrors `BatchIndices::colliders_len` and is kept in sync by
    /// the incremental [`Self::append_bodies`] / [`Self::remove_bodies`] APIs.
    pub(super) num_active_colliders: u32,
    /// CPU-side mirror of the number of *active* rigid bodies per batch.
    /// Mirrors `BatchIndices::bodies_len`. Always `<= num_active_colliders`.
    pub(super) num_active_bodies: u32,
    /// The state of the deterministic mode (see [`Self::set_deterministic`]).
    pub(super) determinism: RbdDeterminismState,
}

/// Flags and buffers of the deterministic mode of an [`RbdState`].
pub(super) struct RbdDeterminismState {
    /// See [`RbdState::set_deterministic`].
    pub(super) enabled: bool,
    /// Whether any collider is a trimesh or polyline (several manifolds per pair).
    /// Without them, the contact sort skips its sub-shape pass.
    pub(super) has_composite_shapes: bool,
    /// Color picked by each constraint in the current round, before the conflict pass.
    /// Only used in deterministic mode, but always allocated (the kernels bind it).
    pub(super) pending_colors: Tensor<u32>,
    /// Output of the deterministic contact sort, swapped with [`RbdState::contacts`].
    /// Only allocated in deterministic mode.
    pub(super) contacts_scratch: Tensor<GpuIndexedContact>,
    /// Output of the deterministic sort of the body constraint lists.
    /// Only allocated in deterministic mode.
    pub(super) stable_body_constraint_ids: Tensor<u32>,
    /// Output of the deterministic sort of the multibody contact index.
    /// Only allocated in deterministic mode.
    #[cfg(feature = "dim3")]
    pub(super) stable_mb_contact_index: Tensor<MbContactIndexEntry>,
    /// Key and value buffers of the contact sort. Only allocated in deterministic mode.
    pub(super) contact_sort_keys: Tensor<u32>,
    pub(super) contact_sort_ids: Tensor<u32>,
    pub(super) contact_sort_keys_out: Tensor<u32>,
    pub(super) contact_sort_ids_out: Tensor<u32>,
    pub(super) contact_sort_n: Tensor<u32>,
    pub(super) contact_sort_workspace: RadixSortWorkspace,
    /// `key_selector` uniforms of the contact sort (`[i] == i`), built once so no
    /// buffer is rewritten between dispatches.
    pub(super) contact_key_selectors: Vec<Tensor<u32>>,
}

impl RbdDeterminismState {
    /// The deterministic mode off: only `pending_colors` is allocated.
    pub(super) fn new(
        backend: &GpuBackend,
        has_composite_shapes: bool,
        pending_colors: Tensor<u32>,
    ) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        Self {
            enabled: false,
            has_composite_shapes,
            pending_colors,
            contacts_scratch: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            stable_body_constraint_ids: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            #[cfg(feature = "dim3")]
            stable_mb_contact_index: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            contact_sort_keys: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            contact_sort_ids: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            contact_sort_keys_out: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            contact_sort_ids_out: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            contact_sort_n: Tensor::vector_uninit(backend, 0, storage).unwrap(),
            contact_sort_workspace: RadixSortWorkspace::new(backend),
            contact_key_selectors: Vec::new(),
        }
    }
}

impl RbdState {
    /// Re-upload the shared `BatchIndices` uniform after any of its
    /// constituent per-batch capacities has changed (e.g. after the contacts
    /// buffer grows in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers), or after multibody
    /// impulse-joint capacities are updated via
    /// `GpuMultibodySet::set_impulse_joints`). Call whenever a cap edit
    /// happens that any kernel reads via its `batch_ids` uniform.
    pub(super) fn rebuild_batch_indices(&mut self, backend: &GpuBackend) {
        self.graph_generation += 1;
        #[allow(unused_mut)] // Only mutated with the dim3 (multibody) feature.
        let mut bi = BatchIndices {
            num_batches: self.num_batches,
            colliders_batch_capacity: self.num_colliders_per_batch,
            colliders_len: self.num_active_colliders,
            bodies_len: self.num_active_bodies,
            collision_pairs_capacity: self.collision_pairs_capacity_cpu,
            contacts_capacity: self.contacts_capacity_cpu,
            impulse_joints_len: self.joints.num_active_joints(),
            solver_color_buckets_stride: self.max_colors + 3,
            deterministic: self.determinism.enabled as u32,
            contact_sort_collider_shift: crate::dynamics::contact_sort_collider_shift(
                self.num_active_colliders * self.num_batches,
            ),
            ..Default::default()
        };
        #[cfg(feature = "dim3")]
        self.multibodies.fill_batch_indices(&mut bi);
        backend
            .write_buffer(self.batch_indices.buffer_mut(), 0, &[bi])
            .unwrap();
    }

    /// Shared per-batch index uniform.
    pub fn batch_indices(&self) -> &Tensor<BatchIndices> {
        &self.batch_indices
    }

    /// Sets the maximum number of constraint colors used by the per-step
    /// graph coloring + Gauss-Seidel solver loop. Lower values cap solver
    /// time at the cost of dropping over-budget constraints.
    pub fn set_max_colors(&mut self, max_colors: u32) {
        self.max_colors = max_colors.max(1);
    }

    /// Grows [`Self::color_uniforms`] so indices `0..n` are available.
    pub(super) fn ensure_color_uniforms(&mut self, backend: &GpuBackend, n: u32) {
        if (self.color_uniforms.len() as u32) < n {
            self.graph_generation += 1;
        }
        for c in self.color_uniforms.len() as u32..n {
            self.color_uniforms
                .push(Tensor::scalar(backend, c, BufferUsages::UNIFORM).unwrap());
        }
    }

    /// Returns the configured max color count.
    pub fn max_colors(&self) -> u32 {
        self.max_colors
    }

    /// Makes two runs of the same scene give identical results (same machine and build).
    /// Off by default. Adds a few sort passes and forces the fixed resize policies.
    pub fn set_deterministic(&mut self, backend: &GpuBackend, deterministic: bool) {
        // Does nothing if unchanged: the viewer calls this every frame.
        if self.determinism.enabled == deterministic {
            return;
        }

        self.determinism.enabled = deterministic;
        // Reallocates buffers and changes the dispatched passes.
        self.graph_generation += 1;

        if deterministic {
            self.capacities.collisions_resize_policy = RbdResizePolicy::Fixed;
            self.capacities.solver_colors_resize_policy = RbdResizePolicy::Fixed;
            self.alloc_deterministic_buffers(backend);
            self.zero_index_buffers(backend);
        } else {
            // Frees the deterministic buffers, except the always-bound `pending_colors`.
            let pending_colors = std::mem::replace(
                &mut self.determinism.pending_colors,
                Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE).unwrap(),
            );
            let has_composite_shapes = self.determinism.has_composite_shapes;
            self.determinism =
                RbdDeterminismState::new(backend, has_composite_shapes, pending_colors);
        }

        self.rebuild_batch_indices(backend);
    }

    /// Whether the pipeline runs in deterministic mode.
    pub fn deterministic(&self) -> bool {
        self.determinism.enabled
    }

    /// Allocates the buffers of the deterministic passes, sized from the contact capacity.
    fn alloc_deterministic_buffers(&mut self, backend: &GpuBackend) {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let contacts_total = self.contacts.len() as u32;
        #[cfg(feature = "dim3")]
        let mb_contact_index_len = self.mb_contact_index.len() as u32;
        let det = &mut self.determinism;

        det.contacts_scratch = Tensor::vector_uninit(backend, contacts_total, storage).unwrap();
        det.stable_body_constraint_ids =
            Tensor::vector_uninit(backend, contacts_total * 2, storage).unwrap();
        #[cfg(feature = "dim3")]
        {
            det.stable_mb_contact_index =
                Tensor::vector_uninit(backend, mb_contact_index_len, storage).unwrap();
        }
        det.contact_sort_keys = Tensor::vector_uninit(backend, contacts_total, storage).unwrap();
        det.contact_sort_ids = Tensor::vector_uninit(backend, contacts_total, storage).unwrap();
        det.contact_sort_keys_out =
            Tensor::vector_uninit(backend, contacts_total, storage).unwrap();
        det.contact_sort_ids_out = Tensor::vector_uninit(backend, contacts_total, storage).unwrap();
        det.contact_sort_n = Tensor::vector(backend, vec![0u32; 1], storage).unwrap();

        // One uniform per key selector: rewriting one uniform between dispatches
        // makes all of them read the last value on some backends.
        det.contact_key_selectors = (0..8)
            .map(|i| Tensor::scalar(backend, i, BufferUsages::UNIFORM).unwrap())
            .collect();
    }

    /// Zeroes the `u32` scratch buffers, which are allocated uninitialized.
    /// The big manifold and constraint buffers are skipped: their reads are bounded.
    fn zero_index_buffers(&mut self, backend: &GpuBackend) {
        // Reallocated instead of written: some buffers don't have `COPY_DST`.
        fn zero(backend: &GpuBackend, t: &mut Tensor<u32>) {
            let len = t.len() as u32;
            if len != 0 {
                let usages =
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
                *t = Tensor::vector(backend, vec![0u32; len as usize], usages).unwrap();
            }
        }

        zero(backend, &mut self.new_body_constraint_ids);
        zero(backend, &mut self.old_body_constraint_ids);
        zero(backend, &mut self.determinism.stable_body_constraint_ids);
        zero(backend, &mut self.constraints_colors);
        zero(backend, &mut self.determinism.pending_colors);
        zero(backend, &mut self.old_constraints_colors);
        zero(backend, &mut self.colored);
        zero(backend, &mut self.color_sorted_ids);
        zero(backend, &mut self.determinism.contact_sort_keys);
        zero(backend, &mut self.determinism.contact_sort_ids);
        zero(backend, &mut self.determinism.contact_sort_keys_out);
        zero(backend, &mut self.determinism.contact_sort_ids_out);
    }

    /// `true` when every rigid-body contact constraint is provably a no-op.
    pub fn rb_contacts_inert(&self) -> bool {
        self.rb_contacts_inert
    }
}

impl RbdState {
    /// Per-collider world pose (= `body_poses[i] * collider_local_poses[i]`).
    /// This is what rendering / debug tooling typically wants — the actual
    /// pose of each collider's shape in world space.
    ///
    /// Refreshed once per step before broad-phase / narrow-phase / contact
    /// constraint init; not mutated during the substep loop.
    pub fn collider_poses(&self) -> &Tensor<Pose> {
        &self.collider_world_poses
    }

    /// Per-body world-origin pose (matches rapier's `RigidBody::position`).
    pub fn body_poses(&self) -> &Tensor<Pose> {
        &self.body_poses
    }

    /// Mutable access to the per-body world-origin poses, for bodies this
    /// pipeline doesn't own and that another solver integrates itself.
    ///
    /// Only valid between steps: the solver works on `solver_body_poses` and
    /// overwrites this buffer wholesale in its finalize pass.
    pub fn body_poses_mut(&mut self) -> &mut Tensor<Pose> {
        &mut self.body_poses
    }

    /// Per-body world-space velocities, indexed like [`Self::body_poses`].
    pub fn vels(&self) -> &Tensor<GpuVelocity> {
        &self.vels
    }

    /// Mutable access to the per-body velocities, for teleports and resets.
    /// Only valid between steps, like [`Self::body_poses_mut`].
    pub fn vels_mut(&mut self) -> &mut Tensor<GpuVelocity> {
        &mut self.vels
    }

    /// Live collision-pair count (total across all batches) most recently
    /// harvested by the non-blocking readback in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers). Lags the GPU by a
    /// frame or two; `0` until the first readback completes.
    pub fn collision_pairs_len(&self) -> u32 {
        self.collision_pairs_len_cpu
    }

    /// GPU buffer of the broad-phase collision pairs found this step.
    pub fn collision_pairs(&self) -> &Tensor<CollisionPair> {
        &self.collision_pairs
    }

    /// GPU buffer holding the single global collision-pair count. Unlike
    /// [`Self::collision_pairs_len`], which returns the CPU mirror from the
    /// last readback, this is the value the current step wrote.
    pub fn collision_pairs_len_gpu(&self) -> &Tensor<u32> {
        &self.collision_pairs_len
    }

    /// The max number a collision pairs the state can currently store.
    pub fn collision_pairs_capacity(&self) -> u32 {
        self.collision_pairs.capacity() as u32
    }

    /// Uploads a new gravity vector, e.g. `[0.0, 0.0, -9.81]` for a Z-up scene.
    /// Every solver path reads this one uniform, so it applies to free
    /// rigid-bodies and multibody links alike. In 2D the third component is
    /// ignored.
    pub fn set_gravity(&mut self, backend: &GpuBackend, gravity: [f32; 3]) {
        self.gravity = Self::gravity_tensor(backend, gravity);
    }

    /// Sets how nearly parallel two contact normals must be for their
    /// manifolds to be clustered by `gpu_reduce_contacts`, as a cosine.
    ///
    /// Defaults to [`COS_MERGE_ANGLE`](crate::shaders::broad_phase::COS_MERGE_ANGLE)
    /// (~5.1 degrees), matching rapier. Pass `-1.0` to merge every manifold of
    /// a collider pair regardless of normal: cheaper, but a single averaged
    /// normal then stands in for a ridge or a step edge.
    pub fn set_contact_merge_cos(&mut self, backend: &GpuBackend, cos: f32) {
        let mut params = self.sim_params_cpu;
        params.contact_merge_cos = cos;
        let _ = backend.write_buffer(self.sim_params.buffer_mut(), 0, &[params]);
        self.sim_params_cpu = params;
    }

    /// Sets how many PGS iterations the biased pass runs per substep (rigid-body
    /// and multibody sweeps alike), without rebuilding the GPU state.
    #[cfg(feature = "dim3")]
    pub fn set_num_internal_pgs_iterations(&mut self, backend: &GpuBackend, n: u32) {
        let n = n.max(1);
        self.multibodies.set_num_internal_pgs_iterations(n);
        // Keep the mirror (and the uniform it backs) honest, even though no
        // shader reads this field.
        let mut params = self.sim_params_cpu;
        params.num_internal_pgs_iterations = n;
        let _ = backend.write_buffer(self.sim_params.buffer_mut(), 0, &[params]);
        self.sim_params_cpu = params;
    }

    /// PGS iterations per substep in the biased pass.
    #[cfg(feature = "dim3")]
    pub fn num_internal_pgs_iterations(&self) -> u32 {
        self.multibodies.num_internal_pgs_iterations()
    }

    /// The gravity uniform shared by every solver kernel.
    pub fn gravity(&self) -> &Tensor<glamx::Vec4> {
        &self.gravity
    }

    pub(super) fn gravity_tensor(backend: &GpuBackend, gravity: [f32; 3]) -> Tensor<glamx::Vec4> {
        Tensor::scalar(
            backend,
            glamx::Vec4::new(gravity[0], gravity[1], gravity[2], 0.0),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap()
    }

    /// Per-collider world pose.
    pub fn collider_world_poses(&self) -> &Tensor<Pose> {
        &self.collider_world_poses
    }

    /// The set of joints part of the simulation.
    pub fn joints(&self) -> &GpuImpulseJointSet {
        &self.joints
    }

    /// Mutable access to the multibody set, useful for runtime mutations like
    /// per-step motor changes.
    #[cfg(feature = "dim3")]
    pub fn multibodies_mut(&mut self) -> &mut crate::dynamics::GpuMultibodySet {
        &mut self.multibodies
    }

    /// Last known multibody contact-constraint slot demand (each contact point
    /// costs one normal + `DIM - 1` friction slots). Refreshed by the same
    /// asynchronous readback as [`Self::collision_pairs_len`], so it lags a
    /// frame or two behind the GPU.
    #[cfg(feature = "dim3")]
    pub fn mb_contact_constraints_len(&self) -> u32 {
        self.mb_cons_demand_cpu
    }

    /// Current capacity (in slots) of the flat multibody contact-constraint
    /// buffer (see [`RbdCapacities::mb_contact_constraints_capacity`]).
    #[cfg(feature = "dim3")]
    pub fn mb_contact_constraints_capacity(&self) -> u32 {
        self.multibodies.contact_constraints_capacity()
    }

    /// Immutable access to the multibody set (e.g. to read back `dof_state`).
    #[cfg(feature = "dim3")]
    pub fn multibodies(&self) -> &crate::dynamics::GpuMultibodySet {
        &self.multibodies
    }

    /// Enables or disables the implicit treatment of multibody coriolis forces.
    #[cfg(feature = "dim3")]
    pub fn set_implicit_coriolis(&mut self, backend: &GpuBackend, enabled: bool) {
        self.multibodies.set_implicit_coriolis(enabled);
        self.rebuild_batch_indices(backend);
    }

    /// Sets the multibody refresh cadence: `refresh` rebuilds the joint and
    /// contact constraints, mass matrix and LU factors every substep (the
    /// default); off, they are built once per step and later substeps only
    /// refresh the joint rhs and limit activity. `light` (ignored while
    /// `refresh` is on) keeps the constraints per substep but the mass matrix
    /// per step. Pure dispatch gating: no GPU layout changes.
    #[cfg(feature = "dim3")]
    pub fn set_substep_refresh(&mut self, refresh: bool, light: bool) {
        self.multibodies.set_substep_refresh(refresh);
        self.multibodies.set_substep_refresh_light(light);
    }

    /// Sets the per-DoF dry joint friction (N·m).
    #[cfg(feature = "dim3")]
    pub fn set_dof_frictionloss(&mut self, backend: &GpuBackend, values: &[f32]) {
        self.multibodies.set_dof_frictionloss(backend, values);
        self.rebuild_batch_indices(backend);
        self.multibodies.constraint_caps_dirty = false;
    }

    /// Returns a reference to the GPU buffer containing collision shapes.
    ///
    /// Each shape corresponds to one rigid body in the simulation.
    pub fn shapes(&self) -> &Tensor<Shape> {
        &self.shapes
    }

    /// Per-collider parent rigid-body slot map (env-local). For debugging.
    pub fn collider_parent(&self) -> &Tensor<u32> {
        &self.collider_parent
    }

    /// GPU buffer holding the contact manifolds.
    pub fn contacts(&self) -> &Tensor<GpuIndexedContact> {
        &self.contacts
    }

    /// GPU buffer holding the rigid-body contact constraints of the current
    /// step (impulses included). For debugging.
    pub fn rigid_contact_constraints(&self) -> &Tensor<TwoBodyConstraint> {
        &self.new_constraints
    }

    /// Debug: reads the contacts back from the GPU, as world-space points of all batches.
    /// Slow: copies the whole contact and collider-pose buffers to the CPU.
    pub async fn debug_contacts(&self, backend: &GpuBackend) -> Vec<DebugContact> {
        let Ok(manifolds) = backend
            .slow_read_vec::<GpuIndexedContact>(self.contacts.buffer())
            .await
        else {
            return Vec::new();
        };
        let Ok(poses) = backend
            .slow_read_vec::<Pose>(self.collider_world_poses.buffer())
            .await
        else {
            return Vec::new();
        };

        // All slots of the flat contact buffer are written each frame (`len == 0` if empty).
        // Collider and body ids are global: `global = local * num_batches + batch`.
        let nb = self.num_batches.max(1);
        let mut result = Vec::new();

        for manifold in &manifolds {
            let collider_a = manifold.colliders.x;
            let Some(pose_a) = poses.get(collider_a as usize) else {
                continue;
            };
            let normal = pose_a.transform_vector(manifold.contact.normal_a);

            for k in 0..(manifold.contact.len as usize).min(MAX_MANIFOLD_POINTS) {
                let point = manifold.contact.points_a[k];
                result.push(DebugContact {
                    point: pose_a.transform_point(point.pt),
                    normal,
                    dist: point.dist,
                    batch: collider_a % nb,
                    colliders: [manifold.colliders.x / nb, manifold.colliders.y / nb],
                    bodies: [manifold.bodies.x / nb, manifold.bodies.y / nb],
                });
            }
        }

        result
    }

    /// Whether the broad phase tests all collider pairs instead of using the LBVH.
    /// This happens for small scenes; `NEXUS_DISABLE_BF` forces the LBVH.
    pub fn uses_brute_force_broad_phase(&self) -> bool {
        self.num_active_colliders <= BRUTE_FORCE_MAX_COLLIDERS
            && std::env::var("NEXUS_DISABLE_BF").is_err()
    }

    /// Debug: reads the LBVH back from the GPU, with the depth of each node.
    /// `None` if the brute-force broad phase is used (there is no tree then). Slow.
    pub async fn debug_lbvh(&self, backend: &GpuBackend) -> Option<Vec<DebugLbvhNode>> {
        let n = self.num_active_colliders as usize;
        if self.uses_brute_force_broad_phase() || n == 0 {
            return None;
        }
        let tree = backend
            .slow_read_vec::<LbvhNode>(self.lbvh.tree().buffer())
            .await
            .ok()?;

        let stride = 2 * self.num_colliders_per_batch as usize;
        let num_internal = n - 1;
        let mut result = Vec::with_capacity(2 * n * self.num_batches as usize);
        let mut stack = Vec::new();

        for batch in 0..self.num_batches as usize {
            let Some(nodes) = tree.get(batch * stride..batch * stride + 2 * n - 1) else {
                break;
            };
            stack.clear();
            stack.push((0usize, 0u32));
            // Bounded by the node count, so a broken tree can't loop forever.
            for _ in 0..nodes.len() {
                let Some((id, depth)) = stack.pop() else {
                    break;
                };
                let Some(node) = nodes.get(id) else { continue };
                let leaf = id >= num_internal;
                result.push(DebugLbvhNode {
                    mins: node.aabb.mins,
                    maxs: node.aabb.maxs,
                    depth,
                    leaf,
                    batch: batch as u32,
                });
                if !leaf {
                    stack.push((node.right as usize, depth + 1));
                    stack.push((node.left as usize, depth + 1));
                }
            }
        }

        Some(result)
    }

    /// Debug: read back active contacts as `(collider_a, collider_b, body_a,
    /// body_b, manifold_len)` tuples (only `len > 0` entries).
    pub fn debug_contact_pairs(&self, backend: &GpuBackend) -> Vec<(u32, u32, u32, u32, u32)> {
        let v: Vec<GpuIndexedContact> =
            futures::executor::block_on(backend.slow_read_vec(self.contacts.buffer()))
                .unwrap_or_default();
        v.iter()
            .filter(|c| c.contact.len > 0)
            .map(|c| {
                (
                    c.colliders.x,
                    c.colliders.y,
                    c.bodies.x,
                    c.bodies.y,
                    c.contact.len,
                )
            })
            .collect()
    }

    /// Debug: per active constraint `(index, solver_body_a, solver_body_b, color, len)`.
    /// Used to check the graph coloring never gives two constraints that share
    /// a body the same color.
    ///
    /// Reads the `old_*` buffers, i.e. the constraints and colors of the last step.
    pub fn debug_constraint_colors(&self, backend: &GpuBackend) -> Vec<(u32, u32, u32, u32, u32)> {
        let cons: Vec<TwoBodyConstraint> =
            futures::executor::block_on(backend.slow_read_vec(self.old_constraints.buffer()))
                .unwrap_or_default();
        let colors: Vec<u32> = futures::executor::block_on(
            backend.slow_read_vec(self.old_constraints_colors.buffer()),
        )
        .unwrap_or_default();
        // Only the slots below the contact bound of the last step are valid.
        let bound = futures::executor::block_on(
            backend.slow_read_vec::<ContactPlan>(self.contact_plan.buffer()),
        )
        .ok()
        .and_then(|plan| plan.first().map(|p| p.bound as usize))
        .unwrap_or(0);

        let mut out = Vec::new();
        for (i, c) in cons.iter().enumerate().take(bound) {
            if c.len == 0 {
                continue;
            }
            let color = colors.get(i).copied().unwrap_or(u32::MAX);
            out.push((i as u32, c.solver_body_a, c.solver_body_b, color, c.len));
        }
        out
    }

    /// The number of colliders per batch.
    pub fn num_colliders_per_batch(&self) -> u32 {
        self.num_colliders_per_batch
    }

    /// The number of *active* colliders per batch — i.e. how many of the
    /// `num_colliders_per_batch` capacity slots are currently in use. Bodies
    /// added via [`Self::append_bodies`] increase this up to the capacity.
    pub fn num_active_colliders(&self) -> u32 {
        self.num_active_colliders
    }

    /// The number of batches.
    pub fn num_batches(&self) -> u32 {
        self.num_batches
    }

    /// The number of solver iterations (max across all environments).
    pub fn num_solver_iterations(&self) -> u32 {
        self.num_solver_iterations
    }
}

/// Extracts a [`GpuColliderMaterial`] from a rapier collider: friction,
/// restitution and their `CoefficientCombineRule`s (stored as `rule as u32`).
pub(super) fn collider_material_from_rapier(
    co: &crate::rapier::geometry::Collider,
) -> GpuColliderMaterial {
    GpuColliderMaterial {
        friction: co.friction(),
        restitution: co.restitution(),
        friction_combine_rule: co.friction_combine_rule() as u32,
        restitution_combine_rule: co.restitution_combine_rule() as u32,
    }
}

pub(super) fn local_mprops_from_rapier(
    mprops: &crate::rapier::prelude::MassProperties,
) -> GpuLocalMassProperties {
    #[cfg(feature = "dim2")]
    {
        GpuLocalMassProperties {
            inv_mass: glamx::Vec2::splat(mprops.inv_mass),
            com: mprops.local_com,
            padding2: 0,
            inv_inertia: mprops.inv_principal_inertia,
        }
    }
    #[cfg(feature = "dim3")]
    {
        GpuLocalMassProperties {
            inertia_ref_frame: mprops.principal_inertia_local_frame,
            inv_principal_inertia: mprops.inv_principal_inertia,
            padding0: 0,
            inv_mass: glamx::Vec3::splat(mprops.inv_mass),
            padding1: 0,
            com: mprops.local_com,
            padding2: 0,
        }
    }
}

/// Computes the world-space mass properties of a body from its body-origin world
/// pose and body-local mass properties. Mirrors the GPU `update_mprops` shader
/// so the buffer is consistent the moment the simulation starts.
pub(super) fn world_mprops_from_local(
    pose: &Pose,
    local: &GpuLocalMassProperties,
) -> GpuWorldMassProperties {
    #[cfg(feature = "dim2")]
    {
        GpuWorldMassProperties {
            inv_inertia: local.inv_inertia,
            inv_mass: local.inv_mass,
            padding1: 0,
            com: *pose * local.com,
        }
    }
    #[cfg(feature = "dim3")]
    {
        // Build the world-space inverse inertia tensor: R * diag * R^T, with R
        // the rotation taking body space to the world principal-inertia frame.
        // Mirrors the GPU `update_mprops` shader so the buffer is consistent
        // before the first `update_mprops` dispatch.
        let world_principal_frame = pose.rotation * local.inertia_ref_frame;
        let rot_mat = glamx::Mat3::from_quat(world_principal_frame);
        let scaled = glamx::Mat3::from_cols(
            rot_mat.x_axis * local.inv_principal_inertia.x,
            rot_mat.y_axis * local.inv_principal_inertia.y,
            rot_mat.z_axis * local.inv_principal_inertia.z,
        );
        let inv_inertia_3 = scaled * rot_mat.transpose();
        let inv_inertia = glamx::Mat4::from_mat3(inv_inertia_3);
        GpuWorldMassProperties {
            inv_inertia,
            inv_mass: local.inv_mass,
            padding0: 0,
            com: *pose * local.com,
            padding1: 0,
        }
    }
}

/// GPU-resident rigid-body reset templates (see
/// [`RbdState::publish_reset_templates`]).
#[cfg(feature = "dim3")]
pub(super) struct ResetTemplatesBodies {
    poses: Tensor<Pose>,
    vels: Tensor<GpuVelocity>,
    mask: Tensor<u32>,
    kernel: crate::shaders::dynamics::GpuEnvResetBodies,
}

/// CPU snapshot of one (single-batch) physics template: body poses, velocities
/// and the multibody joint-space state, read off the GPU once so per-env resets
/// need no readback. See [`RbdState::snapshot`].
#[cfg(feature = "dim3")]
#[derive(Clone)]
pub struct RbdSnapshot {
    body_poses: Vec<Pose>,
    vels: Vec<GpuVelocity>,
    mb: crate::dynamics::GpuMultibodySnapshot,
}

#[cfg(feature = "dim3")]
impl RbdSnapshot {
    /// Debug/test accessor: body `body_id`'s snapshotted (dense per-env) pose.
    pub fn debug_body_pose(&self, body_id: usize) -> Pose {
        self.body_poses[body_id]
    }

    /// A copy with every floating-base multibody translated by `offset`: the
    /// affected links' `body_poses` plus the multibody workspace (root
    /// free-joint coords, local-to-parent, per-link local-to-world). Fixed
    /// bodies (ground, terrain) and velocities are untouched.
    pub fn translated(&self, offset: crate::math::Vector) -> RbdSnapshot {
        let mut out = self.clone();
        out.mb = self.mb.translated(offset);
        self.mb.for_each_link_rb_id(|rb_id| {
            if let Some(p) = out.body_poses.get_mut(rb_id as usize) {
                p.translation += offset;
            }
        });
        out
    }
}

#[cfg(feature = "dim3")]
impl RbdState {
    /// Reads this (template) physics state off the GPU into a CPU snapshot.
    /// Call it once per template at setup and pass the result to
    /// [`Self::reset_env_from_snapshot`] for readback-free per-env resets.
    pub async fn snapshot(&self, backend: &GpuBackend) -> RbdSnapshot {
        let nb = self.num_batches as usize;
        let mut all_poses: Vec<Pose> = bytemuck::zeroed_vec(self.body_poses.len() as usize);
        backend
            .slow_read_buffer(self.body_poses.buffer(), &mut all_poses)
            .await
            .unwrap();
        let mut all_vels: Vec<GpuVelocity> = bytemuck::zeroed_vec(self.vels.len() as usize);
        backend
            .slow_read_buffer(self.vels.buffer(), &mut all_vels)
            .await
            .unwrap();
        // The live buffers are batch-interleaved; a snapshot holds environment
        // 0's state as a dense per-env array.
        let bps = all_poses.len() / nb;
        let body_poses = (0..bps).map(|i| all_poses[i * nb]).collect();
        let vs = all_vels.len() / nb;
        let vels = (0..vs).map(|i| all_vels[i * nb]).collect();
        let mb = self.multibodies.snapshot(backend).await;
        RbdSnapshot {
            body_poses,
            vels,
            mb,
        }
    }

    /// Resets env `dst_env` from a CPU snapshot using `write_buffer` only.
    ///
    /// The per-body buffers are batch-interleaved, so this issues one strided
    /// write per body: fine for the documented slow path, but prefer
    /// [`Self::reset_envs_from_templates`] in reset loops.
    pub fn reset_env_from_snapshot(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &RbdSnapshot,
    ) {
        let nb = self.num_batches as u64;
        let bps = (self.body_poses.len() / nb) as usize;
        for (i, pose) in snap.body_poses[..bps].iter().enumerate() {
            backend
                .write_buffer(
                    self.body_poses.buffer_mut(),
                    i as u64 * nb + dst_env as u64,
                    core::slice::from_ref(pose),
                )
                .unwrap();
        }
        let vs = (self.vels.len() / nb) as usize;
        for (i, vel) in snap.vels[..vs].iter().enumerate() {
            backend
                .write_buffer(
                    self.vels.buffer_mut(),
                    i as u64 * nb + dst_env as u64,
                    core::slice::from_ref(vel),
                )
                .unwrap();
        }
        self.multibodies
            .reset_env_from_snapshot(backend, dst_env, &snap.mb);
    }

    /// [`Self::reset_env_from_snapshot`] with the robot rigidly translated by
    /// `offset` (world frame): the teleport primitive for terrain-curriculum
    /// spawn placement. Only floating-base multibody links move; fixed bodies
    /// keep their snapshot poses. Costs one single-env-sized snapshot clone per
    /// call, so prefer [`Self::reset_envs_from_templates`] in reset loops.
    pub fn reset_env_from_snapshot_offset(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &RbdSnapshot,
        offset: crate::math::Vector,
    ) {
        let moved = snap.translated(offset);
        self.reset_env_from_snapshot(backend, dst_env, &moved);
    }

    /// Uploads the reset templates once (rigid-body poses and velocities here,
    /// the multibody blobs via
    /// [`GpuMultibodySet::publish_reset_templates`][mb]), enabling the batched
    /// [`Self::reset_envs_from_templates`].
    ///
    /// [mb]: crate::dynamics::GpuMultibodySet::publish_reset_templates
    pub fn publish_reset_templates(&mut self, backend: &GpuBackend, snaps: &[&RbdSnapshot]) {
        use crate::shaders::dynamics::GpuEnvResetBodies;
        use khal::Shader as _;
        if snaps.is_empty() {
            return;
        }
        let nb = self.num_batches as usize;
        let bps = self.body_poses.len() as usize / nb;
        let vs = self.vels.len() as usize / nb;
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        let mut poses = Vec::with_capacity(snaps.len() * bps);
        let mut vels = Vec::with_capacity(snaps.len() * vs);
        for snap in snaps {
            poses.extend_from_slice(&snap.body_poses[..bps]);
            vels.extend_from_slice(&snap.vels[..vs]);
        }
        // The bodies a teleport offset applies to: free-multibody links, per
        // `RbdSnapshot::translated`. Ground and terrain stay put.
        let mut mask = vec![0u32; bps];
        snaps[0].mb.for_each_link_rb_id(|rb_id| {
            if let Some(m) = mask.get_mut(rb_id as usize) {
                *m = 1;
            }
        });

        /// `#[derive(Shader)]` supplies `from_backend` for the embedded entry.
        #[derive(khal::Shader)]
        struct EnvResetBodiesShader {
            kernel: GpuEnvResetBodies,
        }
        let shader = EnvResetBodiesShader::from_backend(backend).unwrap();
        self.reset_templates_bodies = Some(ResetTemplatesBodies {
            poses: Tensor::vector(backend, &poses, storage).unwrap(),
            vels: Tensor::vector(backend, &vels, storage).unwrap(),
            mask: Tensor::vector(backend, &mask, storage).unwrap(),
            kernel: shader.kernel,
        });
        let mb_snaps: Vec<&crate::dynamics::GpuMultibodySnapshot> =
            snaps.iter().map(|s| &s.mb).collect();
        self.multibodies.publish_reset_templates(backend, &mb_snaps);
    }

    /// Batched reset: restores every `(dst_env, template)` in `resets` from the
    /// GPU-resident templates, translated by the matching `offsets` entry, with
    /// `dof_vels` (`dofs_per_batch` floats per reset, a randomized reset draw or
    /// zeros) written into the generalized-velocity section.
    ///
    /// One compact upload, two dispatches and one submit for the whole batch,
    /// replacing the per-env snapshot clone, staging uploads and strided
    /// velocity writes. [`Self::publish_reset_templates`] must have run first.
    pub fn reset_envs_from_templates(
        &mut self,
        backend: &GpuBackend,
        resets: &[(u32, u32)],
        offsets: &[crate::math::Vector],
        dof_vels: &[f32],
    ) {
        use crate::shaders::dynamics::{EnvResetBodiesParams, EnvResetRecord};
        use glamx::Vec4;
        use khal::backend::Encoder as _;
        let n = resets.len() as u32;
        if n == 0 {
            return;
        }
        let nb = self.num_batches;
        let bps = self.body_poses.len() as u32 / nb;
        let vs = self.vels.len() as u32 / nb;
        let meta: Vec<EnvResetRecord> = resets
            .iter()
            .map(|&(env, template)| EnvResetRecord { env, template })
            .collect();
        let offs: Vec<Vec4> = offsets
            .iter()
            .map(|o| Vec4::new(o.x, o.y, o.z, 0.0))
            .collect();
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let t_meta = Tensor::vector(backend, &meta, storage).unwrap();
        let t_offs = Tensor::vector(backend, &offs, storage).unwrap();
        let params = Tensor::scalar(
            backend,
            EnvResetBodiesParams {
                bodies_per_env: bps,
                vels_per_env: vs,
                num_resets: n,
                num_batches: nb,
            },
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();

        let tpl = self
            .reset_templates_bodies
            .take()
            .expect("publish_reset_templates must run first");
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("[RBD] env-reset-bodies", None);
            tpl.kernel
                .call(
                    &mut pass,
                    [bps.max(vs), n, 1],
                    &tpl.poses,
                    &tpl.vels,
                    &tpl.mask,
                    &t_meta,
                    &t_offs,
                    &mut self.body_poses,
                    &mut self.vels,
                    &params,
                )
                .unwrap();
        }
        self.multibodies
            .encode_reset_envs_batch(backend, &mut enc, &meta, &offs, dof_vels);
        backend.submit(enc).unwrap();
        self.reset_templates_bodies = Some(tpl);
    }
}

/// Everything that shapes the GPU work recorded by one
/// [`RbdPipeline::step`](crate::pipeline::RbdPipeline::step) run
/// `steps_per_frame` times: buffer identities (any reallocation bumps a
/// generation), the live counts and capacities that size dispatches or
/// host-side loops, and the solver path taken. A cached compute graph of the
/// step is valid exactly as long as this key is unchanged.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RbdGraphKey {
    /// Bumped by every (re)allocation or capacity change of the state's buffers.
    pub generation: u64,
    /// Bumped by every (re)allocation of the LBVH buffers.
    pub lbvh_generation: u64,
    /// `(active colliders per batch, batches)` the radix sort was last told about.
    pub lbvh_n_sort_active: Option<(u32, u32)>,
    /// Number of steps recorded per frame.
    pub steps_per_frame: u32,
    /// Number of simulation environments.
    pub num_batches: u32,
    /// Number of active colliders over all environments.
    pub num_active_colliders: u32,
    /// Collider slots per environment.
    pub num_colliders_per_batch: u32,
    /// Maximum number of graph colors the solver sweeps.
    pub max_colors: u32,
    /// Number of per-color uniform buffers.
    pub num_color_uniforms: usize,
    /// Capacity of the contact buffer.
    pub contacts_capacity: u32,
    /// Capacity of the collision-pair buffer.
    pub collision_pairs_capacity: u32,
    /// Solver iterations per step.
    pub num_solver_iterations: u32,
    /// Whether rigid-body contacts are skipped by the solver.
    pub rb_contacts_inert: bool,
    /// Number of graph colors of the impulse joints.
    pub joints_num_colors: u32,
    /// Whether there are no impulse joints.
    pub joints_empty: bool,
    /// Capacity of the multibody contact-constraint slabs.
    #[cfg(feature = "dim3")]
    pub mb_contact_constraints_capacity: u32,
    /// Whether there are no multibodies.
    #[cfg(feature = "dim3")]
    pub multibodies_empty: bool,
    /// Number of graph colors of the multibody impulse joints.
    #[cfg(feature = "dim3")]
    pub mb_imp_joint_num_colors: u32,
    /// Whether the fused colored-sweep kernels are used.
    pub fused_color_sweeps: bool,
}

impl RbdState {
    /// The [`RbdGraphKey`] of this state for `steps_per_frame` steps per frame.
    pub fn graph_key(&self, steps_per_frame: u32) -> RbdGraphKey {
        RbdGraphKey {
            generation: self.graph_generation,
            lbvh_generation: self.lbvh.generation,
            lbvh_n_sort_active: self.lbvh.n_sort_active(),
            steps_per_frame,
            num_batches: self.num_batches(),
            num_active_colliders: self.num_active_colliders(),
            num_colliders_per_batch: self.num_colliders_per_batch(),
            max_colors: self.max_colors,
            num_color_uniforms: self.color_uniforms.len(),
            contacts_capacity: self.contacts_capacity_cpu,
            collision_pairs_capacity: self.collision_pairs_capacity_cpu,
            num_solver_iterations: self.num_solver_iterations,
            rb_contacts_inert: self.rb_contacts_inert(),
            joints_num_colors: self.joints.num_colors(),
            joints_empty: self.joints.is_empty(),
            // NOTE: not `mb_cons_demand_cpu`: it is a readback mirror that
            // changes as the scene evolves; the resize it may trigger bumps
            // `graph_generation`, which is what the graph depends on.
            #[cfg(feature = "dim3")]
            mb_contact_constraints_capacity: self.mb_contact_constraints_capacity(),
            #[cfg(feature = "dim3")]
            multibodies_empty: self.multibodies.is_empty(),
            #[cfg(feature = "dim3")]
            mb_imp_joint_num_colors: self.multibodies.mb_imp_joint_num_colors(),
            fused_color_sweeps: crate::pipeline::RbdPipeline::fused_color_sweeps(self),
        }
    }
}
