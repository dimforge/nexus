//! GPU-resident rigid-body state ([`RbdState`]): buffer definitions, accessors,
//! run statistics and capacity/resize policies.
use crate::broad_phase::LbvhState;
use crate::dynamics::GpuImpulseJointSet;
#[cfg(feature = "dim3")]
use crate::dynamics::GpuMultibodySet;
use crate::math::Pose;
use crate::queries::{GpuColliderMaterial, GpuIndexedContact};
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{CollisionPair, NarrowPhasePfmPair};
use crate::shaders::dynamics::{
    LocalMassProperties as GpuLocalMassProperties, RbdSimParams, TwoBodyConstraint,
    TwoBodyConstraintBuilder, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use crate::shaders::shapes::Shape;
use crate::shaders::utils::BatchIndices;
use crate::utils::PrefixSumWorkspace;

use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuReadback};
use std::time::Duration;
use vortx::shaders::linalg::Shape as TensorShape;
use vortx::tensor::Tensor;

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
    /// Per-body world-origin pose (matches rapier's `RigidBody::position`). This
    /// is the canonical pose stored between steps and the input to per-step
    /// mass-properties update and multibody FK. The substep loop does NOT
    /// touch this — see [`Self::solver_body_poses`].
    pub(super) body_poses: Tensor<Pose>,
    /// Per-body COM-centered pose (rapier's `SolverPose`). Equals
    /// `body_poses[i].prepend_translation(local_mprops[i].com)`. Seeded from
    /// `body_poses` at step start, mutated by the solver substep loop, and
    /// converted back to `body_poses` by `finalize` at step end.
    pub(super) solver_body_poses: Tensor<Pose>,
    pub(super) local_mprops: Tensor<GpuLocalMassProperties>,
    pub(super) mprops: Tensor<GpuWorldMassProperties>,
    pub(super) vels: Tensor<GpuVelocity>,
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
    /// Per-collider broad-phase pair-filter key, mirroring rapier's built-in
    /// narrow-phase filters at pair-emission time so filtered pairs never
    /// reach the (much more expensive) narrow phase:
    /// - `[0]`: env-local parent body slot — two colliders of the same body
    ///   never collide;
    /// - `[1]`: multibody ordinal + 1 when the parent belongs to a multibody
    ///   with self-contacts disabled (rapier's
    ///   `Multibody::self_contacts_enabled`, MJCF `DISABLE_SELF_CONTACTS`),
    ///   `0` otherwise — equal nonzero keys never collide.
    pub(super) pair_filter: Tensor<[u32; 2]>,
    /// Per-collider friction / restitution coefficients (+ combine rules),
    pub(super) collider_materials: Tensor<GpuColliderMaterial>,
    pub(super) collision_pairs: Tensor<CollisionPair>,
    /// Per-batch live collision-pair counts (length `num_batches`).
    pub(super) collision_pairs_len: Tensor<u32>,
    /// Single-element scratch holding the max of `collision_pairs_len` across all
    /// batches, computed on the GPU (only used when `num_batches > 1`).
    pub(super) collision_pairs_len_max: Tensor<u32>,
    /// `num_batches` as a uniform, the scan length for the max reduction.
    pub(super) num_batches_uniform: Tensor<TensorShape>,
    /// Non-blocking readback of `[max collision_pairs_len, uncolored]` used by
    /// [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers)
    /// to grow buffers without stalling.
    pub(super) resize_readback: GpuReadback<u32>,
    pub(super) collision_pairs_indirect: Tensor<[u32; 3]>,
    /// CPU-side mirrors of the dynamic batch capacities. The capacity values
    /// live in the [`BatchIndices`] uniform; these mirrors let
    /// [`Self::rebuild_batch_indices`] re-emit it whenever a buffer grows.
    pub(super) contacts_per_batch_cpu: u32,
    pub(super) collision_pairs_per_batch_cpu: u32,
    /// Most recently read live collision-pair count — the max across all batches,
    /// harvested by the non-blocking readback in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers).
    /// Surfaced in the viewer UI; lags the GPU by a frame or two like the resize.
    pub(super) collision_pairs_len_cpu: u32,
    /// Single uniform aggregating every per-batch capacity and packed-buffer
    /// section offset consumed by the compute kernels (multibody and RBD
    /// sides). Rebuilt by [`Self::rebuild_batch_indices`] whenever any of its
    /// constituent caps changes (e.g. when the contacts buffer grows).
    pub(super) batch_indices: Tensor<BatchIndices>,
    pub(super) pfm_pairs: Tensor<NarrowPhasePfmPair>,
    pub(super) pfm_pairs_len: Tensor<u32>,
    pub(super) pfm_pairs_indirect: Tensor<[u32; 3]>,
    pub(super) contacts: Tensor<GpuIndexedContact>,
    pub(super) contacts_len: Tensor<u32>,
    pub(super) contacts_indirect: Tensor<[u32; 3]>,
    /// Workgroup grid for the per-multibody contact-constraint dispatches:
    /// `[multibodies_batch_capacity, num_batches, 1]`, or all-zero when no
    /// batch has any contact this step (written by the narrow-phase reduce).
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
    /// Previous frame's constraint colors (aligned with `old_constraints`),
    /// used to seed the topo-gc coloring. Swapped with `constraints_colors`
    /// at the end of each step.
    pub(super) old_constraints_colors: Tensor<u32>,
    pub(super) colored: Tensor<u32>,
    pub(super) constraints_rands: Tensor<u32>,
    /// Per-batch per-color constraint counts (stride `max_colors + 3`), see
    /// the `gpu_color_buckets_*` kernels.
    pub(super) color_bucket_counts: Tensor<u32>,
    /// Per-batch per-color exclusive prefix sums over the counts: color `c`
    /// owns `color_sorted_ids[starts[c]..starts[c + 1]]`.
    pub(super) color_bucket_starts: Tensor<u32>,
    /// Scatter cursors (seeded from the starts each step).
    pub(super) color_bucket_cursors: Tensor<u32>,
    /// Constraint indices bucket-sorted by color (contacts layout).
    pub(super) color_sorted_ids: Tensor<u32>,
    pub(super) curr_color: Tensor<u32>,
    /// Pre-built per-color-index uniforms: `color_uniforms[c]` holds the
    /// constant `c`. Bound by every colored sweep (contacts, impulse joints,
    /// multibody impulse joints) instead of a GPU-incremented cursor, removing
    /// two 1-thread dispatches per color per sweep. Grown on demand by
    /// [`Self::ensure_color_uniforms`].
    pub(super) color_uniforms: Vec<Tensor<u32>>,
    pub(super) uncolored: Tensor<u32>,
    pub(super) uncolored_staging: Tensor<u32>,
    pub(super) lbvh: LbvhState,
    pub(super) joints: GpuImpulseJointSet,
    #[cfg(feature = "dim3")]
    pub(super) multibodies: GpuMultibodySet,
    /// Per-body "graph group" id, used by graph coloring to treat all bodies of
    /// the same multibody as a single node. For free bodies, `body_group[i] = i`;
    /// bodies of a multibody all share the group id of the root link, so two
    /// contacts touching different bodies of the same multibody can never be
    /// assigned the same color.
    pub(super) body_group: Tensor<u32>,
    pub(super) prefix_sum_workspace: PrefixSumWorkspace,
    /// Maximum number of constraint colors the solver will iterate.
    pub(super) max_colors: u32,
    /// `true` when every body is either non-dynamic or multibody-controlled
    /// (its rb-side `inv_mass` is zero) — i.e. every rigid-body CONTACT
    /// constraint is provably a no-op, so the contact-constraint pipeline
    /// (build, coloring, warmstart, colored sweeps, integration) is skipped
    /// entirely. Multibody contacts are unaffected (they are solved by the
    /// multibody solver, which reads the manifolds directly), and so are
    /// impulse joints (the joint solver runs regardless).
    ///
    /// Computed by `from_rapier`; conservatively cleared whenever
    /// [`Self::append_bodies`] appends a dynamic body (never set back by
    /// removals — stale `false` only costs performance, never correctness).
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
}

impl RbdState {
    /// Re-upload the shared `BatchIndices` uniform after any of its
    /// constituent per-batch capacities has changed (e.g. after the contacts
    /// buffer grows in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers), or after multibody
    /// impulse-joint capacities are updated via
    /// `GpuMultibodySet::set_impulse_joints`). Call whenever a cap edit
    /// happens that any kernel reads via its `batch_ids` uniform.
    pub(super) fn rebuild_batch_indices(&mut self, backend: &GpuBackend) {
        #[allow(unused_mut)] // Only mutated with the dim3 (multibody) feature.
        let mut bi = BatchIndices {
            num_batches: self.num_batches,
            colliders_batch_capacity: self.num_colliders_per_batch,
            colliders_len: self.num_active_colliders,
            bodies_len: self.num_active_bodies,
            collision_pairs_batch_capacity: self.collision_pairs_per_batch_cpu,
            contacts_batch_capacity: self.contacts_per_batch_cpu,
            impulse_joints_batch_capacity: self.joints.joints_per_batch(),
            impulse_joints_len: self.joints.num_active_joints(),
            solver_color_buckets_stride: self.max_colors + 3,
            ..Default::default()
        };
        #[cfg(feature = "dim3")]
        self.multibodies.fill_batch_indices(&mut bi);
        // The buffer carries COPY_DST — update it in place instead of
        // reallocating (which would also invalidate cached bind groups).
        backend
            .write_buffer(self.batch_indices.buffer_mut(), 0, &[bi])
            .unwrap();
    }

    /// Shared per-batch index uniform — see `Self::rebuild_batch_indices`.
    pub fn batch_indices(&self) -> &Tensor<BatchIndices> {
        &self.batch_indices
    }

    /// Sets the maximum number of constraint colors used by the per-step
    /// graph coloring + Gauss-Seidel solver loop. Lower values cap solver
    /// time at the cost of dropping over-budget constraints.
    pub fn set_max_colors(&mut self, max_colors: u32) {
        self.max_colors = max_colors.max(1);
    }

    /// Grows [`Self::color_uniforms`] so indices `0..n` are available. Each
    /// entry is a tiny immutable uniform holding its own index; existing
    /// entries are never reallocated.
    pub(super) fn ensure_color_uniforms(&mut self, backend: &GpuBackend, n: u32) {
        for c in self.color_uniforms.len() as u32..n {
            self.color_uniforms
                .push(Tensor::scalar(backend, c, BufferUsages::UNIFORM).unwrap());
        }
    }

    /// Returns the configured max color count.
    pub fn max_colors(&self) -> u32 {
        self.max_colors
    }

    /// `true` when every rigid-body contact constraint is provably a no-op
    /// (robot-only scenes) — see the field docs.
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

    /// Live collision-pair count (batch 0) most recently harvested by the
    /// non-blocking readback in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers). Lags the GPU by a
    /// frame or two; `0` until the first readback completes.
    pub fn collision_pairs_len(&self) -> u32 {
        self.collision_pairs_len_cpu
    }

    /// The max number a collision pairs the state can currently store.
    pub fn collision_pairs_capacity(&self) -> u32 {
        self.collision_pairs.capacity() as u32
    }

    /// Uploads a new gravity vector for the multibody solver, e.g.
    /// `[0.0, 0.0, -9.81]` for a Z-up scene. Affects multibody links; free
    /// (non-multibody) rigid-bodies use a fixed gravity baked into the solver
    /// shader.
    #[cfg(feature = "dim3")]
    pub fn set_gravity(&mut self, backend: &GpuBackend, gravity: [f32; 3]) {
        self.multibodies.set_gravity(backend, gravity);
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

    /// Immutable access to the multibody set (e.g. to read back `dof_state`).
    #[cfg(feature = "dim3")]
    pub fn multibodies(&self) -> &crate::dynamics::GpuMultibodySet {
        &self.multibodies
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

    /// The contact manifold buffer (post narrow-phase + body resolution).
    pub fn contacts(&self) -> &Tensor<GpuIndexedContact> {
        &self.contacts
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
    pub fn debug_constraint_colors(&self, backend: &GpuBackend) -> Vec<(u32, u32, u32, u32, u32)> {
        let cons: Vec<TwoBodyConstraint> =
            futures::executor::block_on(backend.slow_read_vec(self.new_constraints.buffer()))
                .unwrap_or_default();
        let colors: Vec<u32> =
            futures::executor::block_on(backend.slow_read_vec(self.constraints_colors.buffer()))
                .unwrap_or_default();
        let mut out = Vec::new();
        for (i, c) in cons.iter().enumerate() {
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
