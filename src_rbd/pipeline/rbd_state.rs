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
    /// CPU mirror of `sim_params`, so runtime setters can patch one field and
    /// re-upload without rebuilding the whole state.
    pub(super) all_sim_params: Vec<RbdSimParams>,
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
    /// Per-batch live collision-pair counts (length `num_batches`).
    pub(super) collision_pairs_len: Tensor<u32>,
    /// Single-element scratch holding the max of `collision_pairs_len` across all
    /// batches, computed on the GPU (only used when `num_batches > 1`).
    pub(super) collision_pairs_len_max: Tensor<u32>,
    /// Cosine of the maximum angle between two contact normals for their
    /// manifolds to be clustered together (see `gpu_reduce_contacts`).
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
    pub(super) prefix_sum_workspace: PrefixSumWorkspace,
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
        for c in self.color_uniforms.len() as u32..n {
            self.color_uniforms
                .push(Tensor::scalar(backend, c, BufferUsages::UNIFORM).unwrap());
        }
    }

    /// Returns the configured max color count.
    pub fn max_colors(&self) -> u32 {
        self.max_colors
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

    /// Live collision-pair count (batch 0) most recently harvested by the
    /// non-blocking readback in [`RbdPipeline::auto_resize_buffers`](crate::pipeline::RbdPipeline::auto_resize_buffers). Lags the GPU by a
    /// frame or two; `0` until the first readback completes.
    pub fn collision_pairs_len(&self) -> u32 {
        self.collision_pairs_len_cpu
    }

    /// GPU buffer of the broad-phase collision pairs found this step.
    pub fn collision_pairs(&self) -> &Tensor<CollisionPair> {
        &self.collision_pairs
    }

    /// GPU buffer holding the per-batch collision-pair counts. Unlike
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
        let mut params = self.all_sim_params.clone();
        for p in &mut params {
            p.contact_merge_cos = cos;
        }
        let _ = backend.write_buffer(self.sim_params.buffer_mut(), 0, &params);
        self.all_sim_params = params;
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

    /// Sets the per-DoF dry joint friction (MJCF `frictionloss`, N·m), in the
    /// env-major `dofs_per_batch * num_batches` layout described on
    /// [`GpuMultibodySet::set_dof_frictionloss`](crate::dynamics::GpuMultibodySet::set_dof_frictionloss).
    ///
    /// The first non-zero call grows the joint-constraint bank, so the shared
    /// `BatchIndices` uniform is rebuilt here.
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

    /// The contact manifold buffer (post narrow-phase + body resolution).
    /// Per-batch number of manifolds currently in [`Self::contacts`].
    pub fn contacts_len(&self) -> &Tensor<u32> {
        &self.contacts_len
    }

    /// GPU buffer holding the contact manifolds.
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
        let mut body_poses = bytemuck::zeroed_vec(self.body_poses.len() as usize);
        backend
            .slow_read_buffer(self.body_poses.buffer(), &mut body_poses)
            .await
            .unwrap();
        let mut vels = bytemuck::zeroed_vec(self.vels.len() as usize);
        backend
            .slow_read_buffer(self.vels.buffer(), &mut vels)
            .await
            .unwrap();
        let mb = self.multibodies.snapshot(backend).await;
        RbdSnapshot {
            body_poses,
            vels,
            mb,
        }
    }

    /// Resets env `dst_env` from a CPU snapshot using `write_buffer` only, with
    /// no GPU to CPU readback. This is what removes the handful of per-reset
    /// sync stalls that otherwise dominate reset cost on the WebGPU backend.
    pub fn reset_env_from_snapshot(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &RbdSnapshot,
    ) {
        let nb = self.num_batches as u64;
        let bps = (self.body_poses.len() / nb) as usize;
        backend
            .write_buffer(
                self.body_poses.buffer_mut(),
                dst_env as u64 * bps as u64,
                &snap.body_poses[..bps],
            )
            .unwrap();
        let vs = (self.vels.len() / nb) as usize;
        backend
            .write_buffer(
                self.vels.buffer_mut(),
                dst_env as u64 * vs as u64,
                &snap.vels[..vs],
            )
            .unwrap();
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
        use glamx::{UVec4, Vec4};
        use khal::backend::Encoder as _;
        let n = resets.len() as u32;
        if n == 0 {
            return;
        }
        let nb = self.num_batches;
        let bps = self.body_poses.len() as u32 / nb;
        let vs = self.vels.len() as u32 / nb;
        let meta: Vec<UVec4> = resets
            .iter()
            .map(|&(env, t)| UVec4::new(env, t, 0, 0))
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
            UVec4::new(bps, vs, n, 0),
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
