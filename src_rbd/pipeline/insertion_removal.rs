//! Incremental construction of [`RbdState`]: empty allocation, append and removal of bodies.

use crate::broad_phase::{LbvhState, PfmSortState};
use crate::dynamics::GpuImpulseJointSet;
#[cfg(feature = "dim3")]
use crate::dynamics::GpuMultibodySet;
use crate::math::{Pose, Vector};
use crate::queries::GpuColliderMaterial;
use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::{
    LocalMassProperties as GpuLocalMassProperties, RbdSimParams, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use crate::shaders::shapes::Shape;
use crate::shaders::utils::BatchIndices;
use crate::utils::PrefixSumWorkspace;
use std::ops::Range;

use super::rbd_state::*;
#[cfg(feature = "dim3")]
use crate::rapier::dynamics::{MultibodyJointSet, RigidBodySet};
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuReadback};
use vortx::tensor::Tensor;
use {
    crate::math::Point, crate::rapier::dynamics::ImpulseJointSet, crate::shapes::shape_from_parry,
    std::collections::HashMap,
};

impl RbdState {
    /// Creates an empty rbd state preallocated for `num_batches` batches of up
    /// to `capacity_per_batch` colliders each.
    ///
    /// No body is active initially (every slot is reserved padding). Bodies are
    /// added later with [`Self::append_bodies`]; the simulation parameters use
    /// neutral defaults ([`RbdSimParams::default`]) — adjust them via the usual
    /// `sim_params` buffer if needed. Joints and multibodies start empty.
    ///
    /// NOTE: the per-batch collider count is fixed at `capacity_per_batch`;
    /// inserting more than that into a single batch panics (the batched buffers
    /// are not grown/restrided by the incremental path).
    pub fn empty(backend: &GpuBackend, capacities: RbdCapacities) -> Self {
        let num_batches = capacities.batches;
        let capacity_per_batch = capacities.body_capacity;
        let collisions_capacity = capacities.collisions_capacity;
        let num_colliders_per_batch = capacity_per_batch;
        let num_bodies_total = (capacity_per_batch * num_batches) as usize;

        let num_solver_iterations = 4u32;
        let mut base_sim_params = RbdSimParams::default();
        base_sim_params.dt /= num_solver_iterations as f32;

        // Inactive (padding) slots use empty collision groups so the broad-phase
        // never matches them with anything.
        let none_groups = crate::rapier::geometry::InteractionGroups::new(
            crate::rapier::geometry::Group::NONE,
            crate::rapier::geometry::Group::NONE,
            crate::rapier::geometry::InteractionTestMode::And,
        );
        let dummy_shape = Shape::cuboid(Vector::splat(0.5));
        let all_poses = vec![Pose::default(); num_bodies_total];
        let all_collider_local_poses = vec![Pose::IDENTITY; num_bodies_total];
        let all_local_mprops = vec![GpuLocalMassProperties::default(); num_bodies_total];
        let all_mprops = vec![GpuWorldMassProperties::default(); num_bodies_total];
        let all_shapes = vec![dummy_shape; num_bodies_total];
        let all_collision_groups = vec![none_groups; num_bodies_total];
        let all_vels = vec![GpuVelocity::default(); num_bodies_total];

        let all_body_group: Vec<u32> = (0..num_bodies_total as u32).collect();

        let all_collider_parent: Vec<u32> = (0..num_bodies_total as u32).collect();
        let all_pair_filter: Vec<[u32; 2]> = (0..num_bodies_total as u32)
            .map(|i| [i / num_batches, 0u32])
            .collect();

        // Empty joints / multibodies, one (empty) environment per batch.
        let empty_joints = ImpulseJointSet::new();
        let empty_body_ids: HashMap<crate::rapier::dynamics::RigidBodyHandle, u32> = HashMap::new();
        let joint_env_refs: Vec<_> = (0..num_batches as usize)
            .map(|_| (&empty_joints, &empty_body_ids))
            .collect();
        let joints = GpuImpulseJointSet::from_rapier_filtered(backend, &joint_env_refs, &[], &[]);

        #[cfg(feature = "dim3")]
        let mut multibodies = {
            let empty_mb = MultibodyJointSet::new();
            let empty_bodies = RigidBodySet::new();
            let mb_refs: Vec<_> = (0..num_batches as usize)
                .map(|_| (&empty_mb, &empty_body_ids, &empty_bodies))
                .collect();
            let mut mb = GpuMultibodySet::from_rapier(
                backend,
                &mb_refs,
                capacity_per_batch,
                capacities.mb_contact_constraints_capacity,
            );
            mb.set_constraint_softness(backend, &base_sim_params);
            mb
        };

        // Shared shape vertex/index buffers: dummy data to avoid empty bindings.
        let vertex_buffers =
            Tensor::vector(backend, [Point::ZERO.into()], BufferUsages::STORAGE).unwrap();
        let index_buffers = Tensor::vector(backend, [0u32, 0, 0], BufferUsages::STORAGE).unwrap();
        let body_group = Tensor::vector(backend, &all_body_group, BufferUsages::STORAGE).unwrap();

        // Per-body buffers carry COPY_DST | COPY_SRC so `append_bodies` /
        // `remove_bodies` can write / relocate slots in place.
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        let shapes = Tensor::vector(backend, &all_shapes, rw).unwrap();
        let collider_local_poses = Tensor::vector(backend, &all_collider_local_poses, rw).unwrap();
        let collider_parent = Tensor::vector(backend, &all_collider_parent, rw).unwrap();
        let collision_groups = Tensor::vector(backend, &all_collision_groups, rw).unwrap();
        let pair_filter = Tensor::vector(backend, &all_pair_filter, rw).unwrap();
        // Padding colliders are inert; default material keeps the buffer sized.
        let all_collider_materials = vec![GpuColliderMaterial::default(); num_bodies_total];
        let collider_materials = Tensor::vector(backend, &all_collider_materials, rw).unwrap();

        let pairs_capacity = collisions_capacity * num_batches;
        let contacts_capacity = pairs_capacity * 2;
        let collision_pairs = Tensor::vector_uninit(backend, pairs_capacity, storage).unwrap();
        let collision_pairs_len = Tensor::vector(
            backend,
            &[0u32],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        #[cfg(feature = "dim3")]
        let resize_readback = GpuReadback::new(backend, 4).unwrap();
        #[cfg(not(feature = "dim3"))]
        let resize_readback = GpuReadback::new(backend, 3).unwrap();
        let collision_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let contacts = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        #[cfg(feature = "dim3")]
        let mb_contact_index = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let contacts_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let mb_sweep_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs = Tensor::vector_uninit(backend, pairs_capacity, storage).unwrap();
        let pfm_sort = PfmSortState::new(backend, pairs_capacity);
        let pfm_pairs_len = Tensor::vector(
            backend,
            &[0u32],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let old_constraints = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let old_constraint_builders =
            Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let new_constraints = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let new_constraint_builders =
            Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let constraints_colors =
            Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let old_constraints_colors =
            Tensor::vector(backend, vec![0u32; contacts_capacity as usize], storage).unwrap();
        let colored = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let constraints_rands = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let color_buckets_stride = capacities.solver_colors + 3;
        let color_buckets =
            Tensor::vector_uninit(backend, color_buckets_stride * num_batches, storage).unwrap();
        let contact_plan =
            Tensor::scalar(backend, ContactPlan::default(), storage | BufferUsages::UNIFORM)
                .unwrap();
        let color_sorted_ids = Tensor::vector_uninit(backend, contacts_capacity, storage).unwrap();
        let old_constraints_counts = Tensor::vector(
            backend,
            vec![0u32; (num_colliders_per_batch * num_batches) as usize],
            storage,
        )
        .unwrap();
        let new_constraints_counts =
            Tensor::vector_uninit(backend, num_colliders_per_batch * num_batches, storage).unwrap();
        let old_body_constraint_ids =
            Tensor::vector_uninit(backend, contacts_capacity * 2, storage).unwrap();
        let new_body_constraint_ids =
            Tensor::vector_uninit(backend, contacts_capacity * 2, storage).unwrap();

        let lbvh_usages = if crate::VALIDATE_LBVH_TOPOLOGY {
            BufferUsages::STORAGE | BufferUsages::COPY_SRC
        } else {
            BufferUsages::STORAGE
        };

        let contacts_capacity_cpu = contacts_capacity;
        let collision_pairs_capacity_cpu = pairs_capacity;
        #[allow(unused_mut)] // Only mutated with the dim3 (multibody) feature.
        let mut bi = BatchIndices {
            num_batches,
            colliders_batch_capacity: num_colliders_per_batch,
            // No body is active initially; bodies are added later via `append_bodies`.
            colliders_len: 0,
            bodies_len: 0,
            collision_pairs_capacity: collision_pairs_capacity_cpu,
            contacts_capacity: contacts_capacity_cpu,
            impulse_joints_len: joints.num_active_joints(),
            solver_color_buckets_stride: color_buckets_stride,
            ..Default::default()
        };
        #[cfg(feature = "dim3")]
        multibodies.fill_batch_indices(&mut bi);
        let batch_indices = Tensor::scalar(
            backend,
            bi,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();

        Self {
            capacities,
            num_batches,
            num_colliders_per_batch,
            num_solver_iterations,
            sim_params: Tensor::scalar(
                backend,
                base_sim_params,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
            .unwrap(),
            sim_params_cpu: base_sim_params,
            vels: Tensor::vector(backend, &all_vels, rw).unwrap(),
            #[cfg(feature = "dim3")]
            reset_templates_bodies: None,
            solver_vels: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_inc: Tensor::vector(backend, &all_vels, storage).unwrap(),
            joints,
            #[cfg(feature = "dim3")]
            multibodies,
            gravity: Self::gravity_tensor(backend, [0.0, -9.81, 0.0]),
            body_group,
            local_mprops: Tensor::vector(backend, &all_local_mprops, rw).unwrap(),
            mprops: Tensor::vector(backend, &all_mprops, rw).unwrap(),
            body_poses: Tensor::vector(backend, &all_poses, rw).unwrap(),
            solver_body_poses: Tensor::vector(backend, &all_poses, rw).unwrap(),
            collider_world_poses: Tensor::vector(backend, &all_poses, rw).unwrap(),
            vertex_buffers,
            index_buffers,
            shapes,
            collider_local_poses,
            collider_parent,
            collision_groups,
            pair_filter,
            collider_materials,
            collision_pairs,
            collision_pairs_len,
            resize_readback,
            collision_pairs_indirect,
            contacts_capacity_cpu,
            collision_pairs_capacity_cpu,
            collision_pairs_len_cpu: 0,
            #[cfg(feature = "dim3")]
            mb_cons_demand_cpu: 0,
            batch_indices,
            contacts,
            contacts_indirect,
            contact_plan,
            pfm_sort,
            #[cfg(feature = "dim3")]
            mb_contact_index,
            mb_sweep_indirect,
            pfm_pairs,
            pfm_pairs_len,
            pfm_pairs_indirect,
            old_constraints,
            old_constraint_builders,
            old_constraints_counts,
            new_constraints,
            new_constraint_builders,
            new_constraints_counts,
            constraints_colors,
            old_constraints_colors,
            colored,
            constraints_rands,
            color_buckets,
            color_sorted_ids,
            curr_color: Tensor::scalar(
                backend,
                0u32,
                BufferUsages::STORAGE
                    | BufferUsages::UNIFORM
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            uncolored: Tensor::scalar(
                backend,
                0,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            uncolored_staging: Tensor::scalar(
                backend,
                0,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )
            .unwrap(),
            old_body_constraint_ids,
            new_body_constraint_ids,
            color_uniforms: Vec::new(),
            prefix_sum_workspace: PrefixSumWorkspace::default(),
            bucket_prefix_workspace: PrefixSumWorkspace::default(),
            lbvh: LbvhState::with_usages(backend, lbvh_usages),
            max_colors: capacities.solver_colors,
            rb_contacts_inert: false,
            num_active_colliders: 0,
            num_active_bodies: 0,
        }
    }

    /// Appends rigid-bodies (each given as a `(RigidBody, Collider)` pair) into
    /// *every* simulation batch, returning the per-batch local slot range the
    /// new bodies occupy (identical across batches). The same topology is added
    /// to all environments, preserving the equal-topology invariant; per-batch
    /// dynamic divergence can be applied afterwards via the usual buffers.
    ///
    /// Only primitive (vertex-less) colliders are currently supported; mesh
    /// colliders would require growing the shared vertex/index buffers.
    ///
    /// # Panics
    /// Panics if any batch would exceed `num_colliders_per_batch`.
    pub fn append_bodies(
        &mut self,
        backend: &GpuBackend,
        bodies: &[(
            crate::rapier::dynamics::RigidBody,
            crate::rapier::geometry::Collider,
        )],
    ) -> Result<Range<u32>, GpuBackendError> {
        let cap = self.num_colliders_per_batch as usize;
        let active = self.num_active_colliders as usize;
        assert!(
            active + bodies.len() <= cap,
            "rbd batch capacity ({}) exceeded",
            cap
        );

        let mut poses = Vec::with_capacity(bodies.len());
        let mut collider_local_poses = Vec::with_capacity(bodies.len());
        let mut local_mprops = Vec::with_capacity(bodies.len());
        let mut mprops = Vec::with_capacity(bodies.len());
        let mut shapes = Vec::with_capacity(bodies.len());
        let mut collision_groups = Vec::with_capacity(bodies.len());
        let mut materials = Vec::with_capacity(bodies.len());
        let mut vels = Vec::with_capacity(bodies.len());

        for (rb, co) in bodies {
            let body_pose = *rb.position();
            let collider_local_pose = co.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
            let is_dynamic = rb.is_dynamic();
            if is_dynamic {
                // A free dynamic body: its contacts need the rigid-body
                // constraint pipeline. Never set back on removal; a stale
                // `false` only costs performance.
                self.rb_contacts_inert = false;
            }
            let (local, world) = if is_dynamic {
                // A standalone rigid-body carries no collider mass: rapier only
                // folds a collider's mass into the body once the collider is
                // attached in a world (which `append_bodies` bodies are not).
                // Combine the body's own (additional) mass with the collider's,
                // expressed in the body frame, so the appended body falls under
                // gravity exactly like a `from_rapier` body.
                let m = rb.mass_properties();
                let combined =
                    m.local_mprops + co.mass_properties().transform_by(&collider_local_pose);
                let local = local_mprops_from_rapier(&combined);
                let world = world_mprops_from_local(&body_pose, &local);
                (local, world)
            } else {
                let mut local = GpuLocalMassProperties::default();
                let mut world = GpuWorldMassProperties {
                    com: body_pose.translation,
                    ..Default::default()
                };
                local.inv_mass = Vector::ZERO;
                world.inv_mass = Vector::ZERO;
                #[cfg(feature = "dim3")]
                {
                    local.inv_principal_inertia = glamx::Vec3::ZERO;
                    world.inv_inertia = glamx::Mat4::ZERO;
                }
                #[cfg(feature = "dim2")]
                {
                    local.inv_inertia = 0.0;
                    world.inv_inertia = 0.0;
                }
                (local, world)
            };

            let mut shape_buffers = crate::shapes::ShapeBuffers::default();
            let shape =
                shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape");
            assert!(
                shape_buffers.vertices.is_empty(),
                "RbdState::append_bodies currently supports primitive (vertex-less) colliders only."
            );

            poses.push(body_pose);
            collider_local_poses.push(collider_local_pose);
            local_mprops.push(local);
            mprops.push(world);
            shapes.push(shape);
            collision_groups.push(co.collision_groups());
            materials.push(collider_material_from_rapier(co));
            vels.push(GpuVelocity::new(
                rb.linvel(),
                #[cfg(feature = "dim2")]
                rb.angvel(),
                #[cfg(feature = "dim3")]
                rb.angvel(),
            ));
        }

        // The incremental path attaches exactly one collider per body, so a
        // body's collider slot equals its body slot: `collider_parent` is the
        let nb = self.num_batches as usize;
        let parents: Vec<u32> =
            ((active * nb) as u32..((active + bodies.len()) * nb) as u32).collect();
        let pair_filters: Vec<[u32; 2]> = (0..bodies.len() * nb)
            .map(|i| [(active + i / nb) as u32, 0u32])
            .collect();

        fn replicate<T: Copy>(v: &[T], nb: usize) -> Vec<T> {
            let mut out = Vec::with_capacity(v.len() * nb);
            for &x in v {
                for _ in 0..nb {
                    out.push(x);
                }
            }
            out
        }
        let base = (active * nb) as u64;
        backend.write_buffer(self.body_poses.buffer_mut(), base, &replicate(&poses, nb))?;
        backend.write_buffer(
            self.solver_body_poses.buffer_mut(),
            base,
            &replicate(&poses, nb),
        )?;
        backend.write_buffer(
            self.collider_world_poses.buffer_mut(),
            base,
            &replicate(&poses, nb),
        )?;
        backend.write_buffer(
            self.collider_local_poses.buffer_mut(),
            base,
            &replicate(&collider_local_poses, nb),
        )?;
        backend.write_buffer(self.collider_parent.buffer_mut(), base, &parents)?;
        backend.write_buffer(self.pair_filter.buffer_mut(), base, &pair_filters)?;
        backend.write_buffer(
            self.local_mprops.buffer_mut(),
            base,
            &replicate(&local_mprops, nb),
        )?;
        backend.write_buffer(self.mprops.buffer_mut(), base, &replicate(&mprops, nb))?;
        backend.write_buffer(self.shapes.buffer_mut(), base, &replicate(&shapes, nb))?;
        backend.write_buffer(
            self.collision_groups.buffer_mut(),
            base,
            &replicate(&collision_groups, nb),
        )?;
        backend.write_buffer(
            self.collider_materials.buffer_mut(),
            base,
            &replicate(&materials, nb),
        )?;
        backend.write_buffer(self.vels.buffer_mut(), base, &replicate(&vels, nb))?;

        let new_active = (active + bodies.len()) as u32;
        self.num_active_colliders = new_active;
        // One collider per body on this path → body count tracks collider count.
        self.num_active_bodies = new_active;
        self.rebuild_batch_indices(backend);

        Ok(active as u32..new_active)
    }

    /// Removes the bodies at the given per-batch local slot indices using a
    /// swap-remove (the last active body is moved into the freed slot). The
    /// removal is applied identically to every batch, preserving the
    /// equal-topology invariant. Returns the list of `(from, to)` local-slot
    /// relocations performed so callers can patch their slot bookkeeping.
    pub fn remove_bodies(
        &mut self,
        backend: &GpuBackend,
        local_indices: &[u32],
    ) -> Result<Vec<(u32, u32)>, GpuBackendError> {
        let mut remaps = Vec::new();

        // Process local slots in descending order so removing one doesn't
        // disturb the not-yet-removed (lower) slots.
        let mut locals: Vec<usize> = local_indices.iter().map(|&l| l as usize).collect();
        locals.sort_unstable_by(|a, b| b.cmp(a));
        locals.dedup();

        let none_groups = crate::rapier::geometry::InteractionGroups::new(
            crate::rapier::geometry::Group::NONE,
            crate::rapier::geometry::Group::NONE,
            crate::rapier::geometry::InteractionTestMode::And,
        );

        let nb = self.num_batches as usize;
        let staging_usages =
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut enc = backend.begin_encoding();
        let mut any_copy = false;
        let mut staging_pose = backend.uninit_buffer::<Pose>(nb, staging_usages)?;
        let mut staging_local_mprops =
            backend.uninit_buffer::<GpuLocalMassProperties>(nb, staging_usages)?;
        let mut staging_mprops =
            backend.uninit_buffer::<GpuWorldMassProperties>(nb, staging_usages)?;
        let mut staging_vels = backend.uninit_buffer::<GpuVelocity>(nb, staging_usages)?;
        let mut staging_shapes = backend.uninit_buffer::<Shape>(nb, staging_usages)?;
        let mut staging_groups = backend
            .uninit_buffer::<crate::rapier::geometry::InteractionGroups>(nb, staging_usages)?;
        let mut staging_materials =
            backend.uninit_buffer::<GpuColliderMaterial>(nb, staging_usages)?;
        let mut neutralize: Vec<usize> = Vec::new();

        for local in locals {
            let active = self.num_active_colliders as usize;
            if active == 0 || local >= active {
                continue;
            }
            let last = active - 1;

            if local != last {
                let hole_global = local * nb;
                let last_global = last * nb;
                macro_rules! relocate {
                    ($t:expr, $staging:expr) => {{
                        enc.copy_buffer_to_buffer($t.buffer(), last_global, &mut $staging, 0, nb)?;
                        enc.copy_buffer_to_buffer(&$staging, 0, $t.buffer_mut(), hole_global, nb)?;
                    }};
                }
                any_copy = true;
                relocate!(self.body_poses, staging_pose);
                relocate!(self.solver_body_poses, staging_pose);
                relocate!(self.collider_world_poses, staging_pose);
                relocate!(self.collider_local_poses, staging_pose);
                relocate!(self.local_mprops, staging_local_mprops);
                relocate!(self.mprops, staging_mprops);
                relocate!(self.vels, staging_vels);
                relocate!(self.shapes, staging_shapes);
                relocate!(self.collision_groups, staging_groups);
                relocate!(self.collider_materials, staging_materials);
            }

            neutralize.push(last);

            if local != last {
                remaps.push((last as u32, local as u32));
            }
            self.num_active_colliders = (active - 1) as u32;
        }

        if any_copy {
            backend.submit(enc)?;
        }

        // The now-topmost slots become inactive padding: neutralize them so
        // they never participate in collisions even if a kernel scans up to
        // the per-batch capacity. Done after the relocation submit so the
        // copies read the pre-neutralisation data.
        for slot in neutralize {
            let base = (slot * nb) as u64;
            backend.write_buffer(
                self.collision_groups.buffer_mut(),
                base,
                &vec![none_groups; nb],
            )?;
            backend.write_buffer(
                self.local_mprops.buffer_mut(),
                base,
                &vec![GpuLocalMassProperties::default(); nb],
            )?;
            backend.write_buffer(
                self.mprops.buffer_mut(),
                base,
                &vec![GpuWorldMassProperties::default(); nb],
            )?;
        }

        // `collider_parent` is the identity mapping on the incremental (one
        // collider per body) path and stays identity under swap-remove, so it
        // needs no relocation; only the active body count tracks the colliders.
        // The same holds for `pair_filter` (`[identity, 0]` on this path).
        self.num_active_bodies = self.num_active_colliders;
        self.rebuild_batch_indices(backend);
        Ok(remaps)
    }
}
