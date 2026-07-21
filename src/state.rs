use crate::rapier::data::{Coarena, Index};
use crate::rapier::prelude::{
    Collider, ColliderHandle, GenericJoint, ImpulseJointHandle, MultibodyJointHandle, PhysicsWorld,
    RigidBody, RigidBodyHandle,
};
use crate::rbd::dynamics::RbdSimParams;
use crate::rbd::pipeline::{RbdCapacities, RbdResizePolicy, RbdState, RunStats};
use khal::backend::{GpuBackend, GpuBackendError};

/// Handle referencing a rigid-body managed by a [`NexusState`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusRbdHandle(Index);

/// Initial capacities used when allocating the GPU-resident physics states.
///
/// Groups the per-subsystem capacities, each owned by its own crate
/// ([`RbdCapacities`] in nexus-rbd) and forwarded to that subsystem when its
/// scene is first created.
#[derive(Copy, Clone, Debug, Default)]
pub struct NexusCapacities {
    /// Rigid-body subsystem capacities.
    pub rbd: RbdCapacities,
}

impl NexusCapacities {
    pub fn rbd_batches(mut self, num_batches: u32) -> Self {
        self.rbd.batches = num_batches;
        self
    }

    pub fn rbd_bodies(mut self, capacity: u32) -> Self {
        self.rbd.body_capacity = capacity;
        self
    }

    pub fn rbd_collisions(mut self, capacity: u32) -> Self {
        self.rbd.collisions_capacity = capacity;
        self
    }

    pub fn rbd_resize_policy(mut self, resize_policy: RbdResizePolicy) -> Self {
        self.rbd.collisions_resize_policy = resize_policy;
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GpuRigidBodyRef {
    pub gpu_id: u32,
}

impl Default for GpuRigidBodyRef {
    fn default() -> Self {
        Self { gpu_id: u32::MAX }
    }
}

/// Entity counts for the current scene, surfaced in the viewer UI. Rigid-body
/// counts are summed across all environments (batches).
#[derive(Clone, Copy, Default, Debug)]
pub struct NexusCounts {
    pub num_environments: usize,
    pub rigid_bodies: usize,
    pub colliders: usize,
    pub impulse_joints: usize,
    pub multibodies: usize,
    pub multibody_dofs: usize,
    pub collision_pairs: usize,
    pub collision_pairs_capacity: usize,
}

/// High-level, GPU-resident state of a physics simulation.
///
/// The rigid-body sub-state is lazily allocated the first time content is
/// added. The `rbd2gpu` maps translate the stable public handles into the
/// (unstable) GPU buffer slots, which shift around as bodies are inserted and
/// removed.
pub struct NexusState {
    /// Rigid-body sub-state, allocated on the first [`Self::add_rigid_bodies`].
    pub rbd: Option<RbdState>,

    pub run_stats: RunStats,

    /// Handle → GPU-slot map, one [`Coarena`] per simulation environment
    /// (batch).
    pub rbd2gpu: Vec<Coarena<GpuRigidBodyRef>>,

    // Initial capacities used to allocate the states lazily.
    capacities: NexusCapacities,

    /// One rapier world per simulation environment (batch). Environment 0
    /// always exists; the non-`*_in` insert helpers target it. Batched demos
    /// add more via [`Self::add_environment`].
    rbd_envs: Vec<PhysicsWorld>,
    /// Per-environment simulation parameters (same length as `rbd_envs`).
    rbd_sim_params: Vec<RbdSimParams>,
    /// Set whenever the rapier worlds change; consumed by [`Self::finalize`] to
    /// decide whether the GPU [`RbdState`] needs rebuilding.
    rbd_dirty: bool,
    /// Number of rigid-body solver steps advanced per [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call.
    pub rbd_steps_per_frame: u32,
    /// Per-environment GPU collider-slot reservation. When > 0, the GPU
    /// [`RbdState`] is built with this many slots (rather than exactly the
    /// current body count), leaving room for [`Self::add_rigid_body`] to append
    /// bodies in place — without rebuilding the whole scene. Set via
    /// [`Self::reserve_rigid_bodies`].
    rbd_reserve_per_env: usize,
    // TODO: keep track of whether there is any non-fixed rigid-body (if there isn’t, we can
    //       skip the rbd pipeline entirely).
}

impl Default for NexusState {
    fn default() -> Self {
        Self::new(NexusCapacities::default())
    }
}

impl NexusState {
    /// Creates an empty state. The GPU sub-states are allocated lazily (sized
    /// from `capacities`) the first time matching content is added.
    pub fn new(capacities: NexusCapacities) -> Self {
        Self {
            rbd: None,
            run_stats: RunStats::default(),
            rbd_envs: vec![PhysicsWorld::default()],
            rbd_sim_params: vec![RbdSimParams::tgs_soft()],
            rbd_dirty: false,
            rbd_steps_per_frame: 1,
            rbd_reserve_per_env: 0,
            rbd2gpu: vec![Coarena::new()],
            capacities,
        }
    }

    /// Reserves additional capacity in the handle maps to avoid reallocations
    /// when a known number of bodies is about to be inserted.
    pub fn reserve(&mut self, additional: NexusCapacities) {
        for env in &mut self.rbd2gpu {
            env.reserve(additional.rbd.body_capacity as usize);
        }
        // TODO: resize the GPU buffers too.
    }

    /// Sets the rigid-body multibody gravity vector, e.g. `[0.0, 0.0, -9.81]`
    /// for a Z-up scene. No-op until the rigid-body state is built, so call it
    /// after [`Self::finalize`]. Free (non-multibody) bodies keep the solver's
    /// fixed gravity.
    #[cfg(all(feature = "dim3", feature = "rbd"))]
    pub fn set_rbd_gravity(&mut self, backend: &GpuBackend, gravity: [f32; 3]) {
        if let Some(rbd) = self.rbd.as_mut() {
            rbd.set_gravity(backend, gravity);
        }
    }

    // ── Rigid-body runtime settings ─────────────────────────────────────

    /// Sets the number of rigid-body solver steps advanced per
    /// [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call (default 1). Acts as a simulation-speed control.
    pub fn set_rbd_steps_per_frame(&mut self, steps: u32) {
        self.rbd_steps_per_frame = steps.max(1);
    }

    /// Overrides the per-environment collision-pair capacity used when the
    /// GPU rigid-body state is (re)allocated at `finalize`. The default (4096)
    /// is sized for one busy scene, not thousands of small batched envs —
    /// pair-keyed workspaces scale as `capacity x num_envs x sizeof(manifold)`,
    /// which at 2048 envs binds ~9 GiB unless this is lowered.
    pub fn set_rbd_collisions_capacity(&mut self, capacity: u32) {
        self.capacities.rbd.collisions_capacity = capacity.max(1);
    }

    /// Number of rigid-body solver steps per [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call.
    pub fn rbd_steps_per_frame(&self) -> u32 {
        self.rbd_steps_per_frame
    }

    /// Current entity counts (rigid bodies, colliders, joints, multibody DOFs)
    /// for display in the UI. Rigid-body counts are summed across all
    /// environments.
    pub fn counts(&self) -> NexusCounts {
        let mut c = NexusCounts {
            num_environments: self.rbd_envs.len(),
            ..Default::default()
        };
        for world in &self.rbd_envs {
            c.rigid_bodies += world.bodies.len();
            c.colliders += world.colliders.len();
            c.impulse_joints += world.impulse_joints.len();
            for mb in world.multibody_joints.multibodies() {
                c.multibodies += 1;
                c.multibody_dofs += mb.ndofs();
            }
        }
        if let Some(rbd) = self.rbd.as_ref() {
            c.collision_pairs = rbd.collision_pairs_len() as usize;
            c.collision_pairs_capacity = rbd.collision_pairs_capacity() as usize;
        }
        c
    }

    /// Adds a new (empty) simulation environment (batch) and returns its index.
    ///
    /// Environment 0 always exists; batched demos call this once per extra
    /// environment, then insert into it with the `*_in` helpers. Every
    /// environment is solved independently on the GPU and rendered at its own
    /// poses.
    pub fn add_environment(&mut self) -> usize {
        self.rbd_envs.push(PhysicsWorld::default());
        self.rbd_sim_params.push(RbdSimParams::tgs_soft());
        self.rbd2gpu.push(Coarena::new());
        self.rbd_dirty = true;
        self.rbd_envs.len() - 1
    }

    /// Number of simulation environments (batches).
    pub fn num_environments(&self) -> usize {
        self.rbd_envs.len()
    }

    /// Overwrite environment `env`'s solver parameters (default `tgs_soft`).
    /// Marks the rbd state dirty so [`Self::finalize`] rebuilds with them.
    /// Mainly for tests that need to match an external engine's
    /// `IntegrationParameters` exactly (e.g. `num_solver_iterations = 1`).
    pub fn set_rbd_sim_params(&mut self, env: usize, params: RbdSimParams) {
        self.rbd_sim_params[env] = params;
        self.rbd_dirty = true;
    }

    /// Read-only access to environment `env`'s rapier world. Does NOT mark the
    /// rbd state dirty (unlike [`Self::rbd_world_mut`]), so it's safe to use
    /// after [`Self::finalize`] — e.g. to clone the finalized world for an
    /// external reference simulation without forcing a GPU rebuild.
    pub fn rbd_world(&self, env: usize) -> &PhysicsWorld {
        &self.rbd_envs[env]
    }

    /// Mutable access to environment `env`'s rapier world, e.g. for loaders
    /// (URDF) that insert directly into the rapier sets. Marks the rbd state
    /// dirty so [`Self::finalize`] rebuilds the GPU buffers.
    pub fn rbd_world_mut(&mut self, env: usize) -> &mut PhysicsWorld {
        self.rbd_dirty = true;
        &mut self.rbd_envs[env]
    }

    pub fn insert_rigid_body(&mut self, body: RigidBody, collider: Collider) -> RigidBodyHandle {
        self.insert_rigid_body_in(0, body, collider)
    }

    /// Inserts a body + collider into environment `env`.
    pub fn insert_rigid_body_in(
        &mut self,
        env: usize,
        body: RigidBody,
        collider: Collider,
    ) -> RigidBodyHandle {
        let (handle, _) = self.rbd_envs[env].insert(body, collider);
        self.rbd2gpu[env].insert(handle.0, GpuRigidBodyRef { gpu_id: u32::MAX });
        self.rbd_dirty = true;
        handle
    }

    /// Reserves `per_env` GPU collider slots per environment so that bodies can
    /// later be added with [`Self::add_rigid_body`] *in place* — appended to the
    /// existing GPU buffers instead of rebuilding the whole scene.
    ///
    /// Call this before the first [`Self::finalize`]/[`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate). Intended
    /// for single-environment scenes (the appended body data is shared across
    /// batches). `per_env` is a hard cap: once it's full, `add_rigid_body` falls
    /// back to a full rebuild.
    pub fn reserve_rigid_bodies(&mut self, per_env: usize) {
        self.rbd_reserve_per_env = per_env;
    }

    /// Adds a body + collider to environment 0, appending it directly to the GPU
    /// [`RbdState`] **without rebuilding the scene** — provided the state already
    /// exists and has spare capacity (see [`Self::reserve_rigid_bodies`]). If
    /// there is no GPU state yet, or the reservation is full, it falls back to a
    /// normal insert (a full rebuild on the next `finalize`).
    ///
    /// Only primitive (vertex-less) colliders are supported on the fast path.
    pub fn add_rigid_body(
        &mut self,
        backend: &GpuBackend,
        body: RigidBody,
        collider: Collider,
    ) -> Result<RigidBodyHandle, GpuBackendError> {
        let handles = self.add_rigid_bodies(backend, [(body, collider)])?;
        Ok(handles[0])
    }

    /// Adds several body + collider pairs to environment 0 in a single in-place
    /// GPU append — the batched form of [`Self::add_rigid_body`]. One
    /// `append_bodies` call (one buffer upload + one `rebuild_batch_indices`)
    /// covers the whole batch, so it's much cheaper than calling `add_rigid_body`
    /// in a loop. Returns the handles in input order.
    ///
    /// Like the single-body version it appends without rebuilding the scene when
    /// the GPU state exists and has room for the *entire* batch; otherwise it
    /// falls back to a full rebuild on the next `finalize`. Only primitive
    /// (vertex-less) colliders are supported on the fast path.
    pub fn add_rigid_bodies(
        &mut self,
        backend: &GpuBackend,
        bodies: impl IntoIterator<Item = (RigidBody, Collider)>,
    ) -> Result<Vec<RigidBodyHandle>, GpuBackendError> {
        // Keep copies for the GPU append before the rapier world consumes them.
        let mut gpu_pairs: Vec<(RigidBody, Collider)> = Vec::new();
        let mut handles: Vec<RigidBodyHandle> = Vec::new();
        for (body, collider) in bodies {
            gpu_pairs.push((body.clone(), collider.clone()));
            let (handle, _) = self.rbd_envs[0].insert(body, collider);
            handles.push(handle);
        }
        if handles.is_empty() {
            return Ok(handles);
        }

        let appended = match self.rbd.as_mut() {
            Some(rbd)
                if (rbd.num_active_colliders() as usize) + gpu_pairs.len()
                    <= rbd.num_colliders_per_batch() as usize =>
            {
                let range = rbd.append_bodies(backend, &gpu_pairs)?;
                // Single environment: the per-batch local slot is the gpu_id.
                for (i, &handle) in handles.iter().enumerate() {
                    self.rbd2gpu[0].insert(
                        handle.0,
                        GpuRigidBodyRef {
                            gpu_id: range.start + i as u32,
                        },
                    );
                }
                true
            }
            _ => false,
        };

        if !appended {
            // No GPU state yet, or not enough room for the whole batch: fall back
            // to a full rebuild on the next `finalize`.
            for &handle in handles.iter() {
                self.rbd2gpu[0].insert(handle.0, GpuRigidBodyRef { gpu_id: u32::MAX });
            }
            self.rbd_dirty = true;
        }
        Ok(handles)
    }

    /// Inserts a rigid-body without any attached collider (e.g. a joint anchor).
    pub fn insert_body(&mut self, body: RigidBody) -> RigidBodyHandle {
        self.insert_body_in(0, body)
    }

    // TODO: remove this. Inserting a collider should insert into all envs.
    //       (though we should also have a variant that allows specifying different
    //       shapes per env).
    /// Inserts a collider-less rigid-body into environment `env`.
    pub fn insert_body_in(&mut self, env: usize, body: RigidBody) -> RigidBodyHandle {
        let handle = self.rbd_envs[env].insert_body(body);
        self.rbd2gpu[env].insert(handle.0, GpuRigidBodyRef { gpu_id: u32::MAX });
        self.rbd_dirty = true;
        handle
    }

    // TODO: remove this. Inserting a collider should insert into all envs.
    //       (though we should also have a variant that allows specifying different
    //       shapes per env).
    /// Attaches a collider to an existing body (or inserts a parent-less one) in
    /// environment `env`.
    pub fn insert_collider_in(
        &mut self,
        env: usize,
        collider: Collider,
        parent: Option<RigidBodyHandle>,
    ) -> ColliderHandle {
        self.rbd_dirty = true;
        self.rbd_envs[env].insert_collider(collider, parent)
    }

    /// Inserts an impulse joint into environment 0.
    pub fn insert_impulse_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.insert_impulse_joint_in(0, body1, body2, joint)
    }

    /// Inserts an impulse joint between two bodies of environment `env`.
    pub fn insert_impulse_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.rbd_dirty = true;
        self.rbd_envs[env].insert_impulse_joint(body1, body2, joint)
    }

    /// Inserts a multibody joint into environment 0.
    ///
    /// Returns `None` if the joint would create an invalid kinematic chain
    /// (e.g. a cycle).
    pub fn insert_multibody_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> Option<MultibodyJointHandle> {
        self.insert_multibody_joint_in(0, body1, body2, joint)
    }

    /// Inserts a multibody joint between two bodies of environment `env`.
    pub fn insert_multibody_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> Option<MultibodyJointHandle> {
        self.rbd_dirty = true;
        self.rbd_envs[env].insert_multibody_joint(body1, body2, joint)
    }

    /// Number of GPU batches (== number of environments) once finalized.
    pub fn rbd_num_batches(&self) -> u32 {
        self.rbd.as_ref().map(|r| r.num_batches()).unwrap_or(0)
    }

    /// Sets a multibody joint motor's target velocity on the GPU state (used by
    /// the URDF demo for per-frame actuation). No-op until the rbd state exists.
    #[cfg(feature = "dim3")]
    pub fn set_multibody_motor_velocity(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        link_id: u32,
        axis: crate::rapier::dynamics::JointAxis,
        target_vel: f32,
    ) -> Result<(), GpuBackendError> {
        if let Some(rbd) = self.rbd.as_mut() {
            rbd.multibodies_mut()
                .set_motor_velocity(backend, batch, link_id, axis, target_vel)?;
        }
        Ok(())
    }

    pub async fn finalize(&mut self, backend: &GpuBackend) -> Result<(), GpuBackendError> {
        if self.rbd_dirty {
            // Finalize each body's mass properties so additional (`<inertial>`)
            // mass combined with its colliders is reflected in `local_mprops`.
            // rapier only does this during its own step (`update_world_mass_properties`),
            // which we never run — so bodies built MJCF-style (density-0 colliders
            // + additional mass) would otherwise read as zero-mass, making the
            // multibody mass matrix singular and killing gravity. Idempotent for
            // bodies whose mass already comes from dense colliders.
            for world in &mut self.rbd_envs {
                let handles: Vec<RigidBodyHandle> = world.bodies.iter().map(|(h, _)| h).collect();
                for h in handles {
                    let body = &mut world.bodies[h];
                    body.recompute_mass_properties_from_colliders(&world.colliders);
                }
            }
        }
        if self.rbd_dirty {
            // Full (re)build of the GPU rbd state from the rapier worlds. With a
            // reservation (`reserve_rigid_bodies`) the buffers are sized for
            // spare slots so later `add_rigid_body` calls can append in place;
            // otherwise the state is sized exactly to the current body count.
            let rbd_state = if self.rbd_reserve_per_env > 0 {
                let num_envs = self.rbd_envs.len() as u32;
                let max_count = self
                    .rbd_envs
                    .iter()
                    .map(|w| w.colliders.len())
                    .max()
                    .unwrap_or(0);
                let capacity = self.rbd_reserve_per_env.max(max_count) as u32;
                // Per-batch body/batch counts come from the scene; the collision
                // capacity comes from the configured capacities.
                let caps = RbdCapacities {
                    batches: num_envs,
                    body_capacity: capacity, // FIXME: should this be set to match what’s in `self.capacities.rbd`?
                    ..self.capacities.rbd
                };
                let mut st = RbdState::empty(backend, caps);
                // Append environment 0's bodies in collider-iteration order, so
                // the per-batch slot index matches the `from_rapier` layout.
                let world = &self.rbd_envs[0];
                let mut bodies = Vec::new();
                for (_, collider) in world.colliders.iter() {
                    if let Some(bh) = collider.parent() {
                        bodies.push((world.bodies[bh].clone(), collider.clone()));
                    }
                }
                if !bodies.is_empty() {
                    st.append_bodies(backend, &bodies)?;
                }
                st
            } else {
                let environments: Vec<_> = self
                    .rbd_envs
                    .iter()
                    .zip(self.rbd_sim_params.iter())
                    .map(|(w, sp)| {
                        (
                            &w.bodies,
                            &w.colliders,
                            &w.impulse_joints,
                            &w.multibody_joints,
                            sp,
                        )
                    })
                    .collect();
                RbdState::from_rapier(backend, &environments, self.capacities.rbd)
            };

            // Rebuild the per-environment handle → GPU-slot maps. A handle's
            // `gpu_id` is its BODY slot (which indexes the body-keyed buffers
            // such as `body_poses`), NOT a collider slot — a body may own
            // several colliders. Body slots are assigned in the same order
            // `from_rapier` uses: the first time each parent body is seen while
            // iterating colliders (a parentless collider consumes a synthetic
            // body slot, matching `from_rapier`). Bodies are laid out env-major
            // with stride `num_colliders_per_batch`.
            let stride = rbd_state.num_colliders_per_batch();
            for (env_idx, world) in self.rbd_envs.iter().enumerate() {
                let mut body_slot: std::collections::HashMap<_, u32> =
                    std::collections::HashMap::new();
                let mut next_slot = 0u32;
                // Not a plain loop counter: parentless colliders consume a slot
                // without a map entry, and (on dim3) the multibody-link loop
                // below continues the same counter.
                #[allow(clippy::explicit_counter_loop)]
                for (_, collider) in world.colliders.iter() {
                    let Some(body_handle) = collider.parent() else {
                        // Parentless collider → synthetic body slot (no handle
                        // to map, but it still consumes a slot in `from_rapier`).
                        next_slot += 1;
                        continue;
                    };
                    let slot = *body_slot.entry(body_handle).or_insert_with(|| {
                        let s = next_slot;
                        next_slot += 1;
                        s
                    });
                    self.rbd2gpu[env_idx].insert(
                        body_handle.0,
                        GpuRigidBodyRef {
                            gpu_id: env_idx as u32 * stride + slot,
                        },
                    );
                }

                // Mirror `from_rapier`: append a body slot for every multibody
                // link that no collider mapped (collider-less links), in the same
                // multibody-link order.
                #[cfg(feature = "dim3")]
                for mb in world.multibody_joints.multibodies() {
                    for link in mb.links() {
                        let body_handle = link.rigid_body_handle();
                        if body_slot.contains_key(&body_handle) {
                            continue;
                        }
                        let slot = next_slot;
                        next_slot += 1;
                        body_slot.insert(body_handle, slot);
                        self.rbd2gpu[env_idx].insert(
                            body_handle.0,
                            GpuRigidBodyRef {
                                gpu_id: env_idx as u32 * stride + slot,
                            },
                        );
                    }
                }
            }
            self.rbd = Some(rbd_state);
            self.rbd_dirty = false;
        }
        Ok(())
    }

    // /// Removes the given rigid-bodies from the simulation.
    // pub fn remove_rigid_bodies(
    //     &mut self,
    //     backend: &GpuBackend,
    //     bodies: &[NexusRbdHandle],
    // ) -> Result<(), GpuBackendError> {
    //     // 1. Resolve the rbd GPU slots of the bodies to remove.
    //     let gpu_ids: Vec<u32> = bodies
    //         .iter()
    //         .filter_map(|h| self.rbd2gpu.get(h.0).copied())
    //         .collect();
    //
    //     // 2. Swap-remove them from the rbd GPU buffers. `remove_bodies` returns
    //     //    the slot relocations it performed (`(from, to)`) so we can patch the
    //     //    handle map.
    //     let remaps = match self.rbd.as_mut() {
    //         Some(rbd) => rbd.remove_bodies(backend, &gpu_ids)?,
    //         None => Vec::new(),
    //     };
    //
    //     // 3. Drop the removed handles, then patch the relocated slots.
    //     for h in bodies {
    //         self.rbd2gpu.remove(h.0);
    //     }
    //     for (from, to) in remaps {
    //         for (_, slot) in self.rbd2gpu.iter_mut() {
    //             if *slot == from {
    //                 *slot = to;
    //             }
    //         }
    //     }
    //
    //     Ok(())
    // }
}
