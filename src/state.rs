use crate::mpm::pipeline::{MpmCapacities, MpmState};
use crate::mpm::solver::{BoundaryCondition, Particle, SimulationParams};
use crate::rapier::data::{Arena, Coarena, Index};
use crate::rapier::prelude::{
    Collider, ColliderHandle, GenericJoint, ImpulseJointHandle, MultibodyJointHandle, PhysicsWorld,
    RigidBody, RigidBodyHandle,
};
use crate::rbd::dynamics::{
    RbdSimParams,
    body::{BodyCoupling, RapierBodyCouplingEntry},
};
use crate::rbd::pipeline::{RbdCapacities, RbdResizePolicy, RbdState, RunStats};
use khal::backend::{GpuBackend, GpuBackendError};

/// Handle referencing a rigid-body managed by a [`NexusState`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusRbdHandle(Index);

/// Handle referencing a *chunk* of MPM particles managed by a [`NexusState`].
///
/// Particles are addressed by chunk rather than individually (a per-particle
/// handle map would be prohibitive at MPM scale). A chunk is mutable: particles
/// can be appended to it ([`NexusState::extend_chunk`]) or removed from it
/// ([`NexusState::remove_particles_from_chunk`] / [`NexusState::remove_chunk`]).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusParticleChunk(Index);

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum RbdCoupling {
    None,
    MpmOneWay(BoundaryCondition),
    MpmTwoWay(BoundaryCondition),
}

/// Initial capacities used when allocating the GPU-resident physics states.
#[derive(Copy, Clone, Debug, Default)]
pub struct NexusCapacities {
    /// Rigid-body solver capacities.
    pub rbd: RbdCapacities,
    /// MPM solver capacities.
    pub mpm: MpmCapacities,
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

    pub fn mpm_grid_size(mut self, num_chunks: u32) -> Self {
        self.mpm.grid_size = num_chunks;
        self
    }

    pub fn rbd_resize_policy(mut self, resize_policy: RbdResizePolicy) -> Self {
        self.rbd.collisions_resize_policy = resize_policy;
        self
    }

    pub fn mpm_particles(mut self, capacity: u32) -> Self {
        self.mpm.particles_capacity = capacity;
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GpuRigidBodyRef {
    pub coupling: RbdCoupling,
    pub gpu_id: u32,
}

impl Default for GpuRigidBodyRef {
    fn default() -> Self {
        Self {
            coupling: RbdCoupling::None,
            gpu_id: u32::MAX,
        }
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
    pub particles: usize,
}

/// High-level, GPU-resident state of a multiphysics simulation.
///
/// Each sub-state (`rbd`/`mpm`) is lazily allocated the first time content
/// of the corresponding kind is added. The `*2gpu` maps translate the stable
/// public handles into the (unstable) GPU buffer slots, which shift around as
/// bodies/particles are inserted and removed.
pub struct NexusState {
    /// Rigid-body sub-state, allocated on the first [`Self::add_rigid_bodies`].
    pub rbd: Option<RbdState>,
    /// MPM sub-state, allocated on the first [`Self::add_particles`] (or the
    /// first coupled rigid-body insertion).
    pub mpm: Option<MpmState>,

    pub run_stats: RunStats,

    /// Handle → GPU-slot map, one [`Coarena`] per simulation environment
    /// (batch).
    pub rbd2gpu: Vec<Coarena<GpuRigidBodyRef>>,

    /// Live particle count per MPM chunk (the arena key is the public
    /// [`NexusParticleChunk`] handle).
    mpm_chunks: Arena<usize>,
    /// Owning chunk for each GPU particle slot, kept in sync under the
    /// swap-removal performed by [`Self::remove_chunk`] /
    /// [`Self::remove_particles_from_chunk`].
    slot2chunk: Vec<Index>,
    /// MPM simulation params / grid cell width requested before the MPM
    /// sub-state is lazily created.
    mpm_params: Option<SimulationParams>,
    mpm_cell_width: f32,
    /// Number of MPM substeps run per [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call.
    pub mpm_substeps: u32,
    /// Desired CPIC rigid-coupling flag, kept here so it survives until the MPM
    /// sub-state is lazily created (and is what [`Self::mpm_use_cpic`] reports
    /// meanwhile).
    mpm_use_cpic: bool,
    /// Set when particles or MPM-coupled bodies change; consumed by
    /// [`Self::finalize`] to rebuild the MPM↔rapier coupling.
    mpm_dirty: bool,

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
            mpm: None,
            run_stats: RunStats::default(),
            rbd_envs: vec![PhysicsWorld::default()],
            rbd_sim_params: vec![RbdSimParams::tgs_soft()],
            rbd_dirty: false,
            rbd_steps_per_frame: 1,
            rbd_reserve_per_env: 0,
            rbd2gpu: vec![Coarena::new()],
            mpm_chunks: Arena::new(),
            slot2chunk: Vec::new(),
            mpm_params: None,
            mpm_cell_width: 1.0,
            mpm_substeps: 20,
            mpm_use_cpic: true,
            mpm_dirty: false,
            capacities,
        }
    }

    /// Reserves additional capacity in the handle maps to avoid reallocations
    /// when a known number of bodies/particles is about to be inserted.
    pub fn reserve(&mut self, additional: NexusCapacities) {
        for env in &mut self.rbd2gpu {
            env.reserve(additional.rbd.body_capacity as usize);
        }
        // TODO: reserve the MPM handle maps and resize the GPU buffers too.
    }

    /// Sets the MPM simulation parameters (gravity, timestep) and grid cell
    /// width. Call before the first [`Self::add_particles`]; the values are
    /// applied when the MPM sub-state is created. If MPM already exists they are
    /// applied immediately (the grid is reset, so prefer calling this first).
    pub fn set_mpm_params(
        &mut self,
        backend: &GpuBackend,
        params: SimulationParams,
        cell_width: f32,
    ) -> Result<(), GpuBackendError> {
        self.mpm_params = Some(params);
        self.mpm_cell_width = cell_width;
        if let Some(mpm) = self.mpm.as_mut() {
            mpm.set_cell_width(backend, cell_width, self.capacities.mpm.grid_size)?;
            mpm.set_simulation_params(backend, params)?;
        }
        Ok(())
    }

    /// Sets the number of MPM substeps run per [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call (default
    /// 20). More substeps → smaller timestep → more stable but slower.
    pub fn set_mpm_substeps(&mut self, substeps: u32) {
        self.mpm_substeps = substeps.max(1);
    }

    /// Number of MPM substeps run per [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call.
    pub fn mpm_substeps(&self) -> u32 {
        self.mpm_substeps
    }

    /// Enables/disables CPIC (compatible particle-in-cell) rigid coupling. The
    /// preference is stored so it survives until MPM is lazily allocated. Not
    /// overwritten by [`Self::finalize`] unless the coupling set changes.
    pub fn set_mpm_use_cpic(&mut self, enabled: bool) {
        self.mpm_use_cpic = enabled;
        if let Some(mpm) = self.mpm.as_mut() {
            mpm.use_cpic = enabled;
        }
    }

    /// Whether CPIC rigid coupling is enabled. Falls back to the stored
    /// preference before MPM is lazily allocated.
    pub fn mpm_use_cpic(&self) -> bool {
        self.mpm
            .as_ref()
            .map(|m| m.use_cpic)
            .unwrap_or(self.mpm_use_cpic)
    }

    /// Whether this state uses the MPM solver. True once MPM has been configured
    /// via [`Self::set_mpm_params`], even before the sub-state is lazily
    /// allocated on the first [`Self::add_particles`], so a particle emitter
    /// that starts empty still reports its MPM usage.
    pub fn has_mpm(&self) -> bool {
        self.mpm.is_some() || self.mpm_params.is_some()
    }

    /// Sets the MPM gravity vector. Applied on the next [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) (the
    /// per-substep params are re-uploaded each frame), so this is cheap.
    pub fn set_mpm_gravity(&mut self, gravity: crate::rbd::math::Vector) {
        // Keep the stored params authoritative so the gravity survives until MPM
        // is lazily allocated (and is what `mpm_gravity` reports meanwhile).
        if let Some(params) = self.mpm_params.as_mut() {
            params.gravity = gravity;
        }
        if let Some(mpm) = self.mpm.as_mut() {
            mpm.gravity = gravity;
        }
    }

    /// Current MPM gravity vector. Falls back to the gravity configured via
    /// [`Self::set_mpm_params`] before MPM is lazily allocated, and only to zero
    /// if no params were ever set.
    pub fn mpm_gravity(&self) -> crate::rbd::math::Vector {
        self.mpm
            .as_ref()
            .map(|m| m.gravity)
            .or_else(|| self.mpm_params.map(|p| p.gravity))
            .unwrap_or(crate::rbd::math::Vector::ZERO)
    }

    /// Sets the rigid-body gravity vector, e.g. `[0.0, 0.0, -9.81]` for a Z-up
    /// scene. Every solver path reads the same uniform, so this applies to free
    /// rigid-bodies and multibody links alike (in 2D the third component is
    /// ignored). No-op until the rigid-body state is built, so call it after
    /// [`Self::finalize`].
    #[cfg(feature = "rbd")]
    pub fn set_rbd_gravity(&mut self, backend: &GpuBackend, gravity: [f32; 3]) {
        if let Some(rbd) = self.rbd.as_mut() {
            rbd.set_gravity(backend, gravity);
        }
    }

    // ── Rigid-body runtime settings ─────────────────────────────────────

    /// Overrides the per-environment collision-pair capacity used when the
    /// GPU rigid-body state is (re)allocated at `finalize`. The default (4096)
    /// is sized for one busy scene, not thousands of small batched envs —
    /// pair-keyed workspaces scale as `capacity x num_envs x sizeof(manifold)`,
    /// which at 2048 envs binds ~9 GiB unless this is lowered.
    pub fn set_rbd_collisions_capacity(&mut self, capacity: u32) {
        self.capacities.rbd.collisions_capacity = capacity.max(1);
    }

    /// Sets the number of rigid-body solver steps advanced per
    /// [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call (default 1). Acts as a simulation-speed control.
    pub fn set_rbd_steps_per_frame(&mut self, steps: u32) {
        self.rbd_steps_per_frame = steps.max(1);
    }

    /// Number of rigid-body solver steps per [`NexusPipeline::simulate`](crate::pipeline::NexusPipeline::simulate) call.
    pub fn rbd_steps_per_frame(&self) -> u32 {
        self.rbd_steps_per_frame
    }

    /// Current entity counts (rigid bodies, colliders, joints, multibody DOFs,
    /// particles) for display in the UI. Rigid-body
    /// counts are summed across all environments.
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
        if let Some(mpm) = self.mpm.as_ref() {
            c.particles = mpm.particles.len();
        }
        c
    }

    /// Returns a mutable reference to the MPM sub-state, allocating an empty one
    /// (sized from the stored capacities, configured from [`Self::set_mpm_params`])
    /// if it doesn’t exist yet.
    fn mpm_or_insert(&mut self, backend: &GpuBackend) -> Result<&mut MpmState, GpuBackendError> {
        if self.mpm.is_none() {
            let grid_capacity = self.capacities.mpm.grid_size;
            let mut mpm = MpmState::empty(backend, &self.capacities.mpm)?;
            mpm.set_cell_width(backend, self.mpm_cell_width, grid_capacity)?;
            if let Some(params) = self.mpm_params {
                mpm.set_simulation_params(backend, params)?;
            }
            mpm.use_cpic = self.mpm_use_cpic;
            self.mpm = Some(mpm);
        }
        Ok(self.mpm.as_mut().unwrap())
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

    /// Runtime actuation entry point: mutates environment `env`'s rapier
    /// multibody joints through `f` (e.g. `rapier3d-mjcf`'s
    /// `apply_controls_multibody`, which implements MJCF actuator semantics),
    /// then pushes the refreshed joint data — motor targets/gains, limits — to
    /// the GPU multibody links in one buffer write.
    ///
    /// Unlike [`Self::rbd_world_mut`] this does NOT mark the world dirty: motor
    /// updates are per-step control, not a topology change, so no GPU rebuild
    /// is triggered. Call after [`Self::finalize`]; a no-op before it.
    pub fn control_multibody_motors<F>(
        &mut self,
        backend: &GpuBackend,
        env: usize,
        f: F,
    ) -> Result<(), GpuBackendError>
    where
        F: FnOnce(&mut PhysicsWorld),
    {
        let world = &mut self.rbd_envs[env];
        f(world);
        if let Some(rbd) = self.rbd.as_mut() {
            rbd.multibodies_mut().sync_joint_data_from_rapier(
                backend,
                env as u32,
                &world.multibody_joints,
                &world.bodies,
            )?;
        }
        Ok(())
    }

    /// Mutable access to environment `env`'s rapier world that does **not** mark
    /// the rbd state dirty, for use after [`Self::finalize`].
    ///
    /// Nothing written here reaches the GPU on its own: the rapier sets are the
    /// build-time source the GPU buffers were baked from, and marking them dirty
    /// would rebuild those buffers and snap the simulation back to the authored
    /// state. Use this to run rapier-side helpers whose output you then push
    /// through a runtime setter — e.g. driving an MJCF actuator model and
    /// forwarding the resulting motors with
    /// `GpuMultibodySet::set_motors`.
    pub fn rbd_world_mut_untracked(&mut self, env: usize) -> &mut PhysicsWorld {
        &mut self.rbd_envs[env]
    }

    pub fn insert_rigid_body(
        &mut self,
        body: RigidBody,
        collider: Collider,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        self.insert_rigid_body_in(0, body, collider, coupling)
    }

    /// Inserts a body + collider into environment `env`.
    pub fn insert_rigid_body_in(
        &mut self,
        env: usize,
        body: RigidBody,
        collider: Collider,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        let (handle, _) = self.rbd_envs[env].insert(body, collider);
        self.rbd2gpu[env].insert(
            handle.0,
            GpuRigidBodyRef {
                coupling,
                gpu_id: u32::MAX,
            },
        );
        self.rbd_dirty = true;
        // MPM-coupled boundary colliders live only in environment 0 and feed the
        // MPM coupling rebuild in `finalize`.
        if env == 0 && coupling != RbdCoupling::None {
            self.mpm_dirty = true;
        }
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
        coupling: RbdCoupling,
    ) -> Result<RigidBodyHandle, GpuBackendError> {
        let handles = self.add_rigid_bodies(backend, [(body, collider, coupling)])?;
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
        bodies: impl IntoIterator<Item = (RigidBody, Collider, RbdCoupling)>,
    ) -> Result<Vec<RigidBodyHandle>, GpuBackendError> {
        // Keep copies for the GPU append before the rapier world consumes them.
        let mut gpu_pairs: Vec<(RigidBody, Collider)> = Vec::new();
        let mut handles: Vec<RigidBodyHandle> = Vec::new();
        let mut couplings: Vec<RbdCoupling> = Vec::new();
        for (body, collider, coupling) in bodies {
            gpu_pairs.push((body.clone(), collider.clone()));
            let (handle, _) = self.rbd_envs[0].insert(body, collider);
            handles.push(handle);
            couplings.push(coupling);
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
                for (i, (&handle, &coupling)) in handles.iter().zip(&couplings).enumerate() {
                    self.rbd2gpu[0].insert(
                        handle.0,
                        GpuRigidBodyRef {
                            coupling,
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
            for (&handle, &coupling) in handles.iter().zip(&couplings) {
                self.rbd2gpu[0].insert(
                    handle.0,
                    GpuRigidBodyRef {
                        coupling,
                        gpu_id: u32::MAX,
                    },
                );
            }
            self.rbd_dirty = true;
        }
        if couplings.iter().any(|c| *c != RbdCoupling::None) {
            self.mpm_dirty = true;
        }
        Ok(handles)
    }

    /// Inserts a rigid-body without any attached collider (e.g. a joint anchor).
    pub fn insert_body(&mut self, body: RigidBody, coupling: RbdCoupling) -> RigidBodyHandle {
        self.insert_body_in(0, body, coupling)
    }

    // TODO: remove this. Inserting a collider should insert into all envs.
    //       (though we should also have a variant that allows specifying different
    //       shapes per env).
    /// Inserts a collider-less rigid-body into environment `env`.
    pub fn insert_body_in(
        &mut self,
        env: usize,
        body: RigidBody,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        let handle = self.rbd_envs[env].insert_body(body);
        self.rbd2gpu[env].insert(
            handle.0,
            GpuRigidBodyRef {
                coupling,
                gpu_id: u32::MAX,
            },
        );
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

    /// Appends a new chunk of MPM particles (`O(added)`) and returns its handle.
    pub fn add_particles(
        &mut self,
        backend: &GpuBackend,
        particles: Vec<Particle>,
    ) -> Result<NexusParticleChunk, GpuBackendError> {
        let n = particles.len();
        let chunk = self.mpm_chunks.insert(n);
        {
            let mpm = self.mpm_or_insert(backend)?;
            mpm.particles.append(backend, &particles)?;
        }
        self.slot2chunk.extend(std::iter::repeat_n(chunk, n));
        self.mpm_dirty = true;
        Ok(NexusParticleChunk(chunk))
    }

    /// Appends more particles to an existing chunk (`O(added)`).
    pub fn extend_chunk(
        &mut self,
        backend: &GpuBackend,
        chunk: NexusParticleChunk,
        particles: Vec<Particle>,
    ) -> Result<(), GpuBackendError> {
        let n = particles.len();
        {
            let mpm = self.mpm_or_insert(backend)?;
            mpm.particles.append(backend, &particles)?;
        }
        self.slot2chunk.extend(std::iter::repeat_n(chunk.0, n));
        if let Some(c) = self.mpm_chunks.get_mut(chunk.0) {
            *c += n;
        }
        self.mpm_dirty = true;
        Ok(())
    }

    /// MPM background-grid cell width.
    pub fn mpm_cell_width(&self) -> f32 {
        self.mpm_cell_width
    }

    /// Removes every particle of a chunk (`O(removed)`) and drops the handle.
    pub fn remove_chunk(
        &mut self,
        backend: &GpuBackend,
        chunk: NexusParticleChunk,
    ) -> Result<(), GpuBackendError> {
        let slots: Vec<u32> = self
            .slot2chunk
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == chunk.0)
            .map(|(i, _)| i as u32)
            .collect();
        self.swap_remove_particle_slots(backend, &slots)?;
        self.mpm_chunks.remove(chunk.0);
        Ok(())
    }

    /// Removes up to `count` particles from a chunk (`O(removed)`), returning the
    /// number actually removed. The chunk itself is kept (even if emptied).
    pub fn remove_particles_from_chunk(
        &mut self,
        backend: &GpuBackend,
        chunk: NexusParticleChunk,
        count: usize,
    ) -> Result<usize, GpuBackendError> {
        let mut slots: Vec<u32> = self
            .slot2chunk
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == chunk.0)
            .map(|(i, _)| i as u32)
            .collect();
        // Remove the highest GPU slots first, which keeps the swap-removal cheap.
        slots.sort_unstable_by(|a, b| b.cmp(a));
        slots.truncate(count);
        let removed = slots.len();
        self.swap_remove_particle_slots(backend, &slots)?;
        if let Some(c) = self.mpm_chunks.get_mut(chunk.0) {
            *c = c.saturating_sub(removed);
        }
        Ok(removed)
    }

    /// Swap-removes the given GPU particle slots and patches `slot2chunk` to
    /// follow the relocations the GPU performed.
    fn swap_remove_particle_slots(
        &mut self,
        backend: &GpuBackend,
        slots: &[u32],
    ) -> Result<(), GpuBackendError> {
        if slots.is_empty() {
            return Ok(());
        }
        let remaps = {
            let Some(mpm) = self.mpm.as_mut() else {
                return Ok(());
            };
            mpm.particles.swap_remove(backend, slots)?
        };
        // Each `(from, to)`: the tail particle at `from` was moved down to the
        // freed slot `to`, so its chunk ownership moves with it.
        for (from, to) in remaps {
            self.slot2chunk[to as usize] = self.slot2chunk[from as usize];
        }
        let new_len = self.mpm.as_ref().unwrap().particles.len();
        self.slot2chunk.truncate(new_len);
        Ok(())
    }

    pub async fn finalize(&mut self, backend: &GpuBackend) -> Result<(), GpuBackendError> {
        let rbd_was_dirty = self.rbd_dirty;
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

            // Rebuild the per-environment handle to GPU-slot maps. A handle's
            // `gpu_id` is its *body* slot, not a collider slot, since a body may
            // own several colliders. Body slots are assigned in the order
            // `from_rapier` uses (the first time each parent body is seen while
            // iterating colliders) and are laid out env-major with stride
            // `num_colliders_per_batch`.
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
                    let coupling = self.rbd2gpu[env_idx]
                        .get(body_handle.0)
                        .map(|r| r.coupling)
                        .unwrap_or(RbdCoupling::None);
                    self.rbd2gpu[env_idx].insert(
                        body_handle.0,
                        GpuRigidBodyRef {
                            coupling,
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
                        let coupling = self.rbd2gpu[env_idx]
                            .get(body_handle.0)
                            .map(|r| r.coupling)
                            .unwrap_or(RbdCoupling::None);
                        self.rbd2gpu[env_idx].insert(
                            body_handle.0,
                            GpuRigidBodyRef {
                                coupling,
                                gpu_id: env_idx as u32 * stride + slot,
                            },
                        );
                    }
                }
            }
            self.rbd = Some(rbd_state);
            self.rbd_dirty = false;
        }

        // MPM/rapier coupling. Boundary colliders are inserted into environment 0
        // as rigid bodies tagged `RbdCoupling::Mpm*`; rebuild the coupling
        // (sampled rigid particles, uploaded body set) whenever those bodies or
        // the particle set changed.
        if (rbd_was_dirty || self.mpm_dirty) && self.mpm.is_some() {
            let world = &self.rbd_envs[0];
            let mut coupling = Vec::new();
            let mut materials = Vec::new();
            // Rigid-body slot mirroring each coupling entry, so the MPM-owned
            // poses can be written back to the buffer rendering reads.
            let mut rbd_body_slots = Vec::new();
            for (collider_handle, collider) in world.colliders.iter() {
                let Some(body_handle) = collider.parent() else {
                    continue;
                };
                let Some(gpu_ref) = self.rbd2gpu[0].get(body_handle.0) else {
                    continue;
                };

                let (boundary_condition, mode) = match gpu_ref.coupling {
                    RbdCoupling::None => continue,
                    RbdCoupling::MpmOneWay(boundary_condition) => {
                        (boundary_condition, BodyCoupling::OneWay)
                    }
                    RbdCoupling::MpmTwoWay(boundary_condition) => {
                        (boundary_condition, BodyCoupling::TwoWays)
                    }
                };

                coupling.push(RapierBodyCouplingEntry {
                    body: body_handle,
                    collider: collider_handle,
                    mode,
                });
                materials.push(boundary_condition);
                rbd_body_slots.push(gpu_ref.gpu_id);
            }
            if !coupling.is_empty() {
                let cell_width = self.mpm_cell_width;
                let mpm = self.mpm.as_mut().unwrap_or_else(|| unreachable!());
                mpm.set_coupling(
                    backend,
                    &world.bodies,
                    &world.colliders,
                    coupling,
                    &materials,
                    &rbd_body_slots,
                    cell_width,
                )?;
            }
            self.mpm_dirty = false;
        }
        Ok(())
    }
}
