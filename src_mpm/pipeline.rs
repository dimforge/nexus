//! High-level MPM simulation pipeline orchestration.
//!
//! This module provides the main entry point for running MPM simulations. The pipeline
//! coordinates the execution of all MPM algorithm stages on the GPU.

use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::sort::WgSort;
use crate::solver::{
    BoundaryCondition, GpuImpulses, GpuMaterials, GpuParticles, GpuRigidParticles,
    GpuSimulationParams, GpuTimestepBounds, Particle, SimulationParams, WgG2P, WgG2PCdf,
    WgGridUpdate, WgGridUpdateCdf, WgIntegrateBodies, WgP2G, WgP2GCdf, WgParticleUpdate,
    WgRigidParticleUpdate, WgTimestepBounds,
};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuTimestamps};
use khal::{BufferUsages, Shader};
use nexus_rbd::dynamics::GpuBodySet;
use nexus_rbd::math::{Pose, Vector};
use nexus_rbd::utils::{GpuPrefixSum, PrefixSumWorkspace};
use vortx::tensor::Tensor;

use nexus_rbd::dynamics::body::{BodyCoupling, RapierBodyCouplingEntry};

/// Initial capacities of an MPM scene's GPU-resident buffers.
///
/// `NexusState` holds one of these and forwards it on the first particle
/// insertion, when the sub-state is created.
#[derive(Copy, Clone, Debug)]
pub struct MpmCapacities {
    /// Number of grid cells reserved for the background grid.
    pub grid_size: u32,
    /// Number of particles to reserve buffer space for up front, so the initial
    /// particle upload (and early emitter growth) doesn't reallocate.
    pub particles_capacity: u32,
}

impl Default for MpmCapacities {
    fn default() -> Self {
        Self {
            grid_size: 32768,
            particles_capacity: 65536,
        }
    }
}

/// GPU compute pipeline for Material Point Method simulation.
pub struct MpmPipeline {
    grid: WgGrid,
    prefix_sum: GpuPrefixSum,
    sort: WgSort,
    p2g: WgP2G,
    p2g_cdf: WgP2GCdf,
    grid_update_cdf: WgGridUpdateCdf,
    grid_update: WgGridUpdate,
    particles_update: WgParticleUpdate,
    g2p: WgG2P,
    g2p_cdf: WgG2PCdf,
    rigid_particles_update: WgRigidParticleUpdate,
    /// Maximum timestep bound calculation.
    pub timestep_bounds: WgTimestepBounds,
    /// Rigid body impulse computation kernel (publicly accessible for external use).
    pub integrate_bodies: WgIntegrateBodies,
}

/// GPU-resident simulation state for MPM.
pub struct MpmState {
    /// The simulation timestep.
    pub base_dt: f32,
    pub gravity: Vector,
    pub use_cpic: bool,
    /// Global simulation parameters (gravity, timestep).
    pub sim_params: GpuSimulationParams,
    /// Spatial grid for momentum transfer.
    pub grid: GpuGrid,
    /// MPM particles (positions, velocities, masses, material properties).
    pub particles: GpuParticles,
    /// Particles sampled from rigid body collider surfaces for two-way coupling.
    pub rigid_particles: GpuRigidParticles,
    /// Rigid bodies coupled with the MPM simulation.
    pub bodies: GpuBodySet,
    /// MPM materials associated to each rigid-body.
    pub body_materials: GpuMaterials,
    /// Accumulated impulses to apply to rigid bodies from MPM interactions.
    pub impulses: GpuImpulses,
    /// Staging buffer for reading rigid body poses back to CPU.
    pub poses_staging: Tensor<Pose>,
    /// For each coupled body, the rigid-body pipeline slot it mirrors. Empty
    /// unless the coupling was built through the `NexusState` path, which is
    /// the only one that knows the rigid-body slot layout. Consumed by
    /// [`MpmPipeline::writeback_body_poses`].
    pub rbd_body_slots: Tensor<u32>,
    /// The timestep estimate computed from particles and their models.
    pub timestep_bounds: Tensor<GpuTimestepBounds>,
    /// Staging buffer for reading the timestep bound estimate.
    pub timestep_bounds_staging: Tensor<GpuTimestepBounds>,
    prefix_sum: PrefixSumWorkspace,
    coupling: Vec<RapierBodyCouplingEntry>,
}

impl MpmState {
    /// Creates an empty MPM state with no particles and no coupled bodies.
    ///
    /// The grid is preallocated to hold `grid_capacity` cells. Physical
    /// parameters (`gravity`, `base_dt`, the grid `cell_width`) are left at
    /// neutral defaults, so set them before stepping. Particles and coupled
    /// bodies are appended lazily through `NexusState`, growing the GPU buffers
    /// on demand.
    pub fn empty(
        backend: &GpuBackend,
        capacities: &MpmCapacities,
    ) -> Result<Self, GpuBackendError> {
        const DEFAULT_CELL_WIDTH: f32 = 1.0;
        let grid_capacity = capacities.grid_size;
        let params = SimulationParams {
            gravity: Vector::ZERO,
            #[cfg(feature = "dim2")]
            padding: 0.0,
            dt: 1.0 / 60.0,
        };
        let sim_params = GpuSimulationParams::new(backend, params)?;
        let mut particles = GpuParticles::from_particles(backend, &[])?;
        // Reserve room up front so the initial particle upload / early emitter
        // growth doesn't reallocate the per-particle buffers.
        particles.reserve(backend, capacities.particles_capacity as usize)?;
        let rigid_particles = GpuRigidParticles::new(backend)?;
        let bodies = GpuBodySet::empty(backend);
        let body_materials = GpuMaterials::new(backend, &[])?;
        let grid = GpuGrid::with_capacity(backend, grid_capacity, DEFAULT_CELL_WIDTH)?;
        let prefix_sum = PrefixSumWorkspace::with_capacity(backend, grid_capacity);
        let impulses = GpuImpulses::new(backend)?;
        let poses_staging = Tensor::vector_uninit(
            backend,
            bodies.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        let bounds = GpuTimestepBounds::default();
        let timestep_bounds = Tensor::scalar(
            backend,
            bounds,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let timestep_bounds_staging = Tensor::scalar(
            backend,
            bounds,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        Ok(Self {
            base_dt: params.dt,
            gravity: params.gravity,
            use_cpic: false,
            sim_params,
            grid,
            particles,
            rigid_particles,
            bodies,
            body_materials,
            impulses,
            poses_staging,
            rbd_body_slots: Tensor::vector(backend, [], BufferUsages::STORAGE)?,
            timestep_bounds,
            timestep_bounds_staging,
            prefix_sum,
            coupling: Vec::new(),
        })
    }

    /// Updates the global simulation parameters (gravity, timestep) and uploads
    /// them to the GPU.
    pub fn set_simulation_params(
        &mut self,
        backend: &GpuBackend,
        params: SimulationParams,
    ) -> Result<(), GpuBackendError> {
        self.gravity = params.gravity;
        self.base_dt = params.dt;
        self.sim_params = GpuSimulationParams::new(backend, params)?;
        Ok(())
    }

    /// Uploads the per-substep parameters: the visible timestep `base_dt` divided
    /// by `num_substeps`, keeping the current gravity. Cheap (one buffer write),
    /// called each frame by the `NexusState` substep loop.
    pub fn write_substep_params(
        &mut self,
        backend: &GpuBackend,
        num_substeps: u32,
    ) -> Result<(), GpuBackendError> {
        let params = SimulationParams {
            gravity: self.gravity,
            dt: self.base_dt / num_substeps.max(1) as f32,
            #[cfg(feature = "dim2")]
            padding: 0.0,
        };
        backend.write_buffer(self.sim_params.params.buffer_mut(), 0, &[params])?;
        Ok(())
    }

    /// Reallocates the background grid with a new cell width (and capacity). Must
    /// be called before particles are added, since it discards grid state.
    pub fn set_cell_width(
        &mut self,
        backend: &GpuBackend,
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<(), GpuBackendError> {
        self.grid = GpuGrid::with_capacity(backend, grid_capacity, cell_width)?;
        self.prefix_sum = PrefixSumWorkspace::with_capacity(backend, grid_capacity);
        Ok(())
    }

    /// (Re)builds the rigid-body coupling: uploads the coupled bodies, samples
    /// rigid particles from their collider surfaces, and stores the per-collider
    /// boundary materials.
    ///
    /// `rbd_body_slots[i]` is the rigid-body pipeline slot mirroring coupling
    /// entry `i`; it lets [`MpmPipeline::writeback_body_poses`] push the poses
    /// MPM integrates back to the buffer rendering reads.
    ///
    /// Leaves the MPM particles / grid / sim-params untouched.
    pub fn set_coupling(
        &mut self,
        backend: &GpuBackend,
        bodies: &rapier::dynamics::RigidBodySet,
        colliders: &rapier::geometry::ColliderSet,
        coupling: Vec<RapierBodyCouplingEntry>,
        materials: &[BoundaryCondition],
        rbd_body_slots: &[u32],
        cell_width: f32,
    ) -> Result<(), GpuBackendError> {
        assert_eq!(coupling.len(), materials.len());
        assert_eq!(coupling.len(), rbd_body_slots.len());
        let gpu_bodies = GpuBodySet::from_rapier(backend, bodies, colliders, &coupling);
        let rigid_particles =
            GpuRigidParticles::from_rapier(backend, colliders, &gpu_bodies, &coupling, cell_width)?;
        self.body_materials = GpuMaterials::new(backend, materials)?;
        self.poses_staging = Tensor::vector_uninit(
            backend,
            gpu_bodies.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        self.rbd_body_slots = Tensor::vector(backend, rbd_body_slots, BufferUsages::STORAGE)?;
        self.use_cpic = !coupling.is_empty();
        self.bodies = gpu_bodies;
        self.rigid_particles = rigid_particles;
        self.coupling = coupling;
        Ok(())
    }
}

impl MpmState {
    /// Creates new MPM simulation data with default two-way coupling for all colliders.
    pub fn new(
        backend: &GpuBackend,
        params: SimulationParams,
        particles: &[Particle],
        bodies: &rapier::dynamics::RigidBodySet,
        colliders: &rapier::geometry::ColliderSet,
        materials: &[(rapier::geometry::ColliderHandle, BoundaryCondition)],
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, GpuBackendError> {
        let coupling: Vec<_> = colliders
            .iter()
            .filter_map(|(co_handle, co)| {
                let rb_handle = co.parent()?;
                Some(RapierBodyCouplingEntry {
                    body: rb_handle,
                    collider: co_handle,
                    mode: BodyCoupling::OneWay,
                })
            })
            .collect();
        let materials: Vec<_> = coupling
            .iter()
            .map(|c| {
                materials
                    .iter()
                    .find(|e| e.0 == c.collider)
                    .map(|e| e.1)
                    .unwrap_or(BoundaryCondition::separate(1.0))
            })
            .collect();
        Self::with_select_coupling(
            backend,
            params,
            particles,
            bodies,
            colliders,
            coupling,
            &materials,
            cell_width,
            grid_capacity,
        )
    }

    /// Creates new MPM simulation data with custom rigid body coupling configuration.
    pub fn with_select_coupling(
        backend: &GpuBackend,
        params: SimulationParams,
        particles: &[Particle],
        bodies: &rapier::dynamics::RigidBodySet,
        colliders: &rapier::geometry::ColliderSet,
        coupling: Vec<RapierBodyCouplingEntry>,
        materials: &[BoundaryCondition],
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, GpuBackendError> {
        assert_eq!(coupling.len(), materials.len());

        let sampling_step = cell_width;
        let bodies = GpuBodySet::from_rapier(backend, bodies, colliders, &coupling);
        let body_materials = GpuMaterials::new(backend, materials)?;
        let sim_params = GpuSimulationParams::new(backend, params)?;
        let particles = GpuParticles::from_particles(backend, particles)?;
        let rigid_particles =
            GpuRigidParticles::from_rapier(backend, colliders, &bodies, &coupling, sampling_step)?;
        let grid = GpuGrid::with_capacity(backend, grid_capacity, cell_width)?;
        let prefix_sum = PrefixSumWorkspace::with_capacity(backend, grid_capacity);
        let impulses = GpuImpulses::new(backend)?;
        let poses_staging = Tensor::vector_uninit(
            backend,
            bodies.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        let bounds = GpuTimestepBounds::default();
        let timestep_bounds = Tensor::scalar(
            backend,
            bounds,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let timestep_bounds_staging = Tensor::scalar(
            backend,
            bounds,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        Ok(Self {
            sim_params,
            particles,
            gravity: params.gravity,
            use_cpic: true,
            rigid_particles,
            bodies,
            body_materials,
            impulses,
            grid,
            prefix_sum,
            poses_staging,
            // Standalone MPM: no rigid-body pipeline to write poses back to.
            rbd_body_slots: Tensor::vector(backend, [], BufferUsages::STORAGE)?,
            coupling,
            timestep_bounds,
            timestep_bounds_staging,
            base_dt: params.dt,
        })
    }

    /// Returns the list of rigid body coupling entries.
    pub fn coupling(&self) -> &[RapierBodyCouplingEntry] {
        &self.coupling
    }
}

impl MpmPipeline {
    /// Creates a new MPM compute pipeline by compiling all necessary shaders.
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            grid: WgGrid::from_backend(backend)?,
            prefix_sum: GpuPrefixSum::from_backend(backend)?,
            sort: WgSort::from_backend(backend)?,
            p2g: WgP2G::from_backend(backend)?,
            p2g_cdf: WgP2GCdf::from_backend(backend)?,
            grid_update: WgGridUpdate::from_backend(backend)?,
            grid_update_cdf: WgGridUpdateCdf::from_backend(backend)?,
            particles_update: WgParticleUpdate::from_backend(backend)?,
            rigid_particles_update: WgRigidParticleUpdate::from_backend(backend)?,
            g2p: WgG2P::from_backend(backend)?,
            g2p_cdf: WgG2PCdf::from_backend(backend)?,
            integrate_bodies: WgIntegrateBodies::from_backend(backend)?,
            timestep_bounds: WgTimestepBounds::from_backend(backend)?,
        })
    }

    /// Executes one complete MPM simulation timestep.
    pub fn step(
        &self,
        backend: &GpuBackend,
        data: &mut MpmState,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        let mut encoder = backend.begin_encoding();

        {
            let mut pass = encoder.begin_pass("[MPM] Rigid update", timestamps.as_deref_mut());
            self.integrate_bodies.launch_update_world_mass_properties(
                &mut pass,
                &mut data.impulses,
                &mut data.bodies,
            )?;
            self.rigid_particles_update.launch(
                &mut pass,
                &mut data.bodies,
                &mut data.rigid_particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("[MPM] Grid sort", timestamps.as_deref_mut());
            data.grid.swap_buffers();
            self.grid.launch_sort(
                backend,
                &mut pass,
                &mut data.particles,
                data.use_cpic.then_some(&mut data.rigid_particles),
                &mut data.grid,
                &mut data.prefix_sum,
                &self.sort,
                &self.prefix_sum,
            )?;

            if data.use_cpic {
                self.sort.launch_sort_rigid_particles(
                    backend,
                    &mut pass,
                    &mut data.rigid_particles,
                    &mut data.grid,
                    &mut data.prefix_sum,
                    &self.prefix_sum,
                )?;
            }
        }

        if data.use_cpic {
            {
                let mut pass =
                    encoder.begin_pass("[MPM] CDF grid update", timestamps.as_deref_mut());
                self.grid_update_cdf
                    .launch(&mut pass, &mut data.grid, &data.bodies)?;
            }

            {
                let mut pass = encoder.begin_pass("[MPM] CDF P2G", timestamps.as_deref_mut());
                self.p2g_cdf.launch(
                    &mut pass,
                    &mut data.grid,
                    &data.rigid_particles,
                    &data.bodies,
                )?;
            }

            {
                let mut pass = encoder.begin_pass("[MPM] CDF G2P", timestamps.as_deref_mut());
                self.g2p_cdf.launch(
                    &mut pass,
                    &data.sim_params,
                    &data.grid,
                    &mut data.particles,
                )?;
            }
        }

        {
            let mut pass = encoder.begin_pass("[MPM] P2G", timestamps.as_deref_mut());
            self.p2g.launch(
                &mut pass,
                data.use_cpic,
                &mut data.grid,
                &data.particles,
                &mut data.impulses,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("[MPM] Grid update", timestamps.as_deref_mut());
            self.grid_update.launch(
                &mut pass,
                data.use_cpic,
                &data.sim_params,
                &mut data.grid,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("[MPM] G2P", timestamps.as_deref_mut());
            self.g2p.launch(
                &mut pass,
                data.use_cpic,
                &data.sim_params,
                &data.grid,
                &mut data.particles,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("[MPM] Particle update", timestamps.as_deref_mut());
            self.particles_update.launch(
                &mut pass,
                &data.sim_params,
                &data.grid,
                &mut data.particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("[MPM] Integrate bodies", timestamps.as_deref_mut());
            self.integrate_bodies.launch(
                &mut pass,
                &data.grid,
                &data.sim_params,
                &mut data.impulses,
                &mut data.bodies,
            )?;
        }

        if let Some(timestamps) = timestamps {
            timestamps.resolve(&mut encoder);
        }

        backend.submit(encoder)
    }

    /// Pushes the coupled bodies' MPM-integrated poses into `rbd_poses`, the
    /// rigid-body pipeline's body-pose buffer.
    ///
    /// MPM integrates its own copy of every coupled body (see
    /// [`WgIntegrateBodies::launch`]) while the rigid-body pipeline treats them
    /// as static, so the two copies diverge as soon as a body moves. Call this
    /// once per visible frame, after the substep loop, so rendering and the next
    /// step's broad phase see where the body really is.
    pub fn writeback_body_poses(
        &self,
        backend: &GpuBackend,
        data: &MpmState,
        rbd_poses: &mut Tensor<Pose>,
    ) -> Result<(), GpuBackendError> {
        if data.bodies.is_empty() || data.rbd_body_slots.is_empty() {
            return Ok(());
        }

        let mut encoder = backend.begin_encoding();
        {
            let mut pass = encoder.begin_pass("[MPM] Body pose writeback", None);
            self.integrate_bodies.launch_writeback_body_poses(
                &mut pass,
                &data.bodies,
                &data.rbd_body_slots,
                rbd_poses,
            )?;
        }
        backend.submit(encoder)
    }
}
