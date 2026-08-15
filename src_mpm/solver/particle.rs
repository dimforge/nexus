use crate::mpm_shaders::solver::particle::{
    Kinematics, ParticleProperties, Position, RigidParticleIndices,
};
use crate::mpm_shaders::{PaddedMatrix, PaddingExt};
use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError};
use nexus_rbd::dynamics::GpuBodySet;
use nexus_rbd::math::{DIM, Matrix, Vector};
use std::ops::RangeBounds;
use vortx::tensor::Tensor;

use crate::solver::{GpuParticleModel, ParticleModel};
use {
    crate::sampling::{self, SamplingBuffers, SamplingParams},
    nexus_rbd::dynamics::body::RapierBodyCouplingEntry,
};

/// Particle position type used on the GPU.
///
/// In 2D: `Position` contains a Vec2.
/// In 3D: `Position` contains a Vec3.
pub type ParticlePosition = Position;

/// A single MPM particle with position, dynamics, and material model.
#[derive(Copy, Clone, Debug)]
pub struct Particle {
    /// Spatial position.
    pub position: Vector,
    /// Physical state (velocity, deformation, mass, etc.).
    pub dynamics: ParticleDynamics,
    /// Material model defining constitutive behavior.
    pub model: ParticleModel,
}

impl Particle {
    /// Creates a new particle with the given properties.
    pub fn new(position: Vector, radius: f32, density: f32, model: ParticleModel) -> Self {
        Particle {
            position,
            dynamics: ParticleDynamics::new(radius, density),
            model,
        }
    }

    /// Creates a new particle belonging to the given render group.
    ///
    /// See [`ParticleDynamics::group_id`].
    pub fn with_group(
        position: Vector,
        radius: f32,
        density: f32,
        model: ParticleModel,
        group_id: u32,
    ) -> Self {
        let mut result = Self::new(position, radius, density, model);
        result.dynamics.group_id = group_id;
        result
    }
}

/// CPU-side particle dynamics for initialization.
///
/// Splits into GPU `Kinematics`, `Cdf`, deformation gradient, and `ParticleProperties` buffers on upload.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParticleDynamics {
    /// Current velocity.
    pub velocity: Vector,
    /// Deformation gradient.
    pub def_grad: Matrix,
    /// APIC affine velocity matrix.
    pub affine: Matrix,
    /// Additional force * dt.
    pub force_dt: Vector,
    /// Determinant of velocity gradient.
    pub vel_grad_det: f32,
    /// Collision detection field.
    pub cdf: crate::mpm_shaders::solver::particle::Cdf,
    /// Initial particle volume.
    pub init_volume: f32,
    /// Initial particle radius.
    pub init_radius: f32,
    /// Particle mass.
    pub mass: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Particle phase.
    pub phase: f32,
    /// Multiplier on a collider's friction for this particle. See
    /// [`Kinematics::boundary_friction`](crate::mpm_shaders::solver::particle::Kinematics::boundary_friction).
    pub boundary_friction: f32,
    /// Group this particle belongs to. Carries no physics: the viewer looks the
    /// group up in its color palette to shade the particle.
    pub group_id: u32,
    /// Whether this particle is active.
    pub enabled: u32,
    /// Whether this particle is fixed.
    pub fixed: u32,
}

impl ParticleDynamics {
    /// Creates new particle dynamics from radius and material density.
    pub fn new(radius: f32, density: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (radius * 2.0).powi(exponent);
        Self {
            velocity: Vector::ZERO,
            def_grad: Matrix::IDENTITY,
            affine: Matrix::ZERO,
            force_dt: Vector::ZERO,
            vel_grad_det: 0.0,
            init_volume,
            init_radius: radius,
            mass: init_volume * density,
            damping: 0.0,
            cdf: crate::mpm_shaders::solver::particle::Cdf::zero(),
            phase: 1.0,
            boundary_friction: 1.0,
            group_id: 0,
            enabled: 1,
            fixed: 0,
        }
    }

    /// Sets whether this particle is fixed.
    pub fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed as u32;
    }

    /// Sets how much of a collider's friction this particle feels.
    ///
    /// 1 (the default) takes the collider's friction as given; 0 lets the
    /// particle slide freely along every surface without passing through it,
    /// which is what a fluid sharing a floor with granular material wants.
    ///
    /// Requires CPIC coupling, which is on by default: see
    /// [`Kinematics::boundary_friction`](crate::mpm_shaders::solver::particle::Kinematics::boundary_friction).
    pub fn set_boundary_friction(&mut self, factor: f32) {
        self.boundary_friction = factor;
    }

    /// Sets the group this particle belongs to.
    pub fn set_group_id(&mut self, group_id: u32) {
        self.group_id = group_id;
    }

    /// Sets the damping coefficient.
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping;
    }

    /// Updates the particle mass based on a new density.
    pub fn set_density(&mut self, density: f32) {
        self.mass = self.init_volume * density;
    }

    /// Converts to the GPU `Kinematics` struct.
    fn to_gpu_kinematics(self) -> Kinematics {
        Kinematics {
            affine: PaddedMatrix::add_padding(self.affine),
            velocity: self.velocity,
            vel_grad_det: self.vel_grad_det,
            force_dt: self.force_dt,
            mass: self.mass,
            enabled: self.enabled,
            boundary_friction: self.boundary_friction,
            #[cfg(feature = "dim3")]
            _padding: Default::default(),
            cdf: self.cdf,
            #[cfg(feature = "dim2")]
            _tail_padding: Default::default(),
        }
    }

    /// Converts the deformation gradient to a GPU `PaddedMatrix`.
    fn to_gpu_def_grad(self) -> PaddedMatrix {
        PaddedMatrix::add_padding(self.def_grad)
    }

    /// Converts to the GPU `ParticleProperties` struct.
    fn to_gpu_properties(self) -> ParticleProperties {
        ParticleProperties {
            init_volume: self.init_volume,
            init_radius: self.init_radius,
            damping: self.damping,
            phase: self.phase,
            fixed: self.fixed,
            group_id: self.group_id,
            padding: Default::default(),
        }
    }
}

struct SoAParticles {
    positions: Vec<Position>,
    kinematics: Vec<Kinematics>,
    def_grad: Vec<PaddedMatrix>,
    properties: Vec<ParticleProperties>,
    models: Vec<GpuParticleModel>,
}

impl SoAParticles {
    pub fn new(particles: &[Particle]) -> Self {
        let positions: Vec<_> = particles
            .iter()
            .map(|p| Position::new(p.position))
            .collect();
        let kinematics: Vec<_> = particles
            .iter()
            .map(|p| p.dynamics.to_gpu_kinematics())
            .collect();
        let def_grad: Vec<_> = particles
            .iter()
            .map(|p| p.dynamics.to_gpu_def_grad())
            .collect();
        let properties: Vec<_> = particles
            .iter()
            .map(|p| p.dynamics.to_gpu_properties())
            .collect();
        let models: Vec<_> = particles
            .iter()
            .map(|p| GpuParticleModel::from(p.model))
            .collect();

        Self {
            positions,
            kinematics,
            def_grad,
            properties,
            models,
        }
    }
}

/// GPU buffers for particles sampled from rigid body surfaces.
pub struct GpuRigidParticles {
    /// Sample points in local (body-relative) coordinates.
    pub local_sample_points: Tensor<Position>,
    /// Sample points transformed to world coordinates.
    pub sample_points: Tensor<Position>,
    /// Bitmask indicating which rigid particles need grid cell blocking.
    pub rigid_particle_needs_block: Tensor<u32>,
    /// Rigid particle indices sorted by grid block (with room for per-block "extras").
    pub sorted_ids: Tensor<u32>,
    /// Metadata associating each sample with its source collider and body.
    pub sample_ids: Tensor<RigidParticleIndices>,
}

impl GpuRigidParticles {
    /// Creates an empty set of rigid particles.
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        let empty_positions: &[Position] = &[];
        let empty_ids: &[RigidParticleIndices] = &[];
        Ok(Self {
            local_sample_points: Tensor::vector(backend, empty_positions, BufferUsages::STORAGE)?,
            sample_points: Tensor::vector(backend, empty_positions, BufferUsages::STORAGE)?,
            sorted_ids: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE)?,
            sample_ids: Tensor::vector(backend, empty_ids, BufferUsages::STORAGE)?,
            rigid_particle_needs_block: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE)?,
        })
    }

    fn from_buffers(
        backend: &GpuBackend,
        sampling_buffers: &SamplingBuffers,
    ) -> Result<Self, GpuBackendError> {
        Ok(Self {
            local_sample_points: Tensor::vector(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            sample_points: Tensor::vector(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            sorted_ids: Tensor::vector_uninit(
                backend,
                sampling_buffers.samples.len() as u32 * 2_u32.pow(DIM as u32),
                BufferUsages::STORAGE,
            )?,
            sample_ids: Tensor::vector(
                backend,
                &sampling_buffers.samples_ids,
                BufferUsages::STORAGE,
            )?,
            rigid_particle_needs_block: Tensor::vector_uninit(
                backend,
                sampling_buffers.samples.len().div_ceil(32) as u32,
                BufferUsages::STORAGE,
            )?,
        })
    }

    /// Samples particles from collider surfaces for MPM coupling.
    pub fn from_rapier(
        backend: &GpuBackend,
        colliders: &rapier::geometry::ColliderSet,
        gpu_bodies: &GpuBodySet,
        coupling: &[RapierBodyCouplingEntry],
        sampling_step: f32,
    ) -> Result<Self, GpuBackendError> {
        let mut sampling_buffers = SamplingBuffers::default();

        for (collider_id, (coupling, gpu_data)) in coupling
            .iter()
            .zip(gpu_bodies.shapes_data().iter())
            .enumerate()
        {
            let collider = &colliders[coupling.collider];

            #[cfg(feature = "dim2")]
            if let Some(polyline) = collider.shape().as_polyline() {
                // Use polyline_vertex_start() to get the correct base index,
                // which accounts for BVH AABB data preceding the actual vertices.
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: gpu_data.polyline_vertex_start(),
                    sampling_step,
                };
                sampling::sample_polyline(polyline, &sampling_params, &mut sampling_buffers)
            }

            #[cfg(feature = "dim3")]
            if let Some(trimesh) = collider.shape().as_trimesh() {
                // Use trimesh_vertex_start() to get the correct base index,
                // which accounts for BVH AABB data preceding the actual vertices.
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: gpu_data.trimesh_vertex_start(),
                    sampling_step,
                };
                sampling::sample_trimesh(trimesh, &sampling_params, &mut sampling_buffers)
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let (vtx, idx) = heightfield.to_trimesh();
                let trimesh = rapier::geometry::TriMesh::new(vtx, idx).unwrap();
                // Use trimesh_vertex_start() to get the correct base index,
                // which accounts for BVH AABB data preceding the actual vertices.
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: gpu_data.trimesh_vertex_start(),
                    sampling_step,
                };
                sampling::sample_trimesh(&trimesh, &sampling_params, &mut sampling_buffers)
            }
        }

        Self::from_buffers(backend, &sampling_buffers)
    }

    /// Returns the number of rigid body particles.
    pub fn len(&self) -> u64 {
        self.sample_points.len()
    }

    /// Returns true if there are no rigid body particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// GPU buffers storing all MPM particle data in Structure-of-Arrays layout.
pub struct GpuParticles {
    len: usize,
    pub gpu_len: Tensor<u32>,
    pub positions: Tensor<Position>,
    pub kinematics: Tensor<Kinematics>,
    pub def_grad: Tensor<PaddedMatrix>,
    pub properties: Tensor<ParticleProperties>,
    pub models: Tensor<GpuParticleModel>,
    pub sorted_ids: Tensor<u32>,
}

impl GpuParticles {
    /// Returns true if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of particles.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns reference to GPU buffer containing particle count.
    pub fn gpu_len(&self) -> &Tensor<u32> {
        &self.gpu_len
    }

    /// Uploads CPU-side particles to GPU buffers.
    pub fn from_particles(
        backend: &GpuBackend,
        particles: &[Particle],
    ) -> Result<Self, GpuBackendError> {
        let data = SoAParticles::new(particles);
        let resizeable = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        Ok(Self {
            len: particles.len(),
            gpu_len: Tensor::scalar(
                backend,
                particles.len() as u32,
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            positions: Tensor::vector(backend, &data.positions, resizeable)?,
            kinematics: Tensor::vector(backend, &data.kinematics, resizeable)?,
            def_grad: Tensor::vector(backend, &data.def_grad, resizeable)?,
            properties: Tensor::vector(backend, &data.properties, resizeable)?,
            models: Tensor::vector(backend, &data.models, resizeable)?,
            sorted_ids: Tensor::vector_uninit(
                backend,
                particles.len() as u32 * 2_u32.pow(DIM as u32),
                resizeable,
            )?,
        })
    }

    /// Reserves GPU buffer room for at least `capacity` particles, so a
    /// subsequent [`append`](Self::append) of up to `capacity` particles doesn't
    /// reallocate. Must be called while empty (right after construction, say):
    /// it replaces the per-particle buffers with empty, capacity-sized ones.
    pub fn reserve(
        &mut self,
        backend: &GpuBackend,
        capacity: usize,
    ) -> Result<(), GpuBackendError> {
        assert_eq!(
            self.len, 0,
            "GpuParticles::reserve must be called while empty"
        );
        let cap = capacity as u32;
        let resizeable = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        self.positions = Tensor::with_capacity(backend, cap, resizeable)?;
        self.kinematics = Tensor::with_capacity(backend, cap, resizeable)?;
        self.def_grad = Tensor::with_capacity(backend, cap, resizeable)?;
        self.properties = Tensor::with_capacity(backend, cap, resizeable)?;
        self.models = Tensor::with_capacity(backend, cap, resizeable)?;
        // `sorted_ids` holds one entry per particle per touched grid node.
        self.sorted_ids = Tensor::with_capacity(backend, cap * 2u32.pow(DIM as u32), resizeable)?;
        Ok(())
    }

    /// Returns reference to material model buffer.
    pub fn models(&self) -> &Tensor<GpuParticleModel> {
        &self.models
    }

    /// Returns mutable reference to material model buffer.
    pub fn models_mut(&mut self) -> &mut Tensor<GpuParticleModel> {
        &mut self.models
    }

    /// Returns reference to position buffer.
    pub fn positions(&self) -> &Tensor<Position> {
        &self.positions
    }

    /// Returns mutable reference to position buffer.
    pub fn positions_mut(&mut self) -> &mut Tensor<Position> {
        &mut self.positions
    }

    /// Returns reference to kinematics buffer.
    pub fn kinematics(&self) -> &Tensor<Kinematics> {
        &self.kinematics
    }

    /// Returns mutable reference to kinematics buffer.
    pub fn kinematics_mut(&mut self) -> &mut Tensor<Kinematics> {
        &mut self.kinematics
    }

    /// Returns reference to deformation gradient buffer.
    pub fn def_grad(&self) -> &Tensor<PaddedMatrix> {
        &self.def_grad
    }

    /// Returns mutable reference to deformation gradient buffer.
    pub fn def_grad_mut(&mut self) -> &mut Tensor<PaddedMatrix> {
        &mut self.def_grad
    }

    /// Returns reference to particle properties buffer (read-only on GPU).
    pub fn properties(&self) -> &Tensor<ParticleProperties> {
        &self.properties
    }

    /// Returns mutable reference to particle properties buffer.
    pub fn properties_mut(&mut self) -> &mut Tensor<ParticleProperties> {
        &mut self.properties
    }

    /// Returns reference to sorted particle ID buffer.
    pub fn sorted_ids(&self) -> &Tensor<u32> {
        &self.sorted_ids
    }

    /// Returns mutable reference to sorted particle ID buffer.
    pub fn sorted_ids_mut(&mut self) -> &mut Tensor<u32> {
        &mut self.sorted_ids
    }

    /// Removes a range of particles from the GPU buffers, shifting elements to fill the gap.
    ///
    /// Returns the number of removed particles on success.
    pub fn shift_remove(
        &mut self,
        backend: &GpuBackend,
        range: impl RangeBounds<usize> + Clone,
    ) -> Result<usize, GpuBackendError> {
        let Self {
            len,
            gpu_len,
            positions,
            kinematics,
            def_grad,
            properties,
            models,
            sorted_ids: _,
        } = self;

        let removed = positions.shift_remove(backend, range.clone())?;
        kinematics.shift_remove(backend, range.clone())?;
        def_grad.shift_remove(backend, range.clone())?;
        properties.shift_remove(backend, range.clone())?;
        models.shift_remove(backend, range)?;

        *len -= removed;
        backend.write_buffer(gpu_len.buffer_mut(), 0, &[*len as u32])?;
        Ok(removed)
    }

    /// Appends particles at the end of the GPU buffers.
    pub fn append(
        &mut self,
        backend: &GpuBackend,
        particles: &[Particle],
    ) -> Result<(), GpuBackendError> {
        let Self {
            len,
            gpu_len,
            positions,
            kinematics,
            def_grad,
            properties,
            models,
            sorted_ids,
        } = self;

        let data = SoAParticles::new(particles);
        // `sorted_ids` is the spatial-sort scratch; it must stay sized
        // `total_particles * 2^DIM` to match `from_particles` (one entry per
        // particle per touched grid node). Undersizing it corrupts the sort.
        let zeros = vec![0u32; particles.len() * 2usize.pow(DIM as u32)];

        positions.append(backend, &data.positions)?;
        kinematics.append(backend, &data.kinematics)?;
        def_grad.append(backend, &data.def_grad)?;
        properties.append(backend, &data.properties)?;
        models.append(backend, &data.models)?;
        sorted_ids.append(backend, &zeros)?;

        *len += particles.len();
        backend.write_buffer(gpu_len.buffer_mut(), 0, &[*len as u32])?;
        Ok(())
    }

    /// Removes the given particle slots by *swap-removing* them: the live tail
    /// particles are moved into the freed slots and the buffers are truncated.
    /// This is `O(number of removed slots)` with no full-buffer shift, which
    /// makes runtime removal cheap. The reordering is invisible to the solver
    /// because particles are spatially re-sorted at the start of every step.
    ///
    /// Returns the relocations performed as `(from, to)` pairs (a tail particle
    /// moved from slot `from` down to freed slot `to`) so callers can patch any
    /// slot→handle maps. `from` is always a now-truncated tail slot and `to` a
    /// freed slot below the new length.
    pub fn swap_remove(
        &mut self,
        backend: &GpuBackend,
        slots: &[u32],
    ) -> Result<Vec<(u32, u32)>, GpuBackendError> {
        // Process descending so truncating the tail never disturbs a
        // not-yet-processed (lower) slot.
        let mut targets: Vec<u32> = slots.to_vec();
        targets.sort_unstable_by(|a, b| b.cmp(a));
        targets.dedup();

        let mut remaps = Vec::new();
        for slot in targets {
            let slot = slot as usize;
            if slot >= self.len {
                continue;
            }
            let last = self.len - 1;
            if slot != last {
                // Relocate the live tail particle into the freed slot. A staging
                // buffer avoids same-buffer overlapping copies.
                macro_rules! relocate {
                    ($t:expr) => {{
                        let mut staging = backend.uninit_buffer(
                            1,
                            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                        )?;
                        let mut enc = backend.begin_encoding();
                        enc.copy_buffer_to_buffer($t.buffer(), last, &mut staging, 0, 1)?;
                        enc.copy_buffer_to_buffer(&staging, 0, $t.buffer_mut(), slot, 1)?;
                        backend.submit(enc)?;
                    }};
                }
                relocate!(self.positions);
                relocate!(self.kinematics);
                relocate!(self.def_grad);
                relocate!(self.properties);
                relocate!(self.models);
                remaps.push((last as u32, slot as u32));
            }

            // Drop the (now-duplicated) tail element. Removing the last element
            // shifts nothing, so this is O(1). `sorted_ids` is solver scratch
            // (resized by the spatial sort) and is left untouched.
            self.positions.shift_remove(backend, last..)?;
            self.kinematics.shift_remove(backend, last..)?;
            self.def_grad.shift_remove(backend, last..)?;
            self.properties.shift_remove(backend, last..)?;
            self.models.shift_remove(backend, last..)?;
            self.len -= 1;
        }

        backend.write_buffer(self.gpu_len.buffer_mut(), 0, &[self.len as u32])?;
        Ok(remaps)
    }

    /// Reads the current particle world positions back to the CPU.
    pub async fn read_positions(
        &self,
        backend: &GpuBackend,
    ) -> Result<Vec<Vector>, GpuBackendError> {
        // `append` grows the positions buffer to a power-of-two capacity and
        // `slow_read_buffer` reads all of it, so size the destination to the
        // capacity and keep only the `len` live particles.
        let mut data = vec![Position::default(); self.positions.capacity() as usize];
        backend
            .slow_read_buffer(self.positions.buffer(), &mut data)
            .await?;
        data.truncate(self.len);
        Ok(data.iter().map(|p| p.pt).collect())
    }
}
