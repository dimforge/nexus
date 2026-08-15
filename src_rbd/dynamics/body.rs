//! Rigid-body definitions, mass properties, velocities, and GPU storage.
//!
//! This module provides the core data structures for representing rigid bodies on the GPU,
//! including their poses, velocities, forces, and mass properties. It also provides
//! [`GpuBodySet`] for managing collections of rigid bodies in GPU memory.

use crate::math::{Pose, Vector};
use crate::shapes::ShapeBuffers;

use crate::shaders::dynamics::{LocalMassProperties, Velocity, WorldMassProperties};
use crate::shaders::shapes::Shape;
use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuBackendError};
use vortx::tensor::Tensor;

use crate::shaders::PaddedVector;
/// Re-export types from the shader crate for convenience.
pub use crate::shaders::dynamics::{
    Force, Impulse, LocalMassProperties as GpuLocalMassProperties, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use {
    crate::rapier::dynamics::{RigidBodyHandle, RigidBodySet},
    crate::rapier::geometry::{ColliderHandle, ColliderSet},
    crate::rapier::prelude::MassProperties,
    crate::shapes::shape_from_parry,
};

/// A set of rigid-bodies stored on the gpu.
pub struct GpuBodySet {
    len: u32,
    shapes_data: Vec<Shape>,
    /// World-space mass properties for each body.
    pub mprops: Tensor<WorldMassProperties>,
    /// Local-space mass properties for each body.
    pub local_mprops: Tensor<LocalMassProperties>,
    /// Velocities (linear + angular) for each body.
    pub vels: Tensor<Velocity>,
    /// Poses (position + orientation) for each body.
    pub poses: Tensor<Pose>,
    /// Shape descriptors for each collider.
    pub shapes: Tensor<Shape>,
    /// Vertex positions in local space for each collider shape.
    pub shapes_local_vertex_buffers: Tensor<PaddedVector>,
    /// Vertex positions in world space for each collider shape.
    pub shapes_vertex_buffers: Tensor<PaddedVector>,
    /// Triangle index buffers for each collider shape.
    pub shapes_index_buffers: Tensor<u32>,
    /// Collider ID for each vertex.
    pub shapes_vertex_collider_id: Tensor<u32>,
}

#[derive(Copy, Clone)]
/// Helper struct for defining a rigid-body to be added to a [`GpuBodySet`].
pub struct BodyDesc {
    /// The rigid-body's mass-properties in local-space.
    pub local_mprops: LocalMassProperties,
    /// The rigid-body's mass-properties in world-space.
    pub mprops: WorldMassProperties,
    /// The rigid-body's linear and angular velocities.
    pub vel: Velocity,
    /// The rigid-body's world-space pose.
    pub pose: Pose,
    /// The rigid-body's shape.
    pub shape: Shape,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            local_mprops: Default::default(),
            mprops: Default::default(),
            vel: Default::default(),
            pose: Default::default(),
            shape: Shape::cuboid(Vector::splat(0.5)),
        }
    }
}

/// Coupling mode between a GPU body and the physics simulation.
///
/// This controls whether a body is affected by physics forces or acts as a kinematic body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BodyCoupling {
    /// One-way coupling: the body affects other bodies but is not affected by them.
    ///
    /// This is useful for kinematic bodies that move independently of physics forces.
    OneWay,
    /// Two-way coupling: the body both affects and is affected by other bodies.
    ///
    /// This is the standard mode for dynamic rigid bodies.
    #[default]
    TwoWays,
}

/// Associates a body/collider pair with a coupling mode, one entry per body a
/// [`GpuBodySet`] couples.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BodyCouplingEntry {
    /// The rigid body index.
    pub body: usize,
    /// The collider index.
    pub collider: usize,
    /// The coupling mode for this body.
    pub mode: BodyCoupling,
}

/// Associates a Rapier body/collider pair with a coupling mode. The rapier-typed
/// counterpart of [`BodyCouplingEntry`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RapierBodyCouplingEntry {
    /// The Rapier rigid body handle.
    pub body: RigidBodyHandle,
    /// The Rapier collider handle.
    pub collider: ColliderHandle,
    /// The coupling mode for this body.
    pub mode: BodyCoupling,
}

impl GpuBodySet {
    /// Returns `true` if this set contains no rigid bodies.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of rigid bodies in this set.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Creates a new GPU body set from Rapier rigid bodies and colliders.
    pub fn from_rapier(
        backend: &GpuBackend,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: &[RapierBodyCouplingEntry],
    ) -> Self {
        let mut shape_buffers = ShapeBuffers::default();
        let mut gpu_bodies = vec![];
        let mut pt_collider_ids = vec![];

        for (co_id, coupling) in coupling.iter().enumerate() {
            let co = &colliders[coupling.collider];
            let rb = &bodies[coupling.body];

            let prev_len = shape_buffers.vertices.len();
            let shape =
                shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape type");
            for _ in prev_len..shape_buffers.vertices.len() {
                pt_collider_ids.push(co_id as u32);
            }

            let zero_mprops = MassProperties::default();
            let two_ways_coupling = rb.is_dynamic() && coupling.mode == BodyCoupling::TwoWays;
            let desc = BodyDesc {
                vel: Velocity::new(
                    rb.linvel(),
                    #[cfg(feature = "dim2")]
                    rb.angvel(),
                    #[cfg(feature = "dim3")]
                    rb.angvel(),
                ),
                pose: *rb.position(),
                shape,
                local_mprops: if two_ways_coupling {
                    convert_local_mprops(&rb.mass_properties().local_mprops)
                } else {
                    convert_local_mprops(&zero_mprops)
                },
                mprops: Default::default(),
            };
            gpu_bodies.push(desc);
        }

        Self::new(backend, &gpu_bodies, &pt_collider_ids, &shape_buffers)
    }

    /// Create a set of `bodies` on the gpu.
    pub fn new(
        backend: &GpuBackend,
        bodies: &[BodyDesc],
        pt_collider_ids: &[u32],
        shape_buffers: &ShapeBuffers,
    ) -> Self {
        #[allow(clippy::type_complexity)]
        let (local_mprops, (mprops, (vels, (poses, shapes_data)))): (
            Vec<_>,
            (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))),
        ) = bodies
            .iter()
            .copied()
            .map(|b| (b.local_mprops, (b.mprops, (b.vel, (b.pose, b.shape)))))
            .collect();

        // Avoid empty buffer bindings.
        let vertex_buffer = if !shape_buffers.vertices.is_empty() {
            &shape_buffers.vertices[..]
        } else {
            &[PaddedVector::default()]
        };
        let index_buffer = if !shape_buffers.indices.is_empty() {
            &shape_buffers.indices[..]
        } else {
            &[0, 0, 0]
        };

        // All per-body buffers carry COPY_SRC | COPY_DST so the incremental
        // `append` / `shift_remove` paths can write and relocate slots in place.
        let resizeable = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        Self {
            len: bodies.len() as u32,
            mprops: Tensor::vector(backend, &mprops, resizeable).unwrap(),
            local_mprops: Tensor::vector(backend, &local_mprops, resizeable).unwrap(),
            vels: Tensor::vector(backend, &vels, resizeable).unwrap(),
            poses: Tensor::vector(backend, &poses, resizeable).unwrap(),
            shapes: Tensor::vector(backend, &shapes_data, resizeable).unwrap(),
            shapes_local_vertex_buffers: Tensor::vector(backend, vertex_buffer, resizeable)
                .unwrap(),
            shapes_vertex_buffers: Tensor::vector(backend, vertex_buffer, resizeable).unwrap(),
            shapes_index_buffers: Tensor::vector(backend, index_buffer, resizeable).unwrap(),
            shapes_vertex_collider_id: Tensor::vector(backend, pt_collider_ids, resizeable)
                .unwrap(),
            shapes_data,
        }
    }

    /// Creates an empty body set.
    pub fn empty(backend: &GpuBackend) -> Self {
        Self::new(backend, &[], &[], &ShapeBuffers::default())
    }

    /// Removes a range of body slots from this set, shifting later bodies down
    /// to fill the gap. Returns the number of removed bodies.
    ///
    /// NOTE: this updates the per-body buffers only. Shapes referencing the
    /// shared vertex/index buffers (trimesh, heightfield, polyline) are *not*
    /// compacted, so the orphaned vertices remain allocated. Primitive
    /// (vertex-less) colliders are fully handled.
    pub fn shift_remove(
        &mut self,
        backend: &GpuBackend,
        range: impl std::ops::RangeBounds<usize> + Clone,
    ) -> Result<usize, GpuBackendError> {
        let removed = self.poses.shift_remove(backend, range.clone())?;
        self.vels.shift_remove(backend, range.clone())?;
        self.mprops.shift_remove(backend, range.clone())?;
        self.local_mprops.shift_remove(backend, range.clone())?;
        self.shapes.shift_remove(backend, range.clone())?;

        // Mirror the CPU-side shape cache.
        let start = match range.start_bound() {
            std::ops::Bound::Included(i) => *i,
            std::ops::Bound::Excluded(i) => *i + 1,
            std::ops::Bound::Unbounded => 0,
        };
        self.shapes_data.drain(start..start + removed);
        self.len -= removed as u32;
        Ok(removed)
    }

    /// GPU storage buffer containing the poses of every rigid-body.
    pub fn poses(&self) -> &Tensor<Pose> {
        &self.poses
    }

    /// GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels(&self) -> &Tensor<Velocity> {
        &self.vels
    }

    /// GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops(&self) -> &Tensor<WorldMassProperties> {
        &self.mprops
    }

    /// GPU storage buffer containing the local-space mass-properties of every rigid-body.
    pub fn local_mprops(&self) -> &Tensor<LocalMassProperties> {
        &self.local_mprops
    }

    /// GPU storage buffer containing the shape of every rigid-body.
    pub fn shapes(&self) -> &Tensor<Shape> {
        &self.shapes
    }

    /// Mutable reference to the GPU storage buffer containing the poses of every rigid-body.
    pub fn poses_mut(&mut self) -> &mut Tensor<Pose> {
        &mut self.poses
    }

    /// Mutable reference to the GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels_mut(&mut self) -> &mut Tensor<Velocity> {
        &mut self.vels
    }

    /// Mutable reference to the GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops_mut(&mut self) -> &mut Tensor<WorldMassProperties> {
        &mut self.mprops
    }

    /// Returns the GPU buffer containing shape vertices in world-space.
    ///
    /// This buffer is updated each frame as bodies move.
    pub fn shapes_vertex_buffers(&self) -> &Tensor<PaddedVector> {
        &self.shapes_vertex_buffers
    }

    /// Mutable reference to the GPU buffer containing shape vertices in world-space.
    pub fn shapes_vertex_buffers_mut(&mut self) -> &mut Tensor<PaddedVector> {
        &mut self.shapes_vertex_buffers
    }

    /// Returns the GPU buffer containing shape vertices in local-space.
    ///
    /// These are the original vertex positions before transformation.
    pub fn shapes_local_vertex_buffers(&self) -> &Tensor<PaddedVector> {
        &self.shapes_local_vertex_buffers
    }

    /// Returns the GPU buffer mapping each vertex to its collider ID.
    pub fn shapes_vertex_collider_id(&self) -> &Tensor<u32> {
        &self.shapes_vertex_collider_id
    }

    /// Returns a CPU-side slice of the shape data.
    ///
    /// Useful for accessing shape information without GPU readback.
    pub fn shapes_data(&self) -> &[Shape] {
        &self.shapes_data
    }
}

impl GpuBodySet {
    /// Appends rigid-bodies (converted from rapier) to this set and returns the
    /// indices of the newly inserted bodies.
    ///
    /// Only primitive (vertex-less) colliders are supported by this incremental
    /// path; mesh-based colliders (trimesh / heightfield / polyline) would
    /// require growing the shared vertex/index buffers and offsetting the shape
    /// references, which isn't handled yet.
    pub fn append_rapier(
        &mut self,
        backend: &GpuBackend,
        bodies: &[(
            crate::rapier::dynamics::RigidBody,
            crate::rapier::geometry::Collider,
            BodyCoupling,
        )],
    ) -> Result<Vec<u32>, GpuBackendError> {
        let start = self.len;
        let mut poses = Vec::with_capacity(bodies.len());
        let mut vels = Vec::with_capacity(bodies.len());
        let mut mprops = Vec::with_capacity(bodies.len());
        let mut local_mprops = Vec::with_capacity(bodies.len());
        let mut shapes = Vec::with_capacity(bodies.len());

        for (rb, co, coupling) in bodies {
            let mut shape_buffers = ShapeBuffers::default();
            let shape =
                shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape type");
            assert!(
                shape_buffers.vertices.is_empty(),
                "GpuBodySet::append_rapier currently supports primitive (vertex-less) colliders only."
            );

            let two_ways_coupling = rb.is_dynamic() && *coupling == BodyCoupling::TwoWays;
            let local = if two_ways_coupling {
                convert_local_mprops(&rb.mass_properties().local_mprops)
            } else {
                convert_local_mprops(&MassProperties::default())
            };

            poses.push(*rb.position());
            vels.push(Velocity::new(
                rb.linvel(),
                #[cfg(feature = "dim2")]
                rb.angvel(),
                #[cfg(feature = "dim3")]
                rb.angvel(),
            ));
            mprops.push(WorldMassProperties::default());
            local_mprops.push(local);
            shapes.push(shape);
        }

        self.poses.append(backend, &poses)?;
        self.vels.append(backend, &vels)?;
        self.mprops.append(backend, &mprops)?;
        self.local_mprops.append(backend, &local_mprops)?;
        self.shapes.append(backend, &shapes)?;
        self.shapes_data.extend_from_slice(&shapes);
        self.len += bodies.len() as u32;

        Ok((start..self.len).collect())
    }
}

fn convert_local_mprops(mprops: &MassProperties) -> LocalMassProperties {
    #[cfg(feature = "dim2")]
    {
        LocalMassProperties {
            inv_mass: glamx::Vec2::splat(mprops.inv_mass),
            com: mprops.local_com,
            padding2: 0,
            inv_inertia: mprops.inv_principal_inertia,
        }
    }
    #[cfg(feature = "dim3")]
    {
        LocalMassProperties {
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
