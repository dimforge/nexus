//! Rigid-body definition and set.

use crate::math::{AngularInertia, GpuSim};
use crate::shapes::{GpuShape, ShapeBuffers};
use num_traits::Zero;
use rapier::geometry::ColliderHandle;
use rapier::math::{AngVector, Point, Vector};
use rapier::prelude::MassProperties;
use rapier::{
    dynamics::{RigidBodyHandle, RigidBodySet},
    geometry::ColliderSet,
};
use slang_hal::backend::Backend;
use gla::tensor::GpuTensor;
use wgpu::BufferUsages;

#[derive(Copy, Clone, PartialEq, encase::ShaderType)]
#[repr(C)]
/// Linear and angular forces with a layout compatible with the corresponding WGSL struct.
pub struct GpuForce {
    /// The linear part of the force.
    pub linear: Vector<f32>,
    /// The angular part of the force (aka. the torque).
    pub angular: AngVector<f32>,
}

#[derive(Copy, Clone, PartialEq, Default, encase::ShaderType)]
#[repr(C)]
/// Linear and angular velocities with a layout compatible with the corresponding WGSL struct.
pub struct GpuVelocity {
    /// The linear (translational) velocity.
    pub linear: Vector<f32>,
    /// The angular (rotational) velocity.
    pub angular: AngVector<f32>,
}

#[derive(Copy, Clone, PartialEq, encase::ShaderType)]
#[repr(C)]
/// Rigid-body mass-properties, with a layout compatible with the corresponding WGSL struct.
pub struct GpuMassProperties {
    /// The inverse angular inertia tensor.
    pub inv_inertia: AngularInertia<f32>,
    /// The inverse mass.
    pub inv_mass: Vector<f32>,
    /// The center-of-mass.
    pub com: Vector<f32>, // ShaderType isn’t implemented for Point
}

impl From<MassProperties> for GpuMassProperties {
    fn from(props: MassProperties) -> Self {
        GpuMassProperties {
            #[cfg(feature = "dim2")]
            inv_inertia: props.inv_principal_inertia_sqrt * props.inv_principal_inertia_sqrt,
            #[cfg(feature = "dim3")]
            inv_inertia: props.reconstruct_inverse_inertia_matrix(),
            inv_mass: Vector::repeat(props.inv_mass),
            com: props.local_com.coords,
        }
    }
}

impl Default for GpuMassProperties {
    fn default() -> Self {
        GpuMassProperties {
            #[rustfmt::skip]
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
            #[cfg(feature = "dim3")]
            inv_inertia: AngularInertia::identity(),
            inv_mass: Vector::repeat(1.0),
            com: Vector::zeros(),
        }
    }
}

/// A set of rigid-bodies stored on the gpu.
pub struct GpuBodySet<B: Backend> {
    len: u32,
    shapes_data: Vec<GpuShape>, // TODO: exists only for convenience in the MPM simulation.
    pub(crate) mprops: GpuTensor<GpuMassProperties, B>,
    pub(crate) local_mprops: GpuTensor<GpuMassProperties, B>,
    pub(crate) vels: GpuTensor<GpuVelocity, B>,
    pub(crate) poses: GpuTensor<GpuSim, B>,
    // TODO: support other shape types.
    // TODO: support a shape with a shift relative to the body.
    pub(crate) shapes: GpuTensor<GpuShape, B>,
    // TODO: it’s a bit weird that we store the vertex buffer but not the
    //       index buffer. This is because our only use-case currently
    //       is from wgsparkl which has its own way of storing indices.
    pub(crate) shapes_local_vertex_buffers: GpuTensor<Point<f32>, B>,
    pub(crate) shapes_vertex_buffers: GpuTensor<Point<f32>, B>,
    pub(crate) shapes_vertex_collider_id: GpuTensor<u32, B>, // NOTE: this is a bit of a hack for wgsparkl
}

#[derive(Copy, Clone)]
/// Helper struct for defining a rigid-body to be added to a [`GpuBodySet`].
pub struct BodyDesc {
    /// The rigid-body’s mass-properties in local-space.
    pub local_mprops: GpuMassProperties,
    /// The rigid-body’s mass-properties in world-space.
    pub mprops: GpuMassProperties,
    /// The rigid-body’s linear and angular velocities.
    pub vel: GpuVelocity,
    /// The rigid-body’s world-space pose.
    pub pose: GpuSim,
    /// The rigid-body’s shape.
    pub shape: GpuShape,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            local_mprops: Default::default(),
            mprops: Default::default(),
            vel: Default::default(),
            pose: Default::default(),
            shape: GpuShape::cuboid(Vector::repeat(0.5)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BodyCoupling {
    OneWay,
    #[default]
    TwoWays,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BodyCouplingEntry {
    pub body: RigidBodyHandle,
    pub collider: ColliderHandle,
    pub mode: BodyCoupling,
}

impl<B: Backend> GpuBodySet<B> {
    /// Is this set empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of rigid-bodies in this set.
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn from_rapier(
        backend: &B,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: &[BodyCouplingEntry],
    ) -> Result<Self, B::Error> {
        let mut shape_buffers = ShapeBuffers::default();
        let mut gpu_bodies = vec![];
        let mut pt_collider_ids = vec![];

        for (co_id, coupling) in coupling.iter().enumerate() {
            let co = &colliders[coupling.collider];
            let rb = &bodies[coupling.body];

            let prev_len = shape_buffers.vertices.len();
            let shape = GpuShape::from_parry(co.shape(), &mut shape_buffers)
                .expect("Unsupported shape type");
            for _ in prev_len..shape_buffers.vertices.len() {
                pt_collider_ids.push(co_id as u32);
            }

            let zero_mprops = MassProperties::zero();
            let two_ways_coupling = rb.is_dynamic() && coupling.mode == BodyCoupling::TwoWays;
            let desc = BodyDesc {
                vel: GpuVelocity {
                    linear: *rb.linvel(),
                    #[allow(clippy::clone_on_copy)] // Needed for 2D/3D switch.
                    angular: rb.angvel().clone(),
                },
                #[cfg(feature = "dim2")]
                pose: (*rb.position()).into(),
                #[cfg(feature = "dim3")]
                pose: GpuSim::from_isometry(*rb.position(), 1.0),
                shape,
                local_mprops: if two_ways_coupling {
                    rb.mass_properties().local_mprops.into()
                } else {
                    zero_mprops.into()
                },
                mprops: if two_ways_coupling {
                    rb.mass_properties()
                        .local_mprops
                        .transform_by(rb.position())
                        .into()
                } else {
                    zero_mprops.into()
                },
            };
            gpu_bodies.push(desc);
        }

        Self::new(backend, &gpu_bodies, &pt_collider_ids, &shape_buffers)
    }

    /// Create a set of `bodies` on the gpu.
    pub fn new(
        backend: &B,
        bodies: &[BodyDesc],
        pt_collider_ids: &[u32],
        shape_buffers: &ShapeBuffers,
    ) -> Result<Self, B::Error> {
        #[allow(clippy::type_complexity)]
        let (local_mprops, (mprops, (vels, (poses, shapes_data)))): (
            Vec<_>,
            (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))),
        ) = bodies
            .iter()
            .copied()
            // NOTE: Looks silly, but we can’t just collect into (Vec, Vec, Vec).
            .map(|b| (b.local_mprops, (b.mprops, (b.vel, (b.pose, b.shape)))))
            .collect();
        // TODO: (api design) how can we let the user pick the buffer usages?
        Ok(Self {
            len: bodies.len() as u32,
            mprops: GpuTensor::vector_encased(backend, &mprops, BufferUsages::STORAGE)?,
            local_mprops: GpuTensor::vector_encased(backend, &local_mprops, BufferUsages::STORAGE)?,
            vels: GpuTensor::vector_encased(
                backend,
                &vels,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            poses: GpuTensor::vector(
                backend,
                &poses,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )?,
            shapes: GpuTensor::vector(backend, &shapes_data, BufferUsages::STORAGE)?,
            shapes_local_vertex_buffers: GpuTensor::vector_encased(
                backend,
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            )?,
            shapes_vertex_buffers: GpuTensor::vector_encased(
                backend,
                // TODO: init in world-space directly?
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            )?,
            shapes_vertex_collider_id: GpuTensor::vector(
                backend,
                pt_collider_ids,
                BufferUsages::STORAGE,
            )?,
            shapes_data,
        })
    }

    /// GPU storage buffer containing the poses of every rigid-body.
    pub fn poses(&self) -> &GpuTensor<GpuSim, B> {
        &self.poses
    }

    /// GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels(&self) -> &GpuTensor<GpuVelocity, B> {
        &self.vels
    }

    /// GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops(&self) -> &GpuTensor<GpuMassProperties, B> {
        &self.mprops
    }

    /// GPU storage buffer containing the local-space mass-properties of every rigid-body.
    pub fn local_mprops(&self) -> &GpuTensor<GpuMassProperties, B> {
        &self.local_mprops
    }

    /// GPU storage buffer containing the shape of every rigid-body.
    pub fn shapes(&self) -> &GpuTensor<GpuShape, B> {
        &self.shapes
    }

    pub fn shapes_vertex_buffers(&self) -> &GpuTensor<Point<f32>, B> {
        &self.shapes_vertex_buffers
    }

    pub fn shapes_local_vertex_buffers(&self) -> &GpuTensor<Point<f32>, B> {
        &self.shapes_local_vertex_buffers
    }

    pub fn shapes_vertex_collider_id(&self) -> &GpuTensor<u32, B> {
        &self.shapes_vertex_collider_id
    }

    pub fn shapes_data(&self) -> &[GpuShape] {
        &self.shapes_data
    }
}
