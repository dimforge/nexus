//! Typed ordinary-body inputs shared with alternative execution backends.
use super::RbdState;
use crate::math::Pose;
use crate::shaders::{
    PaddedVector,
    dynamics::{ImpulseJoint, LocalMassProperties, RbdSimParams, Velocity},
    queries::ColliderMaterial,
    shapes::Shape,
    utils::BatchIndices,
};
use khal::backend::{Backend, GpuBackend, GpuBackendError, GpuBuffer};
#[cfg(any(feature = "webgpu", feature = "metal", feature = "cuda"))]
use khal::backend::{Buffer, Encoder};

/// Native typed inputs for starting an ordinary-body simulation from current state.
///
/// Arrays retain Nexus's batch-interleaved layout and allocated padding. Body and
/// collider indices are distinct; use `batch_indices` and `collider_parents` rather
/// than assuming one shared ordering. Geometry indices and joint coloring are the
/// original Nexus conversion results. No shape or joint conversion is duplicated.
///
/// This is a cold-start input, not a continuation checkpoint: contact impulses,
/// recycling, coloring and solver scratch are rebuilt by the receiving backend.
/// Articulated multibodies are outside this representation.
#[derive(Clone)]
pub struct RbdSceneUpload {
    /// Active counts, per-batch stride and native capacities.
    pub batch_indices: BatchIndices,
    /// Native solver parameters; `dt` is the substep duration.
    pub params: RbdSimParams,
    /// The native gravity uniform, including its padding component.
    pub gravity: glamx::Vec4,
    /// Current body poses at the body origin, including allocated padding.
    pub body_poses: Vec<Pose>,
    /// Current body velocities, including allocated padding.
    pub velocities: Vec<Velocity>,
    /// Local centers of mass, inverse masses and principal inertia.
    pub local_mass: Vec<LocalMassProperties>,
    /// Collider poses relative to their parent bodies.
    pub collider_local_poses: Vec<Pose>,
    /// Native global batch-interleaved parent indices, in collider order.
    pub collider_parents: Vec<u32>,
    /// Converted shape descriptors referring to `vertices` and `indices`.
    pub shapes: Vec<Shape>,
    /// Per-collider friction, restitution and combination rules.
    pub materials: Vec<ColliderMaterial>,
    /// Native interaction-group membership, filters and test modes.
    pub groups: Vec<crate::rapier::geometry::InteractionGroups>,
    /// Native same-body and adjacent-link exclusion keys.
    pub pair_filters: Vec<[u32; 2]>,
    /// Shared native geometry vertices.
    pub vertices: Vec<PaddedVector>,
    /// Shared native mesh/polyline/convex geometry indices.
    pub indices: Vec<u32>,
    /// Converted ordinary joints in the native batch-interleaved color order.
    pub joints: Vec<ImpulseJoint>,
    /// Per-batch allocated joint count, distinct from the active joint count.
    pub joints_per_batch: u32,
    /// Native exclusive end of each joint color, shared by all batches.
    pub joint_color_ends: Vec<u32>,
}

impl RbdState {
    /// Read native inputs for a cold-start simulation on another execution backend.
    ///
    /// This does not mutate or step the state. Submitted work must be complete
    /// before calling, and callers must not mutate its buffers concurrently. The
    /// receiving backend must validate its supported features and rebuild solver
    /// scratch and contact persistence. This method rejects articulated multibodies
    /// instead of silently discarding their dynamics.
    pub async fn read_scene_upload(
        &self,
        backend: &GpuBackend,
    ) -> Result<RbdSceneUpload, GpuBackendError> {
        if !self.multibodies.is_empty() {
            return Err(GpuBackendError::Unsupported(
                "ordinary scene upload does not include articulated multibodies",
            ));
        }
        let batch_indices = backend.slow_read_vec(self.batch_indices.buffer()).await?[0];
        let gravity = backend.slow_read_vec(self.gravity.buffer()).await?[0];
        let joints = self.joints.read_upload(backend).await?;
        Ok(RbdSceneUpload {
            batch_indices,
            params: self.sim_params_cpu,
            gravity,
            body_poses: backend.slow_read_vec(self.body_poses.buffer()).await?,
            velocities: backend.slow_read_vec(self.vels.buffer()).await?,
            local_mass: backend.slow_read_vec(self.local_mprops.buffer()).await?,
            collider_local_poses: backend
                .slow_read_vec(self.collider_local_poses.buffer())
                .await?,
            collider_parents: backend.slow_read_vec(self.collider_parent.buffer()).await?,
            shapes: backend.slow_read_vec(self.shapes.buffer()).await?,
            materials: backend
                .slow_read_vec(self.collider_materials.buffer())
                .await?,
            groups: read_groups(backend, self.collision_groups.buffer()).await?,
            pair_filters: backend.slow_read_vec(self.pair_filter.buffer()).await?,
            vertices: backend.slow_read_vec(self.vertex_buffers.buffer()).await?,
            indices: backend.slow_read_vec(self.index_buffers.buffer()).await?,
            joints,
            joints_per_batch: self.joints.joints_per_batch(),
            joint_color_ends: self.joints.color_ends().to_vec(),
        })
    }
}

// InteractionGroups contains an enum and is NoUninit, not AnyBitPattern. Read
// plain words first and validate that enum instead of constructing unchecked Rust
// values from device memory. The staging copy also supports private GPU storage.
async fn read_groups(
    backend: &GpuBackend,
    source: &GpuBuffer<crate::rapier::geometry::InteractionGroups>,
) -> Result<Vec<crate::rapier::geometry::InteractionGroups>, GpuBackendError> {
    #[cfg(feature = "cpu")]
    if let (GpuBackend::Cpu, GpuBuffer::Cpu(values)) = (backend, source) {
        return Ok(values.clone());
    }
    #[cfg(any(feature = "webgpu", feature = "metal", feature = "cuda"))]
    {
        read_groups_gpu(backend, source).await
    }
    #[cfg(not(any(feature = "webgpu", feature = "metal", feature = "cuda")))]
    {
        let _ = (backend, source);
        Err(GpuBackendError::Unsupported(
            "scene group readback backend mismatch",
        ))
    }
}

#[cfg(any(feature = "webgpu", feature = "metal", feature = "cuda"))]
async fn read_groups_gpu(
    backend: &GpuBackend,
    source: &GpuBuffer<crate::rapier::geometry::InteractionGroups>,
) -> Result<Vec<crate::rapier::geometry::InteractionGroups>, GpuBackendError> {
    use crate::rapier::geometry::{Group, InteractionGroups, InteractionTestMode};
    const {
        assert!(size_of::<InteractionGroups>() == 12);
        assert!(std::mem::offset_of!(InteractionGroups, memberships) == 0);
        assert!(std::mem::offset_of!(InteractionGroups, filter) == 4);
        assert!(std::mem::offset_of!(InteractionGroups, test_mode) == 8);
    }
    let len = source.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut staging = backend.uninit_buffer::<InteractionGroups>(
        len,
        khal::BufferUsages::COPY_DST | khal::BufferUsages::MAP_READ,
    )?;
    let mut encoder = backend.begin_encoding();
    encoder.copy_buffer_to_buffer(source, 0, &mut staging, 0, len)?;
    backend.submit(encoder)?;
    backend.synchronize()?;
    let mut words = vec![[0u32; 3]; len];
    match (backend, &staging) {
        #[cfg(feature = "webgpu")]
        (GpuBackend::WebGpu(device), GpuBuffer::WebGpu(buffer)) => {
            device.read_buffer(buffer, &mut words).await?;
        }
        #[cfg(feature = "metal")]
        (GpuBackend::Metal(_), GpuBuffer::Metal(buffer)) => {
            // SAFETY: MAP_READ requests host-visible staging, the completed copy
            // wrote exactly len complete records, and the separately allocated
            // words have exactly that byte length. No enum is read through this
            // pointer; all bit patterns of u32 are valid.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.raw().contents().cast::<u8>(),
                    words.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(words.as_slice()),
                );
            }
        }
        #[cfg(feature = "cuda")]
        (GpuBackend::Cuda(device), GpuBuffer::Cuda(buffer)) => {
            if let Some(bytes) = buffer.inner() {
                device
                    .stream()
                    .memcpy_dtoh(bytes, bytemuck::cast_slice_mut(&mut words))
                    .map_err(|error| GpuBackendError::Cuda(error.into()))?;
                device.synchronize()?;
            }
        }
        _ => {
            return Err(GpuBackendError::Unsupported(
                "scene group readback backend mismatch",
            ));
        }
    }
    words
        .into_iter()
        .map(|[memberships, filter, mode]| {
            let mode = match mode {
                0 => InteractionTestMode::And,
                1 => InteractionTestMode::Or,
                _ => {
                    return Err(GpuBackendError::Unsupported(
                        "invalid scene interaction test mode",
                    ));
                }
            };
            Ok(InteractionGroups::new(
                Group::from_bits_retain(memberships),
                Group::from_bits_retain(filter),
                mode,
            ))
        })
        .collect()
}
