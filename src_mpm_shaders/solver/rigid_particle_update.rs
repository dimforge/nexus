//! Rigid particle update kernels: transforms sample/shape points from local to world space.

use crate::solver::particle::{Position, RigidParticleIndices};
use crate::{Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use nexus_rbd_shaders::PaddedVector;

/// Transforms rigid body sample points from local space to world space.
///
/// Each thread transforms one sample point by applying the pose of the collider
/// that owns the corresponding rigid particle.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_transform_sample_points(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    rigid_particle_indices: &[RigidParticleIndices],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] local_pts: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] world_pts: &mut [Position],
) {
    let id = invocation_id.x;

    if id < local_pts.len() as u32 {
        let collider_id = rigid_particle_indices.read(id as usize).collider;
        let pose = poses.read(collider_id as usize);
        let local_pt = local_pts.read(id as usize);
        world_pts.write(id as usize, Position::new(pose * local_pt.pt));
    }
}

/// Transforms rigid body shape (collider mesh) vertices from local space to world space.
///
/// Each thread transforms one vertex by applying the pose of the collider
/// identified by the vertex-to-collider mapping.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_transform_shape_points(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] vertex_collider_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] local_pts: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] world_pts: &mut [PaddedVector],
) {
    let id = invocation_id.x;

    if id < local_pts.len() as u32 {
        let collider_id = vertex_collider_ids.read(id as usize);
        let pose = poses.read(collider_id as usize);
        let local_pt = local_pts.read(id as usize);
        world_pts.write(id as usize, (pose * local_pt.0).into());
    }
}
