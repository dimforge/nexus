//! Rigid body particle transformation kernels.

use crate::mpm_shaders::solver::rigid_particle_update::{
    GpuTransformSamplePoints, GpuTransformShapePoints,
};
use crate::solver::GpuRigidParticles;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;
/// GPU kernels for updating rigid body particle positions.
///
/// Transforms surface-sampled particles from local to world coordinates
/// as rigid bodies move.
#[derive(Shader)]
pub struct WgRigidParticleUpdate {
    /// Kernel for transforming sample points.
    transform_sample_points: GpuTransformSamplePoints,
    /// Kernel for transforming collider mesh vertices.
    transform_shape_points: GpuTransformShapePoints,
}

impl WgRigidParticleUpdate {
    /// Transforms rigid body particles from local to world space.
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        bodies: &mut GpuBodySet,
        rigid_particles: &mut GpuRigidParticles,
    ) -> Result<(), GpuBackendError> {
        if rigid_particles.is_empty() {
            return Ok(());
        }

        let sample_len = rigid_particles.local_sample_points.len() as u32;
        self.transform_sample_points.call(
            pass,
            [sample_len, 1, 1],
            &rigid_particles.sample_ids,
            &bodies.poses,
            &rigid_particles.local_sample_points,
            &mut rigid_particles.sample_points,
        )?;

        // Keep the world-space vertex buffer in sync with the body poses; `p2g_cdf`
        // projects world-space grid nodes on these vertices. The BVH AABB entries
        // interleaved in this buffer get transformed as plain points, which is fine:
        // consumers needing the BVH read `shapes_local_vertex_buffers` instead.
        let vtx_len = bodies.shapes_vertex_buffers.len() as u32;
        self.transform_shape_points.call(
            pass,
            [vtx_len, 1, 1],
            &bodies.shapes_vertex_collider_id,
            &bodies.poses,
            &bodies.shapes_local_vertex_buffers,
            &mut bodies.shapes_vertex_buffers,
        )?;

        Ok(())
    }
}
