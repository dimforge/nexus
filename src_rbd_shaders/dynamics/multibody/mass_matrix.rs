#[cfg(feature = "dim3")]
use glamx::Mat3;

use crate::dynamics::body::LocalMassProperties;
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;

use super::types::MultibodyLinkWorkspace;

impl MultibodyLinkWorkspace {
    /// World-space inertia for this link.
    ///
    /// In 3D returns a `Mat3` (`I_world = R · diag(principal_inertia) · Rᵀ`). In 2D
    /// returns the scalar moment of inertia (already in world frame because there
    /// is only one rotational DOF).
    #[cfg(feature = "dim3")]
    #[inline]
    pub(super) fn link_world_inertia(&self, lmp: &LocalMassProperties) -> Mat3 {
        let ipi = lmp.inv_principal_inertia;
        let px = if ipi.x != 0.0 { 1.0 / ipi.x } else { 0.0 };
        let py = if ipi.y != 0.0 { 1.0 / ipi.y } else { 0.0 };
        let pz = if ipi.z != 0.0 { 1.0 / ipi.z } else { 0.0 };
        let r = rotation_to_matrix(self.local_to_world.rotation * lmp.inertia_ref_frame);
        // M = r · diag(px, py, pz) (column-scale); I = M · rᵀ.
        let m = Mat3::from_cols(r.x_axis * px, r.y_axis * py, r.z_axis * pz);
        m * r.transpose()
    }

    #[cfg(feature = "dim2")]
    #[inline]
    pub(super) fn link_world_inertia(&self, lmp: &LocalMassProperties) -> f32 {
        if lmp.inv_inertia != 0.0 {
            1.0 / lmp.inv_inertia
        } else {
            0.0
        }
    }
}
