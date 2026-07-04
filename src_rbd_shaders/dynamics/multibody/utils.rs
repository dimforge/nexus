//! Small math / coordinate helpers shared across multibody kernels.

use crate::dynamics::joint::{ANG_AXES_MASK, LIN_AXES_MASK};
use crate::{DIM, Pose, Rotation, Vector};
use khal_std::index::MaybeIndexUnchecked;
use parry::math::VectorExt;

use super::types::{MAX_JOINT_DOFS, MultibodyLinkStatic};

/// Number of free DOFs implied by a `locked_axes` bitmask.
#[inline]
pub fn count_free_dofs(locked: u32) -> u32 {
    crate::dynamics::joint::SPATIAL_DIM as u32
        - (locked & (LIN_AXES_MASK | ANG_AXES_MASK)).count_ones()
}

/// Number of free linear DOFs (bits 0..DIM).
#[inline]
pub fn count_free_lin_dofs(locked: u32) -> u32 {
    DIM - (locked & LIN_AXES_MASK).count_ones()
}

/// Number of free angular DOFs.
#[inline]
pub fn count_free_ang_dofs(locked: u32) -> u32 {
    crate::ANG_DIM - (locked & ANG_AXES_MASK).count_ones()
}

impl MultibodyLinkStatic {
    /// Compute the link's `local_to_parent` pose given its current joint coords/rotation.
    ///
    /// Mirrors rapier's `MultibodyJoint::body_to_parent`.
    #[inline]
    pub fn body_to_parent(&self, joint_rot: Rotation, coords: &[f32; MAX_JOINT_DOFS]) -> Pose {
        let locked = self.data.locked_axes;
        let mut transform =
            Pose::from_parts(Vector::ZERO, joint_rot) * self.data.local_frame_b.inverse();

        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                let t = Vector::ith(i as usize, coords.read(i as usize));
                transform = Pose::from_parts(t, Rotation::IDENTITY) * transform;
            }
        }

        self.data.local_frame_a * transform
    }
}
