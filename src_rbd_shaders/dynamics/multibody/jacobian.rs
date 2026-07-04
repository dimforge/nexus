//! Body jacobian assembly.
//!
//! For each link i, the `SPATIAL_DIM × ndofs` body jacobian J_i maps generalized
//! velocities to the link's world-frame spatial velocity at its COM.

use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;
use crate::utils::linalg::MatSlice;
use crate::{ANG_DIM, DIM, Rotation, Vector};
use parry::math::VectorExt;

use super::types::MultibodyLinkStatic;

impl MultibodyLinkStatic {
    /// Adds this joint's jacobian (world-frame) to the first `ndofs` columns of
    /// `view` inside `out` (a `+=` of the joint jacobian on `view`'s columns).
    ///
    /// Each free DOF column is written by exactly one lane (the one whose ID
    /// matches `curr_free_dof`); lanes outside `[0, total_free)` pass through with
    /// no writes. Structured this way to avoid subtle rust-gpu lowering bugs.
    #[inline]
    pub(super) fn joint_jacobian_accumulate_par(
        &self,
        transform_rot: Rotation,
        out: &mut [f32],
        view: MatSlice,
        lane: u32,
        _lanes: u32,
    ) {
        let locked = self.data.locked_axes;
        let mut curr_free_dof = 0u32;

        // Linear DOFs (axis order). Only the linear rows are nonzero.
        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                if lane == curr_free_dof {
                    let axis = transform_rot * Vector::ith(i as usize, 1.0);
                    #[cfg(feature = "dim3")]
                    {
                        let i0 = view.idx(0, curr_free_dof);
                        let i1 = view.idx(1, curr_free_dof);
                        let i2 = view.idx(2, curr_free_dof);
                        out.write(i0, out.read(i0) + axis.x);
                        out.write(i1, out.read(i1) + axis.y);
                        out.write(i2, out.read(i2) + axis.z);
                    }
                    #[cfg(feature = "dim2")]
                    {
                        let i0 = view.idx(0, curr_free_dof);
                        let i1 = view.idx(1, curr_free_dof);
                        out.write(i0, out.read(i0) + axis.x);
                        out.write(i1, out.read(i1) + axis.y);
                    }
                }
                curr_free_dof += 1;
            }
        }

        // Angular DOFs.
        let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
        let num_ang = ANG_DIM - ang_locked.count_ones();
        if num_ang == 1 {
            if lane == curr_free_dof {
                #[cfg(feature = "dim3")]
                {
                    let dof_id = (!ang_locked & 0x7).trailing_zeros();
                    let axis = transform_rot * Vector::ith(dof_id as usize, 1.0);
                    let i3 = view.idx(3, curr_free_dof);
                    let i4 = view.idx(4, curr_free_dof);
                    let i5 = view.idx(5, curr_free_dof);
                    out.write(i3, out.read(i3) + axis.x);
                    out.write(i4, out.read(i4) + axis.y);
                    out.write(i5, out.read(i5) + axis.z);
                }
                #[cfg(feature = "dim2")]
                {
                    let i2 = view.idx(2, curr_free_dof);
                    out.write(i2, out.read(i2) + 1.0);
                }
            }
        } else if num_ang == 3 {
            #[cfg(feature = "dim3")]
            {
                let rotmat = rotation_to_matrix(transform_rot);
                for k in 0..3u32 {
                    if lane == curr_free_dof + k {
                        let axis = if k == 0 {
                            rotmat.x_axis
                        } else if k == 1 {
                            rotmat.y_axis
                        } else {
                            rotmat.z_axis
                        };
                        let i3 = view.idx(3, curr_free_dof + k);
                        let i4 = view.idx(4, curr_free_dof + k);
                        let i5 = view.idx(5, curr_free_dof + k);
                        out.write(i3, out.read(i3) + axis.x);
                        out.write(i4, out.read(i4) + axis.y);
                        out.write(i5, out.read(i5) + axis.z);
                    }
                }
            }
        } // TODO: num_ang == 2
    }

    /// Computes column `c` (`0 ≤ c < stat.ndofs`) of this joint's body jacobian as
    /// the pair `(linear_part, angular_part)`, in registers. In 3D, returns
    /// `(Vec3, Vec3)`. In 2D, returns `(Vec2, f32)`.
    ///
    /// Single-exit form (no early `return` inside the loop) — rust-gpu's
    /// structured control flow lowering can produce silently incorrect results
    /// for `return` statements in the middle of a `for` loop.
    #[cfg(feature = "dim3")]
    #[inline]
    pub(super) fn joint_jacobian_column(
        &self,
        transform_rot: Rotation,
        c: u32,
    ) -> (Vector, glamx::Vec3) {
        let locked = self.data.locked_axes;
        let mut curr_free_dof = 0u32;
        let mut result_lin = Vector::ZERO;
        let mut result_ang = glamx::Vec3::ZERO;

        // Linear DOFs in axis order.
        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                if curr_free_dof == c {
                    result_lin = transform_rot * Vector::ith(i as usize, 1.0);
                }
                curr_free_dof += 1;
            }
        }

        // Angular DOFs.
        let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
        let num_ang = ANG_DIM - ang_locked.count_ones();
        if num_ang == 1 {
            if curr_free_dof == c {
                let dof_id = (!ang_locked & 0x7).trailing_zeros();
                result_ang = transform_rot * Vector::ith(dof_id as usize, 1.0);
            }
        } else if num_ang == 3 {
            let rotmat = rotation_to_matrix(transform_rot);
            for k in 0..3u32 {
                if curr_free_dof + k == c {
                    result_ang = if k == 0 {
                        rotmat.x_axis
                    } else if k == 1 {
                        rotmat.y_axis
                    } else {
                        rotmat.z_axis
                    };
                }
            }
        }

        (result_lin, result_ang)
    }

    #[cfg(feature = "dim2")]
    #[inline]
    pub(super) fn joint_jacobian_column(&self, transform_rot: Rotation, c: u32) -> (Vector, f32) {
        let locked = self.data.locked_axes;
        let mut curr_free_dof = 0u32;
        let mut result_lin = Vector::ZERO;
        let mut result_ang = 0.0f32;

        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                if curr_free_dof == c {
                    result_lin = transform_rot * Vector::ith(i as usize, 1.0);
                }
                curr_free_dof += 1;
            }
        }
        let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
        let num_ang = ANG_DIM - ang_locked.count_ones();
        if num_ang == 1 && curr_free_dof == c {
            result_ang = 1.0;
        }
        (result_lin, result_ang)
    }
}
