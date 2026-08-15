//! Linear (corotated) elasticity model.

use super::utils::{
    ElasticitySoundSpeedTimestepBound, bulk_modulus_from_lame, shear_modulus_from_lame,
};
use crate::glamx::MatExt;
use crate::{Matrix, Vector, diag};

/// Corotated linear elastic constitutive model.
///
/// Uses SVD-based corotated formulation for computing the Kirchoff stress.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct LinearElasticModel {
    pub lambda: f32,
    pub mu: f32,
    pub cfl_coeff: f32,
}

impl LinearElasticModel {
    /// Computes the Kirchoff stress tensor using corotated linear elasticity.
    ///
    /// Performs SVD on the deformation gradient, computes the strain from the
    /// singular values, and assembles the stress tensor.
    #[inline]
    pub fn kirchoff_stress(&self, deformation_gradient: Matrix) -> Matrix {
        let svd = deformation_gradient.svd();

        #[cfg(feature = "dim2")]
        let j = svd.s.x * svd.s.y;
        #[cfg(feature = "dim3")]
        let j = svd.s.x * svd.s.y * svd.s.z;

        let modified_s = svd.s - Vector::ONE;

        // Recompose with modified singular values: U * diag(modified_s) * Vt
        let recomposed = svd.u * diag(modified_s) * svd.vt;

        let diag_val = self.lambda * (j - 1.0) * j;
        let mut result = recomposed * deformation_gradient.transpose() * (2.0 * self.mu);

        result.x_axis.x += diag_val;
        result.y_axis.y += diag_val;
        #[cfg(feature = "dim3")]
        {
            result.z_axis.z += diag_val;
        }

        result
    }

    /// Computes the CFL-based timestep bound for this elastic model.
    #[inline]
    pub fn timestep_bound(
        &self,
        particle_density0: f32,
        particle_velocity: Vector,
        particle_def_grad_det: f32,
        elastic_hardening: f32,
        cell_width: f32,
    ) -> f32 {
        let bulk_modulus = bulk_modulus_from_lame(self.lambda, self.mu);
        let shear_modulus = shear_modulus_from_lame(self.lambda, self.mu);

        let bound = ElasticitySoundSpeedTimestepBound::new(
            self.cfl_coeff,
            bulk_modulus * elastic_hardening,
            shear_modulus * elastic_hardening,
        );
        bound.timestep_bound(
            particle_density0,
            particle_def_grad_det,
            particle_velocity,
            cell_width,
        )
    }

    /// Computes the positive part of the elastic energy density, keeping only
    /// the tensile contributions.
    #[inline]
    pub fn pos_energy_density(&self, def_grad: Matrix, elastic_hardening: f32) -> f32 {
        let j = def_grad.determinant();
        let svd = def_grad.svd();

        let sig = (svd.s - Vector::ONE).max(Vector::ZERO);
        let pos_dev_part = self.mu * elastic_hardening * sig.dot(sig);
        let spherical_part = self.lambda * elastic_hardening * 0.5 * (j - 1.0) * (j - 1.0);

        if j < 1.0 {
            pos_dev_part
        } else {
            pos_dev_part + spherical_part
        }
    }
}
