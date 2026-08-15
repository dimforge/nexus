//! Neo-Hookean elasticity model.

use super::utils::{
    ElasticitySoundSpeedTimestepBound, bulk_modulus_from_lame, shear_modulus_from_lame,
};
use crate::glamx::MatExt;
use crate::{Matrix, Vector};
use khal_std::num_traits::Float;

/// Neo-Hookean hyperelastic constitutive model.
///
/// Computes stress based on the deformation gradient using the
/// neo-Hookean strain energy density function.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NeoHookeanModel {
    pub lambda: f32,
    pub mu: f32,
    pub cfl_coeff: f32,
}

impl NeoHookeanModel {
    /// Computes the Kirchoff stress tensor for the neo-Hookean model.
    #[inline]
    pub fn kirchoff_stress(&self, deformation_gradient: Matrix) -> Matrix {
        let j = f32::max(deformation_gradient.determinant(), 1.0e-10);
        let diag_val = self.lambda * j.ln() - self.mu;
        let mut stress = deformation_gradient * deformation_gradient.transpose() * self.mu;

        stress.x_axis.x += diag_val;
        stress.y_axis.y += diag_val;
        #[cfg(feature = "dim3")]
        {
            stress.z_axis.z += diag_val;
        }

        stress
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
