//! Weakly-compressible fluid model: Tait equation of state plus viscosity.

use super::utils::{ElasticitySoundSpeedTimestepBound, deviatoric_part};
use crate::glamx::MatExt;
use crate::{Matrix, Vector};
use khal_std::num_traits::Float;

/// Weakly-compressible Newtonian fluid.
///
/// The pressure comes from the Tait equation of state, so the fluid resists
/// compression stiffly without requiring a pressure solve; the deviatoric part
/// of the stress is a plain Newtonian viscous term.
///
/// Particles using this model only track the volumetric part of the deformation
/// gradient (see [`MODEL_FLAGS_FLUID`](super::interfaces::MODEL_FLAGS_FLUID)),
/// which avoids the drift a full tensor would accumulate under the large shear
/// a fluid undergoes.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct FluidModel {
    /// Bulk modulus `k` of the equation of state (Pa). Higher values make the
    /// fluid less compressible but shorten the stable timestep.
    pub bulk_modulus: f32,
    /// Stiffness exponent `gamma` of the equation of state (7 for water).
    pub gamma: f32,
    /// Dynamic viscosity (Pa.s).
    pub viscosity: f32,
    /// CFL coefficient scaling the stable timestep.
    pub cfl_coeff: f32,
    /// Stiffness of the tensile branch, as a fraction of `bulk_modulus`.
    ///
    /// A free surface biases the divergence the grid reports (the stencil
    /// reaches into empty cells), so an explicitly integrated volume ratio
    /// ratchets upward every step and the fluid slowly inflates. A soft
    /// pull-back at `J > 1` holds the volume without making the surface clump
    /// the way a full tensile equation of state would.
    pub tensile_stiffness: f32,
}

impl FluidModel {
    /// Pressure as a function of the volume ratio `J`.
    ///
    /// Compression follows the Tait equation of state,
    /// `k ((rho / rho0)^gamma - 1)` with `rho / rho0 = 1 / J`, which stiffens
    /// sharply and keeps the fluid nearly incompressible. Expansion uses a much
    /// softer linear branch instead: the Tait curve is far too strong in tension
    /// and would pull the fluid into blobs, but some restoring force is still
    /// needed (see [`Self::tensile_stiffness`]).
    #[inline]
    pub fn pressure(&self, j: f32) -> f32 {
        let j = f32::max(j, 1.0e-6);
        if j <= 1.0 {
            // `exp(-gamma * ln(j))` rather than `powf`, matching the
            // transcendentals the other models already rely on.
            let ratio = (-self.gamma * j.ln()).exp();
            self.bulk_modulus * (ratio - 1.0)
        } else {
            -self.bulk_modulus * self.tensile_stiffness * (j - 1.0)
        }
    }

    /// Computes the Kirchoff stress `J * (-p I + 2 mu dev(strain_rate))`.
    #[inline]
    pub fn kirchoff_stress(&self, deformation_gradient: Matrix, strain_rate: Matrix) -> Matrix {
        let j = f32::max(deformation_gradient.determinant(), 1.0e-6);
        // Only the deviatoric part of the strain rate contributes: the
        // volumetric response is entirely governed by the equation of state.
        let mut stress = deviatoric_part(strain_rate) * (2.0 * self.viscosity * j);
        let diag_val = -self.pressure(j) * j;

        stress.x_axis.x += diag_val;
        stress.y_axis.y += diag_val;
        #[cfg(feature = "dim3")]
        {
            stress.z_axis.z += diag_val;
        }

        stress
    }

    /// Computes the CFL-based timestep bound from the speed of sound of the
    /// equation of state, `sqrt(gamma * k / rho)`.
    #[inline]
    pub fn timestep_bound(
        &self,
        particle_density0: f32,
        particle_velocity: Vector,
        particle_def_grad_det: f32,
        cell_width: f32,
    ) -> f32 {
        let bound = ElasticitySoundSpeedTimestepBound::new(
            self.cfl_coeff,
            self.bulk_modulus * self.gamma,
            0.0,
        );
        bound.timestep_bound(
            particle_density0,
            particle_def_grad_det,
            particle_velocity,
            cell_width,
        )
    }
}
