//! Material constitutive models for MPM particles.
//!
//! This module provides material models that define how particles respond to deformation.
//! The actual model implementations live in the shader crate; this module re-exports
//! them and provides CPU-side convenience constructors.

pub use crate::mpm_shaders::models::drucker_prager::{
    DruckerPragerPlasticState, DruckerPragerPlasticity,
};
pub use crate::mpm_shaders::models::fluid::FluidModel;
pub use crate::mpm_shaders::models::linear_elasticity::LinearElasticModel;
pub use crate::mpm_shaders::models::snow::{SnowPlasticState, SnowPlasticity};

pub use drucker_prager::DruckerPrager;

mod drucker_prager;

/// Computes Lamé parameters (λ, μ) from Young's modulus and Poisson's ratio.
pub(crate) fn lame_lambda_mu(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        shear_modulus(young_modulus, poisson_ratio),
    )
}

/// Computes shear modulus μ (also called G) from Young's modulus and Poisson's ratio.
fn shear_modulus(young_modulus: f32, poisson_ratio: f32) -> f32 {
    young_modulus / (2.0 * (1.0 + poisson_ratio))
}

/// Lamé parameters for linear elastic materials.
///
/// CPU-side convenience type wrapping `LinearElasticModel`.
pub type ElasticCoefficients = LinearElasticModel;

/// Extension trait for creating `LinearElasticModel` from engineering parameters.
pub trait ElasticCoefficientsExt {
    /// Creates elastic coefficients from engineering parameters.
    fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self;
}

impl ElasticCoefficientsExt for LinearElasticModel {
    fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_lambda_mu(young_modulus, poisson_ratio);
        Self {
            lambda,
            mu,
            cfl_coeff: 0.5,
        }
    }
}
