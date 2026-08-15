use crate::trace;
use crate::{DIM, DIM_USIZE, Matrix, Vector, sqrt};

/// Computes the Lame parameters (lambda, mu) from the Young modulus and Poisson ratio.
/// Returns (lambda, mu).
#[inline]
pub fn lame_lambda_mu(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    let lambda =
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    let mu = hook_to_shear_modulus(young_modulus, poisson_ratio);
    (lambda, mu)
}

/// Computes the shear modulus from the Young modulus and Poisson ratio.
#[inline]
pub fn hook_to_shear_modulus(young_modulus: f32, poisson_ratio: f32) -> f32 {
    young_modulus / (2.0 * (1.0 + poisson_ratio))
}

/// Computes the bulk modulus from the Young modulus and Poisson ratio.
#[inline]
pub fn hook_to_bulk_modulus(young_modulus: f32, poisson_ratio: f32) -> f32 {
    young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
}

/// Returns the shear modulus from the Lame parameters.
#[inline]
pub fn shear_modulus_from_lame(_lambda: f32, mu: f32) -> f32 {
    mu
}

/// Computes the bulk modulus from the Lame parameters.
#[inline]
pub fn bulk_modulus_from_lame(lambda: f32, mu: f32) -> f32 {
    lambda + 2.0 * mu / 3.0
}

/// Solves the quadratic equation `a*x^2 + b*x + c = 0`.
/// Returns the two roots as (x1, x2).
#[inline]
pub fn solve_quadratic(a: f32, b: f32, c: f32) -> (f32, f32) {
    let discr_sqr = sqrt(b * b - 4.0 * a * c);
    ((-b + discr_sqr) / (2.0 * a), (-b - discr_sqr) / (2.0 * a))
}

/// Computes the spin tensor (antisymmetric part) of a velocity gradient.
#[inline]
pub fn spin_tensor(velocity_gradient: Matrix) -> Matrix {
    (velocity_gradient - velocity_gradient.transpose()) * 0.5
}

/// Computes the strain rate (symmetric part) of a velocity gradient.
#[inline]
pub fn strain_rate(velocity_gradient: Matrix) -> Matrix {
    (velocity_gradient + velocity_gradient.transpose()) * 0.5
}

/// Computes the deviatoric part of a tensor.
#[inline]
pub fn deviatoric_part(tensor: Matrix) -> Matrix {
    DecomposedTensor::new(tensor).deviatoric_part
}

/// Computes the spherical part (mean diagonal value) of a tensor.
#[inline]
pub fn spherical_part(tensor: Matrix) -> f32 {
    trace(tensor) / DIM as f32
}

/// A tensor decomposed into its deviatoric and spherical parts.
#[derive(Clone, Copy)]
pub struct DecomposedTensor {
    pub deviatoric_part: Matrix,
    pub spherical_part: f32,
}

impl DecomposedTensor {
    /// Decomposes a tensor into its deviatoric and spherical parts.
    #[inline]
    pub fn new(tensor: Matrix) -> Self {
        let spherical_part = trace(tensor) / DIM as f32;
        let mut deviatoric_part = tensor;

        #[cfg(feature = "dim2")]
        {
            deviatoric_part.x_axis.x -= spherical_part;
            deviatoric_part.y_axis.y -= spherical_part;
        }
        #[cfg(feature = "dim3")]
        {
            deviatoric_part.x_axis.x -= spherical_part;
            deviatoric_part.y_axis.y -= spherical_part;
            deviatoric_part.z_axis.z -= spherical_part;
        }

        Self {
            deviatoric_part,
            spherical_part,
        }
    }

    /// Recomposes the tensor from its deviatoric and spherical parts.
    #[inline]
    pub fn recompose(&self) -> Matrix {
        let mut result = self.deviatoric_part;

        #[cfg(feature = "dim2")]
        {
            result.x_axis.x += self.spherical_part;
            result.y_axis.y += self.spherical_part;
        }
        #[cfg(feature = "dim3")]
        {
            result.x_axis.x += self.spherical_part;
            result.y_axis.y += self.spherical_part;
            result.z_axis.z += self.spherical_part;
        }

        result
    }
}

/// CFL-based timestep bound using the speed of sound in an elastic material.
#[derive(Clone, Copy)]
pub struct ElasticitySoundSpeedTimestepBound {
    pub alpha: f32,
    pub bulk_modulus: f32,
    pub shear_modulus: f32,
}

impl ElasticitySoundSpeedTimestepBound {
    /// Creates a new timestep bound from the CFL coefficient and Lame-derived moduli.
    #[inline]
    pub fn new(alpha: f32, bulk_modulus: f32, shear_modulus: f32) -> Self {
        Self {
            alpha,
            bulk_modulus,
            shear_modulus,
        }
    }

    /// Creates a new timestep bound from the CFL coefficient, Young modulus, and Poisson ratio.
    #[inline]
    pub fn from_elasticity(alpha: f32, young_modulus: f32, poisson_ratio: f32) -> Self {
        Self {
            alpha,
            bulk_modulus: hook_to_bulk_modulus(young_modulus, poisson_ratio),
            shear_modulus: hook_to_shear_modulus(young_modulus, poisson_ratio),
        }
    }

    /// Computes the CFL-based timestep bound.
    ///
    /// Uses the speed of sound for pressure waves and the physical velocity
    /// to determine the maximum stable timestep.
    #[inline]
    pub fn timestep_bound(
        &self,
        density0: f32,
        def_grad_det: f32,
        velocity: Vector,
        cell_width: f32,
    ) -> f32 {
        // Avoid division by zero.
        let curr_density = density0 / f32::max(def_grad_det, 1.0e-6);

        // Speed of sound of pressure waves.
        let sound_speed = sqrt((self.bulk_modulus + self.shear_modulus * 4.0 / 3.0) / curr_density);

        // Take the max between sound speed and physical velocity.
        // Note that we don't calculate the speed of sound for the shear waves since that's
        // always slower than for the pressure wave.
        let max_speed = f32::max(velocity.length(), sound_speed);
        self.alpha * cell_width / max_speed
    }
}
