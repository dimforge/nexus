use crate::models::lame_lambda_mu;
use crate::mpm_shaders::models::drucker_prager::DruckerPragerPlasticity;

/// CPU-side convenience wrapper for constructing `DruckerPragerPlasticity`.
pub struct DruckerPrager;

impl DruckerPrager {
    /// Creates a Drucker-Prager model with default sand parameters.
    // Factory constructing the configured plasticity struct; kept as `new` for
    // API stability.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(young_modulus: f32, poisson_ratio: f32) -> DruckerPragerPlasticity {
        let (lambda, mu) = if young_modulus > 0.0 {
            lame_lambda_mu(young_modulus, poisson_ratio)
        } else {
            (-1.0, -1.0)
        };

        Self::from_lame(lambda, mu)
    }

    /// Friction coefficient `alpha` at the initial hardening state.
    ///
    /// Mirrors `DruckerPragerPlasticity::alpha` on the GPU, evaluated at the
    /// accumulated plastic strain a fresh particle starts with. It is the factor
    /// relating a cohesion strain to the shear strength it implies.
    pub fn initial_alpha() -> f32 {
        let plasticity = Self::from_lame(1.0, 1.0);
        let q = 1.0f32;
        let angle =
            plasticity.ha + (plasticity.hb * q - plasticity.hd) * (-plasticity.hc * q).exp();
        let s = angle.sin();
        (2.0f32 / 3.0).sqrt() * (2.0 * s) / (3.0 - s)
    }

    /// Creates a Drucker-Prager model from Lamé parameters with default plasticity settings.
    pub fn from_lame(lambda: f32, mu: f32) -> DruckerPragerPlasticity {
        Self::from_lame_with_cohesion(lambda, mu, 0.0)
    }

    /// Creates a Drucker-Prager model from Lamé parameters and a cohesion.
    ///
    /// `cohesion` is the volumetric log-strain the material sustains in tension
    /// before separating; 0 gives dry sand, and a few 1e-3 already lets a pile
    /// stand at a much steeper angle than its friction angle allows.
    pub fn from_lame_with_cohesion(lambda: f32, mu: f32, cohesion: f32) -> DruckerPragerPlasticity {
        DruckerPragerPlasticity {
            ha: 35.0f32.to_radians(),
            hb: 9.0f32.to_radians(),
            hc: 0.2,
            hd: 10.0f32.to_radians(),
            lambda,
            mu,
            cohesion,
        }
    }
}
