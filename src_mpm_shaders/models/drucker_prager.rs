//! Drucker-Prager plasticity model.

use crate::glamx::MatExt;
use crate::{Matrix, Vector, diag, sin, sqrt};
use khal_std::num_traits::Float;

/// Persistent plastic state for a Drucker-Prager particle.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct DruckerPragerPlasticState {
    pub plastic_deformation_gradient_det: f32,
    pub plastic_hardening: f32,
    pub log_vol_gain: f32,
}

/// Result of a Drucker-Prager plasticity projection.
///
/// Contains the updated plastic state and the projected deformation gradient.
#[derive(Clone, Copy)]
pub struct DruckerPragerResult {
    pub state: DruckerPragerPlasticState,
    pub deformation_gradient: Matrix,
}

/// Intermediate result of the return mapping on singular values.
#[derive(Clone, Copy)]
pub struct DruckerPragerProjectionResult {
    pub singular_values: Vector,
    pub plastic_hardening: f32,
    pub valid: bool,
}

/// Drucker-Prager plasticity model with hardening.
///
/// The hardening law is parameterized by (ha, hb, hc, hd) which control
/// how the friction angle evolves with accumulated plastic strain.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct DruckerPragerPlasticity {
    pub ha: f32,
    pub hb: f32,
    pub hc: f32,
    pub hd: f32,
    pub lambda: f32,
    pub mu: f32,
    /// Cohesion, expressed as the volumetric log-strain the material can sustain
    /// in tension before it separates. Zero recovers cohesionless sand; positive
    /// values let the material hold a shape at zero confining pressure (wet sand,
    /// mud, snow that packs).
    pub cohesion: f32,
}

impl DruckerPragerPlasticity {
    /// Computes the friction-angle-based alpha parameter given accumulated plastic strain `q`.
    #[inline]
    pub fn alpha(&self, q: f32) -> f32 {
        let angle = self.ha + (self.hb * q - self.hd) * (-self.hc * q).exp();
        let s_angle = sin(angle);
        sqrt(2.0 / 3.0) * (2.0 * s_angle) / (3.0 - s_angle)
    }

    /// Projects the singular values of the deformation gradient onto the yield surface (2D).
    #[cfg(feature = "dim2")]
    #[inline]
    fn project_deformation_gradient(
        &self,
        singular_values: Vector,
        log_vol_gain: f32,
        alpha: f32,
    ) -> DruckerPragerProjectionResult {
        let d = 2.0;
        let strain = glamx::Vec2::new(singular_values.x.ln(), singular_values.y.ln())
            + Vector::splat(log_vol_gain / d);
        let strain_trace = strain.x + strain.y;
        let deviatoric_strain = strain - Vector::splat(strain_trace / d);
        // Cohesion shifts the yield surface along the pressure axis: the material
        // behaves as if it were under `cohesion` more compression than it is.
        let shifted_trace = strain_trace - self.cohesion;

        if shifted_trace > 0.0 || deviatoric_strain == Vector::ZERO {
            // Past the tensile capacity: project onto the tensile limit rather
            // than all the way back to the undeformed state.
            return DruckerPragerProjectionResult {
                singular_values: Vector::splat((self.cohesion / d).exp()),
                plastic_hardening: strain.length(),
                valid: true,
            };
        }

        let deviatoric_strain_norm = deviatoric_strain.length();
        let gamma = deviatoric_strain_norm
            + (d * self.lambda + 2.0 * self.mu) / (2.0 * self.mu) * shifted_trace * alpha;

        if gamma <= 0.0 {
            return DruckerPragerProjectionResult {
                singular_values: Vector::ZERO,
                plastic_hardening: 0.0,
                valid: false,
            };
        }

        let h = strain - deviatoric_strain * (gamma / deviatoric_strain_norm);
        DruckerPragerProjectionResult {
            singular_values: glamx::Vec2::new(h.x.exp(), h.y.exp()),
            plastic_hardening: gamma,
            valid: true,
        }
    }

    /// Projects the singular values of the deformation gradient onto the yield surface (3D).
    #[cfg(feature = "dim3")]
    #[inline]
    fn project_deformation_gradient(
        &self,
        singular_values: Vector,
        log_vol_gain: f32,
        alpha: f32,
    ) -> DruckerPragerProjectionResult {
        let d = 3.0;
        let strain = glamx::Vec3::new(
            singular_values.x.ln(),
            singular_values.y.ln(),
            singular_values.z.ln(),
        ) + Vector::splat(log_vol_gain / d);
        let strain_trace = strain.x + strain.y + strain.z;
        let deviatoric_strain = strain - Vector::splat(strain_trace / d);
        // Cohesion shifts the yield surface along the pressure axis: the material
        // behaves as if it were under `cohesion` more compression than it is.
        let shifted_trace = strain_trace - self.cohesion;

        if shifted_trace > 0.0 || deviatoric_strain == Vector::ZERO {
            // Past the tensile capacity: project onto the tensile limit rather
            // than all the way back to the undeformed state.
            return DruckerPragerProjectionResult {
                singular_values: Vector::splat((self.cohesion / d).exp()),
                plastic_hardening: strain.length(),
                valid: true,
            };
        }

        let deviatoric_strain_norm = deviatoric_strain.length();
        let gamma = deviatoric_strain_norm
            + (d * self.lambda + 2.0 * self.mu) / (2.0 * self.mu) * shifted_trace * alpha;

        if gamma <= 0.0 {
            return DruckerPragerProjectionResult {
                singular_values: Vector::ZERO,
                plastic_hardening: 0.0,
                valid: false,
            };
        }

        let h = strain - deviatoric_strain * (gamma / deviatoric_strain_norm);
        DruckerPragerProjectionResult {
            singular_values: glamx::Vec3::new(h.x.exp(), h.y.exp(), h.z.exp()),
            plastic_hardening: gamma,
            valid: true,
        }
    }

    /// Projects the deformation gradient through the Drucker-Prager yield surface (2D).
    ///
    /// If plasticity is disabled (lambda == 0), returns the input unchanged.
    /// Otherwise, performs SVD, projects the singular values, and recomposes.
    #[cfg(feature = "dim2")]
    #[inline]
    pub fn project(
        &self,
        state: DruckerPragerPlasticState,
        deformation_gradient: Matrix,
    ) -> DruckerPragerResult {
        if self.lambda == 0.0 {
            // Plasticity is disabled on this particle.
            return DruckerPragerResult {
                state,
                deformation_gradient,
            };
        }

        let svd = deformation_gradient.svd();
        let alpha = self.alpha(state.plastic_hardening);
        let projection = self.project_deformation_gradient(svd.s, state.log_vol_gain, alpha);

        if projection.valid {
            let prev_det = svd.s.x * svd.s.y;
            let new_det = projection.singular_values.x * projection.singular_values.y;

            let new_plastic_deformation_gradient_det =
                state.plastic_deformation_gradient_det * prev_det / new_det;
            let new_log_vol_gain = state.log_vol_gain + prev_det.ln() - new_det.ln();
            let new_plastic_hardening = state.plastic_hardening + projection.plastic_hardening;
            let new_deformation_gradient = svd.u * diag(projection.singular_values) * svd.vt;

            DruckerPragerResult {
                state: DruckerPragerPlasticState {
                    plastic_deformation_gradient_det: new_plastic_deformation_gradient_det,
                    plastic_hardening: new_plastic_hardening,
                    log_vol_gain: new_log_vol_gain,
                },
                deformation_gradient: new_deformation_gradient,
            }
        } else {
            DruckerPragerResult {
                state,
                deformation_gradient,
            }
        }
    }

    /// Projects the deformation gradient through the Drucker-Prager yield surface (3D).
    ///
    /// If plasticity is disabled (lambda == 0), returns the input unchanged.
    /// Otherwise, performs SVD, projects the singular values, and recomposes.
    #[cfg(feature = "dim3")]
    #[inline]
    pub fn project(
        &self,
        state: DruckerPragerPlasticState,
        deformation_gradient: Matrix,
    ) -> DruckerPragerResult {
        if self.lambda == 0.0 {
            // Plasticity is disabled on this particle.
            return DruckerPragerResult {
                state,
                deformation_gradient,
            };
        }

        let svd = deformation_gradient.svd();
        let alpha = self.alpha(state.plastic_hardening);
        let projection = self.project_deformation_gradient(svd.s, state.log_vol_gain, alpha);

        if projection.valid {
            let prev_det = svd.s.x * svd.s.y * svd.s.z;
            let new_det = projection.singular_values.x
                * projection.singular_values.y
                * projection.singular_values.z;

            let new_plastic_deformation_gradient_det =
                state.plastic_deformation_gradient_det * prev_det / new_det;
            let new_log_vol_gain = state.log_vol_gain + prev_det.ln() - new_det.ln();
            let new_plastic_hardening = state.plastic_hardening + projection.plastic_hardening;
            let new_deformation_gradient = svd.u * diag(projection.singular_values) * svd.vt;

            DruckerPragerResult {
                state: DruckerPragerPlasticState {
                    plastic_deformation_gradient_det: new_plastic_deformation_gradient_det,
                    plastic_hardening: new_plastic_hardening,
                    log_vol_gain: new_log_vol_gain,
                },
                deformation_gradient: new_deformation_gradient,
            }
        } else {
            DruckerPragerResult {
                state,
                deformation_gradient,
            }
        }
    }
}
