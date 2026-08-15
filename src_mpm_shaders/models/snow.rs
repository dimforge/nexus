//! Snow elastoplasticity (Stomakhin et al. 2013).

use crate::glamx::MatExt;
use crate::{Matrix, PaddedMatrix, PaddingExt, Vector, diag};
use khal_std::num_traits::Float;

/// Persistent plastic state for a snow particle.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct SnowPlasticState {
    /// Determinant of the plastic part of the deformation gradient. Below 1 the
    /// particle has been compacted, which hardens it.
    pub plastic_det: f32,
}

/// Snow plasticity: singular-value clamping with compaction hardening.
///
/// The elastic deformation is confined to a box in singular-value space and
/// whatever is clamped away becomes plastic. Compaction is tracked separately,
/// so a packed snowball becomes stiffer than the loose snow it was made from.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct SnowPlasticity {
    /// Compression the elastic response sustains before yielding (`theta_c`).
    pub critical_compression: f32,
    /// Stretch the elastic response sustains before yielding (`theta_s`).
    /// Small values make the snow break apart readily under tension.
    pub critical_stretch: f32,
    /// Hardening coefficient (`xi`): how sharply the moduli grow with compaction.
    pub hardening: f32,
}

/// Result of a snow return mapping.
///
/// The deformation gradient is a [`PaddedMatrix`]: a bare `Matrix` member is laid
/// out with 36-byte offsets in 3D while SPIR-V pads each column of a 3x3 to 16
/// bytes, so any member after it overlaps and the module fails validation. The
/// `#[repr(C)]` keeps Rust from reordering it back into that position.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SnowResult {
    pub state: SnowPlasticState,
    pub deformation_gradient: PaddedMatrix,
    /// Multiplier to apply to the elastic moduli for this particle.
    pub hardening: f32,
}

/// Product of the components of a vector, i.e. the determinant of `diag(v)`.
#[inline]
fn component_product(v: Vector) -> f32 {
    #[cfg(feature = "dim2")]
    {
        v.x * v.y
    }
    #[cfg(feature = "dim3")]
    {
        v.x * v.y * v.z
    }
}

impl SnowPlasticity {
    /// Lowest and highest plastic compaction tracked. Left unbounded, a particle
    /// squeezed in a corner can harden without limit and blow up the timestep.
    const MIN_PLASTIC_DET: f32 = 0.1;
    const MAX_PLASTIC_DET: f32 = 4.0;

    /// Multiplier applied to the Lamé parameters for the given compaction.
    #[inline]
    pub fn hardening_factor(&self, state: SnowPlasticState) -> f32 {
        (self.hardening * (1.0 - state.plastic_det)).exp()
    }

    /// Clamps the elastic singular values into the yield box, moving the excess
    /// into the plastic determinant.
    #[inline]
    pub fn project(&self, state: SnowPlasticState, deformation_gradient: Matrix) -> SnowResult {
        if self.critical_compression <= 0.0 && self.critical_stretch <= 0.0 {
            // Plasticity disabled: purely elastic snow.
            return SnowResult {
                state,
                deformation_gradient: PaddedMatrix::add_padding(deformation_gradient),
                hardening: self.hardening_factor(state),
            };
        }

        let svd = deformation_gradient.svd();
        let lo = 1.0 - self.critical_compression;
        let hi = 1.0 + self.critical_stretch;
        let clamped = svd.s.clamp(Vector::splat(lo), Vector::splat(hi));

        let prev_det = component_product(svd.s);
        let new_det = component_product(clamped);

        // Everything the clamp removed is absorbed into the plastic part, so the
        // total deformation is preserved.
        let plastic_det = (state.plastic_det * prev_det / new_det)
            .clamp(Self::MIN_PLASTIC_DET, Self::MAX_PLASTIC_DET);
        let state = SnowPlasticState { plastic_det };

        SnowResult {
            state,
            deformation_gradient: PaddedMatrix::add_padding(svd.u * diag(clamped) * svd.vt),
            hardening: self.hardening_factor(state),
        }
    }
}
