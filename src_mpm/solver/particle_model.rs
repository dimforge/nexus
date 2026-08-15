use crate::models::{
    DruckerPrager, DruckerPragerPlasticState, DruckerPragerPlasticity, ElasticCoefficients,
    ElasticCoefficientsExt, FluidModel, SnowPlasticState, SnowPlasticity,
};
pub use crate::mpm_shaders::models::default::{GpuParticleModel, MODEL_DATA_WORDS};
use nexus_rbd::math::DIM;

/// Material model for MPM particles.
///
/// Defines the constitutive behavior (how stress relates to deformation) for particles.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticleModel {
    /// Linear elastic material (St. Venant-Kirchhoff).
    ElasticLinear(ElasticCoefficients),
    /// Neo-Hookean hyperelastic material (better for large deformations).
    ElasticNeoHookean(ElasticCoefficients),
    /// Sand/granular material with linear elasticity and Drucker-Prager plasticity.
    SandLinear(SandModel),
    /// Sand with Neo-Hookean elasticity and Drucker-Prager plasticity.
    SandNeoHookean(SandModel),
    /// Weakly-compressible Newtonian fluid (Tait equation of state).
    Fluid(FluidModel),
    /// Snow: elasticity with singular-value clamping and compaction hardening.
    Snow(SnowModel),
}

impl Default for ParticleModel {
    fn default() -> Self {
        Self::elastic(Self::DEFAULT_YOUNG_MODULUS, Self::DEFAULT_POISSON_RATIO)
    }
}

impl ParticleModel {
    /// Default Young's modulus for elastic materials (Pa).
    pub const DEFAULT_YOUNG_MODULUS: f32 = 1_000.0;
    /// Default Poisson's ratio for elastic materials (dimensionless).
    pub const DEFAULT_POISSON_RATIO: f32 = 0.2;
    /// Default tensile stiffness of a fluid, as a fraction of its bulk modulus.
    ///
    /// Enough to stop the volume drifting at a free surface without pulling the
    /// fluid into blobs: raising it to 1 barely improves the drift further but
    /// visibly clumps a settled column.
    pub const DEFAULT_FLUID_TENSILE_STIFFNESS: f32 = 0.25;
    /// Default snow compression yield threshold (Stomakhin et al. 2013).
    pub const DEFAULT_SNOW_CRITICAL_COMPRESSION: f32 = 2.5e-2;
    /// Default snow stretch yield threshold (Stomakhin et al. 2013).
    pub const DEFAULT_SNOW_CRITICAL_STRETCH: f32 = 7.5e-3;
    /// Default snow hardening coefficient (Stomakhin et al. 2013).
    pub const DEFAULT_SNOW_HARDENING: f32 = 10.0;

    /// Creates a linear elastic material model.
    pub fn elastic(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticLinear(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    pub fn elastic_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticNeoHookean(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    /// Creates a sand/granular material model with Drucker-Prager plasticity.
    pub fn sand(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandLinear(SandModel {
            plastic_state: DruckerPragerPlasticState {
                plastic_deformation_gradient_det: 1.0,
                plastic_hardening: 1.0,
                log_vol_gain: 0.0,
            },
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }

    /// Cohesion parameter that gives a material the requested cohesive shear
    /// strength, in Pascals.
    ///
    /// [`Self::cohesive_sand`] takes cohesion as a *strain*, so the strength it
    /// produces scales with the elastic moduli: the same value on stiffer sand
    /// is a far stronger material. This inverts
    /// `tau_c = cohesion * (d*lambda + 2*mu) * alpha` so the material can be
    /// specified by the shear stress it should sustain at zero confining
    /// pressure. Damp sand is a few kPa; approaching the material's own weight
    /// stress makes it immovable.
    pub fn sand_cohesion_for_strength(
        young_modulus: f32,
        poisson_ratio: f32,
        shear_strength: f32,
    ) -> f32 {
        let (lambda, mu) = crate::models::lame_lambda_mu(young_modulus, poisson_ratio);
        let alpha = DruckerPrager::initial_alpha();
        let scale = (DIM as f32 * lambda + 2.0 * mu) * alpha;
        shear_strength / scale.max(1.0e-6)
    }

    /// Creates a cohesive granular material from the shear strength it should
    /// sustain at zero confining pressure, in Pascals.
    ///
    /// Prefer this over [`Self::cohesive_sand`] unless you already know what
    /// cohesion strain the material needs; see
    /// [`Self::sand_cohesion_for_strength`].
    pub fn cohesive_sand_with_strength(
        young_modulus: f32,
        poisson_ratio: f32,
        shear_strength: f32,
    ) -> Self {
        Self::cohesive_sand(
            young_modulus,
            poisson_ratio,
            Self::sand_cohesion_for_strength(young_modulus, poisson_ratio, shear_strength),
        )
    }

    /// Creates a cohesive granular material (wet sand, mud, packed snow).
    ///
    /// `cohesion` is the volumetric log-strain the material sustains in tension
    /// before separating: 0 reproduces [`Self::sand`], while positive values let
    /// the material hold a shape with no confining pressure. Being a strain, the
    /// strength it implies depends on the elastic moduli; use
    /// [`Self::cohesive_sand_with_strength`] to specify a stress instead.
    pub fn cohesive_sand(young_modulus: f32, poisson_ratio: f32, cohesion: f32) -> Self {
        let (lambda, mu) = if young_modulus > 0.0 {
            crate::models::lame_lambda_mu(young_modulus, poisson_ratio)
        } else {
            (-1.0, -1.0)
        };
        ParticleModel::SandLinear(SandModel {
            plastic_state: DruckerPragerPlasticState {
                plastic_deformation_gradient_det: 1.0,
                plastic_hardening: 1.0,
                log_vol_gain: 0.0,
            },
            plastic: DruckerPrager::from_lame_with_cohesion(lambda, mu, cohesion),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }

    /// Creates a sand/granular material model with Neo-Hookean elasticity.
    pub fn sand_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandNeoHookean(SandModel {
            plastic_state: DruckerPragerPlasticState {
                plastic_deformation_gradient_det: 1.0,
                plastic_hardening: 1.0,
                log_vol_gain: 0.0,
            },
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }

    /// Creates a weakly-compressible fluid.
    ///
    /// `bulk_modulus` sets how strongly compression is resisted (and therefore
    /// how small the stable timestep gets), `gamma` how sharply that resistance
    /// grows (7 is the usual value for water), and `viscosity` the dynamic
    /// viscosity in Pa.s (~0.001 for water, orders of magnitude more for honey).
    pub fn fluid(bulk_modulus: f32, gamma: f32, viscosity: f32) -> Self {
        ParticleModel::Fluid(FluidModel {
            bulk_modulus,
            gamma,
            viscosity,
            cfl_coeff: 0.5,
            tensile_stiffness: Self::DEFAULT_FLUID_TENSILE_STIFFNESS,
        })
    }

    /// Creates a water-like fluid with the given bulk modulus.
    ///
    /// Prefer [`Self::water_for_depth`] unless you already know what bulk modulus
    /// the scene needs: too low a value is the usual cause of a fluid that
    /// visibly loses volume under its own weight.
    pub fn water(bulk_modulus: f32) -> Self {
        Self::fluid(bulk_modulus, 7.0, 0.001)
    }

    /// Creates a water-like fluid stiff enough to stay nearly incompressible
    /// under its own weight.
    ///
    /// A weakly-compressible fluid trades incompressibility for an explicit
    /// solve: the deeper the pool, the higher the pressure at the bottom, and the
    /// more the equation of state lets it compress. `depth` is the deepest the
    /// fluid will get, and `max_compression` the volume loss tolerated there
    /// (0.01 is a good default, since 1% is invisible).
    ///
    /// The cost is the stable timestep, which scales as `1 / sqrt(bulk_modulus)`.
    pub fn water_for_depth(density: f32, gravity: f32, depth: f32, max_compression: f32) -> Self {
        Self::water(Self::fluid_bulk_modulus(
            density,
            gravity,
            depth,
            max_compression,
            7.0,
        ))
    }

    /// Bulk modulus that limits compression to `max_compression` at the bottom of
    /// a column of fluid `depth` deep.
    ///
    /// Inverts the equation of state at the hydrostatic pressure `rho g h`.
    pub fn fluid_bulk_modulus(
        density: f32,
        gravity: f32,
        depth: f32,
        max_compression: f32,
        gamma: f32,
    ) -> f32 {
        let pressure = density * gravity * depth.max(0.0);
        let j = (1.0 - max_compression.clamp(1.0e-4, 0.5)).max(1.0e-4);
        // `p = k (J^-gamma - 1)` solved for `k`.
        let response = j.powf(-gamma) - 1.0;
        pressure / response.max(1.0e-6)
    }

    /// Creates a snow material with the default yield box and hardening from
    /// Stomakhin et al. 2013.
    pub fn snow(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::snow_with_params(
            young_modulus,
            poisson_ratio,
            Self::DEFAULT_SNOW_CRITICAL_COMPRESSION,
            Self::DEFAULT_SNOW_CRITICAL_STRETCH,
            Self::DEFAULT_SNOW_HARDENING,
        )
    }

    /// Creates a snow material with an explicit yield box and hardening.
    ///
    /// Widening `critical_stretch` makes the snow hold together under tension
    /// (wet, packing snow); shrinking it makes it powdery. `hardening` controls
    /// how much stiffer compacted snow becomes than loose snow.
    pub fn snow_with_params(
        young_modulus: f32,
        poisson_ratio: f32,
        critical_compression: f32,
        critical_stretch: f32,
        hardening: f32,
    ) -> Self {
        ParticleModel::Snow(SnowModel {
            plastic_state: SnowPlasticState { plastic_det: 1.0 },
            plastic: SnowPlasticity {
                critical_compression,
                critical_stretch,
                hardening,
            },
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }
}

/// Combined elastic-plastic model for snow.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SnowModel {
    /// Current plastic compaction state.
    pub plastic_state: SnowPlasticState,
    /// Yield box and hardening parameters.
    pub plastic: SnowPlasticity,
    /// Elastic coefficients (Lamé parameters) before hardening.
    pub elastic: ElasticCoefficients,
}

/// Combined elastic-plastic model for sand and granular materials.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SandModel {
    /// Current plastic deformation state.
    pub plastic_state: DruckerPragerPlasticState,
    /// Drucker-Prager plasticity model parameters.
    pub plastic: DruckerPragerPlasticity,
    /// Elastic coefficients (Lamé parameters).
    pub elastic: ElasticCoefficients,
}

// NOTE: keeps `GpuParticleModel` (tag + [u32; MODEL_DATA_WORDS]) in step with
// the GPU-side layout.
static_assertions::assert_eq_size!(GpuParticleModel, [u8; 4 + 4 * MODEL_DATA_WORDS]);

impl From<ParticleModel> for GpuParticleModel {
    fn from(val: ParticleModel) -> Self {
        let mut data = [0u32; MODEL_DATA_WORDS];
        let tag = match val {
            ParticleModel::ElasticLinear(elastic) => {
                let bytes = bytemuck::bytes_of(&elastic);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                0
            }
            ParticleModel::ElasticNeoHookean(elastic) => {
                let bytes = bytemuck::bytes_of(&elastic);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                1
            }
            ParticleModel::SandLinear(sand) => {
                let bytes = bytemuck::bytes_of(&sand);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                2
            }
            ParticleModel::SandNeoHookean(sand) => {
                let bytes = bytemuck::bytes_of(&sand);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                3
            }
            ParticleModel::Fluid(fluid) => {
                let bytes = bytemuck::bytes_of(&fluid);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                4
            }
            ParticleModel::Snow(snow) => {
                let bytes = bytemuck::bytes_of(&snow);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                5
            }
        };
        GpuParticleModel { tag, data }
    }
}
