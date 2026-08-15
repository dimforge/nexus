//! Default particle model: a tagged union dispatching to different constitutive models.

use super::drucker_prager::*;
use super::fluid::FluidModel;
use super::interfaces::*;
use super::linear_elasticity::LinearElasticModel;
use super::neo_hookean_elasticity::NeoHookeanModel;
use super::snow::{SnowPlasticState, SnowPlasticity};
use crate::{Matrix, PaddedMatrix, PaddingExt, Vector};

/// Number of `u32` words of per-particle storage available to a constitutive
/// model. Holds both the model parameters and any state the model mutates
/// (plastic state, damage, viscous strain, temperature, ...).
pub const MODEL_DATA_WORDS: usize = 24;

/// The byte stride of `GpuParticleModel` in GPU buffers. Has to match the size
/// of `GpuParticleModel` on the host side.
pub const DEFAULT_MODEL_BYTES_STRIDE: usize = 100;

/// Model type tag: corotated linear elasticity.
pub const MODEL_ELASTIC_LINEAR: u32 = 0;
/// Model type tag: neo-Hookean elasticity.
pub const MODEL_ELASTIC_NEO_HOOKEAN: u32 = 1;
/// Model type tag: Drucker-Prager sand with linear elastic backbone.
pub const MODEL_SAND_LINEAR: u32 = 2;
/// Model type tag: Drucker-Prager sand with neo-Hookean elastic backbone.
pub const MODEL_SAND_NEO_HOOKEAN: u32 = 3;
/// Model type tag: weakly-compressible fluid.
pub const MODEL_FLUID: u32 = 4;
/// Model type tag: snow (singular-value clamping with compaction hardening).
pub const MODEL_SNOW: u32 = 5;

/// GPU particle model stored as a tagged union in a fixed-size buffer.
///
/// The `tag` field selects the constitutive model variant, and `data` holds the
/// model parameters in a flat `[u32; 24]` (96 bytes) that is reinterpreted
/// as the appropriate model struct.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct GpuParticleModel {
    pub tag: u32,
    pub data: [u32; MODEL_DATA_WORDS],
}

/// Sand model combining Drucker-Prager plasticity with an elastic backbone.
///
/// The `plastic_state` field must be first because the state offset
/// assumes offset 0 within `SandModel`.
#[derive(Clone, Copy)]
#[repr(C)]
struct SandModel<E: Copy> {
    plastic_state: DruckerPragerPlasticState,
    plastic: DruckerPragerPlasticity,
    elastic: E,
}

/// Loads a `LinearElasticModel` from the raw data array (3 words at offset 0).
#[inline]
fn load_elastic(data: &[u32; MODEL_DATA_WORDS], offset: usize) -> LinearElasticModel {
    LinearElasticModel {
        lambda: f32::from_bits(data[offset]),
        mu: f32::from_bits(data[offset + 1]),
        cfl_coeff: f32::from_bits(data[offset + 2]),
    }
}

/// Loads a `NeoHookeanModel` from the raw data array (3 words at offset 0).
#[inline]
fn load_neo_hookean(data: &[u32; MODEL_DATA_WORDS], offset: usize) -> NeoHookeanModel {
    NeoHookeanModel {
        lambda: f32::from_bits(data[offset]),
        mu: f32::from_bits(data[offset + 1]),
        cfl_coeff: f32::from_bits(data[offset + 2]),
    }
}

/// Snow model: plastic state, plasticity parameters, and elastic backbone.
///
/// As for `SandModel`, `plastic_state` must come first: the store helper assumes
/// it sits at offset 0.
#[derive(Clone, Copy)]
#[repr(C)]
struct SnowModel {
    plastic_state: SnowPlasticState,
    plastic: SnowPlasticity,
    elastic: LinearElasticModel,
}

/// Loads a `SnowModel` from the raw data array (7 words).
#[inline]
fn load_snow(data: &[u32; MODEL_DATA_WORDS]) -> SnowModel {
    SnowModel {
        plastic_state: SnowPlasticState {
            plastic_det: f32::from_bits(data[0]),
        },
        plastic: SnowPlasticity {
            critical_compression: f32::from_bits(data[1]),
            critical_stretch: f32::from_bits(data[2]),
            hardening: f32::from_bits(data[3]),
        },
        elastic: load_elastic(data, 4),
    }
}

/// Writes a `SnowPlasticState` back into the raw data array at offset 0.
#[inline]
fn store_snow_state(data: &mut [u32; MODEL_DATA_WORDS], state: SnowPlasticState) {
    data[0] = state.plastic_det.to_bits();
}

/// Scales the Lamé parameters of an elastic model by a hardening factor.
#[inline]
fn harden(elastic: LinearElasticModel, hardening: f32) -> LinearElasticModel {
    LinearElasticModel {
        lambda: elastic.lambda * hardening,
        mu: elastic.mu * hardening,
        cfl_coeff: elastic.cfl_coeff,
    }
}

/// Loads a `FluidModel` from the raw data array (5 words).
#[inline]
fn load_fluid(data: &[u32; MODEL_DATA_WORDS], offset: usize) -> FluidModel {
    FluidModel {
        bulk_modulus: f32::from_bits(data[offset]),
        gamma: f32::from_bits(data[offset + 1]),
        viscosity: f32::from_bits(data[offset + 2]),
        cfl_coeff: f32::from_bits(data[offset + 3]),
        tensile_stiffness: f32::from_bits(data[offset + 4]),
    }
}

/// Loads a `DruckerPragerPlasticState` from the raw data array (3 words).
#[inline]
fn load_plastic_state(data: &[u32; MODEL_DATA_WORDS], offset: usize) -> DruckerPragerPlasticState {
    DruckerPragerPlasticState {
        plastic_deformation_gradient_det: f32::from_bits(data[offset]),
        plastic_hardening: f32::from_bits(data[offset + 1]),
        log_vol_gain: f32::from_bits(data[offset + 2]),
    }
}

/// Loads a `DruckerPragerPlasticity` from the raw data array (7 words).
#[inline]
fn load_plasticity(data: &[u32; MODEL_DATA_WORDS], offset: usize) -> DruckerPragerPlasticity {
    DruckerPragerPlasticity {
        ha: f32::from_bits(data[offset]),
        hb: f32::from_bits(data[offset + 1]),
        hc: f32::from_bits(data[offset + 2]),
        hd: f32::from_bits(data[offset + 3]),
        lambda: f32::from_bits(data[offset + 4]),
        mu: f32::from_bits(data[offset + 5]),
        cohesion: f32::from_bits(data[offset + 6]),
    }
}

/// Loads a `SandModel<LinearElasticModel>` from the raw data array (13 words).
#[inline]
fn load_sand_linear(data: &[u32; MODEL_DATA_WORDS]) -> SandModel<LinearElasticModel> {
    SandModel {
        plastic_state: load_plastic_state(data, 0),
        plastic: load_plasticity(data, 3),
        elastic: load_elastic(data, 10),
    }
}

/// Loads a `SandModel<NeoHookeanModel>` from the raw data array (13 words).
#[inline]
fn load_sand_neo_hookean(data: &[u32; MODEL_DATA_WORDS]) -> SandModel<NeoHookeanModel> {
    SandModel {
        plastic_state: load_plastic_state(data, 0),
        plastic: load_plasticity(data, 3),
        elastic: load_neo_hookean(data, 10),
    }
}

/// Writes a `DruckerPragerPlasticState` back into the raw data array at offset 0.
#[inline]
fn store_plastic_state(data: &mut [u32; MODEL_DATA_WORDS], state: DruckerPragerPlasticState) {
    data[0] = state.plastic_deformation_gradient_det.to_bits();
    data[1] = state.plastic_hardening.to_bits();
    data[2] = state.log_vol_gain.to_bits();
}

/// Default particle model dispatcher.
///
/// Reads the model tag from the `GpuParticleModel` array and dispatches
/// to the appropriate constitutive model implementation.
pub struct DefaultParticleModel;

impl DefaultParticleModel {
    /// Returns the model flags for a given particle.
    #[inline]
    pub fn model_flags(models: &[GpuParticleModel], particle_id: u32) -> u32 {
        if models[particle_id as usize].tag == MODEL_FLUID {
            MODEL_FLAGS_FLUID
        } else {
            MODEL_FLAGS_NONE
        }
    }

    /// Runs the constitutive model update for a particle.
    ///
    /// Reads the model data, computes the Kirchoff stress, and for plastic models
    /// also updates the plastic state and deformation gradient in place.
    #[inline]
    pub fn update(
        models: &mut [GpuParticleModel],
        data: &ParticleUpdateData,
        def_grad_padded: &mut PaddedMatrix,
    ) -> ModelUpdateResult {
        let model = &mut models[data.particle_id as usize];
        let tag = model.tag;
        let def_grad = def_grad_padded.remove_padding();

        match tag {
            MODEL_ELASTIC_LINEAR => {
                let elastic = load_elastic(&model.data, 0);
                let stress = elastic.kirchoff_stress(def_grad);
                ModelUpdateResult::new(stress)
            }
            MODEL_ELASTIC_NEO_HOOKEAN => {
                let elastic = load_neo_hookean(&model.data, 0);
                let stress = elastic.kirchoff_stress(def_grad);
                ModelUpdateResult::new(stress)
            }
            MODEL_SAND_LINEAR => {
                let sand = load_sand_linear(&model.data);
                let projection = sand.plastic.project(sand.plastic_state, def_grad);
                store_plastic_state(&mut model.data, projection.state);
                *def_grad_padded = PaddedMatrix::add_padding(projection.deformation_gradient);
                let stress = sand
                    .elastic
                    .kirchoff_stress(projection.deformation_gradient);
                ModelUpdateResult::new(stress)
            }
            MODEL_SAND_NEO_HOOKEAN => {
                let sand = load_sand_neo_hookean(&model.data);
                let projection = sand.plastic.project(sand.plastic_state, def_grad);
                store_plastic_state(&mut model.data, projection.state);
                *def_grad_padded = PaddedMatrix::add_padding(projection.deformation_gradient);
                let stress = sand
                    .elastic
                    .kirchoff_stress(projection.deformation_gradient);
                ModelUpdateResult::new(stress)
            }
            MODEL_FLUID => {
                let fluid = load_fluid(&model.data, 0);
                let stress = fluid.kirchoff_stress(def_grad, data.strain_rate());
                ModelUpdateResult::new(stress)
            }
            MODEL_SNOW => {
                let snow = load_snow(&model.data);
                let projection = snow.plastic.project(snow.plastic_state, def_grad);
                store_snow_state(&mut model.data, projection.state);
                *def_grad_padded = projection.deformation_gradient;
                let stress = harden(snow.elastic, projection.hardening)
                    .kirchoff_stress(projection.deformation_gradient.remove_padding());
                ModelUpdateResult::new(stress)
            }
            _ => ModelUpdateResult::new(Matrix::ZERO),
        }
    }

    /// Computes the CFL-based timestep bound for a given particle's model.
    #[inline]
    pub fn timestep_bound(
        models: &[GpuParticleModel],
        particle_id: u32,
        particle_density0: f32,
        def_grad: Matrix,
        particle_velocity: Vector,
        cell_width: f32,
    ) -> f32 {
        let model = &models[particle_id as usize];
        let tag = model.tag;
        let def_grad_det = def_grad.determinant();

        match tag {
            MODEL_ELASTIC_LINEAR => {
                let elastic = load_elastic(&model.data, 0);
                elastic.timestep_bound(
                    particle_density0,
                    particle_velocity,
                    def_grad_det,
                    1.0,
                    cell_width,
                )
            }
            MODEL_ELASTIC_NEO_HOOKEAN => {
                let elastic = load_neo_hookean(&model.data, 0);
                elastic.timestep_bound(
                    particle_density0,
                    particle_velocity,
                    def_grad_det,
                    1.0,
                    cell_width,
                )
            }
            MODEL_SAND_LINEAR => {
                let sand = load_sand_linear(&model.data);
                sand.elastic.timestep_bound(
                    particle_density0,
                    particle_velocity,
                    def_grad_det,
                    1.0,
                    cell_width,
                )
            }
            MODEL_SAND_NEO_HOOKEAN => {
                let sand = load_sand_neo_hookean(&model.data);
                sand.elastic.timestep_bound(
                    particle_density0,
                    particle_velocity,
                    def_grad_det,
                    1.0,
                    cell_width,
                )
            }
            MODEL_FLUID => {
                let fluid = load_fluid(&model.data, 0);
                fluid.timestep_bound(
                    particle_density0,
                    particle_velocity,
                    def_grad_det,
                    cell_width,
                )
            }
            MODEL_SNOW => {
                let snow = load_snow(&model.data);
                snow.elastic.timestep_bound(
                    particle_density0,
                    particle_velocity,
                    def_grad_det,
                    snow.plastic.hardening_factor(snow.plastic_state),
                    cell_width,
                )
            }
            _ => 0.0,
        }
    }
}
