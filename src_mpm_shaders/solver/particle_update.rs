//! Particle update kernel: advection, deformation gradient update, constitutive model,
//! and APIC affine matrix computation.
//!
//! The main per-particle update kernel, run after G2P has transferred grid velocities
//! back to particles. Diverged particles (NaN position or non-positive deformation
//! gradient determinant) are automatically disabled.

use crate::PaddingExt;
use crate::grid::grid::Grid;
use crate::grid::kernel::QuadraticKernel;
use crate::models::default::{DefaultParticleModel, GpuParticleModel};
use crate::models::interfaces::{MODEL_FLAGS_FLUID, ParticleUpdateData};
use crate::solver::boundary_condition::{BOUNDARY_CONDITION_SLIP, BoundaryCondition};
use crate::solver::params::SimulationParams;
use crate::solver::particle::{Kinematics, ParticleProperties, Position};
use crate::{DIM, Matrix, PaddedMatrix, Vector, diag};

/// Largest relative volume change a fluid particle may undergo in one substep.
const MAX_VOLUME_RATE: f32 = 0.1;
/// Bounds on the per-axis stretch of a fluid particle, i.e. on `J^(1/DIM)`.
/// A particle that leaves this range has diverged, not deformed.
const MIN_FLUID_STRETCH: f32 = 0.5;
const MAX_FLUID_STRETCH: f32 = 2.0;
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Phase data for multi-material mixing (currently unused placeholder).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Phase {
    pub phase: f32,
    pub max_stretch: f32,
}

/// Penalty coefficient for collision response.
const PENALTY_COEFF: f32 = 1.0e3;

/// Checks if a Vector contains any NaN components.
#[inline]
fn vector_has_nan(v: Vector) -> bool {
    #[cfg(feature = "dim2")]
    {
        v.x.is_nan() || v.y.is_nan()
    }
    #[cfg(feature = "dim3")]
    {
        v.x.is_nan() || v.y.is_nan() || v.z.is_nan()
    }
}

/// Main particle update kernel.
///
/// Each thread processes one particle, performing the full update cycle:
/// advection, deformation gradient update, constitutive model evaluation,
/// and APIC affine matrix computation.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_particle_update(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimulationParams,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    particles_model: &mut [GpuParticleModel],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_pos: &mut [Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] particles_kin: &mut [Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    particles_def_grad: &mut [PaddedMatrix],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    particles_props: &[ParticleProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] particles_len: &u32,
) {
    let particle_id = invocation_id.x;

    if particle_id >= *particles_len {
        return;
    }

    let flags = DefaultParticleModel::model_flags(particles_model, particle_id);
    let dt = params.dt;
    let cell_width = grid.cell_width;
    let mut kin = particles_kin.read(particle_id as usize);
    let cdf = kin.cdf;
    let mut def_grad = particles_def_grad.read(particle_id as usize);
    let props = particles_props.read(particle_id as usize);
    let particle_pos = particles_pos.at(particle_id as usize).pt;

    /*
     * Update velocity.
     */
    // Reproject velocity if the particle is penetrating a rigid collider.
    // TODO: double check that we never need the reprojection below.
    //       This reprojection isn't part of the original MPM-MLS/CPIC paper but we added
    //       it at some point as it appeared that we'd still get some penetrating particles.
    //       However, that might have been caused by other bugs so it is unsure if we need to
    //       keep it now.
    if cdf.signed_distance < -0.05 * cell_width {
        let slip = BoundaryCondition::new(BOUNDARY_CONDITION_SLIP, 0.0);
        kin.velocity =
            cdf.rigid_vel + slip.project_velocity(kin.velocity - cdf.rigid_vel, cdf.normal);
    }

    // Clamp the max velocity a particle can get.
    // TODO: clamp the grid velocities instead?
    let vel_len = kin.velocity.length();
    if vel_len > cell_width / dt {
        kin.velocity = kin.velocity / vel_len * cell_width / dt;
    }

    // Apply Rayleigh mass-proportional damping (implicit integration for stability).
    // v_new = v / (1 + damping * dt)
    kin.velocity /= 1.0 + props.damping * dt;

    // If the particle is fixed, clear its velocity.
    // This isn't ideal (this should typically be handled on the grid) but we sometimes
    // need sub-grid-sized fixed particles.
    if props.fixed != 0 {
        kin.velocity = Vector::ZERO;
    }

    /*
     * Update position.
     */
    let new_particle_pos = particle_pos + kin.velocity * dt;

    /*
     * Penalty impulse.
     */
    // TODO: apply the penalty impulse as an extra force on the grid instead of
    //       changing the particle velocity directly?
    if cdf.signed_distance < -0.05 * cell_width {
        let corrected_dist = cdf.signed_distance.max(-0.3 * cell_width);
        let impulse = (dt * -corrected_dist * PENALTY_COEFF) * cdf.normal;
        kin.velocity += impulse;
    }

    /*
     * Deformation gradient update.
     */
    if (flags & MODEL_FLAGS_FLUID) == 0 {
        // Solid path: F_new = F + (vel_grad * dt) * F
        // NOTE: the velocity gradient was stored in the affine buffer.
        def_grad = def_grad + (kin.affine * dt) * def_grad;
    } else {
        // Fluid path: only the volumetric part is tracked, as `F = d * I`.
        //
        // `vel_grad_det` is the divergence, so the volume ratio obeys
        // `J' = J tr(grad v)`. Since `J = d^DIM`, the diagonal entry grows at
        // `1 / DIM` of that rate; dividing here is what keeps
        // `def_grad.determinant()` equal to `J`, which the constitutive model,
        // the CFL bound and the rendering all assume.
        let def_grad0 = def_grad.x_axis.x;
        let rate = kin.vel_grad_det * dt / DIM as f32;
        // Bound the per-substep change so one bad divergence estimate at a free
        // surface cannot send a particle's volume to zero or to infinity.
        let rate = rate.clamp(-MAX_VOLUME_RATE, MAX_VOLUME_RATE);
        let new_def_grad_diag_elt =
            (def_grad0 + rate * def_grad0).clamp(MIN_FLUID_STRETCH, MAX_FLUID_STRETCH);
        def_grad = PaddedMatrix::add_padding(diag(Vector::splat(new_def_grad_diag_elt)));
    }

    /*
     * Constitutive model.
     */
    // `kin.affine` still holds the velocity gradient at this point; it is
    // overwritten with the APIC affine matrix further down.
    let velocity_gradient = kin.affine.remove_padding();
    let update_data = ParticleUpdateData::new(dt, cell_width, particle_id, velocity_gradient);
    let update_result = DefaultParticleModel::update(particles_model, &update_data, &mut def_grad);

    /*
     * Affine matrix for APIC transfer.
     */
    let inv_d = QuadraticKernel::inv_d(cell_width);
    // NOTE: the velocity gradient was stored in the affine buffer.
    let affine = kin.affine * kin.mass
        - PaddedMatrix::add_padding(
            update_result.kirchoff_stress * (props.init_volume * inv_d * dt),
        );

    /*
     * Write back the new particle properties.
     */
    // Check for NaN and invalid deformation gradients.
    if !vector_has_nan(new_particle_pos) && def_grad.determinant() > 0.0 {
        particles_pos.at_mut(particle_id as usize).pt = new_particle_pos;
        kin.affine = affine;
    } else {
        // This particle diverged, disable it.
        kin.enabled = 0;
        kin.velocity = Vector::ZERO;
        def_grad = PaddedMatrix::IDENTITY;
        kin.affine = PaddedMatrix::ZERO;
        kin.mass = 0.0;
    }
    kin.force_dt = Vector::ZERO;

    particles_kin.write(particle_id as usize, kin);
    particles_def_grad.write(particle_id as usize, def_grad);
}
