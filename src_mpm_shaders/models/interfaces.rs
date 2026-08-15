use crate::Matrix;

/// Result of a constitutive model update, containing the Kirchoff stress tensor.
#[derive(Clone, Copy, Default)]
pub struct ModelUpdateResult {
    pub kirchoff_stress: Matrix,
}

impl ModelUpdateResult {
    #[inline]
    pub fn new(kirchoff_stress: Matrix) -> Self {
        Self { kirchoff_stress }
    }
}

/// Data passed to the particle model update function.
#[derive(Clone, Copy)]
pub struct ParticleUpdateData {
    pub dt: f32,
    pub cell_width: f32,
    pub particle_id: u32,
    /// Velocity gradient at the particle, as gathered by G2P. Rate-dependent
    /// models (viscosity, viscoplasticity, viscoelasticity) need it; purely
    /// deformation-driven models can ignore it.
    pub velocity_gradient: Matrix,
}

impl ParticleUpdateData {
    #[inline]
    pub fn new(dt: f32, cell_width: f32, particle_id: u32, velocity_gradient: Matrix) -> Self {
        Self {
            dt,
            cell_width,
            particle_id,
            velocity_gradient,
        }
    }

    /// Symmetric part of the velocity gradient (the strain rate).
    #[inline]
    pub fn strain_rate(&self) -> Matrix {
        (self.velocity_gradient + self.velocity_gradient.transpose()) * 0.5
    }
}

/// Model behavior flags (bitflags stored as u32).
pub const MODEL_FLAGS_NONE: u32 = 0;
pub const MODEL_FLAGS_FLUID: u32 = 1;
