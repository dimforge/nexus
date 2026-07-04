//! Simulation and constraint regularization parameters.

use crate::MAX_FLT;

/// Two times pi (2π), used for converting natural frequency to angular frequency.
pub const TWO_PI: f32 = core::f32::consts::TAU;

/// Precomputed soft-constraint coefficients (contact + joint), matching rapier's
/// TGS-soft `SpringCoefficients`. Computed once per step on the host from
/// [`RbdSimParams`] and passed to the multibody contact/joint kernels as a small
/// uniform. Exactly 32 bytes (8 scalars) for std140 uniform layout.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ConstraintSoftness {
    /// Contact soft ERP `× 1/dt` (from `contact_natural_frequency` + damping) —
    /// the bias velocity coefficient. Much smaller than `1/dt`; a rigid `1/dt`
    /// overshoots and jitters.
    pub erp_inv_dt: f32,
    /// Contact `1 / (1 + cfm_coeff)` — multiplies the contact impulse each PGS
    /// sweep for constraint-force-mixing compliance.
    pub cfm_factor: f32,
    /// Penetration the solver won't try to correct.
    pub allowed_lin_err: f32,
    /// Max corrective velocity applied for penetration recovery.
    pub max_corr_velocity: f32,
    /// `1/dt` (substep), used for the speculative-contact (`dist > 0`) rhs and
    /// the joint-motor target-velocity clamp.
    pub inv_dt: f32,
    /// Joint soft ERP `× 1/dt` (from `joint_natural_frequency` + damping) — used
    /// for joint limit/lock positional bias. With the default `joint_natural_
    /// frequency = 1e6` this is ≈ `1/dt` (near-rigid), but it is now configurable.
    pub joint_erp_inv_dt: f32,
    /// Joint CFM coeff (rapier's `joint.softness.cfm_coeff(dt)`) — folded into the
    /// limit/lock constraint's `inv_lhs` for compliance.
    pub joint_cfm_coeff: f32,
    /// Substep `dt`, needed by `motor_params` for joint motors.
    pub dt: f32,
}

#[cfg(not(target_arch_is_gpu))]
impl ConstraintSoftness {
    /// Computes the soft coefficients from the (substep) sim params, mirroring
    /// rapier's contact + joint softness.
    pub fn from_params(params: &RbdSimParams) -> Self {
        Self {
            erp_inv_dt: params.contact_erp_inv_dt(),
            cfm_factor: params.contact_cfm_factor(),
            allowed_lin_err: params.allowed_linear_error(),
            max_corr_velocity: params.max_corrective_velocity(),
            inv_dt: params.inv_dt(),
            joint_erp_inv_dt: params.joint_erp_inv_dt(),
            joint_cfm_coeff: params.joint_cfm_coeff(),
            dt: params.dt,
        }
    }
}

/// Parameters for a time-step of the physics engine.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct RbdSimParams {
    /// The timestep length (default: `1.0 / 60.0`).
    pub dt: f32,

    /// > 0: the damping ratio used by the springs for contact constraint stabilization.
    ///
    /// Larger values make the constraints more compliant (allowing more visible
    /// penetrations before stabilization).
    /// (default `5.0`).
    pub contact_damping_ratio: f32,

    /// > 0: the natural frequency used by the springs for contact constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly at the
    /// expense of potential jitter effects due to overshooting. In order to make the simulation
    /// look stiffer, it is recommended to increase the `contact_damping_ratio` instead of this
    /// value.
    /// (default: `30.0`).
    pub contact_natural_frequency: f32,

    /// > 0: the natural frequency used by the springs for joint constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly.
    /// (default: `1.0e6`).
    pub joint_natural_frequency: f32,

    /// The fraction of critical damping applied to the joint for constraints regularization.
    ///
    /// Larger values make the constraints more compliant (allowing more joint
    /// drift before stabilization).
    /// (default `1.0`).
    pub joint_damping_ratio: f32,

    /// The coefficient in `[0, 1]` applied to warmstart impulses, i.e., impulses that are used as the
    /// initial solution (instead of 0) at the next simulation step.
    ///
    /// This should generally be set to 1.
    ///
    /// (default `1.0`).
    pub warmstart_coefficient: f32,

    /// The approximate size of most dynamic objects in the scene.
    ///
    /// This value is used internally to estimate some length-based tolerance. In particular, the
    /// values `allowed_linear_error`, `max_corrective_velocity`, `prediction_distance`,
    /// `normalized_linear_threshold` are scaled by this value implicitly.
    ///
    /// This value can be understood as the number of units-per-meter in your physical world compared
    /// to a human-sized world in meter. For example, in a 2d game, if your typical object size is 100
    /// pixels, set the `length_unit` parameter to 100.0. The physics engine will interpret
    /// it as if 100 pixels is equivalent to 1 meter in its various internal threshold.
    /// (default `1.0`).
    pub length_unit: f32,

    /// Amount of penetration the engine won't attempt to correct (default: `0.001m`).
    ///
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_allowed_linear_error: f32,

    /// Maximum amount of penetration the solver will attempt to resolve in one timestep (default: `10.0`).
    ///
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_max_corrective_velocity: f32,

    /// The maximal distance separating two objects that will generate predictive contacts (default: `0.002m`).
    ///
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_prediction_distance: f32,

    /// The number of solver iterations run by the constraints solver for calculating forces (default: `4`).
    pub num_solver_iterations: u32,
}

impl RbdSimParams {
    /// Initialize the simulation parameters with settings matching the TGS-soft solver
    /// with warmstarting.
    ///
    /// This is the default configuration, equivalent to [`RbdSimParams::default()`].
    pub fn tgs_soft() -> Self {
        Self {
            dt: 1.0 / 60.0,
            contact_natural_frequency: 30.0,
            contact_damping_ratio: 5.0,
            joint_natural_frequency: 1.0e6,
            joint_damping_ratio: 1.0,
            warmstart_coefficient: 1.0,
            num_solver_iterations: 4,
            normalized_allowed_linear_error: 0.001,
            normalized_max_corrective_velocity: 10.0,
            normalized_prediction_distance: 0.002,
            length_unit: 1.0,
        }
    }
}

impl Default for RbdSimParams {
    fn default() -> Self {
        Self::tgs_soft()
    }
}

impl RbdSimParams {
    /// Computes the inverse timestep (1/dt). Returns 0.0 if dt is zero.
    pub fn inv_dt(&self) -> f32 {
        if self.dt == 0.0 { 0.0 } else { 1.0 / self.dt }
    }

    /// Computes the contact constraint angular frequency (rad/s).
    pub fn contact_angular_frequency(&self) -> f32 {
        self.contact_natural_frequency * TWO_PI
    }

    /// The `contact_erp` coefficient, multiplied by the inverse timestep length.
    pub fn contact_erp_inv_dt(&self) -> f32 {
        let ang_freq = self.contact_angular_frequency();
        ang_freq / (self.dt * ang_freq + 2.0 * self.contact_damping_ratio)
    }

    /// The effective Error Reduction Parameter applied for calculating regularization forces
    /// on contacts.
    ///
    /// This parameter is computed automatically from `contact_natural_frequency`,
    /// `contact_damping_ratio` and the substep length.
    pub fn contact_erp(&self) -> f32 {
        self.dt * self.contact_erp_inv_dt()
    }

    /// The joint's spring angular frequency for constraint regularization.
    pub fn joint_angular_frequency(&self) -> f32 {
        self.joint_natural_frequency * TWO_PI
    }

    /// The `joint_erp` coefficient, multiplied by the inverse timestep length.
    pub fn joint_erp_inv_dt(&self) -> f32 {
        let ang_freq = self.joint_angular_frequency();
        ang_freq / (self.dt * ang_freq + 2.0 * self.joint_damping_ratio)
    }

    /// The effective Error Reduction Parameter applied for calculating regularization forces
    /// on joints.
    ///
    /// This parameter is computed automatically from `joint_natural_frequency`,
    /// `joint_damping_ratio` and the substep length.
    pub fn joint_erp(&self) -> f32 {
        self.dt * self.joint_erp_inv_dt()
    }

    /// The CFM factor to be used in the constraint resolution.
    ///
    /// This parameter is computed automatically from `contact_natural_frequency`,
    /// `contact_damping_ratio` and the substep length.
    pub fn contact_cfm_factor(&self) -> f32 {
        // Compute CFM assuming a critically damped spring multiplied by the damping ratio.
        // The logic is similar to `joint_cfm_coeff`.
        let contact_erp = self.contact_erp();
        if contact_erp == 0.0 {
            return 0.0;
        }
        let inv_erp_minus_one = 1.0 / contact_erp - 1.0;

        // let stiffness = 4.0 * damping_ratio * damping_ratio * projected_mass
        //     / (dt * dt * inv_erp_minus_one * inv_erp_minus_one);
        // let damping = 4.0 * damping_ratio * damping_ratio * projected_mass
        //     / (dt * inv_erp_minus_one);
        // let cfm = 1.0 / (dt * dt * stiffness + dt * damping);
        // NOTE: This simplifies to cfm = cfm_coeff / projected_mass:
        let cfm_coeff = inv_erp_minus_one * inv_erp_minus_one
            / ((1.0 + inv_erp_minus_one)
                * 4.0
                * self.contact_damping_ratio
                * self.contact_damping_ratio);

        // Furthermore, we use this coefficient inside of the impulse resolution.
        // Surprisingly, several simplifications happen there.
        // Let `m` the projected mass of the constraint.
        // Let `m'` the projected mass that includes CFM: `m' = 1 / (1 / m + cfm_coeff / m) = m / (1 + cfm_coeff)`
        // We have:
        // new_impulse = old_impulse - m' (delta_vel - cfm * old_impulse)
        //             = old_impulse - m / (1 + cfm_coeff) * (delta_vel - cfm_coeff / m * old_impulse)
        //             = old_impulse * (1 - cfm_coeff / (1 + cfm_coeff)) - m / (1 + cfm_coeff) * delta_vel
        //             = old_impulse / (1 + cfm_coeff) - m * delta_vel / (1 + cfm_coeff)
        //             = 1 / (1 + cfm_coeff) * (old_impulse - m * delta_vel)
        // So, setting cfm_factor = 1 / (1 + cfm_coeff).
        // We obtain:
        // new_impulse = cfm_factor * (old_impulse - m * delta_vel)
        //
        // The value returned by this function is this cfm_factor that can be used directly
        // in the constraint solver.
        1.0 / (1.0 + cfm_coeff)
    }

    /// The CFM (constraints force mixing) coefficient applied to all joints for constraints regularization.
    ///
    /// This parameter is computed automatically from `joint_natural_frequency`,
    /// `joint_damping_ratio` and the substep length.
    pub fn joint_cfm_coeff(&self) -> f32 {
        // Compute CFM assuming a critically damped spring multiplied by the damping ratio.
        // The logic is similar to `contact_cfm_factor`.
        let joint_erp = self.joint_erp();
        if joint_erp == 0.0 {
            return 0.0;
        }
        let inv_erp_minus_one = 1.0 / joint_erp - 1.0;
        inv_erp_minus_one * inv_erp_minus_one
            / ((1.0 + inv_erp_minus_one)
                * 4.0
                * self.joint_damping_ratio
                * self.joint_damping_ratio)
    }

    /// Amount of penetration the engine won't attempt to correct (default: `0.001` multiplied by
    /// `length_unit`).
    pub fn allowed_linear_error(&self) -> f32 {
        self.normalized_allowed_linear_error * self.length_unit
    }

    /// Maximum amount of penetration the solver will attempt to resolve in one timestep.
    ///
    /// This is equal to `normalized_max_corrective_velocity` multiplied by
    /// `length_unit`.
    pub fn max_corrective_velocity(&self) -> f32 {
        if self.normalized_max_corrective_velocity != MAX_FLT {
            self.normalized_max_corrective_velocity * self.length_unit
        } else {
            MAX_FLT
        }
    }

    /// The maximal distance separating two objects that will generate predictive contacts
    /// (default: `0.002m` multiplied by `length_unit`).
    pub fn prediction_distance(&self) -> f32 {
        self.normalized_prediction_distance * self.length_unit
    }
}
