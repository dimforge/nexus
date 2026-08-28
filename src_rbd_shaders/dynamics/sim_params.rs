//! Simulation and constraint regularization parameters.

use crate::MAX_FLT;

/// Two times pi (2π), used for converting natural frequency to angular frequency.
pub const TWO_PI: f32 = core::f32::consts::TAU;

/// Precomputed soft-constraint coefficients (contact + joint), matching rapier's
/// TGS-soft `SpringCoefficients`. Computed once per step on the host from
/// [`RbdSimParams`] and passed to the multibody contact/joint kernels as a small
/// uniform. Exactly 48 bytes (12 scalars) for std140 uniform layout.
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
    /// Geometric slop distance.
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
    /// [`Self::erp_inv_dt`] for contacts touching a fixed body.
    pub static_erp_inv_dt: f32,
    /// [`Self::cfm_factor`] for contacts touching a fixed body.
    pub static_cfm_factor: f32,
    /// Coefficient in `[0, 1]` applied to a contact impulse before it is
    /// re-used as the next substep's (or next frame's) initial guess.
    pub warmstart_coefficient: f32,
    /// Unused; keeps the uniform a 16-byte multiple.
    pub _padding1: f32,
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
            static_erp_inv_dt: params.static_contact_erp_inv_dt(),
            static_cfm_factor: params.static_contact_cfm_factor(),
            warmstart_coefficient: params.warmstart_coefficient,
            _padding1: 0.0,
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
    /// (default `10.0`).
    pub contact_damping_ratio: f32,

    /// > 0: the natural frequency used by the springs for contact constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly at the
    /// expense of potential jitter effects due to overshooting. In order to make the simulation
    /// look stiffer, it is recommended to increase the `contact_damping_ratio` instead of this
    /// value.
    /// (default: `30.0`).
    pub contact_natural_frequency: f32,

    /// [`Self::contact_damping_ratio`] for contacts touching a fixed body.
    /// (default `10.0`).
    pub static_contact_damping_ratio: f32,

    /// [`Self::contact_natural_frequency`] for contacts touching a fixed body.
    /// (default: `60.0`).
    pub static_contact_natural_frequency: f32,

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

    /// Geometric slop distance (default: `0.005m`).
    ///
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_allowed_linear_error: f32,

    /// Maximum speed at which contact penetration is pushed out by the biased solve (default: `3.0`).
    ///
    /// Capping this recovery velocity keeps deep penetrations from being resolved explosively.
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_max_corrective_velocity: f32,

    /// The maximal distance separating two objects that will generate predictive contacts
    /// (default: `0.02m`)
    ///
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_prediction_distance: f32,

    /// Maximum linear velocity a body may have after each solver substep (default: `400.0` m/s).
    /// Bounding per-step travel keeps speculative contacts reliable; set to `f32::MAX` to disable.
    ///
    /// This value is implicitly scaled by `length_unit`.
    pub normalized_max_linear_velocity: f32,

    /// The number of solver iterations run by the constraints solver for calculating forces (default: `4`).
    pub num_solver_iterations: u32,

    /// Minimum cosine between two manifold normals for the contact-reduction
    /// pass to cluster them (default: `0.996`, ~5.1 degrees, matching rapier).
    ///
    /// `-1.0` merges every manifold of a collider pair regardless of normal:
    /// cheaper, but one averaged normal then stands in for a ridge or a step.
    pub contact_merge_cos: f32,
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
            contact_damping_ratio: 10.0,
            static_contact_natural_frequency: 60.0,
            static_contact_damping_ratio: 10.0,
            joint_natural_frequency: 1.0e6,
            joint_damping_ratio: 1.0,
            warmstart_coefficient: 1.0,
            num_solver_iterations: 4,
            normalized_allowed_linear_error: 0.005,
            normalized_max_corrective_velocity: 3.0,
            normalized_prediction_distance: 0.02,
            contact_merge_cos: crate::broad_phase::COS_MERGE_ANGLE,
            normalized_max_linear_velocity: 400.0,
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

    /// The soft-constraint `erp / dt` for a spring with the given natural
    /// frequency (Hz) and damping ratio.
    fn spring_erp_inv_dt(&self, natural_frequency: f32, damping_ratio: f32) -> f32 {
        let ang_freq = natural_frequency * TWO_PI;
        ang_freq / (self.dt * ang_freq + 2.0 * damping_ratio)
    }

    /// The soft-constraint CFM factor `1 / (1 + cfm_coeff)` for a spring with
    /// the given natural frequency (Hz) and damping ratio.
    ///
    /// See [`Self::contact_cfm_factor`] for the derivation.
    fn spring_cfm_factor(&self, natural_frequency: f32, damping_ratio: f32) -> f32 {
        let erp = self.dt * self.spring_erp_inv_dt(natural_frequency, damping_ratio);
        if erp == 0.0 {
            return 0.0;
        }
        let inv_erp_minus_one = 1.0 / erp - 1.0;
        let cfm_coeff = inv_erp_minus_one * inv_erp_minus_one
            / ((1.0 + inv_erp_minus_one) * 4.0 * damping_ratio * damping_ratio);
        1.0 / (1.0 + cfm_coeff)
    }

    /// Computes the contact constraint angular frequency (rad/s).
    pub fn contact_angular_frequency(&self) -> f32 {
        self.contact_natural_frequency * TWO_PI
    }

    /// The `contact_erp` coefficient, multiplied by the inverse timestep length.
    pub fn contact_erp_inv_dt(&self) -> f32 {
        self.spring_erp_inv_dt(self.contact_natural_frequency, self.contact_damping_ratio)
    }

    /// [`Self::contact_erp_inv_dt`] for contacts touching a fixed body
    /// (rapier's `static_contact_softness`).
    pub fn static_contact_erp_inv_dt(&self) -> f32 {
        self.spring_erp_inv_dt(
            self.static_contact_natural_frequency,
            self.static_contact_damping_ratio,
        )
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
        self.spring_cfm_factor(self.contact_natural_frequency, self.contact_damping_ratio)
    }

    /// [`Self::contact_cfm_factor`] for contacts touching a fixed body
    /// (rapier's `static_contact_softness`).
    pub fn static_contact_cfm_factor(&self) -> f32 {
        self.spring_cfm_factor(
            self.static_contact_natural_frequency,
            self.static_contact_damping_ratio,
        )
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

    /// Geometric slop distance (default: `0.005` multiplied by `length_unit`).
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
    /// (default: `0.02m` multiplied by `length_unit`).
    pub fn prediction_distance(&self) -> f32 {
        self.normalized_prediction_distance * self.length_unit
    }

    /// Maximum linear velocity a body may have after each solver substep.
    ///
    /// This is equal to `normalized_max_linear_velocity` multiplied by
    /// `length_unit`, or `f32::MAX` when the linear speed cap is disabled.
    pub fn max_linear_velocity(&self) -> f32 {
        if self.normalized_max_linear_velocity != MAX_FLT {
            self.normalized_max_linear_velocity * self.length_unit
        } else {
            MAX_FLT
        }
    }

    /// Per-substep angular speed cap: bounds the total rotation per step to `π/4`
    /// regardless of the substep count.
    pub fn max_angular_velocity(&self) -> f32 {
        const MAX_ROTATION: f32 = core::f32::consts::FRAC_PI_4;
        MAX_ROTATION * self.inv_dt() / self.num_solver_iterations.max(1) as f32
    }
}
