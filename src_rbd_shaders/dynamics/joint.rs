//! Joint data structures
//!
//! This module defines the data structures for articulated joints between rigid bodies.

use crate::Pose;

#[cfg(feature = "dim2")]
/// Spatial dimension (3 DOFs in 2D: 2 linear + 1 angular).
pub const SPATIAL_DIM: usize = 3;

#[cfg(feature = "dim3")]
/// Spatial dimension (6 DOFs in 3D: 3 linear + 3 angular).
pub const SPATIAL_DIM: usize = 6;

#[cfg(feature = "dim2")]
/// Bitmask for linear axes (X and Y in 2D).
pub const LIN_AXES_MASK: u32 = 1 + (1 << 1);

#[cfg(feature = "dim2")]
/// Bitmask for angular axis (Z rotation in 2D).
pub const ANG_AXES_MASK: u32 = 1 << 2;

#[cfg(feature = "dim3")]
/// Bitmask for linear axes (X, Y, Z in 3D).
pub const LIN_AXES_MASK: u32 = 1 + (1 << 1) + (1 << 2);

#[cfg(feature = "dim3")]
/// Bitmask for angular axes (Rx, Ry, Rz in 3D).
pub const ANG_AXES_MASK: u32 = (1 << 3) + (1 << 4) + (1 << 5);

/// An impulse-based joint connecting two rigid bodies.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ImpulseJoint {
    /// Index of the first body.
    pub body_a: u32,
    /// Index of the second body.
    pub body_b: u32,
    /// Padding for alignment (GenericJoint starts with Pose which has 16-byte alignment in 3D).
    #[cfg(feature = "dim3")]
    pub padding: [u32; 2],
    /// Joint configuration data.
    pub data: GenericJoint,
}

/// A generic (6 DOFs in 3D or 3 DOFs in 2D) joint.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct GenericJoint {
    /// The joint's frame, expressed in the first rigid-body's local-space.
    pub local_frame_a: Pose,
    /// The joint's frame, expressed in the second rigid-body's local-space.
    pub local_frame_b: Pose,
    /// The degrees-of-freedoms locked by this joint.
    pub locked_axes: u32,
    /// The degrees-of-freedoms limited by this joint.
    pub limit_axes: u32,
    /// The degrees-of-freedoms motorised by this joint.
    pub motor_axes: u32,
    /// The coupled degrees of freedom of this joint.
    ///
    /// Note that coupling degrees of freedoms (DoF) changes the interpretation of the coupled joint's limits and motors.
    /// If multiple linear DoF are limited/motorized, only the limits/motor configuration for the first
    /// coupled linear DoF is applied to all coupled linear DoF. Similarly, if multiple angular DoF are limited/motorized
    /// only the limits/motor configuration for the first coupled angular DoF is applied to all coupled angular DoF.
    pub coupled_axes: u32,
    /// The limits, along each degree of freedoms of this joint.
    ///
    /// Note that the limit must also be explicitly enabled by the `limit_axes` bitmask.
    /// For coupled degrees of freedoms (DoF), only the first linear (resp. angular) coupled DoF limit and `limit_axis`
    /// bitmask is applied to the coupled linear (resp. angular) axes.
    pub limits: [JointLimits; SPATIAL_DIM],
    /// The motors, along each degree of freedoms of this joint.
    ///
    /// Note that the motor must also be explicitly enabled by the `motor_axes` bitmask.
    /// For coupled degrees of freedoms (DoF), only the first linear (resp. angular) coupled DoF motor and `motor_axes`
    /// bitmask is applied to the coupled linear (resp. angular) axes.
    pub motors: [JointMotor; SPATIAL_DIM],
}

/// Limits that restrict a joint's range of motion along one axis.
///
/// Use to constrain how far a joint can move/rotate. Examples:
/// - Door that only opens 90°: revolute joint with limits `[0.0, PI/2.0]`
/// - Piston with 2-unit stroke: prismatic joint with limits `[0.0, 2.0]`
/// - Elbow that bends 0-150°: revolute joint with limits `[0.0, 5*PI/6]`
///
/// When a joint hits its limit, forces are applied to prevent further movement in that direction.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct JointLimits {
    /// Minimum allowed value (angle for revolute, distance for prismatic).
    pub min: f32,
    /// Maximum allowed value (angle for revolute, distance for prismatic).
    pub max: f32,
    /// Internal: impulse being applied to enforce the limit.
    pub impulse: f32,
}

/// A powered motor that drives a joint toward a target position/velocity.
///
/// Motors add actuation to joints - they apply forces to make the joint move toward
/// a desired state. Think of them as servos, electric motors, or hydraulic actuators.
///
/// ## Two control modes
///
/// 1. **Velocity control**: Set `target_vel` to make the motor spin/slide at constant speed
/// 2. **Position control**: Set `target_pos` with `stiffness`/`damping` to reach a target angle/position
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct JointMotor {
    /// Target velocity (units/sec for prismatic, rad/sec for revolute).
    pub target_vel: f32,
    /// Target position (units for prismatic, radians for revolute).
    pub target_pos: f32,
    /// Spring constant - how strongly to pull toward target position.
    pub stiffness: f32,
    /// Damping coefficient - resistance to motion (prevents oscillation).
    pub damping: f32,
    /// Maximum force the motor can apply (Newtons for prismatic, Nm for revolute).
    pub max_force: f32,
    /// Internal: current impulse being applied.
    pub impulse: f32,
    /// Force-based or acceleration-based motor model.
    pub model: u32,
}

/// Spring constants auto-scale with mass (easier to tune, recommended).
pub const ACCELERATION_BASED: u32 = 0;
/// Spring constants produce absolute forces (mass-dependent).
pub const FORCE_BASED: u32 = 1;

/// Parameters computed from motor configuration for constraint solving.
#[derive(Clone, Copy, Default)]
pub struct MotorParameters {
    pub erp_inv_dt: f32,
    pub cfm_coeff: f32,
    pub cfm_gain: f32,
    pub target_pos: f32,
    pub target_vel: f32,
    pub max_impulse: f32,
}

/// Helper function for pseudo inverse.
fn pseudo_inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

impl JointMotor {
    /// Computes motor parameters from this motor's configuration.
    pub fn motor_params(&self, dt: f32) -> MotorParameters {
        if self.model == ACCELERATION_BASED {
            let erp_inv_dt = self.stiffness * pseudo_inv(dt * self.stiffness + self.damping);
            let cfm_coeff = pseudo_inv(dt * dt * self.stiffness + dt * self.damping);

            MotorParameters {
                erp_inv_dt,
                cfm_coeff,
                cfm_gain: 0.0,
                target_pos: self.target_pos,
                target_vel: self.target_vel,
                max_impulse: self.max_force * dt,
            }
        } else {
            // FORCE_BASED
            let erp_inv_dt = self.stiffness * pseudo_inv(dt * self.stiffness + self.damping);
            let cfm_gain = pseudo_inv(dt * dt * self.stiffness + dt * self.damping);

            MotorParameters {
                erp_inv_dt,
                cfm_coeff: 0.0,
                cfm_gain,
                target_pos: self.target_pos,
                target_vel: self.target_vel,
                max_impulse: self.max_force * dt,
            }
        }
    }
}
