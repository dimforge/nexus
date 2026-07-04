//! Rigid-body dynamics data structures and integration.

use crate::{AngVector, Pose, Rotation, Vector};
#[cfg(feature = "dim3")]
use crate::{rotation_from_scaled_axis, rotation_renormalize_fast, rotation_to_matrix};

#[cfg(feature = "dim3")]
use glamx::{Mat3, Mat4, Quat, Vec3};
#[cfg(feature = "dim2")]
use glamx::{Rot2, Vec2};

/// The mass-properties of a rigid-body in local (body-space) coordinates.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(bytemuck::Pod, bytemuck::Zeroable, Debug)
)]
#[repr(C)]
pub struct LocalMassProperties {
    #[cfg(feature = "dim3")]
    /// The reference frame for the principal inertia axes (3D only).
    pub inertia_ref_frame: Quat,

    #[cfg(feature = "dim3")]
    /// The inverse principal inertia components (3D only).
    pub inv_principal_inertia: Vec3,

    #[cfg(feature = "dim3")]
    pub padding0: u32,

    /// The rigid-body's inverse mass along each coordinate axis.
    pub inv_mass: Vector,

    #[cfg(feature = "dim3")]
    pub padding1: u32,

    /// The rigid-body's center of mass in local coordinates.
    pub com: Vector,

    pub padding2: u32,

    #[cfg(feature = "dim2")]
    /// The inverse inertia tensor (scalar in 2D).
    pub inv_inertia: f32,
}

impl Default for LocalMassProperties {
    fn default() -> Self {
        Self {
            #[cfg(feature = "dim3")]
            inertia_ref_frame: Quat::IDENTITY,
            #[cfg(feature = "dim3")]
            inv_principal_inertia: Vec3::ONE,
            #[cfg(feature = "dim3")]
            padding0: 0,
            inv_mass: Vector::ONE,
            #[cfg(feature = "dim3")]
            padding1: 0,
            com: Vector::ZERO,
            padding2: 0,
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
        }
    }
}

impl LocalMassProperties {
    #[cfg(feature = "dim2")]
    /// Updates world-space mass properties from local properties and current pose (2D version).
    pub fn to_world(&self, pose: &Pose) -> WorldMassProperties {
        let world_com = pose * self.com;

        WorldMassProperties {
            inv_inertia: self.inv_inertia,
            inv_mass: self.inv_mass,
            com: world_com,
            padding1: 0,
        }
    }

    #[cfg(feature = "dim3")]
    /// Updates world-space mass properties from local properties and current pose (3D version).
    pub fn to_world(&self, pose: &Pose) -> WorldMassProperties {
        let world_com = pose * self.com;
        let rot_mat = rotation_to_matrix(pose.rotation * self.inertia_ref_frame);
        let scaled_rot_mat = Mat3::from_cols(
            rot_mat.x_axis * self.inv_principal_inertia.x,
            rot_mat.y_axis * self.inv_principal_inertia.y,
            rot_mat.z_axis * self.inv_principal_inertia.z,
        );
        let world_inv_inertia = scaled_rot_mat * rot_mat.transpose();

        WorldMassProperties {
            inv_inertia: glamx::Mat4::from_mat3(world_inv_inertia),
            inv_mass: self.inv_mass,
            com: world_com,
            padding0: 0,
            padding1: 0,
        }
    }
}

/// The mass-properties of a rigid-body in world-space coordinates.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct WorldMassProperties {
    #[cfg(feature = "dim3")]
    /// The inverse inertia tensor in world space (3x3 matrix in 3D).
    pub inv_inertia: Mat4, // NOTE using a Mat4 to avoid implicit padding.

    /// The rigid-body's inverse mass along each coordinate axis in world space.
    pub inv_mass: Vector,

    #[cfg(feature = "dim3")]
    pub padding0: u32,

    /// The rigid-body's center of mass in world-space coordinates.
    pub com: Vector,

    pub padding1: u32,

    #[cfg(feature = "dim2")]
    /// The inverse inertia tensor in world space (scalar in 2D).
    pub inv_inertia: f32,
}

impl WorldMassProperties {
    #[inline]
    pub fn inv_inertia_mul(&self, v: AngVector) -> AngVector {
        #[cfg(feature = "dim2")]
        return self.inv_inertia * v;
        // TODO PERF: this is ugly. Only needed to avoid internal implicit padding in Mat3.
        //            Is there a better option?
        //            Does this have a negative performance impact?
        #[cfg(feature = "dim3")]
        return (self.inv_inertia * v.extend(0.0)).truncate();
    }
}

impl Default for WorldMassProperties {
    fn default() -> Self {
        Self {
            #[cfg(feature = "dim3")]
            inv_inertia: Mat4::IDENTITY,
            inv_mass: Vector::ONE,
            #[cfg(feature = "dim3")]
            padding0: 0,
            com: Vector::ZERO,
            padding1: 0,
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
        }
    }
}

/// An impulse (instantaneous change in momentum).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Impulse {
    /// Linear impulse component (change in linear momentum).
    pub linear: Vector,

    /// Angular impulse component (change in angular momentum / torque impulse).
    pub angular: AngVector,
}

impl Impulse {
    pub fn new(linear: Vector, angular: AngVector) -> Self {
        Self { linear, angular }
    }
}

/// A force and torque applied to a rigid body.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Force {
    /// Linear force component.
    pub linear: Vector,

    /// Angular force component (torque).
    pub angular: AngVector,
}

impl Force {
    pub fn new(linear: Vector, angular: AngVector) -> Self {
        Self { linear, angular }
    }

    /// Integrates forces over a timestep to compute velocity changes (explicit Euler).
    pub fn integrate(
        &self,
        mprops: &WorldMassProperties,
        velocity: &Velocity,
        dt: f32,
    ) -> Velocity {
        let acc_lin = mprops.inv_mass * self.linear;
        let acc_ang = mprops.inv_inertia_mul(self.angular);

        Velocity::new(
            velocity.linear + acc_lin * dt,
            velocity.angular + acc_ang * dt,
        )
    }
}

/// Linear and angular velocity of a rigid body.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Velocity {
    /// Linear (translational) velocity.
    pub linear: Vector,
    #[cfg(feature = "dim3")]
    pub padding0: u32,

    /// Angular (rotational) velocity.
    pub angular: AngVector,
    pub padding1: u32,
}

impl Velocity {
    pub fn new(linear: Vector, angular: AngVector) -> Self {
        Self {
            linear,
            angular,
            #[cfg(feature = "dim3")]
            padding0: 0,
            padding1: 0,
        }
    }

    /// Computes the linear velocity at a specific point on a rigid body.
    pub fn velocity_at_point(&self, com: Vector, point: Vector) -> Vector {
        let lever_arm = point - com;

        #[cfg(feature = "dim2")]
        {
            self.linear + self.angular * Vec2::new(-lever_arm.y, lever_arm.x)
        }
        #[cfg(feature = "dim3")]
        {
            self.linear + self.angular.cross(lever_arm)
        }
    }

    /// Integrates velocity over a timestep to compute the new pose.
    pub fn integrate(&self, pose: &Pose, local_com: Vector, dt: f32) -> Pose {
        let init_com = pose * local_com;
        let init_tra = pose.translation;
        #[cfg(feature = "dim2")]
        let delta_ang = Rot2::new(self.angular * dt);
        #[cfg(feature = "dim3")]
        let delta_ang = rotation_from_scaled_axis(self.angular * dt);
        let delta_lin = self.linear * dt;
        let new_translation = init_com + delta_ang * (init_tra - init_com) + delta_lin;
        #[cfg(feature = "dim2")]
        let new_rotation = delta_ang * pose.rotation;
        #[cfg(feature = "dim3")]
        let new_rotation = rotation_renormalize_fast(delta_ang * pose.rotation);

        Pose::from_parts(new_translation, new_rotation)
    }

    /// Same as [`Self::integrate`] but with the angular part linearized and the local
    /// center-of-mass assumed to be zero.
    #[inline]
    #[cfg(feature = "dim2")]
    pub(crate) fn integrate_linearized(
        &self,
        dt: f32,
        translation: &mut Vector,
        rotation: &mut Rotation,
    ) {
        let dang = self.angular * dt;
        let new_cos = rotation.re - dang * rotation.im;
        let new_sin = rotation.im + dang * rotation.re;
        *rotation = Rot2::from_cos_sin_unchecked(new_cos, new_sin);
        // NOTE: don't use renormalize_fast since the linearization might cause more drift.
        rotation.normalize_mut();
        *translation += self.linear * dt;
    }

    /// Same as [`Self::integrate`] but with the angular part linearized and the local
    /// center-of-mass assumed to be zero.
    #[inline]
    #[cfg(feature = "dim3")]
    pub(crate) fn integrate_linearized(
        &self,
        dt: f32,
        translation: &mut Vector,
        rotation: &mut Rotation,
    ) {
        // Rotations linearization is inspired from
        // https://ahrs.readthedocs.io/en/latest/filters/angular.html (not using the matrix form).
        let hang = self.angular * (dt * 0.5);
        // Quaternion identity + `hang` seen as a quaternion.
        let id_plus_hang = Rotation::from_xyzw(hang.x, hang.y, hang.z, 1.0);
        *rotation = id_plus_hang * *rotation;
        *rotation = rotation.normalize();
        *translation += self.linear * dt;
    }

    /// Applies an impulse to a rigid body, computing the resulting velocity change.
    pub fn apply_impulse(&self, mprops: &WorldMassProperties, imp: &Impulse) -> Velocity {
        let acc_lin = mprops.inv_mass * imp.linear;
        let acc_ang = mprops.inv_inertia_mul(imp.angular);
        Velocity::new(self.linear + acc_lin, self.angular + acc_ang)
    }
}

/// Complete state of a rigid body (pose and velocity).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct RigidBodyState {
    /// The rigid-body's pose (position, orientation, and uniform scale).
    pub pose: Pose,

    /// The rigid-body's velocity (linear and angular).
    pub velocity: Velocity,
}
