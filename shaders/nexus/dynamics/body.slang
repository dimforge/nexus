module body;

// NOTE: operator overloading on rotations doesn’t compile unless we import the modules explicitly.
import gla.geometry.quat;
import gla.geometry.rot2;
import nexus.aliases;


/// The mass-properties of a rigid-body.
/// Note that the mass-properties may be expressed either in the rigid-body’s local-space or in world-space,
/// depending on its provenance. Usually, the world-space and local-space mass-properties will be stored in
/// two separate buffers.
public struct BodyMassProperties {
    // TODO: a representation with Quaternion & vec3 (for frame & principal inertia) would be much more compact and make
    //       this struct have the size of a mat4x4
#if DIM == 2
   /// The rigid-body’s inverse inertia tensor.
   public float inv_inertia;
#else
   public Mat inv_inertia;
#endif
   /// The rigid-body’s inverse mass along each coordinate axis.
   ///
   /// Allowing different values along each axis allows the user to specify 0 along each axis.
   /// By setting zero, the linear motion along the corresponding world-space axis will be locked.
   public Vect inv_mass;
   /// The rigid-body’s center of mass.
   public Vect com;
}

/// An impulse (linear and angular/torque)
public struct Impulse {
    /// A linear impulse.
    public Vect linear;
    /// An angular impulse (torque impulse).
    public AngVect angular;
}

/// A force and torque.
public struct Force {
    /// A linear force.
    public Vect linear;
    /// An angular force (torque).
    public AngVect angular;
}

/// A linear and angular velocity.
public struct BodyVelocity {
    /// The linear (translational) part of the velocity.
    public Vect linear;
    /// The angular (rotational) part of the velocity.
    public AngVect angular;
}

/// A rigid-body pose and its velocity.
public struct BodyState {
    /// The rigid-body’s pose (translation, rotation, uniform scale).
    public Pose pose;
    /// The rigid-body’s velocity (translational and rotational).
    public BodyVelocity velocity;
}

/// Computes new velocities after applying the given impulse.
public func apply_impulse(mprops: BodyMassProperties, velocity: BodyVelocity, imp: Impulse) -> BodyVelocity {
    let acc_lin = mprops.inv_mass * imp.linear;
    let acc_ang = mul(imp.angular, mprops.inv_inertia);
    return BodyVelocity(velocity.linear + acc_lin, velocity.angular + acc_ang);
}


/// Computes new velocities after integrating forces by a timestep equal to `dt`.
public func integrate_forces(mprops: BodyMassProperties, velocity: BodyVelocity, force: Force, dt: float) -> BodyVelocity {
    let acc_lin = mprops.inv_mass * force.linear;
    let acc_ang = mul(force.angular, mprops.inv_inertia);
    return BodyVelocity(velocity.linear + acc_lin * dt, velocity.angular + acc_ang * dt);
}

#if DIM == 2
/// Computes a new pose after integrating velocitie by a timestep equal to `dt`.
public func integrate_velocity(pose: Pose, vels: BodyVelocity, local_com: Vect, dt: float) -> Pose {
    let init_com = pose.mul_pt(local_com);
    let init_tra = pose.translation;
    let init_scale = pose.scale;

    let delta_ang = Rot.from_angle(vels.angular * dt);
    let delta_lin = vels.linear * dt;

    let new_translation =
        init_com + (delta_ang * (init_tra - init_com)) * init_scale + delta_lin;
    let new_rotation = delta_ang * pose.rotation;

    return Pose(new_rotation, new_translation, init_scale);
}

/// Computes the new world-space mass-properties based on the local-space mass-properties and its transform.
public func update_mprops(pose: Pose, local_mprops: BodyMassProperties) -> BodyMassProperties {
    let world_com = pose.mul_pt(local_mprops.com);
    return BodyMassProperties(local_mprops.inv_inertia, local_mprops.inv_mass, world_com);
}

/// Computes the linear velocity at a given point.
// TODO: for some reasons the slang compiler doesn’t want `Vect` as the return type here…
public func body_velocity_at_point(Vect center_of_mass, BodyVelocity vels, Vect point) -> float2 {
    let lever_arm = point - center_of_mass;
    return vels.linear + vels.angular * float2(-lever_arm.y, lever_arm.x);
}
#else
/// Computes a new pose after integrating velocitie by a timestep equal to `dt`.
public func integrate_velocity(pose: Pose, vels: BodyVelocity, local_com: Vect, dt: float) -> Pose {
    let init_com = pose.mul_pt(local_com);
    let init_tra = pose.translation_scale.xyz;
    let init_scale = pose.translation_scale.w;

    let delta_ang = Rot.from_scaled_axis(vels.angular * dt);
    let delta_lin = vels.linear * dt;

    let new_translation =
        init_com + (delta_ang * (init_tra - init_com)) * init_scale + delta_lin;
    let new_rotation = (delta_ang * pose.rotation).renormalize_fast();

    return Pose(new_rotation, float4(new_translation, init_scale));
}

/// Computes the linear velocity at a given point.
// TODO: for some reasons the slang compiler doesn’t want `Vect` as the return type here…
public func body_velocity_at_point(Vect com, BodyVelocity vels, Vect point) -> float3 {
    return vels.linear + cross(vels.angular, point - com);
}

/// Computes the new world-space mass-properties based on the local-space mass-properties and its transform.
public func update_mprops(pose: Pose, local_mprops: BodyMassProperties) -> BodyMassProperties {
    let world_com = pose.mul_pt(local_mprops.com);
    let rot_mat = pose.rotation.to_matrix();
    let world_inv_inertia = rot_mat * local_mprops.inv_inertia * transpose(rot_mat);

    return BodyMassProperties(world_inv_inertia, local_mprops.inv_mass, world_com);
}
#endif