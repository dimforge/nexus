//! Rigid body impulse accumulation and integration kernels for coupling MPM particles
//! with rigid bodies.

use crate::grid::grid::Grid;
use crate::nexus_rbd_shaders::dynamics::{
    Impulse, LocalMassProperties, Velocity, WorldMassProperties,
};
use crate::solver::p2g::IntegerImpulse;
use crate::solver::params::SimulationParams;
use crate::{AngVector, IVector, Pose, Vector, ang_length};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Scaling factor for float-to-integer impulse conversion.
pub const FLOAT_TO_INT_FACTOR: f32 = 1e5;

/// Converts a float value to its integer-quantized representation.
#[inline]
pub fn flt2int(flt: f32) -> i32 {
    (flt * FLOAT_TO_INT_FACTOR) as i32
}

/// Converts an integer-quantized value back to float.
#[inline]
pub fn int2flt(i: i32) -> f32 {
    i as f32 / FLOAT_TO_INT_FACTOR
}

impl IntegerImpulse {
    /// Converts this integer-quantized impulse to a floating-point [`Impulse`].
    #[inline]
    pub fn to_float(&self) -> Impulse {
        #[cfg(feature = "dim2")]
        {
            Impulse::new(
                Vec2::new(int2flt(self.linear_x), int2flt(self.linear_y)),
                int2flt(self.angular),
            )
        }
        #[cfg(feature = "dim3")]
        {
            Impulse::new(
                Vec3::new(
                    int2flt(self.linear_x),
                    int2flt(self.linear_y),
                    int2flt(self.linear_z),
                ),
                Vec3::new(
                    int2flt(self.angular_x),
                    int2flt(self.angular_y),
                    int2flt(self.angular_z),
                ),
            )
        }
    }
}

/// Updates rigid body velocities and poses by applying accumulated impulses, then resets
/// the impulse accumulator for the next substep.
///
/// NOTE: numthreads(16) because we are currently limited to 16 bodies
/// due to the CPIC affinity bitmask size.
#[spirv_bindgen]
#[spirv(compute(threads(16)))]
pub fn gpu_rigid_impulses_update(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] sim_params: &SimulationParams,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mprops: &mut [WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    incremental_impulses: &mut [IntegerImpulse],
) {
    let id = invocation_id.x;

    if id < vels.len() as u32 {
        let idx = id as usize;
        let inc_impulse = incremental_impulses.at(idx).to_float();

        // Reset the incremental impulse to zero for the next substep.
        *incremental_impulses.at_mut(idx) = IntegerImpulse::default();

        // Apply impulse and integrate.
        let current_vel = vels.read(idx);
        let current_mprops = mprops.read(idx);
        let mut new_vel = current_vel.apply_impulse(&current_mprops, &inc_impulse);

        // Cap the velocities to not move more than a fraction of a cell-width in a given substep.
        let linvel_norm = new_vel.linear.length();
        let angvel_norm = ang_length(new_vel.angular);
        let lin_limit = 0.1 * grid.cell_width / sim_params.dt;
        let ang_limit = 1.0; // TODO: what's a good angular limit?

        let impulse_linear_len = inc_impulse.linear.length();
        let impulse_angular_len = ang_length(inc_impulse.angular);

        if impulse_linear_len != 0.0 || impulse_angular_len != 0.0 {
            if linvel_norm > lin_limit {
                new_vel.linear *= lin_limit / linvel_norm;
            }
            if angvel_norm > ang_limit {
                new_vel.angular *= ang_limit / angvel_norm;
            }
        }

        let current_pose = poses.read(idx);
        let local_mp = local_mprops.read(idx);
        let new_pose = new_vel.integrate(&current_pose, local_mp.com, sim_params.dt);

        // Apply gravity.
        // Construct a mask: 1.0 where inv_mass != 0.0, 0.0 otherwise.
        #[cfg(feature = "dim2")]
        let mass_mask = Vec2::new(
            (current_mprops.inv_mass.x != 0.0) as u32 as f32,
            (current_mprops.inv_mass.y != 0.0) as u32 as f32,
        );
        #[cfg(feature = "dim3")]
        let mass_mask = Vec3::new(
            (current_mprops.inv_mass.x != 0.0) as u32 as f32,
            (current_mprops.inv_mass.y != 0.0) as u32 as f32,
            (current_mprops.inv_mass.z != 0.0) as u32 as f32,
        );
        new_vel.linear += sim_params.gravity * mass_mask * sim_params.dt;

        vels.write(idx, new_vel);
        poses.write(idx, new_pose);
    }
}

/// Copies the MPM-integrated poses of the coupled bodies into the rigid-body
/// pipeline's body-pose buffer.
///
/// MPM keeps its own copy of every coupled body and integrates it each substep
/// in [`gpu_rigid_impulses_update`]; that copy is the one the sand collides
/// with. The rigid-body pipeline treats those bodies as static (zero inverse
/// mass), so without this writeback the copy that rendering and the broad phase
/// read stays frozen at the insertion pose.
///
/// `rbd_slots[i]` is the rigid-body slot mirroring coupled body `i`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_writeback_body_poses(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rbd_slots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rbd_poses: &mut [Pose],
) {
    let id = invocation_id.x;

    if id < rbd_slots.len() as u32 {
        let idx = id as usize;
        let slot = rbd_slots.read(idx) as usize;
        rbd_poses.write(slot, poses.read(idx));
    }
}

/// Updates world-space mass properties from local properties and current poses.
///
/// Also writes the updated center of mass into the incremental impulse buffer
/// so that P2G can access it without an extra binding.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_world_mass_properties(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mprops: &mut [WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    incremental_impulses: &mut [IntegerImpulse],
) {
    let id = invocation_id.x;

    if id < mprops.len() as u32 {
        let idx = id as usize;
        let local_mp = local_mprops.read(idx);
        let new_mprops = local_mp.to_world(poses.at(idx));
        incremental_impulses.at_mut(idx).com = new_mprops.com;
        mprops.write(idx, new_mprops);
    }
}
