//! Joint constraint data structures
//!
//! This module defines the constraint structures used by the joint solver.

use crate::{AngVector, Vector};

use super::joint::SPATIAL_DIM;

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::{Mat4, Vec2};

use khal_std::glamx::UVec3;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::Pose;
use crate::utils::{BatchIndices, Slice};
use khal_std::index::MaybeIndexUnchecked;

use super::body::{LocalMassProperties, Velocity, WorldMassProperties};
use super::joint::ImpulseJoint;
use super::joint_constraint_builder::JointConstraintBuilder;
use super::sim_params::RbdSimParams;

const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of constraints an impulse joint can generate.
///
/// TODO(perf): in most cases, SPATIAL_DIM * 2 is very overkill. That would be
///             a free joint with both limits and motors enabled along all axes.
///             The most common case would be SPATIAL_DIM + 1 where a prismatic
///             or revolute joint has a motor.
const MAX_CONSTRAINTS_PER_JOINT: usize = SPATIAL_DIM * 2;

/// Solver body data for joint constraints.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct JointSolverBody {
    /// Inverse mass along each axis.
    pub im: Vector,

    #[cfg(feature = "dim2")]
    /// Inverse inertia (scalar in 2D).
    pub ii: f32,

    #[cfg(feature = "dim3")]
    /// Inverse inertia tensor (3x3 matrix in 3D).
    pub ii: Mat4,

    // TODO: is this still needed now that the solver body poses are expressed at the center of mass?
    /// World-space center of mass.
    pub world_com: Vector,

    /// Index in solver velocity array.
    pub solver_vel: u32,
}

impl JointSolverBody {
    pub fn ii_mul(&self, v: AngVector) -> AngVector {
        #[cfg(feature = "dim2")]
        return self.ii * v;
        #[cfg(feature = "dim3")]
        return (self.ii * v.extend(0.0)).truncate();
    }
}

/// A joint constraint between two rigid bodies.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")] // Same as 3D but with different field ordering to avoid padding.
pub struct JointConstraint {
    /// Index of body A in solver velocity array.
    pub solver_vel_a: u32,
    /// Index of body B in solver velocity array.
    pub solver_vel_b: u32,
    /// Inverse mass of body A.
    pub im_a: Vector,
    /// Inverse mass of body B.
    pub im_b: Vector,

    /// The constraints for a joint. Up to 6 in 3D, and up to 3 in 2D.
    pub elements: [JointConstraintElement; MAX_CONSTRAINTS_PER_JOINT],
    /// The number of active `JointConstraint::elements`.
    pub len: u32,
    pub padding: [u32; 1],
}

/// A joint constraint between two rigid bodies.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")] // Same as 2D but with different field ordering to avoid padding.
pub struct JointConstraint {
    /// Inverse mass of body A.
    pub im_a: Vector,
    /// Index of body A in solver velocity array.
    pub solver_vel_a: u32,
    /// Inverse mass of body B.
    pub im_b: Vector,
    /// Index of body B in solver velocity array.
    pub solver_vel_b: u32,

    /// The constraints for a joint. Up to 6 in 3D, and up to 3 in 2D.
    pub elements: [JointConstraintElement; MAX_CONSTRAINTS_PER_JOINT],
    /// The number of active `JointConstraint::elements`.
    pub len: u32,
    pub padding: [u32; 3],
}

impl Default for JointConstraint {
    fn default() -> Self {
        Self {
            solver_vel_a: 0,
            solver_vel_b: 0,
            im_a: Vector::ZERO,
            im_b: Vector::ZERO,
            elements: [JointConstraintElement::default(); MAX_CONSTRAINTS_PER_JOINT],
            len: 0,
            padding: [0; _],
        }
    }
}

/// A single element (DOF) of a joint constraint.
// NOTE: field order has been selected meticulously to reduce padding in both 2D and 3D versions.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct JointConstraintElement {
    /// Linear Jacobian direction.
    pub lin_jac: Vector,
    /// Joint ID for writeback.
    pub joint_id: u32,
    /// Angular Jacobian for body A.
    pub ang_jac_a: AngVector,
    /// Accumulated impulse.
    pub impulse: f32,
    /// Angular Jacobian for body B.
    pub ang_jac_b: AngVector,
    /// Inverse effective mass (1 / m_eff).
    pub inv_lhs: f32,
    /// Angular Jacobian for body A multiplied by inverse inertia.
    pub ii_ang_jac_a: AngVector,
    /// Right-hand side (target velocity).
    pub rhs: f32,
    /// Angular Jacobian for body B multiplied by inverse inertia.
    pub ii_ang_jac_b: AngVector,
    /// Right-hand side without bias.
    pub rhs_wo_bias: f32,
    /// CFM gain for soft constraints.
    pub cfm_gain: f32,
    /// CFM coefficient for soft constraints.
    pub cfm_coeff: f32,
    #[cfg(feature = "dim2")]
    pub padding: u32,
    /// Impulse bounds (min, max).
    pub impulse_bounds: Vec2,
}

/// Initializes joint constraint builders and constraints.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] joints: &[ImpulseJoint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    builders: &mut [JointConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &mut [JointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    local_mprops: &[LocalMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;

    let joints = batch_ids.impulse_joints_batch(batch_id, joints);
    let mut builders = batch_ids.impulse_joints_batch_mut(batch_id, builders);
    let mut constraints = batch_ids.impulse_joints_batch_mut(batch_id, constraints);
    let local_mprops = batch_ids.coll_batch(batch_id, local_mprops);

    let len = batch_ids.impulse_joints_len;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        let joint = &joints[idx];

        // Mirror rapier `GenericJoint::transform_to_solver_body_space`: the
        // joint's local anchor frames are expressed relative to the body's
        // origin, but the solver works in COM-centered space, so subtract the
        // body local center of mass from each anchor's translation.
        // TODO: handle the rapier "is_fixed" branch (`local_frame = body_pose * local_frame`).
        let mut joint_data = joint.data;
        joint_data.local_frame_a.translation -= local_mprops[joint.body_a as usize].com;
        joint_data.local_frame_b.translation -= local_mprops[joint.body_b as usize].com;

        builders[idx] = JointConstraintBuilder {
            body1: joint.body_a,
            body2: joint.body_b,
            joint_id: i,
            joint: joint_data,
            constraint_id: i,
        };

        constraints[idx].solver_vel_a = joint.body_a;
        constraints[idx].solver_vel_b = joint.body_b;
        constraints[idx].im_a = local_mprops[joint.body_a as usize].inv_mass;
        constraints[idx].im_b = local_mprops[joint.body_b as usize].inv_mass;
        constraints[idx].len = 0; // Constraint elements will be filled later.
    }
}

/// Updates joint constraints for a new substep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[JointConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &mut [JointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] all_params: &[RbdSimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let params = all_params.at(batch_id as usize);

    let builders = batch_ids.impulse_joints_batch(batch_id, builders);
    let mut constraints = batch_ids.impulse_joints_batch_mut(batch_id, constraints);
    let poses = batch_ids.coll_batch(batch_id, poses);
    let mprops = batch_ids.coll_batch(batch_id, mprops);

    let len = batch_ids.impulse_joints_len;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        builders[idx].update_constraint(&mut constraints[idx], &poses, &mprops, params);
    }
}

/// Solves joint constraints.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solve_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints: &mut [JointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] all_color_groups: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] curr_color: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    // 0 for the stabilization sweep (bias-free rhs); non-zero for the biased
    // sweep. Replaces the former `gpu_remove_joint_bias` full-buffer pass.
    #[spirv(uniform, descriptor_set = 0, binding = 5)] use_bias: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;

    let mut constraints = batch_ids.impulse_joints_batch_mut(batch_id, constraints);
    let mut solver_vels = batch_ids.coll_batch_mut(batch_id, solver_vels);
    let use_bias = *use_bias != 0;

    let color = *curr_color as usize;
    // Coloring is identical across batches (enforced on the host), so the
    // color-group prefix sums live in a single-batch buffer read at offset 0.
    let color_groups = Slice(all_color_groups, 0);

    let start = if color > 0 {
        color_groups[color - 1]
    } else {
        0
    };
    let end = color_groups[color];

    for i in StepRng::new(start + invocation_id.x..end, num_threads) {
        constraints[i as usize].solve_joint_constraint(&mut solver_vels, use_bias);
    }
}
