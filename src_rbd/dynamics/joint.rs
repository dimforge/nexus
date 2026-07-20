//! Impulse-based joint constraints for articulated bodies.
//!
//! This module provides GPU-accelerated joint constraint solving, allowing bodies
//! to be connected with various joint types (revolute, prismatic, fixed, etc.).

use crate::math::Pose;
use crate::shaders::dynamics::{
    GpuInitJointConstraints, GpuRemoveJointBias,
    GpuSolveJointConstraints, GpuUpdateJointConstraints, ImpulseJoint, JointConstraint,
    JointConstraintBuilder, LocalMassProperties, RbdSimParams, Velocity, WorldMassProperties,
};
use bytemuck::Zeroable;
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

use {
    crate::rapier::dynamics::{
        GenericJoint as RapierGenericJoint, ImpulseJoint as RapierImpulseJoint, ImpulseJointSet,
        JointLimits as RapierJointLimits, JointMotor as RapierJointMotor, RigidBodyHandle,
    },
    crate::rapier::prelude::MotorModel,
    crate::shaders::dynamics::{GenericJoint, JointLimits, JointMotor},
    khal::BufferUsages,
    std::collections::HashMap,
};

fn convert_joint_limits(limits: RapierJointLimits<f32>) -> JointLimits {
    JointLimits {
        min: limits.min,
        max: limits.max,
        impulse: limits.impulse,
    }
}

fn convert_joint_motor(motor: RapierJointMotor) -> JointMotor {
    JointMotor {
        target_vel: motor.target_vel,
        target_pos: motor.target_pos,
        stiffness: motor.stiffness,
        damping: motor.damping,
        max_force: motor.max_force,
        impulse: motor.impulse,
        model: match motor.model {
            MotorModel::AccelerationBased => 0,
            MotorModel::ForceBased => 1,
        },
    }
}

fn convert_generic_joint(joint: RapierGenericJoint) -> GenericJoint {
    GenericJoint {
        local_frame_a: joint.local_frame1,
        local_frame_b: joint.local_frame2,
        locked_axes: joint.locked_axes.bits() as u32,
        limit_axes: joint.limit_axes.bits() as u32,
        motor_axes: joint.motor_axes.bits() as u32,
        coupled_axes: joint.coupled_axes.bits() as u32,
        limits: joint.limits.map(convert_joint_limits),
        motors: joint.motors.map(convert_joint_motor),
    }
}

fn convert_impulse_joint(
    joint: &RapierImpulseJoint,
    body_ids: &HashMap<RigidBodyHandle, u32>,
) -> ImpulseJoint {
    ImpulseJoint {
        body_a: body_ids[&joint.body1()],
        body_b: body_ids[&joint.body2()],
        #[cfg(feature = "dim3")]
        padding: [0, 0],
        data: convert_generic_joint(joint.data),
    }
}

/// A set of impulse joints simulated on the GPU.
pub struct GpuImpulseJointSet {
    /// Per-batch buffer stride (capacity) of the `joints` / `builders` /
    /// `constraints` slabs.
    len: u32,
    /// Number of *active* impulse joints per batch (the kernel loop bound).
    /// Identical across batches by the equal-topology invariant.
    num_active_joints: u32,
    num_colors: u32,
    /// Host copy of the per-color prefix sums in `color_groups` (identical
    /// across batches), used to size each per-color solve dispatch exactly.
    color_groups_cpu: Vec<u32>,
    color_groups: Tensor<u32>,
    joints: Tensor<ImpulseJoint>,
    builders: Tensor<JointConstraintBuilder>,
    constraints: Tensor<JointConstraint>,
}

impl GpuImpulseJointSet {
    /// Converts per-environment Rapier joints to GPU joints.
    ///
    /// Each environment can have different joints. The joints are padded to the max
    /// joint count across all environments, and graph coloring is done independently
    /// per environment.
    ///
    /// `environments` is a slice of `(impulse_joints, body_ids)`. The optional
    /// `multibody_groups` parameter — when provided — must be aligned with
    /// `environments` and gives, for each body local id, the group it belongs to:
    /// either its own id (free body) or a shared `multibody_id`. Bodies sharing
    /// the same group act as a single node for graph coloring, mirroring rapier:
    /// no two impulse-joint constraints sharing the same multibody group can be
    /// solved in parallel.
    pub fn from_rapier(
        backend: &GpuBackend,
        environments: &[(&ImpulseJointSet, &HashMap<RigidBodyHandle, u32>)],
    ) -> Self {
        Self::from_rapier_with_groups(backend, environments, &[])
    }

    /// Same as [`from_rapier`](Self::from_rapier) but with a per-environment
    /// multibody-grouping table (see method-level docs for the rule).
    pub fn from_rapier_with_groups(
        backend: &GpuBackend,
        environments: &[(&ImpulseJointSet, &HashMap<RigidBodyHandle, u32>)],
        multibody_groups: &[Vec<u32>],
    ) -> Self {
        Self::from_rapier_filtered(backend, environments, multibody_groups, &[])
    }

    /// Build the GPU set, optionally filtering out joints whose body1 OR
    /// body2 is part of any multibody. The skipped joints must instead be
    /// uploaded to the multibody side via
    /// `GpuMultibodySet::set_impulse_joints` — those go through the
    /// multibody generic-constraint path because the regular impulse
    /// solver can't propagate impulses through `M⁻¹·Jᵀ`.
    ///
    /// `is_mb_body[env][body_local_id]` should be `true` iff the body is
    /// part of any multibody. Empty (or shorter-than-`environments`) skip
    /// vector means "no skip" and falls back to the old behavior.
    pub fn from_rapier_filtered(
        backend: &GpuBackend,
        environments: &[(&ImpulseJointSet, &HashMap<RigidBodyHandle, u32>)],
        multibody_groups: &[Vec<u32>],
        is_mb_body: &[Vec<bool>],
    ) -> Self {
        let usage = BufferUsages::STORAGE;
        let num_batches = environments.len() as u32;

        let is_mb_for = |env_idx: usize, body_local: u32| -> bool {
            if env_idx >= is_mb_body.len() {
                return false;
            }
            is_mb_body[env_idx]
                .get(body_local as usize)
                .copied()
                .unwrap_or(false)
        };

        // Per-batch joint count drops the MB-touching joints, since they
        // get routed through the multibody constraint path.
        let mut filtered_lens: Vec<u32> = Vec::with_capacity(num_batches as usize);
        for (env_idx, (joints, body_ids)) in environments.iter().enumerate() {
            let mut count = 0u32;
            for (_, joint) in joints.iter() {
                let a = body_ids.get(&joint.body1()).copied();
                let b = body_ids.get(&joint.body2()).copied();
                let skip_a = a.map(|id| is_mb_for(env_idx, id)).unwrap_or(false);
                let skip_b = b.map(|id| is_mb_for(env_idx, id)).unwrap_or(false);
                if !skip_a && !skip_b {
                    count += 1;
                }
            }
            filtered_lens.push(count);
        }
        let max_joints = filtered_lens.iter().copied().max().unwrap_or(0);

        let mut global_num_colors = 0u32;

        // Per-environment sorted joints and color groups.
        let mut per_env_sorted_joints: Vec<Vec<ImpulseJoint>> = Vec::new();
        let mut per_env_color_groups: Vec<Vec<u32>> = Vec::new();
        // Per-environment structural signature (connectivity + locked/limit/
        // motor/coupled axis masks, in joint order) and coloring, used to
        // assert that every environment shares the same impulse-joint topology,
        // DOFs and coloring — the precondition for the single-batch color-group
        // buffer below.
        let mut per_env_structure: Vec<Vec<[u32; 6]>> = Vec::new();
        let mut per_env_colors: Vec<Vec<u32>> = Vec::new();

        for (env_idx, (joints, body_ids)) in environments.iter().enumerate() {
            let _len = filtered_lens[env_idx];

            // Convert joints, dropping any with at least one multibody side.
            let mut unsorted_gpu_joints = vec![];
            for (_, joint) in joints.iter() {
                let a = body_ids.get(&joint.body1()).copied();
                let b = body_ids.get(&joint.body2()).copied();
                let skip_a = a.map(|id| is_mb_for(env_idx, id)).unwrap_or(false);
                let skip_b = b.map(|id| is_mb_for(env_idx, id)).unwrap_or(false);
                if skip_a || skip_b {
                    continue;
                }
                unsorted_gpu_joints.push(convert_impulse_joint(joint, body_ids));
            }

            per_env_structure.push(
                unsorted_gpu_joints
                    .iter()
                    .map(|j| {
                        [
                            j.body_a,
                            j.body_b,
                            j.data.locked_axes,
                            j.data.limit_axes,
                            j.data.motor_axes,
                            j.data.coupled_axes,
                        ]
                    })
                    .collect(),
            );

            // Build the body-id → graph-group lookup. Without a multibody group
            // table, every body is its own node. With one, bodies that share a
            // multibody collapse to a single node so two impulse-joint contacts
            // touching different bodies of the same multibody must be in
            // different colors.
            let max_body_id = body_ids.values().copied().max().unwrap_or_default();
            let body_group: Vec<u32> =
                if env_idx < multibody_groups.len() && !multibody_groups[env_idx].is_empty() {
                    multibody_groups[env_idx].clone()
                } else {
                    (0..=max_body_id).collect()
                };
            let max_group = body_group.iter().copied().max().unwrap_or(0);

            // Run graph coloring on the multibody-grouped graph.
            let mut colors = vec![];
            let mut group_masks = vec![0u128; max_group as usize + 1];

            for joint in &unsorted_gpu_joints {
                let a = body_group[joint.body_a as usize] as usize;
                let b = body_group[joint.body_b as usize] as usize;
                let mask = group_masks[a] | group_masks[b];
                let color = mask.trailing_ones();
                colors.push(color);
                group_masks[a] |= 1 << color;
                group_masks[b] |= 1 << color;
            }

            let env_num_colors = colors
                .iter()
                .copied()
                .max()
                .map(|n| n + 1)
                .unwrap_or_default();
            let mut color_groups = vec![0u32; env_num_colors as usize];

            for color in &colors {
                color_groups[*color as usize] += 1;
            }

            // Prefix sum.
            for i in 0..color_groups.len().saturating_sub(1) {
                color_groups[i + 1] += color_groups[i];
            }

            // Bucket sort.
            let mut target = color_groups.clone();
            target.insert(0, 0);
            let mut sorted_gpu_joints = unsorted_gpu_joints.clone();

            for (joint, color) in unsorted_gpu_joints.iter().zip(colors.iter()) {
                sorted_gpu_joints[target[*color as usize] as usize] = *joint;
                target[*color as usize] += 1;
            }

            global_num_colors = global_num_colors.max(env_num_colors);

            per_env_sorted_joints.push(sorted_gpu_joints);
            per_env_color_groups.push(color_groups);
            per_env_colors.push(colors);
        }

        // Enforce identical impulse-joint structure across environments: same
        // connectivity + DOFs (axis masks) and therefore the same graph
        // coloring. This lets the color-group prefix sums be stored once
        // (single batch) and read at offset 0 by every batch in the solver.
        if let Some((first_struct, rest)) = per_env_structure.split_first() {
            for (i, s) in rest.iter().enumerate() {
                assert_eq!(
                    s,
                    first_struct,
                    "batched rbd requires identical impulse-joint topology/DOFs in every \
                     environment (env {} differs from env 0)",
                    i + 1
                );
            }
        }
        if let Some((first_colors, rest)) = per_env_colors.split_first() {
            for (i, c) in rest.iter().enumerate() {
                assert_eq!(
                    c,
                    first_colors,
                    "batched rbd requires identical impulse-joint coloring in every \
                     environment (env {} differs from env 0)",
                    i + 1
                );
            }
        }

        // Build flat joint buffer [num_batches * max_joints], padded with zeroed joints.
        let dummy_joint = ImpulseJoint::zeroed();
        let mut all_joints = Vec::with_capacity(num_batches as usize * max_joints as usize);
        for sorted_joints in &per_env_sorted_joints {
            all_joints.extend_from_slice(sorted_joints);
            // Pad to max_joints.
            for _ in sorted_joints.len()..max_joints as usize {
                all_joints.push(dummy_joint);
            }
        }

        // Single-batch color_groups buffer [global_num_colors]: every
        // environment has identical coloring (asserted above), so the prefix
        // sums are stored once and read at offset 0 by all batches.
        let all_color_groups = per_env_color_groups.first().cloned().unwrap_or_default();

        Self {
            len: max_joints,
            // All environments have the same joint count by the equal-topology
            // invariant, so the per-batch active count is any env's count.
            num_active_joints: filtered_lens.first().copied().unwrap_or(0),
            num_colors: global_num_colors,
            color_groups_cpu: all_color_groups.clone(),
            color_groups: Tensor::vector(backend, &all_color_groups, usage).unwrap(),
            joints: Tensor::vector(backend, &all_joints, usage).unwrap(),
            builders: Tensor::matrix_uninit(backend, num_batches, max_joints, usage).unwrap(),
            constraints: Tensor::matrix_uninit(backend, num_batches, max_joints, usage).unwrap(),
        }
    }

    /// Is this set empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of joints in this set.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Per-batch capacity (max joints across batches). Stride for
    /// `joints`, `builders`, `constraints` buffers.
    pub fn joints_per_batch(&self) -> u32 {
        self.len
    }

    /// Number of *active* impulse joints per batch (the kernel loop bound).
    pub fn num_active_joints(&self) -> u32 {
        self.num_active_joints
    }

    /// Number of color groups (also the per-batch stride of `color_groups`).
    pub fn num_colors(&self) -> u32 {
        self.num_colors
    }
}

/// GPU shader for joint constraint solving.
#[derive(Shader)]
pub struct GpuJointSolver {
    /// Initializes joint constraints.
    init_joint_constraints: GpuInitJointConstraints,
    /// Updates joint constraints each substep.
    update_joint_constraints: GpuUpdateJointConstraints,
    /// Solves joint constraints.
    solve_joint_constraints: GpuSolveJointConstraints,
    /// Removes bias from joint constraints.
    remove_joint_bias: GpuRemoveJointBias,
}

/// Arguments given to the joint solver.
pub struct JointSolverArgs<'a> {
    /// Number of constraint solving batches (iterations).
    pub num_batches: u32,
    /// The simulation parameters.
    pub sim_params: &'a Tensor<RbdSimParams>,
    /// The set of joints to solve.
    pub joints: &'a mut GpuImpulseJointSet,
    /// World-space mass properties.
    pub mprops: &'a Tensor<WorldMassProperties>,
    /// Local-space mass properties.
    pub local_mprops: &'a Tensor<LocalMassProperties>,
    /// Shared per-batch capacity / section-offset uniform.
    pub batch_indices: &'a Tensor<crate::shaders::utils::BatchIndices>,
    /// Per-color-index uniform tensors (`color_uniforms[c]` holds `c`),
    /// shared with the contact solver. See [`super::SolverArgs::color_uniforms`].
    pub color_uniforms: &'a [Tensor<u32>],
}

impl GpuJointSolver {
    /// Generate joint constraints for this set of joints.
    pub fn init(
        &self,
        pass: &mut GpuPass,
        args: &mut JointSolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if args.joints.is_empty() {
            return Ok(());
        }

        self.init_joint_constraints.call(
            pass,
            [args.joints.len, args.num_batches, 1],
            &args.joints.joints,
            &mut args.joints.builders,
            &mut args.joints.constraints,
            args.local_mprops,
            args.batch_indices,
        )?;
        Ok(())
    }

    /// Updates the non-linear terms of the joint constraints.
    pub fn update(
        &self,
        pass: &mut GpuPass,
        args: &mut JointSolverArgs<'_>,
        poses: &Tensor<Pose>,
    ) -> Result<(), GpuBackendError> {
        if args.joints.is_empty() {
            return Ok(());
        }

        self.update_joint_constraints.call(
            pass,
            [args.joints.len, args.num_batches, 1],
            &args.joints.builders,
            &mut args.joints.constraints,
            poses,
            args.mprops,
            args.sim_params,
            args.batch_indices,
        )?;
        Ok(())
    }

    /// Apply a single Projected-Gauss-Seidel step for solving joints.
    pub fn solve(
        &self,
        pass: &mut GpuPass,
        args: &mut JointSolverArgs<'_>,
        solver_vels: &mut Tensor<Velocity>,
        use_bias: bool,
    ) -> Result<(), GpuBackendError> {
        if args.joints.is_empty() {
            return Ok(());
        }

        if !use_bias {
            self.remove_joint_bias.call(
                pass,
                [args.joints.len, args.num_batches, 1],
                &mut args.joints.constraints,
                args.batch_indices,
            )?;
        }

        // One dispatch per color, sized exactly to that color's group (the
        // prefix sums are known on the host). The color index is bound as a
        // tiny pre-built uniform instead of a GPU-incremented cursor.
        for c in 0..args.joints.num_colors as usize {
            let start = if c > 0 {
                args.joints.color_groups_cpu[c - 1]
            } else {
                0
            };
            let group_len = args.joints.color_groups_cpu[c] - start;
            if group_len == 0 {
                continue;
            }
            self.solve_joint_constraints.call(
                pass,
                [group_len, args.num_batches, 1],
                &mut args.joints.constraints,
                solver_vels,
                &args.joints.color_groups,
                &args.color_uniforms[c],
                args.batch_indices,
            )?;
        }

        Ok(())
    }
}
