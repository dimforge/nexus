//! Routing of multibody-touching impulse joints into the `MbImpulseJointConstraint` solver path.

use super::multibody_set::*;
use crate::shaders::dynamics::{
    MAX_AXIS_CONSTRAINTS, MbImpulseJointBuilder, MbImpulseJointConstraint, SIDE_KIND_BODY,
    SIDE_KIND_FIXED, SIDE_KIND_MB,
};
use khal::BufferUsages;
use khal::backend::GpuBackend;
use vortx::tensor::Tensor;
use {
    crate::rapier::dynamics::{ImpulseJointSet, MultibodyJointSet, RigidBodyHandle, RigidBodySet},
    std::collections::HashMap,
};

impl GpuMultibodySet {
    /// Upload the per-batch impulse joints whose body1 OR body2 is part
    /// of a multibody. These joints are routed through the
    /// `MbImpulseJointConstraint` solver path (rapier's
    /// `JointGenericExternalConstraintBuilder`); free-only impulse joints
    /// stay in the regular `GpuImpulseJointSet` path because they don't
    /// need `M⁻¹·Jᵀ` propagation.
    ///
    /// `environments` matches the layout used elsewhere in the pipeline:
    /// one entry per batch, in the same order as the multibody envs that
    /// were passed to `from_rapier`. Free-only joints are silently
    /// skipped here.
    pub fn set_impulse_joints(
        &mut self,
        backend: &GpuBackend,
        environments: &[(
            &ImpulseJointSet,
            &MultibodyJointSet,
            &HashMap<RigidBodyHandle, u32>,
            &RigidBodySet,
        )],
    ) {
        assert_eq!(environments.len() as u32, self.num_batches);

        // Stage 1 — per-batch list of touched joints + their side metadata.
        let mut per_env_builders: Vec<Vec<MbImpulseJointBuilder>> =
            Vec::with_capacity(self.num_batches as usize);
        // Per-env color-group prefix sums (one Vec<u32> per batch), built
        // alongside the builders below. `global_num_colors` /
        // `global_max_color_group_len` are the cross-batch maxima used to
        // size the flat buffer and the per-color dispatch width.
        let mut per_env_color_groups: Vec<Vec<u32>> = Vec::with_capacity(self.num_batches as usize);
        let mut global_num_colors = 0u32;
        let mut global_max_color_group_len = 0u32;
        let mut max_joints = 0u32;
        let mut max_jac_floats = 0u32;

        for (batch_idx, (impulse_joints, mb_set, body_ids, bodies)) in
            environments.iter().enumerate()
        {
            let _ = batch_idx;
            // body local id → (mb_index_in_batch, link_index_within_mb).
            let mut body_to_mb_link: HashMap<u32, (u32, u32)> = HashMap::new();
            for (mb_idx, mb) in mb_set.multibodies().enumerate() {
                for (link_idx, link) in mb.links().enumerate() {
                    if let Some(&local) = body_ids.get(&link.rigid_body_handle()) {
                        body_to_mb_link.insert(local, (mb_idx as u32, link_idx as u32));
                    }
                }
            }

            let mut builders = Vec::new();
            let mut jac_offset = 0u32;
            let mut constraint_id = 0u32;

            for (_handle, joint) in impulse_joints.iter() {
                let body1 = joint.body1();
                let body2 = joint.body2();
                let local1 = match body_ids.get(&body1) {
                    Some(&id) => id,
                    None => continue,
                };
                let local2 = match body_ids.get(&body2) {
                    Some(&id) => id,
                    None => continue,
                };

                let mb1 = body_to_mb_link.get(&local1).copied();
                let mb2 = body_to_mb_link.get(&local2).copied();
                if mb1.is_none() && mb2.is_none() {
                    continue; // Free-only joint; existing path handles it.
                }

                let rb1 = bodies.get(body1);
                let rb2 = bodies.get(body2);

                // Mirror rapier's `LinkOrBody` resolution + `transform_to_solver_body_space`.
                // Side A:
                let (side_a_kind, side_a_id, side_a_link, ndofs_a) = match (mb1, rb1) {
                    (Some((mb_idx, link_idx)), _) => {
                        let mb = mb_set.multibodies().nth(mb_idx as usize).unwrap();
                        (SIDE_KIND_MB, mb_idx, link_idx, mb.ndofs() as u32)
                    }
                    (None, Some(rb)) if rb.is_dynamic() => (SIDE_KIND_BODY, local1, 0, 6),
                    _ => (SIDE_KIND_FIXED, u32::MAX, 0, 0),
                };

                let (side_b_kind, side_b_id, side_b_link, ndofs_b) = match (mb2, rb2) {
                    (Some((mb_idx, link_idx)), _) => {
                        let mb = mb_set.multibodies().nth(mb_idx as usize).unwrap();
                        (SIDE_KIND_MB, mb_idx, link_idx, mb.ndofs() as u32)
                    }
                    (None, Some(rb)) if rb.is_dynamic() => (SIDE_KIND_BODY, local2, 0, 6),
                    _ => (SIDE_KIND_FIXED, u32::MAX, 0, 0),
                };

                if ndofs_a + ndofs_b == 0 {
                    continue; // Both sides static — no constraint to solve.
                }

                // Mirror rapier `GenericJoint::transform_to_solver_body_space`:
                // shift the anchor frame's translation into COM space — but ONLY
                // for FREE-BODY sides, whose solver pose IS the center of mass.
                // A multibody-link side is positioned by its `local_to_world`,
                // which is the link ORIGIN frame (not the COM), so the shift must
                // NOT be applied there — the anchor stays origin-relative and the
                // lever arm is taken against the COM separately (see
                // `world_com` in `update_one_joint`). Applying the shift to MB
                // links offsets the anchor by `local_com` (≈0.25 m for Cassie's
                // rods), producing a huge spurious loop-closure violation. This
                // matches rapier's `generic_joint_constraint_builder` (the shift
                // is applied to `LinkOrBody::Body` sides only). Fixed-side fold
                // is still a TODO mirroring rapier's `is_fixed` branch.
                let mut joint_data = convert_generic_joint(joint.data);
                if side_a_kind == SIDE_KIND_BODY
                    && let Some(rb) = rb1
                {
                    let com = rb.mass_properties().local_mprops.local_com;
                    joint_data.local_frame_a.translation -= com;
                }
                if side_b_kind == SIDE_KIND_BODY
                    && let Some(rb) = rb2
                {
                    let com = rb.mass_properties().local_mprops.local_com;
                    joint_data.local_frame_b.translation -= com;
                }

                // Per-axis stride = 2 * (ndofs_a + ndofs_b); reserve
                // MAX_AXIS_CONSTRAINTS slots up front so the kernel can
                // walk them sequentially without rechecking.
                let stride = 2 * (ndofs_a + ndofs_b);
                let cap_floats = stride * MAX_AXIS_CONSTRAINTS;
                let builder = MbImpulseJointBuilder {
                    joint: joint_data,
                    side_a_kind,
                    side_a_id,
                    side_a_link,
                    joint_id: builders.len() as u32,
                    side_b_kind,
                    side_b_id,
                    side_b_link,
                    constraint_id,
                    jacobian_offset: jac_offset,
                    jacobian_capacity: cap_floats,
                    #[cfg(feature = "dim3")]
                    _pad0: [0; 2],
                };
                builders.push(builder);
                constraint_id += MAX_AXIS_CONSTRAINTS;
                jac_offset += cap_floats;
            }

            max_joints = max_joints.max(builders.len() as u32);
            max_jac_floats = max_jac_floats.max(jac_offset);

            // ── Init-time graph coloring (mirrors the rigid-body impulse
            // joint coloring in `dynamics/joint.rs`). Conflict graph: nodes
            // are multibodies and free bodies that appear in an MB joint
            // (FIXED sides touch no mutable state → no node); an edge joins
            // the two sides of every joint. Two joints get the same color
            // only if they share no node, so within a color every joint
            // writes disjoint `dof_state` / `solver_vels`, making the
            // per-color sweep an exact (race-free) Gauss–Seidel step.
            let num_mb = mb_set.multibodies().count() as u32;
            // Unified node id: MB side → mb_idx; free body → num_mb +
            // local_body_id; FIXED → none.
            let node = |kind: u32, id: u32| -> Option<usize> {
                if kind == SIDE_KIND_MB {
                    Some(id as usize)
                } else if kind == SIDE_KIND_BODY {
                    Some((num_mb + id) as usize)
                } else {
                    None
                }
            };
            let max_node = builders
                .iter()
                .flat_map(|b| {
                    [
                        node(b.side_a_kind, b.side_a_id),
                        node(b.side_b_kind, b.side_b_id),
                    ]
                })
                .flatten()
                .max()
                .unwrap_or(0);

            let mut colors = Vec::with_capacity(builders.len());
            let mut group_masks = vec![0u128; max_node + 1];
            for b in &builders {
                let a = node(b.side_a_kind, b.side_a_id);
                let bb = node(b.side_b_kind, b.side_b_id);
                let used = a.map_or(0, |n| group_masks[n]) | bb.map_or(0, |n| group_masks[n]);
                let color = used.trailing_ones();
                colors.push(color);
                if let Some(n) = a {
                    group_masks[n] |= 1 << color;
                }
                if let Some(n) = bb {
                    group_masks[n] |= 1 << color;
                }
            }

            let env_num_colors = colors.iter().copied().max().map(|n| n + 1).unwrap_or(0);
            let mut color_groups = vec![0u32; env_num_colors as usize];
            for c in &colors {
                color_groups[*c as usize] += 1;
            }
            let env_max_color_group_len = color_groups.iter().copied().max().unwrap_or(0);

            // Prefix sum → per-color end offsets in the sorted builder slab.
            for i in 0..color_groups.len().saturating_sub(1) {
                color_groups[i + 1] += color_groups[i];
            }

            // Bucket-sort builders by color (constraint_id / jacobian_offset
            // travel inside each builder, so reordering is safe — every
            // kernel indexes the slab via `builder.constraint_id`).
            let mut target = color_groups.clone();
            target.insert(0, 0);
            let mut sorted_builders = builders.clone();
            for (b, c) in builders.iter().zip(colors.iter()) {
                sorted_builders[target[*c as usize] as usize] = *b;
                target[*c as usize] += 1;
            }

            global_num_colors = global_num_colors.max(env_num_colors);
            global_max_color_group_len = global_max_color_group_len.max(env_max_color_group_len);

            per_env_color_groups.push(color_groups);
            per_env_builders.push(sorted_builders);
        }

        // Stage 2 — flatten with per-batch padding to `max_joints`.
        let joints_cap = max_joints.max(1);
        let cons_cap = (joints_cap * MAX_AXIS_CONSTRAINTS).max(1);
        let jac_cap = max_jac_floats.max(1);

        let mut all_builders: Vec<MbImpulseJointBuilder> =
            Vec::with_capacity((joints_cap * self.num_batches) as usize);
        let mut all_counts: Vec<u32> = Vec::with_capacity(self.num_batches as usize);
        // Padding builder: both sides marked FIXED so the GPU kernel can
        // skip them by sentinel check (replaces the per-batch `num_joints`
        // storage binding the kernel used to read for early-out).
        let mut dummy: MbImpulseJointBuilder = bytemuck::Zeroable::zeroed();
        dummy.side_a_kind = SIDE_KIND_FIXED;
        dummy.side_b_kind = SIDE_KIND_FIXED;
        for env in &per_env_builders {
            all_counts.push(env.len() as u32);
            all_builders.extend_from_slice(env);
            for _ in env.len()..joints_cap as usize {
                all_builders.push(dummy);
            }
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let usage_u = storage | BufferUsages::UNIFORM;
        self.mb_imp_joint_count = Tensor::vector(backend, &all_counts, usage_u).unwrap();
        self.mb_imp_joint_builders = Tensor::vector(backend, &all_builders, storage).unwrap();
        self.mb_imp_joint_constraints = Tensor::vector(
            backend,
            vec![MbImpulseJointConstraint::default(); (cons_cap * self.num_batches) as usize],
            storage,
        )
        .unwrap();
        self.mb_imp_joint_jacobians = Tensor::vector(
            backend,
            vec![0.0f32; (jac_cap * self.num_batches) as usize],
            storage,
        )
        .unwrap();
        self.mb_imp_joints_per_batch = joints_cap;
        self.mb_imp_joint_constraints_per_batch = cons_cap;
        self.mb_imp_joint_jacobians_per_batch = jac_cap;

        // Flat color-groups buffer [num_batches * cols]. Envs with fewer
        // colors are padded with their last prefix value so the extra
        // colors are no-ops (start == end). `cols` is clamped to ≥1 so the
        // buffer is always a valid non-empty binding even with no joints.
        let cols = global_num_colors.max(1);
        let mut all_color_groups = Vec::with_capacity((cols * self.num_batches) as usize);
        for env_cg in &per_env_color_groups {
            let last = env_cg.last().copied().unwrap_or(0);
            all_color_groups.extend_from_slice(env_cg);
            for _ in env_cg.len()..cols as usize {
                all_color_groups.push(last);
            }
        }
        self.mb_imp_joint_color_groups =
            Tensor::vector(backend, &all_color_groups, storage).unwrap();
        self.mb_imp_joint_curr_color = Tensor::scalar(backend, 0u32, usage_u).unwrap();
        self.mb_imp_joint_num_colors = global_num_colors;
        self.mb_imp_joint_max_color_group_len = global_max_color_group_len;
    }
}
