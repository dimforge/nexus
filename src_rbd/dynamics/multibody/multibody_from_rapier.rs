//! Conversion of rapier multibodies into the packed GPU buffers of [`GpuMultibodySet`].

use super::multibody_set::*;
use crate::shaders::dynamics::{
    ConstraintSoftness, MAX_AXIS_CONSTRAINTS, MAX_MB_CONTACT_CONSTRAINTS_PER_MB,
    MbImpulseJointBuilder, MbImpulseJointConstraint, MultibodyContactConstraint, MultibodyInfo,
    MultibodyJointConstraint, MultibodyLinkStatic, MultibodyLinkWorkspace, RbdSimParams,
};
use crate::shaders::utils::linalg::MAX_MB_DOFS;
use glamx::Vec4;
use khal::BufferUsages;
use khal::backend::GpuBackend;
use vortx::tensor::Tensor;
use {
    crate::rapier::dynamics::{MultibodyJointSet, RigidBodyHandle, RigidBodySet},
    std::collections::HashMap,
};

impl GpuMultibodySet {
    /// Convert a slice of per-batch `(MultibodyJointSet, body_ids_map)` pairs into
    /// packed GPU buffers. `body_ids` maps each rapier `RigidBodyHandle` to the
    /// corresponding collider/body index used elsewhere (poses, mprops buffers).
    ///
    /// Root links must be the first link in their multibody (rapier guarantees
    /// this via assembly ids being assigned in traversal order).
    pub fn from_rapier(
        backend: &GpuBackend,
        environments: &[(
            &MultibodyJointSet,
            &HashMap<RigidBodyHandle, u32>,
            &RigidBodySet,
        )],
        gravity: [f32; 3],
        colliders_per_batch: u32,
    ) -> Self {
        let num_batches = environments.len() as u32;

        // Stage 1: per-batch counts.
        let mut per_env_infos: Vec<Vec<MultibodyInfo>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_links_static: Vec<Vec<MultibodyLinkStatic>> =
            Vec::with_capacity(num_batches as usize);
        let mut per_env_links_workspace: Vec<Vec<MultibodyLinkWorkspace>> =
            Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_values: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_vels: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_damping: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_armature: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);

        let mut global_max_mb = 0u32;
        let mut global_max_links = 0u32;
        let mut global_max_dofs = 0u32;
        let mut global_max_jac = 0u32;
        let mut global_max_mm = 0u32;
        let mut global_max_cor = 0u32;
        let mut global_max_icdt = 0u32;
        let mut global_max_cons = 0u32;

        for (set, body_ids, bodies) in environments {
            let mut infos = Vec::new();
            let mut statics = Vec::new();
            let mut workspaces = Vec::new();
            let mut dof_vals = Vec::new();
            let mut dof_vels = Vec::new();
            let mut dof_damping = Vec::new();
            let mut dof_armature = Vec::new();

            let mut first_link = 0u32;
            let mut first_dof = 0u32;
            let mut jac_off = 0u32;
            let mut mm_off = 0u32;
            let mut cor_off = 0u32;
            let mut icdt_off = 0u32;
            let mut cons_off = 0u32;

            for (mb_idx, mb) in set.multibodies().enumerate() {
                if mb.ndofs() > MAX_MB_DOFS {
                    panic!(
                        "Multibody {} dofs {} exceed the maximum supported {}.",
                        mb_idx,
                        mb.ndofs(),
                        MAX_MB_DOFS
                    );
                }

                // rapier always creates the root with a free 6-DOF joint and only
                // converts it to a fixed joint later during its own step. Since we
                // don't run rapier's step here, detect a fixed root body and lock
                // all 6 DOFs ourselves.
                let root_is_dynamic = mb
                    .link(0)
                    .and_then(|r| bodies.get(r.rigid_body_handle()))
                    .map(|rb| rb.is_dynamic())
                    .unwrap_or(false);

                let root_ndof_adjust = if !root_is_dynamic && mb.link(0).is_some() {
                    mb.link(0).unwrap().joint().ndofs() as u32
                } else {
                    0
                };
                let ndofs = mb.ndofs() as u32 - root_ndof_adjust;
                let num_links = mb.num_links() as u32;

                // Count maximum constraint slots this multibody could need: for
                // each non-root non-kinematic joint, every free axis with a limit
                // OR a motor enabled produces one constraint slot, plus an
                // additional one if BOTH limit and motor are enabled on the same
                // axis (rapier emits them as separate constraints).
                let max_constraints = mb
                    .links()
                    .enumerate()
                    .map(|(li, link)| {
                        if link.joint().kinematic {
                            return 0u32;
                        }
                        if li == 0 && !root_is_dynamic {
                            return 0u32;
                        }
                        let j = link.joint().data;
                        let locked = j.locked_axes.bits() as u32;
                        let limit_axes = j.limit_axes.bits() as u32 & !locked;
                        let motor_axes = j.motor_axes.bits() as u32 & !locked;
                        // 1 per active limit + 1 per active motor (axis-wise).
                        let mut n = 0u32;
                        for ax in 0u32..6 {
                            if (limit_axes >> ax) & 1 != 0 {
                                n += 1;
                            }
                            if (motor_axes >> ax) & 1 != 0 {
                                n += 1;
                            }
                        }
                        n
                    })
                    .sum::<u32>();

                infos.push(MultibodyInfo {
                    first_link,
                    num_links,
                    first_dof,
                    ndofs,
                    jacobian_offset: jac_off,
                    mass_matrix_offset: mm_off,
                    root_is_dynamic: if root_is_dynamic { 1 } else { 0 },
                    coriolis_offset: cor_off,
                    i_coriolis_dt_offset: icdt_off,
                    first_constraint: cons_off,
                    max_constraints,
                    self_contacts_enabled: if mb.self_contacts_enabled() { 1 } else { 0 },
                    contact_constraint_count: 0,
                });

                // `assembly_id` is not exposed publicly on `MultibodyLink`, so we
                // reconstruct it ourselves — rapier assigns ids in the same traversal
                // order as `links()`.
                let mut assembly_counter = 0u32;
                // Per-DoF damping and armature (reflected rotor inertia) come from
                // rapier's multibody vectors, which are indexed by *rapier's*
                // assembly id (the cumulative sum of `link.joint().ndofs()` in
                // `links()` order, including the root's DoFs even when the root is
                // fixed). We track that index separately from our re-numbered
                // `assembly_counter` (which drops the fixed root's DoFs).
                let mb_damping = mb.damping();
                let mb_armature = mb.armature();
                let mut rapier_assembly = 0usize;
                for (link_idx, link) in mb.links().enumerate() {
                    let rb_id = body_ids
                        .get(&link.rigid_body_handle())
                        .copied()
                        .unwrap_or(0);
                    let parent_id = match link.parent_id() {
                        Some(p) => p as u32,
                        None => u32::MAX,
                    };

                    // Lock all 6 DOFs on the root if its body is fixed.
                    let mut data = convert_generic_joint(link.joint().data);
                    let link_ndofs = if link_idx == 0 && !root_is_dynamic {
                        data.locked_axes = 0x3f;
                        0u32
                    } else {
                        link.joint().ndofs() as u32
                    };

                    // Per-link mass properties (real masses stored here so the
                    // multibody solver sees correct values even when the shared
                    // body mprops are zeroed out). Fused into the static per-link
                    // record (see `MultibodyLinkStatic::local_mprops`) instead of
                    // a separate `links_mprops` storage buffer, to keep the
                    // dynamics kernels within the 8-storage-buffer WebGPU limit.
                    let mp = bodies
                        .get(link.rigid_body_handle())
                        .map(|rb| convert_link_mprops(&rb.mass_properties().local_mprops))
                        .unwrap_or_default();
                    // For fixed-root links, mass/inertia are zeroed so they don't
                    // contribute to the CRBA mass matrix (rapier skips them too).
                    let mp = if link_idx == 0 && !root_is_dynamic {
                        let mut z = mp;
                        z.inv_mass = glamx::Vec3::ZERO;
                        z.inv_principal_inertia = glamx::Vec3::ZERO;
                        z
                    } else {
                        mp
                    };

                    let stat = MultibodyLinkStatic {
                        rb_id,
                        parent_link_id: parent_id,
                        multibody_id: mb_idx as u32,
                        assembly_id: assembly_counter,
                        ndofs: link_ndofs,
                        kinematic: if link.joint().kinematic { 1 } else { 0 },
                        _pad0: [0; 2],
                        data,
                        local_mprops: mp,
                    };
                    statics.push(stat);
                    assembly_counter += link_ndofs;

                    let mut ws = make_workspace_init();
                    ws.coords = link.joint.coords();
                    ws.joint_rot = link.joint.joint_rot();

                    // For free joints at the root, copy the rigid-body pose directly.
                    if link.joint.data.locked_axes.is_empty()
                        && let Some(rb) = bodies.get(link.rigid_body_handle())
                    {
                        let pos = rb.position();
                        ws.coords[0] = pos.translation.x;
                        ws.coords[1] = pos.translation.y;
                        ws.coords[2] = pos.translation.z;
                        ws.joint_rot = pos.rotation;
                    }

                    workspaces.push(ws);

                    // Per-DOF damping and armature read straight from rapier's
                    // multibody (set by the MJCF loader from `<joint damping>` /
                    // `<joint armature>`). Armature is the reflected rotor inertia
                    // MuJoCo models rely on for stability — omitting it makes the
                    // joint-space inertia far too small and the integrator blows up.
                    // Both are added to the mass-matrix diagonal in the dynamics
                    // shader, exactly as rapier's `update_mass_matrix` does
                    // (`diag = damping·dt + armature`).
                    for d in 0..link_ndofs as usize {
                        dof_vals.push(0.0);
                        dof_vels.push(0.0);
                        dof_damping.push(mb_damping[rapier_assembly + d]);
                        dof_armature.push(mb_armature[rapier_assembly + d]);
                    }
                    // Advance by rapier's full link DoF count (which includes a
                    // fixed root's DoFs, where `link_ndofs` above is 0).
                    rapier_assembly += link.joint().ndofs();
                }

                first_link += num_links;
                first_dof += ndofs;
                jac_off += num_links * 6 * ndofs;
                mm_off += ndofs * ndofs;
                cor_off += num_links * 3 * ndofs;
                icdt_off += 6 * ndofs;
                cons_off += max_constraints;
            }

            global_max_mb = global_max_mb.max(infos.len() as u32);
            global_max_links = global_max_links.max(statics.len() as u32);
            global_max_dofs = global_max_dofs.max(dof_vals.len() as u32);
            global_max_jac = global_max_jac.max(jac_off);
            global_max_mm = global_max_mm.max(mm_off);
            global_max_cor = global_max_cor.max(cor_off);
            global_max_icdt = global_max_icdt.max(icdt_off);
            global_max_cons = global_max_cons.max(cons_off);

            per_env_infos.push(infos);
            per_env_links_static.push(statics);
            per_env_links_workspace.push(workspaces);
            per_env_dof_values.push(dof_vals);
            per_env_dof_vels.push(dof_vels);
            per_env_dof_damping.push(dof_damping);
            per_env_dof_armature.push(dof_armature);
        }

        // Pad capacities (avoid empty buffers — GPU dislikes size-zero storage bindings).
        let mb_cap = global_max_mb.max(1);
        let links_cap = global_max_links.max(1);
        let dofs_cap = global_max_dofs.max(1);
        let jac_cap = global_max_jac.max(1);
        let mm_cap = global_max_mm.max(1);
        let cor_cap = global_max_cor.max(1);
        let icdt_cap = global_max_icdt.max(1);
        let cons_cap = global_max_cons.max(1);
        // One length-`dofs_cap` column of `M⁻¹` per constraint slot.
        let cons_col_cap = cons_cap.saturating_mul(dofs_cap).max(1);

        // Per-multibody contact-constraint banks: every multibody owns a
        // fixed-size slab of `MAX_MB_CONTACT_CONSTRAINTS_PER_MB` slots —
        // each contact point produces 1 normal + (DIM-1) friction tangent
        // constraint slots. The init kernel marks unused slots as `kind = 0`.
        let contact_cons_cap = mb_cap
            .saturating_mul(MAX_MB_CONTACT_CONSTRAINTS_PER_MB)
            .max(1);
        let contact_cons_col_cap = contact_cons_cap.saturating_mul(dofs_cap).max(1);
        let body_to_link_cap = colliders_per_batch.max(1);

        // Build the per-body multibody/link lookup. Free / non-multibody bodies
        // get the sentinel `[u32::MAX, u32::MAX]`. The kernel reads
        // `body_to_link[batch_offset + body_local_id]` and skips the
        // sentinel.
        let mut all_body_to_link: Vec<[u32; 2]> =
            vec![[u32::MAX, u32::MAX]; (body_to_link_cap * num_batches) as usize];
        for (batch_idx, (set, body_ids, _)) in environments.iter().enumerate() {
            let base = batch_idx * body_to_link_cap as usize;
            for (mb_idx, mb) in set.multibodies().enumerate() {
                for (link_idx, link) in mb.links().enumerate() {
                    if let Some(&local) = body_ids.get(&link.rigid_body_handle())
                        && local < body_to_link_cap
                    {
                        all_body_to_link[base + local as usize] = [mb_idx as u32, link_idx as u32];
                    }
                }
            }
        }

        // Flatten, padding each batch to `*_cap`.
        let mut all_infos: Vec<MultibodyInfo> = Vec::with_capacity((mb_cap * num_batches) as usize);
        let mut all_statics: Vec<MultibodyLinkStatic> =
            Vec::with_capacity((links_cap * num_batches) as usize);
        let mut all_ws: Vec<MultibodyLinkWorkspace> =
            Vec::with_capacity((links_cap * num_batches) as usize);
        let mut all_dof_vals: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_vels: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_damping: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_armature: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);

        let dummy_info = MultibodyInfo::default();
        let dummy_stat: MultibodyLinkStatic = bytemuck::Zeroable::zeroed();
        let dummy_ws = make_workspace_init();

        for i in 0..num_batches as usize {
            all_infos.extend_from_slice(&per_env_infos[i]);
            for _ in per_env_infos[i].len()..mb_cap as usize {
                all_infos.push(dummy_info);
            }

            all_statics.extend_from_slice(&per_env_links_static[i]);
            for _ in per_env_links_static[i].len()..links_cap as usize {
                all_statics.push(dummy_stat);
            }

            all_ws.extend_from_slice(&per_env_links_workspace[i]);
            for _ in per_env_links_workspace[i].len()..links_cap as usize {
                all_ws.push(dummy_ws);
            }

            all_dof_vals.extend_from_slice(&per_env_dof_values[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_values[i].len());
            all_dof_vals.resize(all_dof_vals.len() + pad, 0.0);
            all_dof_vels.extend_from_slice(&per_env_dof_vels[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_vels[i].len());
            all_dof_vels.resize(all_dof_vels.len() + pad, 0.0);
            all_dof_damping.extend_from_slice(&per_env_dof_damping[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_damping[i].len());
            all_dof_damping.resize(all_dof_damping.len() + pad, 0.0);
            all_dof_armature.extend_from_slice(&per_env_dof_armature[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_armature[i].len());
            all_dof_armature.resize(all_dof_armature.len() + pad, 0.0);
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let usage_u = storage | BufferUsages::UNIFORM;

        Self {
            num_batches,
            multibodies_per_batch: mb_cap,
            num_active_multibodies: global_max_mb,
            links_per_batch: links_cap,
            dofs_per_batch: dofs_cap,
            jacobian_entries_per_batch: jac_cap,
            mass_matrix_entries_per_batch: mm_cap,
            coriolis_entries_per_batch: cor_cap,
            i_coriolis_dt_entries_per_batch: icdt_cap,
            implicit_coriolis: true,
            has_joint_constraints: all_infos.iter().any(|info| info.max_constraints > 0),

            multibody_info: Tensor::vector(backend, &all_infos, storage).unwrap(),
            links_static: Tensor::vector(backend, &all_statics, storage | BufferUsages::COPY_DST)
                .unwrap(),
            links_static_mirror: all_statics.clone(),
            // COPY_SRC so hosts can read joint/link state back (observation
            // pipelines); see `GpuMultibodySet::links_workspace`.
            links_workspace: Tensor::vector(backend, &all_ws, storage | BufferUsages::COPY_SRC)
                .unwrap(),
            dof_values: Tensor::vector(backend, &all_dof_vals, storage).unwrap(),
            dof_state: {
                // Pack [velocities (N), damping (N), armature (N)] back-to-back
                // where N = dofs_cap * num_batches. The damping section starts at
                // `N` (`dof_damping_section_offset`) and the armature section at
                // `2·N` (computed in-shader as `2 · dof_damping_section_offset`).
                let n = (dofs_cap * num_batches) as usize;
                let mut buf = Vec::with_capacity(3 * n);
                buf.extend_from_slice(&all_dof_vels);
                buf.extend_from_slice(&all_dof_damping);
                buf.extend_from_slice(&all_dof_armature);
                debug_assert_eq!(buf.len(), 3 * n);
                Tensor::vector(backend, &buf, storage).unwrap()
            },
            gen_forces: Tensor::vector(
                backend,
                vec![0.0f32; (dofs_cap * num_batches) as usize],
                storage | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            body_jacobians: Tensor::vector(
                backend,
                vec![0.0f32; (jac_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            mass_matrices: Tensor::vector(
                backend,
                vec![0.0f32; (mm_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            lu_pivots: Tensor::vector(
                backend,
                vec![0u32; (dofs_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            coriolis_packed: {
                // Pack [coriolis_v (A), coriolis_w (A), i_coriolis_dt (B)]
                // back-to-back where A = cor_cap * num_batches and B =
                // icdt_cap * num_batches.
                let a = (cor_cap * num_batches) as usize;
                let b = (icdt_cap * num_batches) as usize;
                Tensor::vector(backend, vec![0.0f32; 2 * a + b], storage).unwrap()
            },
            joint_constraints: Tensor::vector(
                backend,
                vec![MultibodyJointConstraint::default(); (cons_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            joint_constraint_columns: Tensor::vector(
                backend,
                vec![0.0f32; (cons_col_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            body_to_link: Tensor::vector(backend, &all_body_to_link, storage).unwrap(),
            contact_constraints: Tensor::vector(
                backend,
                vec![
                    MultibodyContactConstraint::default();
                    (contact_cons_cap * num_batches) as usize
                ],
                storage,
            )
            .unwrap(),
            contact_constraint_jacs: Tensor::vector(
                backend,
                vec![0.0f32; (contact_cons_col_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            contact_constraint_columns: Tensor::vector(
                backend,
                vec![0.0f32; (contact_cons_col_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),

            // Impulse-joint buffers are sized for "no MB-touching joints" by
            // default — `set_impulse_joints` resizes them at pipeline build
            // time when the host has actually counted the joints.
            mb_imp_joint_count: Tensor::vector(
                backend,
                vec![0u32; num_batches as usize],
                storage | BufferUsages::UNIFORM,
            )
            .unwrap(),
            mb_imp_joint_builders: Tensor::vector(
                backend,
                vec![<MbImpulseJointBuilder as bytemuck::Zeroable>::zeroed(); num_batches as usize],
                storage,
            )
            .unwrap(),
            mb_imp_joint_constraints: Tensor::vector(
                backend,
                vec![
                    MbImpulseJointConstraint::default();
                    (MAX_AXIS_CONSTRAINTS as usize) * (num_batches as usize)
                ],
                storage,
            )
            .unwrap(),
            mb_imp_joint_jacobians: Tensor::vector(
                backend,
                vec![0.0f32; num_batches as usize],
                storage,
            )
            .unwrap(),
            mb_imp_joints_per_batch: 0,
            mb_imp_joint_constraints_per_batch: MAX_AXIS_CONSTRAINTS,
            mb_imp_joint_jacobians_per_batch: 1,
            mb_imp_joint_color_groups: Tensor::vector(
                backend,
                vec![0u32; num_batches as usize],
                storage,
            )
            .unwrap(),
            mb_imp_joint_curr_color: Tensor::scalar(backend, 0u32, usage_u).unwrap(),
            mb_imp_joint_num_colors: 0,
            mb_imp_joint_max_color_group_len: 0,
            joint_constraints_per_batch: cons_cap,
            joint_constraint_columns_per_batch: cons_col_cap,
            contact_constraints_per_batch: contact_cons_cap,
            contact_constraint_columns_per_batch: contact_cons_col_cap,

            num_solver_iterations: 4,

            // FIXME: should be read from the simulation settings.
            gravity: Tensor::scalar(
                backend,
                Vec4::new(gravity[0], gravity[1], gravity[2], 0.0),
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
            .unwrap(),
            dt: Tensor::scalar(
                backend,
                1.0f32 / 60.0,
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
            .unwrap(),
            // Sensible default; the caller overrides via `set_constraint_softness`
            // with the real (substep) sim params after construction.
            constraint_softness: Tensor::scalar(
                backend,
                ConstraintSoftness::from_params(&RbdSimParams::default()),
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
            .unwrap(),
        }
    }
}
