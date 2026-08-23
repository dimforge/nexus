//! Conversion of rapier multibodies into the packed GPU buffers of [`GpuMultibodySet`].

use super::multibody_set::*;
use crate::shaders::dynamics::{
    ConstraintSoftness, MAX_AXIS_CONSTRAINTS, MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MbDofCoupling,
    MbImpulseJointBuilder, MbImpulseJointConstraint, MultibodyContactConstraint, MultibodyInfo,
    MultibodyJointConstraint, MultibodyLinkStatic, MultibodyLinkWorkspace, RbdSimParams,
};
use crate::shaders::utils::linalg::MAX_MB_DOFS;
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
        let mut per_env_dof_stiffness: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_spring_ref: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_kinematic: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_couplings: Vec<Vec<MbDofCoupling>> =
            Vec::with_capacity(num_batches as usize);

        let mut global_max_mb = 0u32;
        let mut global_max_links = 0u32;
        // Per-multibody maxima (not per-env sums) for the uniform loop bounds.
        let mut max_mb_ndofs = 0u32;
        let mut max_mb_links = 0u32;
        let mut max_mb_joint_constraints = 0u32;
        let mut global_max_dofs = 0u32;
        let mut global_max_jac = 0u32;
        let mut global_max_mm = 0u32;
        let mut global_max_cor = 0u32;
        let mut global_max_icdt = 0u32;
        let mut global_max_cons = 0u32;
        let mut global_max_couplings = 0u32;

        for (set, body_ids, bodies) in environments {
            let mut infos = Vec::new();
            let mut statics = Vec::new();
            let mut workspaces = Vec::new();
            let mut dof_vals = Vec::new();
            let mut dof_vels = Vec::new();
            let mut dof_damping = Vec::new();
            let mut dof_armature = Vec::new();
            let mut dof_stiffness = Vec::new();
            let mut dof_spring_ref = Vec::new();
            let mut dof_kinematic = Vec::new();
            let mut couplings = Vec::new();

            let mut first_link = 0u32;
            let mut first_dof = 0u32;
            let mut jac_off = 0u32;
            let mut mm_off = 0u32;
            let mut cor_off = 0u32;
            let mut icdt_off = 0u32;
            let mut cons_off = 0u32;
            let mut coupling_off = 0u32;

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
                max_mb_ndofs = max_mb_ndofs.max(ndofs);
                max_mb_links = max_mb_links.max(num_links);

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

                // One extra constraint slot per DoF coupling (they are solved
                // among the joint constraints).
                let num_couplings = mb.couplings().len() as u32;
                let max_constraints = max_constraints + num_couplings;
                max_mb_joint_constraints = max_mb_joint_constraints.max(max_constraints);

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
                    batch_contacts_len: 0,
                    first_coupling: coupling_off,
                    num_couplings,
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
                let mb_statics_start = statics.len();
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
                    if let Some(rb) = bodies.get(link.rigid_body_handle()) {
                        ws.gravity_scale = rb.gravity_scale();
                        ws.external_force = rb.user_force();
                        ws.external_torque = rb.user_torque();
                    }

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
                    // Per-DoF spring stiffness / rest position (rapier's
                    // `MultibodyJoint::spring`, MuJoCo `<joint stiffness
                    // springref>`): flattened per free axis in DoF order,
                    // exactly like damping. The kinematic mask marks DoFs whose
                    // velocities are user-controlled (identity mass-matrix rows,
                    // zero acceleration).
                    let locked = link.joint().data.locked_axes.bits() as u32;
                    let is_kin = link.joint().kinematic;
                    let mut free_axis_of_dof = [0usize; 6];
                    let mut nfree = 0usize;
                    for ax in 0..6usize {
                        if (locked >> ax) & 1 == 0 {
                            free_axis_of_dof[nfree] = ax;
                            nfree += 1;
                        }
                    }
                    for d in 0..link_ndofs as usize {
                        dof_vals.push(0.0);
                        dof_vels.push(0.0);
                        dof_damping.push(mb_damping[rapier_assembly + d]);
                        dof_armature.push(mb_armature[rapier_assembly + d]);
                        let (k_s, rest) = link.joint().spring(free_axis_of_dof[d]);
                        dof_stiffness.push(k_s);
                        dof_spring_ref.push(rest);
                        dof_kinematic.push(if is_kin { 1.0 } else { 0.0 });
                    }
                    // Advance by rapier's full link DoF count (which includes a
                    // fixed root's DoFs, where `link_ndofs` above is 0).
                    rapier_assembly += link.joint().ndofs();
                }

                // DoF couplings, converted to the GPU's re-numbered assembly
                // ids.
                for c in mb.couplings() {
                    let g1 = statics[mb_statics_start + c.link1].assembly_id + c.dof1 as u32;
                    let g2 = statics[mb_statics_start + c.link2].assembly_id + c.dof2 as u32;
                    couplings.push(MbDofCoupling {
                        dof1: g1,
                        dof2: g2,
                        link_axis1: c.link1 as u32 | ((c.axis1 as u32) << 16),
                        link_axis2: c.link2 as u32 | ((c.axis2 as u32) << 16),
                        coeff: c.coeff,
                        offset: c.offset,
                        _pad: [0; 2],
                    });
                }

                first_link += num_links;
                first_dof += ndofs;
                jac_off += num_links * 6 * ndofs;
                mm_off += ndofs * ndofs;
                cor_off += num_links * 3 * ndofs;
                icdt_off += 6 * ndofs;
                cons_off += max_constraints;
                coupling_off += num_couplings;
            }

            global_max_mb = global_max_mb.max(infos.len() as u32);
            global_max_links = global_max_links.max(statics.len() as u32);
            global_max_dofs = global_max_dofs.max(dof_vals.len() as u32);
            global_max_jac = global_max_jac.max(jac_off);
            global_max_mm = global_max_mm.max(mm_off);
            global_max_cor = global_max_cor.max(cor_off);
            global_max_icdt = global_max_icdt.max(icdt_off);
            global_max_cons = global_max_cons.max(cons_off);
            global_max_couplings = global_max_couplings.max(coupling_off);

            per_env_infos.push(infos);
            per_env_links_static.push(statics);
            per_env_links_workspace.push(workspaces);
            per_env_dof_values.push(dof_vals);
            per_env_dof_vels.push(dof_vels);
            per_env_dof_damping.push(dof_damping);
            per_env_dof_armature.push(dof_armature);
            per_env_dof_stiffness.push(dof_stiffness);
            per_env_dof_spring_ref.push(dof_spring_ref);
            per_env_dof_kinematic.push(dof_kinematic);
            per_env_couplings.push(couplings);
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
        let couplings_cap = global_max_couplings.max(1);
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
        let mut all_dof_stiffness: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_spring_ref: Vec<f32> =
            Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_kinematic: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_couplings: Vec<MbDofCoupling> =
            Vec::with_capacity((couplings_cap * num_batches) as usize);

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
            all_dof_stiffness.extend_from_slice(&per_env_dof_stiffness[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_stiffness[i].len());
            all_dof_stiffness.resize(all_dof_stiffness.len() + pad, 0.0);
            all_dof_spring_ref.extend_from_slice(&per_env_dof_spring_ref[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_spring_ref[i].len());
            all_dof_spring_ref.resize(all_dof_spring_ref.len() + pad, 0.0);
            all_dof_kinematic.extend_from_slice(&per_env_dof_kinematic[i]);
            let pad = (dofs_cap as usize).saturating_sub(per_env_dof_kinematic[i].len());
            all_dof_kinematic.resize(all_dof_kinematic.len() + pad, 0.0);
            all_couplings.extend_from_slice(&per_env_couplings[i]);
            let pad = (couplings_cap as usize).saturating_sub(per_env_couplings[i].len());
            all_couplings.resize(all_couplings.len() + pad, MbDofCoupling::default());
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        // Batch-interleaved (batch-minor) layout for the
        // dynamics buffers: element `k` of batch `b` lives at `k · nb + b`.
        fn interleave<T: Copy>(data: &[T], cap: u32, nb: usize) -> Vec<T> {
            let cap = cap as usize;
            let mut out = Vec::with_capacity(data.len());
            for k in 0..cap {
                for b in 0..nb {
                    out.push(data[b * cap + k]);
                }
            }
            out
        }
        let nb = num_batches as usize;
        let info_mirror = all_infos.clone();
        let all_infos = interleave(&all_infos, mb_cap, nb);
        let all_statics = interleave(&all_statics, links_cap, nb);
        let all_dof_vals = interleave(&all_dof_vals, dofs_cap, nb);
        let all_dof_vels = interleave(&all_dof_vels, dofs_cap, nb);
        let all_dof_damping = interleave(&all_dof_damping, dofs_cap, nb);
        let all_dof_armature = interleave(&all_dof_armature, dofs_cap, nb);
        let all_dof_stiffness = interleave(&all_dof_stiffness, dofs_cap, nb);
        let all_dof_spring_ref = interleave(&all_dof_spring_ref, dofs_cap, nb);
        let all_dof_kinematic = interleave(&all_dof_kinematic, dofs_cap, nb);

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
            // Default: implicit coriolis. The acceleration solve uses a mass
            // matrix augmented with the coriolis/gyroscopic derivatives, while
            // constraints keep the plain one. `set_implicit_coriolis(false)`
            // falls back to a single plain matrix with explicit-only
            // coriolis/gyroscopic forces (cheaper, but less stable).
            implicit_coriolis: true,
            coriolis_in_uniform: true,
            has_joint_constraints: all_infos.iter().any(|info| info.max_constraints > 0),
            frictionloss_slots_reserved: false,
            constraint_caps_dirty: false,

            multibody_info: Tensor::vector(backend, &all_infos, storage).unwrap(),
            max_contact_constraints: Tensor::scalar(
                backend,
                0u32,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap(),
            links_static: Tensor::vector(backend, &all_statics, storage | BufferUsages::COPY_DST)
                .unwrap(),
            links_static_mirror: all_statics.clone(),
            info_mirror,
            // COPY_SRC so hosts can read joint/link state back (observation
            // pipelines); see `GpuMultibodySet::links_workspace`.
            links_workspace: Tensor::vector(
                backend,
                crate::shaders::dynamics::ws_soa_from_structs(&all_ws, links_cap, num_batches),
                storage | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            dof_values: Tensor::vector(backend, &all_dof_vals, storage).unwrap(),
            dof_state: {
                // Pack [velocities, damping, armature, spring stiffness,
                // spring rest, kinematic mask, frictionloss] back-to-back,
                // each section N = dofs_cap * num_batches long. The shaders
                // address section `s` at intra-batch offset
                // `s · dof_batch_capacity`. Frictionloss has no rapier
                // counterpart, so it starts at zero (off) and is filled in by
                // `RbdState::set_dof_frictionloss`.
                let n = (dofs_cap * num_batches) as usize;
                let mut buf = Vec::with_capacity(7 * n);
                buf.extend_from_slice(&all_dof_vels);
                buf.extend_from_slice(&all_dof_damping);
                buf.extend_from_slice(&all_dof_armature);
                buf.extend_from_slice(&all_dof_stiffness);
                buf.extend_from_slice(&all_dof_spring_ref);
                buf.extend_from_slice(&all_dof_kinematic);
                buf.resize(7 * n, 0.0);
                debug_assert_eq!(buf.len(), 7 * n);
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
                // Two sections: the plain matrix (constraint LU) at intra-batch
                // offset 0 and the coriolis-melded acceleration matrix at intra-batch
                // offset `mm_cap`.
                vec![0.0f32; (2 * mm_cap * num_batches) as usize],
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
            dof_couplings: Tensor::vector(backend, &all_couplings, storage).unwrap(),
            couplings_per_batch: couplings_cap,
            body_to_link: Tensor::vector(backend, &all_body_to_link, storage).unwrap(),
            body_to_link_host: all_body_to_link,
            body_to_link_cap,
            motor_delay_state: Tensor::vector(
                backend,
                vec![0.0f32; ((2 + links_cap) * num_batches) as usize],
                storage | BufferUsages::COPY_DST,
            )
            .unwrap(),
            motor_delay_params: Tensor::scalar(
                backend,
                glamx::UVec4::new(num_batches, 2 + links_cap, 0, 0),
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap(),
            delay_update_cache: None,
            contact_sensor_links: Tensor::vector(
                backend,
                &[u32::MAX; crate::shaders::dynamics::MAX_CONTACT_SENSORS as usize],
                storage,
            )
            .unwrap(),
            contact_sensor_out: Tensor::vector(
                backend,
                vec![
                    0.0f32;
                    (mb_cap * num_batches * crate::shaders::dynamics::MAX_CONTACT_SENSORS) as usize
                ],
                storage | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            num_contact_sensors: 0,
            substep_refresh: true,
            substep_refresh_light: false,
            env_reset: None,
            reset_templates: None,
            scatter_caches: Vec::new(),
            contact_constraints: Tensor::vector(
                backend,
                vec![
                    MultibodyContactConstraint::default();
                    (contact_cons_cap * num_batches) as usize
                ],
                storage,
            )
            .unwrap(),
            old_contact_constraints: Tensor::vector(
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
            // Per-multibody Delassus blocks for the constraint-space contact
            // sweep: MAX_MB_CONTACT_CONSTRAINTS_PER_MB² floats each (147 KB
            // in 3D), so only small total multibody counts get them; larger
            // batched scenes keep the dof-space solve.
            contact_delassus: {
                // Sized by the capacity stride (the kernels index blocks by
                // `batch · multibodies_batch_capacity + mb_idx`).
                let total_mbs = mb_cap * num_batches;
                // `MAX_DELASSUS_MULTIBODIES` is currently 0, which disables the
                // path; the bound is kept so raising the constant re-enables it.
                #[allow(clippy::absurd_extreme_comparisons)]
                let use_delassus = global_max_mb > 0 && total_mbs <= MAX_DELASSUS_MULTIBODIES;
                if use_delassus {
                    let block = (MAX_MB_CONTACT_CONSTRAINTS_PER_MB
                        * MAX_MB_CONTACT_CONSTRAINTS_PER_MB)
                        as usize;
                    Some(
                        Tensor::vector_uninit(
                            backend,
                            (total_mbs as usize * block) as u32,
                            storage,
                        )
                        .unwrap(),
                    )
                } else {
                    None
                }
            },

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
            mb_imp_joint_num_colors: 0,
            mb_imp_joint_max_color_group_len: 0,
            max_ndofs: max_mb_ndofs,
            max_links: max_mb_links,
            max_joint_constraints: max_mb_joint_constraints,
            joint_constraints_per_batch: cons_cap,
            joint_constraint_columns_per_batch: cons_col_cap,
            contact_constraints_per_batch: contact_cons_cap,
            contact_constraint_columns_per_batch: contact_cons_col_cap,

            num_solver_iterations: 4,
            num_internal_pgs_iterations: 1,

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
            warmstart_coefficient: RbdSimParams::default().warmstart_coefficient,
        }
    }
}
