//! Initialization of [`RbdState`] from CPU-side Rapier data structures.

use crate::broad_phase::LbvhState;
use crate::dynamics::GpuImpulseJointSet;
#[cfg(feature = "dim3")]
use crate::dynamics::GpuMultibodySet;
use crate::math::{Pose, Vector};
use crate::queries::GpuColliderMaterial;
use crate::shaders::dynamics::{
    LocalMassProperties as GpuLocalMassProperties, RbdSimParams, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use crate::shaders::utils::BatchIndices;
use crate::utils::PrefixSumWorkspace;

use super::rbd_state::*;
use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuReadback};
use vortx::tensor::Tensor;
use {
    crate::math::Point,
    crate::rapier::dynamics::{ImpulseJointSet, MultibodyJointSet, RigidBodySet},
    crate::rapier::geometry::ColliderSet,
    crate::shapes::ShapeBuffers,
    crate::shapes::shape_from_parry,
    std::collections::HashMap,
};

impl RbdState {
    /// Creates a new GPU physics state from per-environment Rapier data structures.
    ///
    /// Environments with fewer colliders/joints are padded with dummy fixed bodies.
    /// Panics if any rigid body has more than one collider attached.
    pub fn from_rapier(
        backend: &GpuBackend,
        environments: &[(
            &RigidBodySet,
            &ColliderSet,
            &ImpulseJointSet,
            &MultibodyJointSet,
            &RbdSimParams,
        )],
        capacities: RbdCapacities,
    ) -> Self {
        // Fixed-grid dispatch default: ON for CUDA (each indirect dispatch there
        // costs a stream sync + host count read), OFF for WebGPU/Metal (native
        // indirect dispatch). `NEXUS_FIXED_GRID` overrides.
        crate::set_fixed_dispatch_grid_default(backend.is_cuda());
        let num_batches = environments.len() as u32;

        // Equal-topology invariant: every environment must share the same
        // collider count, joint count, multibody count and solver-iteration
        // count. Only collider shapes and dynamic state may differ. This lets
        // the per-batch topology counts collapse to scalar uniforms instead of
        // padded per-batch storage arrays.
        if let Some(((b0, c0, ij0, mj0, sp0), rest)) = environments.split_first() {
            for (i, (b, c, ij, mj, sp)) in rest.iter().enumerate() {
                let env = i + 1;
                assert_eq!(
                    c.len(),
                    c0.len(),
                    "batched rbd requires the same collider count in every environment \
                     (env 0 has {}, env {env} has {})",
                    c0.len(),
                    c.len()
                );
                assert_eq!(
                    b.len(),
                    b0.len(),
                    "batched rbd requires the same rigid-body count in every environment \
                     (env 0 has {}, env {env} has {})",
                    b0.len(),
                    b.len()
                );
                assert_eq!(
                    ij.len(),
                    ij0.len(),
                    "batched rbd requires the same impulse-joint count in every environment \
                     (env 0 has {}, env {env} has {})",
                    ij0.len(),
                    ij.len()
                );
                assert_eq!(
                    mj.multibodies().count(),
                    mj0.multibodies().count(),
                    "batched rbd requires the same multibody count in every environment"
                );
                assert_eq!(
                    sp.num_solver_iterations, sp0.num_solver_iterations,
                    "batched rbd requires the same solver-iteration count in every environment \
                     (env 0 has {}, env {env} has {})",
                    sp0.num_solver_iterations, sp.num_solver_iterations
                );
            }
        }

        // Equal across all environments by the invariant above, so the
        // first environment's collider count is the per-batch count and there
        // is no padding.
        let num_colliders = environments
            .first()
            .map(|(_, c, _, _, _)| c.len())
            .unwrap_or(0);
        // Body slots are independent of colliders: every collider-parented body,
        // every parentless collider's synthetic body, AND every multibody link
        // (even collider-less ones — visual-only links, spacers) gets a slot, so
        // rapier-style collider-less bodies need no placeholder collider. Bodies
        // and colliders still share one per-batch stride; size it to fit whichever
        // is larger. Equal topology across batches → env 0's counts are the
        // per-batch counts.
        let num_bodies_capacity = environments
            .first()
            .map(|(_bodies, colliders, _imp, mbj, _sp)| {
                let _ = &mbj;
                let mut parents = std::collections::HashSet::new();
                let mut parentless = 0usize;
                for (_, co) in colliders.iter() {
                    match co.parent() {
                        Some(h) => {
                            parents.insert(h);
                        }
                        None => parentless += 1,
                    }
                }
                #[cfg(feature = "dim3")]
                for mb in mbj.multibodies() {
                    for link in mb.links() {
                        parents.insert(link.rigid_body_handle());
                    }
                }
                parents.len() + parentless
            })
            .unwrap_or(0);
        let max_colliders = num_colliders.max(num_bodies_capacity);

        let mut all_poses = Vec::new();
        let mut all_vels = Vec::new();
        let mut all_local_mprops = Vec::new();
        let mut all_mprops = Vec::new();
        let mut all_shapes = Vec::new();
        let mut all_collision_groups: Vec<crate::rapier::geometry::InteractionGroups> = Vec::new();
        let mut all_collider_materials: Vec<GpuColliderMaterial> = Vec::new();
        let mut all_collider_local_poses: Vec<Pose> = Vec::new();
        // Per-collider map to the owning rigid body's slot (env-local index).
        // Bodies and colliders share the `max_colliders` per-batch stride but
        // form distinct index spaces — a body may own several colliders.
        let mut all_collider_parent: Vec<u32> = Vec::new();
        // Per-environment count of *active* rigid bodies (distinct collider
        // parents + one synthetic body per parentless collider). Used to assert
        // the equal-topology invariant and to set `BatchIndices::bodies_len`.
        let mut all_env_body_counts: Vec<usize> = Vec::new();
        let mut shape_buffers = ShapeBuffers::default();
        let mut joint_envs: Vec<(
            &ImpulseJointSet,
            HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
        )> = Vec::new();

        // Collect per-batch sim params, adjusting dt for substeps.
        let num_solver_iterations = environments
            .iter()
            .map(|(_, _, _, _, sp)| sp.num_solver_iterations)
            .max()
            .unwrap_or(4);
        let all_sim_params: Vec<RbdSimParams> = environments
            .iter()
            .map(|(_, _, _, _, sp)| {
                let mut sp = **sp;
                sp.dt /= sp.num_solver_iterations as f32;
                sp
            })
            .collect();
        // Pick representative dt (outer dt, not the per-substep one) from any batch.
        #[cfg(feature = "dim3")]
        let multibody_dt = environments
            .first()
            .map(|(_, _, _, _, sp)| sp.dt)
            .unwrap_or(1.0 / 60.0);

        // Dummy data for padding shorter environments.
        let dummy_pose = Pose::default();
        let dummy_local_mprops = GpuLocalMassProperties::default();
        let dummy_mprops = GpuWorldMassProperties::default();

        #[cfg(feature = "dim3")]
        let mut multibody_envs: Vec<(
            &MultibodyJointSet,
            HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
            &RigidBodySet,
        )> = Vec::new();

        #[cfg_attr(not(feature = "dim3"), allow(unused_variables))]
        for (bodies, colliders, impulse_joints, multibody_joints, _sim_params) in environments {
            let env_collider_count = colliders.len();
            // `body_ids` maps a rigid-body handle to its env-local body slot.
            // A body slot is allocated the first time one of the body's
            // colliders is visited (so multi-collider bodies get one slot).
            let mut body_ids = HashMap::new();
            // Body-indexed data is appended in body-slot order; the first
            // `env_body_idx` entries of this env's body region are real bodies.
            let mut env_body_idx = 0u32;

            // Helper: compute a body's (local, world) mass properties. The world
            // mass properties aggregate ALL of the body's colliders because
            // rapier's `RigidBody::mass_properties()` already does so (the host
            // calls `recompute_mass_properties_from_colliders` beforehand).
            // Non-dynamic / parentless bodies get a zero (static) mass.
            let make_body_mprops =
                |body_pose: &Pose, dynamic_body: Option<&crate::rapier::dynamics::RigidBody>| {
                    if let Some(b) = dynamic_body {
                        let m = b.mass_properties();
                        let local = local_mprops_from_rapier(&m.local_mprops);
                        let world = world_mprops_from_local(body_pose, &local);
                        (local, world)
                    } else {
                        let mut local = GpuLocalMassProperties::default();
                        let mut world = GpuWorldMassProperties {
                            com: body_pose.translation,
                            ..Default::default()
                        };
                        local.inv_mass = Vector::ZERO;
                        world.inv_mass = Vector::ZERO;
                        #[cfg(feature = "dim3")]
                        {
                            local.inv_principal_inertia = glamx::Vec3::ZERO;
                            world.inv_inertia = glamx::Mat4::ZERO;
                        }
                        #[cfg(feature = "dim2")]
                        {
                            local.inv_inertia = 0.0;
                            world.inv_inertia = 0.0;
                        }
                        (local, world)
                    }
                };

            for (_, co) in colliders.iter() {
                // Resolve (allocating if needed) the parent body's env-local slot.
                //
                // Mirror rapier: bodies hold a world-origin pose, colliders hold
                // their local offset and a world pose `body * local`. Joint /
                // solver / integrator / mprops_update / multibody-FK consume
                // `body_poses`. Broad-phase, narrow-phase and contact-to-constraint
                // consume `collider_world_poses`, which the
                // `gpu_sync_collider_poses` kernel keeps in sync each step.
                // Parentless colliders become an implicit fixed body anchored at
                // the world origin: the body pose is IDENTITY and the collider's
                // world transform is folded into its body-local offset.
                let (body_local, collider_local_pose) = match co.parent() {
                    Some(h) => {
                        let collider_local_pose =
                            co.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
                        let body_local = *body_ids.entry(h).or_insert_with(|| {
                            let idx = env_body_idx;
                            env_body_idx += 1;
                            let b = &bodies[h];
                            let body_pose = *b.position();
                            let (local_mprops, mprops) =
                                make_body_mprops(&body_pose, b.is_dynamic().then_some(b));
                            all_poses.push(body_pose);
                            all_vels.push(GpuVelocity::new(b.linvel(), b.angvel()));
                            all_local_mprops.push(local_mprops);
                            all_mprops.push(mprops);
                            idx
                        });
                        (body_local, collider_local_pose)
                    }
                    None => {
                        let idx = env_body_idx;
                        env_body_idx += 1;
                        let body_pose = Pose::IDENTITY;
                        let (local_mprops, mprops) = make_body_mprops(&body_pose, None);
                        all_poses.push(body_pose);
                        all_vels.push(GpuVelocity::default());
                        all_local_mprops.push(local_mprops);
                        all_mprops.push(mprops);
                        (idx, *co.position())
                    }
                };

                all_shapes.push(
                    shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape"),
                );
                all_collider_local_poses.push(collider_local_pose);
                all_collision_groups.push(co.collision_groups());
                all_collider_materials.push(collider_material_from_rapier(co));
                // Env-local body slot; the kernels apply the per-batch stride.
                all_collider_parent.push(body_local);
            }

            // Give every multibody link a body slot too, even collider-less ones
            // (rapier's bodies exist independently of colliders). Without this a
            // collider-less link's FK pose would fall back to slot 0. Slots are
            // appended after the collider-parented bodies, in multibody-link
            // order; `state.rs`'s `rbd2gpu` rebuild mirrors this exact order.
            #[cfg(feature = "dim3")]
            for mb in multibody_joints.multibodies() {
                for link in mb.links() {
                    let h = link.rigid_body_handle();
                    body_ids.entry(h).or_insert_with(|| {
                        let idx = env_body_idx;
                        env_body_idx += 1;
                        let b = &bodies[h];
                        let body_pose = *b.position();
                        let (local_mprops, mprops) =
                            make_body_mprops(&body_pose, b.is_dynamic().then_some(b));
                        all_poses.push(body_pose);
                        all_vels.push(GpuVelocity::new(b.linvel(), b.angvel()));
                        all_local_mprops.push(local_mprops);
                        all_mprops.push(mprops);
                        idx
                    });
                }
            }

            let env_body_count = env_body_idx as usize;
            all_env_body_counts.push(env_body_count);

            // Pad colliders to `max_colliders` with dummy colliders that never
            // collide (zero membership AND zero filter).
            let dummy_shape = all_shapes.last().copied().unwrap_or_default();
            for _ in env_collider_count..max_colliders {
                all_collider_local_poses.push(Pose::IDENTITY);
                all_shapes.push(dummy_shape);
                all_collision_groups.push(crate::rapier::geometry::InteractionGroups::new(
                    crate::rapier::geometry::Group::NONE,
                    crate::rapier::geometry::Group::NONE,
                    crate::rapier::geometry::InteractionTestMode::And,
                ));
                // Padding colliders are inert; material is irrelevant but must
                // exist to keep the per-collider buffers the same length.
                all_collider_materials.push(GpuColliderMaterial::default());
                // Padding colliders are inert; parent body 0 is a safe placeholder.
                all_collider_parent.push(0);
            }

            // Pad bodies to the shared per-batch stride (`max_colliders` =
            // `max(num_colliders, num_bodies)`) with dummy fixed bodies.
            for _ in env_body_count..max_colliders {
                all_poses.push(dummy_pose);
                all_vels.push(GpuVelocity::default());
                all_local_mprops.push(dummy_local_mprops);
                all_mprops.push(dummy_mprops);
            }

            #[cfg(feature = "dim3")]
            multibody_envs.push((multibody_joints, body_ids.clone(), bodies));
            joint_envs.push((impulse_joints, body_ids));
        }

        // Equal-topology invariant also covers the body count: every batch must
        // expose the same number of rigid bodies in the same slot order.
        let num_bodies = all_env_body_counts.first().copied().unwrap_or(0);
        for (env, count) in all_env_body_counts.iter().enumerate() {
            assert_eq!(
                *count, num_bodies,
                "batched rbd requires the same rigid-body count in every environment \
                 (env {env} has {count}, env 0 has {num_bodies})"
            );
        }

        // NOTE: GPU doesn't like empty storage buffer bindings so add dummy data
        //       instead of leaving them empty (which is fine considering they are
        //       not referenced by any collider).
        if shape_buffers.vertices.is_empty() {
            shape_buffers.vertices.push(Point::ZERO.into());
        }
        if shape_buffers.indices.is_empty() {
            shape_buffers.indices.extend_from_slice(&[0; 3]);
        }

        let vertex_buffers =
            Tensor::vector(backend, &shape_buffers.vertices, BufferUsages::STORAGE).unwrap();
        let index_buffers =
            Tensor::vector(backend, &shape_buffers.indices, BufferUsages::STORAGE).unwrap();

        let joint_env_refs: Vec<(
            &ImpulseJointSet,
            &HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
        )> = joint_envs
            .iter()
            .map(|(joints, body_ids)| (*joints, body_ids))
            .collect();

        // Per-environment "graph group" table: each body local id maps to either
        // its own id (free body) or a shared multibody-group id. Bodies in the
        // same multibody collapse to a single node for impulse-joint coloring,
        // matching rapier's rule.
        //
        // We also build `is_mb_body[env][body_local]` — true iff the body is
        // part of some multibody. The regular `GpuImpulseJointSet` skips any
        // joint touching such a body; those joints are routed to
        // `GpuMultibodySet::set_impulse_joints` instead so they go through
        // the generic `M⁻¹·Jᵀ` solver path.
        #[cfg(feature = "dim3")]
        let (multibody_groups, is_mb_body): (Vec<Vec<u32>>, Vec<Vec<bool>>) = {
            let mut all_groups: Vec<Vec<u32>> = Vec::with_capacity(num_batches as usize);
            let mut all_is_mb: Vec<Vec<bool>> = Vec::with_capacity(num_batches as usize);
            for (env_idx, (mb_set, body_ids, _)) in multibody_envs.iter().enumerate() {
                let _ = env_idx;
                let max_id = body_ids.values().copied().max().unwrap_or_default();
                let mut group: Vec<u32> = (0..=max_id).collect();
                let mut is_mb: Vec<bool> = vec![false; (max_id + 1) as usize];
                for (g, mb) in (max_id + 1..).zip(mb_set.multibodies()) {
                    for link in mb.links() {
                        if let Some(&id) = body_ids.get(&link.rigid_body_handle()) {
                            group[id as usize] = g;
                            is_mb[id as usize] = true;
                        }
                    }
                }
                all_groups.push(group);
                all_is_mb.push(is_mb);
            }
            (all_groups, all_is_mb)
        };
        #[cfg(not(feature = "dim3"))]
        let (multibody_groups, is_mb_body): (Vec<Vec<u32>>, Vec<Vec<bool>>) =
            (Vec::new(), Vec::new());

        let joints = GpuImpulseJointSet::from_rapier_filtered(
            backend,
            &joint_env_refs,
            &multibody_groups,
            &is_mb_body,
        );

        // Convert multibodies (3D only).
        #[cfg(feature = "dim3")]
        let multibodies = {
            let mb_refs: Vec<(
                &MultibodyJointSet,
                &HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
                &RigidBodySet,
            )> = multibody_envs
                .iter()
                .map(|(mb, ids, bodies)| (*mb, ids, *bodies))
                .collect();
            let mut mb = GpuMultibodySet::from_rapier(
                backend,
                &mb_refs,
                [0.0, -9.81, 0.0],
                max_colliders as u32,
            );
            mb.set_visible_dt(backend, multibody_dt);
            // Soft contact coefficients (rapier TGS-soft) from the substep sim
            // params, so multibody-vs-floor contacts use the same soft ERP + CFM
            // as the free-body path (and as rapier) instead of a rigid `1/dt`.
            mb.set_constraint_softness(backend, &all_sim_params[0]);

            // Route MB-touching impulse joints (those skipped by the
            // regular `GpuImpulseJointSet`) to the multibody generic
            // constraint path.
            let imp_refs: Vec<(
                &crate::rapier::dynamics::ImpulseJointSet,
                &MultibodyJointSet,
                &HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
                &RigidBodySet,
            )> = joint_envs
                .iter()
                .zip(multibody_envs.iter())
                .map(|((imp, body_ids), (mb_set, _, bodies))| (*imp, *mb_set, body_ids, *bodies))
                .collect();
            mb.set_impulse_joints(backend, &imp_refs);
            mb
        };

        // Mark multibody-controlled bodies as kinematic (inv_mass = 0) in the shared
        // body buffers so the rigid-body pipeline leaves them alone. The multibody
        // solver owns their masses internally.
        #[cfg(feature = "dim3")]
        {
            for (batch_idx, (mb_set, body_ids, _)) in multibody_envs.iter().enumerate() {
                let batch_offset = batch_idx * max_colliders;
                for mb in mb_set.multibodies() {
                    for link in mb.links() {
                        if let Some(&rb_local_id) = body_ids.get(&link.rigid_body_handle()) {
                            let global = batch_offset + rb_local_id as usize;
                            all_local_mprops[global].inv_mass = Vector::ZERO;
                            all_local_mprops[global].inv_principal_inertia = glamx::Vec3::ZERO;
                            // Multibody links must have a zero local COM in the
                            // shared body buffer: the multibody substep writes
                            // body-origin poses to `solver_body_poses`, and we
                            // need `init_solver_bodies` / `finalize` (which
                            // shift by ±local_com) to be no-ops here so the
                            // multibody apply_substep can treat solver poses
                            // as body-origin poses.
                            all_local_mprops[global].com = Vector::ZERO;
                            all_mprops[global].inv_mass = Vector::ZERO;
                            all_mprops[global].inv_inertia = glamx::Mat4::ZERO;
                        }
                    }
                }
            }
        }

        // Build the per-body "graph group" lookup. Free bodies map to themselves
        // (one body = one graph node). Bodies belonging to a multibody all map
        // to a single shared group id (= the body id of the multibody's root
        // link). Coloring kernels read `body_group[body_id]` instead of `body_id`
        // when computing constraint adjacency, so contacts touching different
        // bodies of the same multibody correctly conflict and never share a
        // color.
        // `body_group` stores PER-BATCH local indices (so a kernel can use the
        // same `Slice(buf, colliders_start)` pattern as for `body_constraint_*`
        // and just index by `group_local`).
        let mut all_body_group: Vec<u32> = Vec::with_capacity(max_colliders * num_batches as usize);
        for _batch_idx in 0..num_batches as usize {
            for b in 0..max_colliders {
                all_body_group.push(b as u32);
            }
        }
        #[cfg(feature = "dim3")]
        for (batch_idx, (mb_set, body_ids, _)) in multibody_envs.iter().enumerate() {
            let base = batch_idx * max_colliders;
            for mb in mb_set.multibodies() {
                let group_local = mb
                    .links()
                    .next()
                    .and_then(|root| body_ids.get(&root.rigid_body_handle()).copied());
                let Some(group_local) = group_local else {
                    continue;
                };
                for link in mb.links() {
                    if let Some(&local) = body_ids.get(&link.rigid_body_handle()) {
                        all_body_group[base + local as usize] = group_local;
                    }
                }
            }
        }
        let body_group = Tensor::vector(backend, &all_body_group, BufferUsages::STORAGE).unwrap();

        let num_colliders_per_batch = max_colliders;
        let num_bodies_total = num_colliders_per_batch * num_batches as usize;

        // Initial body velocities were accumulated in body-slot order alongside
        // `all_poses`; zero-filling here would silently drop each body's initial
        // linvel/angvel (gravity used to mask the linear part, but e.g. an
        // initial spin was lost entirely).
        debug_assert_eq!(all_vels.len(), num_bodies_total);
        let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let shapes = Tensor::vector(backend, &all_shapes, storage).unwrap();
        let collider_local_poses =
            Tensor::vector(backend, &all_collider_local_poses, storage).unwrap();
        let collider_parent = Tensor::vector(backend, &all_collider_parent, storage).unwrap();
        let collision_groups = Tensor::vector(backend, &all_collision_groups, storage).unwrap();
        let collider_materials = Tensor::vector(backend, &all_collider_materials, storage).unwrap();

        let collision_pairs = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let collision_pairs_len = Tensor::vector_uninit(
            backend,
            num_batches,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let collision_pairs_len_max =
            Tensor::vector_uninit(backend, 1, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
                .unwrap();
        let num_batches_uniform = Tensor::scalar(
            backend,
            collision_pairs_len.layout().into(),
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();
        // Two-element readback: the (max) collision-pair count and the uncolored count.
        let resize_readback = GpuReadback::new(backend, 2).unwrap();
        let collision_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();

        let contacts = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let contacts_len = Tensor::vector_uninit(
            backend,
            num_batches,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let contacts_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let pfm_pairs_len = Tensor::vector_uninit(
            backend,
            num_batches,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let old_constraints = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let old_constraint_builders = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let new_constraints = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let new_constraint_builders = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let constraints_colors = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let colored = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let constraints_rands = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * num_batches,
            storage,
        )
        .unwrap();
        let old_constraints_counts = Tensor::vector_uninit(
            backend,
            num_colliders_per_batch as u32 * num_batches,
            storage,
        )
        .unwrap();
        let new_constraints_counts = Tensor::vector_uninit(
            backend,
            num_colliders_per_batch as u32 * num_batches,
            storage,
        )
        .unwrap();
        let old_body_constraint_ids = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * 2 * num_batches,
            storage,
        )
        .unwrap();
        let new_body_constraint_ids = Tensor::vector_uninit(
            backend,
            capacities.collisions_capacity * 2 * num_batches,
            storage,
        )
        .unwrap();

        let lbvh_usages = if crate::VALIDATE_LBVH_TOPOLOGY {
            BufferUsages::STORAGE | BufferUsages::COPY_SRC
        } else {
            BufferUsages::STORAGE
        };

        let contacts_per_batch_cpu = capacities.collisions_capacity;
        let collision_pairs_per_batch_cpu = capacities.collisions_capacity;
        #[allow(unused_mut)] // Only mutated with the dim3 (multibody) feature.
        let mut bi = BatchIndices {
            colliders_batch_capacity: num_colliders_per_batch as u32,
            colliders_len: num_colliders as u32,
            bodies_len: num_bodies as u32,
            collision_pairs_batch_capacity: collision_pairs_per_batch_cpu,
            contacts_batch_capacity: contacts_per_batch_cpu,
            impulse_joints_batch_capacity: joints.joints_per_batch(),
            impulse_joints_len: joints.num_active_joints(),
            ..Default::default()
        };
        #[cfg(feature = "dim3")]
        multibodies.fill_batch_indices(&mut bi);
        let batch_indices = Tensor::scalar(
            backend,
            bi,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();

        Self {
            capacities,
            num_batches,
            num_colliders_per_batch: num_colliders_per_batch as u32,
            num_solver_iterations,
            sim_params: Tensor::vector(backend, &all_sim_params, BufferUsages::STORAGE).unwrap(),
            vels: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_out: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_inc: Tensor::vector(backend, &all_vels, storage).unwrap(),
            joints,
            #[cfg(feature = "dim3")]
            multibodies,
            body_group,
            local_mprops: Tensor::vector(backend, &all_local_mprops, storage).unwrap(),
            mprops: Tensor::vector(backend, &all_mprops, storage).unwrap(),
            body_poses: Tensor::vector(
                backend,
                &all_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            // Sized like `body_poses`. Will be (re-)seeded each step before
            // the solver runs; the initial contents don't matter, but using
            // the body poses here keeps the buffer in a sensible state.
            solver_body_poses: Tensor::vector(
                backend,
                &all_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            // Sized like `body_poses`. Refreshed by `gpu_sync_collider_poses`
            // once per step before broad-phase / narrow-phase /
            // contact-to-constraint init.
            collider_world_poses: Tensor::vector(
                backend,
                &all_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            vertex_buffers,
            index_buffers,
            shapes,
            collider_local_poses,
            collider_parent,
            collision_groups,
            collider_materials,
            collision_pairs,
            collision_pairs_len,
            collision_pairs_len_max,
            num_batches_uniform,
            resize_readback,
            collision_pairs_indirect,
            contacts_per_batch_cpu,
            collision_pairs_per_batch_cpu,
            collision_pairs_len_cpu: 0,
            batch_indices,
            contacts,
            contacts_len,
            contacts_indirect,
            pfm_pairs,
            pfm_pairs_len,
            pfm_pairs_indirect,
            old_constraints,
            old_constraint_builders,
            old_constraints_counts,
            new_constraints,
            new_constraint_builders,
            new_constraints_counts,
            constraints_colors,
            colored,
            constraints_rands,
            curr_color: Tensor::scalar(
                backend,
                0u32,
                BufferUsages::STORAGE
                    | BufferUsages::UNIFORM
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            uncolored: Tensor::scalar(
                backend,
                0,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            uncolored_staging: Tensor::scalar(
                backend,
                0,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )
            .unwrap(),
            old_body_constraint_ids,
            new_body_constraint_ids,
            prefix_sum_workspace: PrefixSumWorkspace::default(),
            lbvh: LbvhState::with_usages(backend, lbvh_usages),
            max_colors: capacities.solver_colors,
            num_active_colliders: num_colliders as u32,
            num_active_bodies: num_bodies as u32,
        }
    }
}
