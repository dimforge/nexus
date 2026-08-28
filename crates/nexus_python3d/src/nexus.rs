//! Core simulation objects: `NexusState`, `NexusPipeline`, `RbdCoupling`,
//! `GpuTimestamps`, and the various entity handles.

use crate::loaders::{MjcfSceneInfo, UrdfLoaderOptions, UrdfRobotHandles};
use crate::math::{Pose, Vec3};
use crate::mpm::{BoundaryCondition, Particle, SimulationParams};
use crate::rbd::{
    Collider, ImpulseJointHandle, JointArg, JointAxis, MultibodyJointHandle, RigidBody,
    RigidBodyHandle, SharedShape,
};
use crate::robot::{Robot, build_robot, free_axes, joint_axis, pose_from_wxyz, to_wxyz};
use crate::viewer::NexusViewer;
use khal::backend::{Backend, GpuTimestamps as RGpuTimestamps};
use nexus3d::mpm::solver::BoundaryCondition as RBoundaryCondition;
use nexus3d::prelude::{
    NexusPipeline as RNexusPipeline, NexusPipelineMask, NexusState as RNexusState,
    RbdCoupling as RRbdCoupling,
};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapier3d::prelude as rp;
use std::collections::HashMap;

/// Maps a GPU backend error to a Python exception.
fn gpu_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}

/// Coupling mode between a rigid body and the MPM simulation.
#[pyclass(name = "RbdCoupling", from_py_object)]
#[derive(Clone, Copy)]
pub struct RbdCoupling(pub RRbdCoupling);

#[pymethods]
impl RbdCoupling {
    #[classattr]
    const NONE: RbdCoupling = RbdCoupling(RRbdCoupling::None);
    // Convenience constants defaulting to a `stick()` boundary; use
    // `mpm_one_way` / `mpm_two_way` to pick a specific boundary condition.
    #[classattr]
    const MPM_ONE_WAY_COUPLING: RbdCoupling =
        RbdCoupling(RRbdCoupling::MpmOneWay(RBoundaryCondition::stick()));
    #[classattr]
    const MPM_TWO_WAY_COUPLING: RbdCoupling =
        RbdCoupling(RRbdCoupling::MpmTwoWay(RBoundaryCondition::stick()));

    /// One-way coupling (MPM pushes the rigid body, not vice-versa) using the
    /// given boundary condition at the collider surface.
    #[staticmethod]
    fn mpm_one_way(boundary: BoundaryCondition) -> RbdCoupling {
        RbdCoupling(RRbdCoupling::MpmOneWay(boundary.0))
    }

    /// Two-way coupling (MPM and the rigid body affect each other) using the
    /// given boundary condition at the collider surface.
    #[staticmethod]
    fn mpm_two_way(boundary: BoundaryCondition) -> RbdCoupling {
        RbdCoupling(RRbdCoupling::MpmTwoWay(boundary.0))
    }
}

/// Entity counts for a `NexusState` (mirrors `NexusCounts`).
#[pyclass(name = "NexusCounts", from_py_object)]
#[derive(Clone, Copy)]
pub struct NexusCounts {
    #[pyo3(get)]
    pub num_environments: usize,
    #[pyo3(get)]
    pub rigid_bodies: usize,
    #[pyo3(get)]
    pub colliders: usize,
    #[pyo3(get)]
    pub impulse_joints: usize,
    #[pyo3(get)]
    pub multibodies: usize,
    #[pyo3(get)]
    pub multibody_dofs: usize,
    #[pyo3(get)]
    pub particles: usize,
}

/// Handle to a chunk of MPM particles added via `NexusState.add_particles`.
#[pyclass(name = "NexusParticleChunk", from_py_object)]
#[derive(Clone, Copy)]
pub struct NexusParticleChunk(pub nexus3d::prelude::NexusParticleChunk);

/// Optional GPU timing-query buffer (`khal::backend::GpuTimestamps`).
#[pyclass(name = "GpuTimestamps", unsendable)]
pub struct GpuTimestamps(pub RGpuTimestamps);

#[pymethods]
impl GpuTimestamps {
    #[new]
    fn new(viewer: PyRef<NexusViewer>, capacity: u32) -> Self {
        GpuTimestamps(RGpuTimestamps::new(viewer.backend(), capacity))
    }
}

/// The GPU-resident state of a multiphysics simulation
/// (`nexus3d::prelude::NexusState`). The second field keeps the
/// `rapier3d-mjcf` robot handles of the last `insert_mjcf`, so
/// `apply_actuator_controls` can drive the robot's actuators per step.
#[pyclass(name = "NexusState", unsendable)]
pub struct NexusState(pub RNexusState, pub Option<crate::loaders::MjcfHandles>);

#[pymethods]
impl NexusState {
    #[new]
    fn new() -> Self {
        NexusState(RNexusState::default(), None)
    }

    // --- rigid bodies -----------------------------------------------------

    fn insert_rigid_body(
        &mut self,
        body: PyRef<RigidBody>,
        collider: PyRef<Collider>,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        RigidBodyHandle(
            self.0
                .insert_rigid_body(body.0.clone(), collider.0.clone(), coupling.0),
        )
    }

    fn insert_rigid_body_in(
        &mut self,
        env: usize,
        body: PyRef<RigidBody>,
        collider: PyRef<Collider>,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_rigid_body_in(
            env,
            body.0.clone(),
            collider.0.clone(),
            coupling.0,
        ))
    }

    fn insert_body(&mut self, body: PyRef<RigidBody>, coupling: RbdCoupling) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_body(body.0.clone(), coupling.0))
    }

    /// Inserts a collider-less body into environment `env`; attach colliders to
    /// it afterwards with `insert_collider_in` (multiple colliders per body).
    fn insert_body_in(
        &mut self,
        env: usize,
        body: PyRef<RigidBody>,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_body_in(env, body.0.clone(), coupling.0))
    }

    /// Attaches a collider to an existing body (`parent`), or inserts a
    /// parent-less one when `parent` is `None`, in environment `env`.
    #[pyo3(signature = (env, collider, parent=None))]
    fn insert_collider_in(
        &mut self,
        env: usize,
        collider: PyRef<Collider>,
        parent: Option<RigidBodyHandle>,
    ) {
        self.0
            .insert_collider_in(env, collider.0.clone(), parent.map(|h| h.0));
    }

    /// Reserves `capacity` spare GPU body slots (in environment 0) so later
    /// `add_rigid_bodies` calls append in place instead of forcing a full scene
    /// rebuild. Call this *before* the first `finalize`.
    fn reserve_rigid_bodies(&mut self, capacity: usize) {
        self.0.reserve_rigid_bodies(capacity);
    }

    /// Appends body+collider pairs to the *live* GPU scene (environment 0) in a
    /// single batch, without rebuilding — the fast path for spawning bodies
    /// mid-simulation. Unlike `insert_rigid_body` (whose bodies only reach the
    /// GPU on the next `finalize`), these are simulated immediately. Reserve
    /// capacity up-front with `reserve_rigid_bodies`; only primitive-shape
    /// colliders are supported on the fast path. Returns the new handles in
    /// input order.
    fn add_rigid_bodies(
        &mut self,
        viewer: PyRef<NexusViewer>,
        bodies: Vec<RigidBody>,
        colliders: Vec<Collider>,
        coupling: RbdCoupling,
    ) -> PyResult<Vec<RigidBodyHandle>> {
        if bodies.len() != colliders.len() {
            return Err(PyRuntimeError::new_err(
                "bodies and colliders must have the same length",
            ));
        }
        let triples = bodies
            .into_iter()
            .zip(colliders)
            .map(|(b, c)| (b.0, c.0, coupling.0));
        self.0
            .add_rigid_bodies(viewer.backend(), triples)
            .map(|hs| hs.into_iter().map(RigidBodyHandle).collect())
            .map_err(gpu_err)
    }

    // --- joints -----------------------------------------------------------

    fn insert_impulse_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> ImpulseJointHandle {
        ImpulseJointHandle(
            self.0
                .insert_impulse_joint(body1.0, body2.0, joint.into_generic()),
        )
    }

    fn insert_impulse_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> ImpulseJointHandle {
        ImpulseJointHandle(self.0.insert_impulse_joint_in(
            env,
            body1.0,
            body2.0,
            joint.into_generic(),
        ))
    }

    fn insert_multibody_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> Option<MultibodyJointHandle> {
        self.0
            .insert_multibody_joint(body1.0, body2.0, joint.into_generic())
            .map(MultibodyJointHandle)
    }

    fn insert_multibody_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> Option<MultibodyJointHandle> {
        self.0
            .insert_multibody_joint_in(env, body1.0, body2.0, joint.into_generic())
            .map(MultibodyJointHandle)
    }

    // --- batched environments ---------------------------------------------

    /// Allocates a new batched simulation environment, returning its index.
    fn add_environment(&mut self) -> usize {
        self.0.add_environment()
    }

    /// Number of GPU batches (== number of environments) once finalized.
    fn rbd_num_batches(&self) -> u32 {
        self.0.rbd_num_batches()
    }

    // --- robot loaders ----------------------------------------------------

    /// Loads a URDF robot into environment 0 as a multibody and returns the
    /// per-collider render shapes plus the link count. Register the shapes with
    /// `viewer.insert_visual_shape(0, body, shape, pose)`.
    ///
    /// With `actuate_angx_motors=True` every joint's `AngX` motor is switched to
    /// acceleration-based mode (initial target velocity 0), ready for per-frame
    /// `set_multibody_motor_velocity` control.
    #[pyo3(signature = (path, options, actuate_angx_motors=false))]
    fn insert_urdf(
        &mut self,
        path: std::path::PathBuf,
        options: PyRef<UrdfLoaderOptions>,
        actuate_angx_motors: bool,
    ) -> PyResult<UrdfRobotHandles> {
        use rapier3d_urdf::{UrdfMultibodyOptions, UrdfRobot};
        let opts = options.to_rapier();
        let (mut robot, _) = UrdfRobot::from_file(&path, opts, None).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to load URDF {}: {e}", path.display()))
        })?;
        if actuate_angx_motors {
            for j in &mut robot.joints {
                j.joint
                    .set_motor_model(rp::JointAxis::AngX, rp::MotorModel::AccelerationBased);
                j.joint.set_motor_velocity(rp::JointAxis::AngX, 0.0, 1.0);
            }
        }
        let world = self.0.rbd_world_mut(0);
        let handles = robot.insert_using_multibody_joints(
            &mut world.bodies,
            &mut world.colliders,
            &mut world.multibody_joints,
            UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
        );
        let num_links = handles.links.len() as u32;
        let mut render_shapes = Vec::new();
        for link in &handles.links {
            for collider in &link.colliders {
                let (shape, local_pose) = match &collider.visual {
                    Some(v) => (v.shape.clone(), v.local_pose),
                    None => (
                        world.colliders[collider.handle].shared_shape().clone(),
                        rp::Pose::IDENTITY,
                    ),
                };
                render_shapes.push((
                    RigidBodyHandle(link.body),
                    SharedShape(shape),
                    Pose(local_pose),
                ));
            }
        }
        Ok(UrdfRobotHandles {
            render_shapes,
            num_links,
        })
    }

    /// Per-environment collision-pair capacity (default 4096). Lower it before
    /// `finalize` when batching many small environments: pair-keyed GPU
    /// workspaces scale with `capacity x num_envs`.
    fn set_rbd_collisions_capacity(&mut self, capacity: u32) {
        self.0.set_rbd_collisions_capacity(capacity);
    }

    /// Loads a MuJoCo MJCF scene into environment `env` as multibodies,
    /// registering its render shapes (and a sized floor) with `viewer`. Returns
    /// scene info (suggested camera + whether the scene is Z-up). Call
    /// `finalize` after.
    #[pyo3(signature = (viewer, scene_path, render_colliders=false, env=0))]
    fn insert_mjcf(
        &mut self,
        viewer: PyRefMut<NexusViewer>,
        scene_path: std::path::PathBuf,
        render_colliders: bool,
        env: usize,
    ) -> PyResult<MjcfSceneInfo> {
        let (info, handles) =
            crate::loaders::insert_mjcf(&mut self.0, viewer, &scene_path, render_colliders, env)?;
        self.1 = handles;
        Ok(info)
    }

    // --- MJCF actuation -----------------------------------------------------

    /// Names of the MJCF `<actuator>`s of the robot loaded by `insert_mjcf`, in
    /// actuator (control-vector) order. Unnamed actuators fall back to the name
    /// of the joint they drive. Empty before `insert_mjcf`.
    fn actuator_names(&self) -> Vec<String> {
        self.1
            .as_ref()
            .map(|h| {
                h.actuators
                    .iter()
                    .map(|a| {
                        a.actuator
                            .name
                            .clone()
                            .or_else(|| a.actuator.joint.clone())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Applies one MJCF control vector (one entry per actuator, in
    /// `actuator_names` order) to every environment's copy of the robot loaded
    /// by `insert_mjcf`, with full MJCF actuator semantics (`<position>` servos
    /// with kp/kv, `<motor>` force/gear, force limits), and pushes the resulting
    /// joint-motor state to the GPU.
    ///
    /// Call once per control step, after `finalize`; the next
    /// `NexusPipeline.simulate` steps the solver against the new targets.
    #[pyo3(signature = (viewer, ctrl))]
    fn apply_actuator_controls(
        &mut self,
        viewer: PyRef<NexusViewer>,
        ctrl: Vec<f32>,
    ) -> PyResult<()> {
        let Some(handles) = self.1.as_ref() else {
            return Err(PyRuntimeError::new_err(
                "no MJCF robot loaded (call insert_mjcf first)",
            ));
        };
        if ctrl.len() != handles.actuators.len() {
            return Err(PyRuntimeError::new_err(format!(
                "ctrl has {} entries but the robot has {} actuators",
                ctrl.len(),
                handles.actuators.len()
            )));
        }
        let handles = handles.clone();
        self.0
            .control_multibody_motors(viewer.backend(), |_, world| {
                handles.apply_controls_multibody(
                    &mut world.bodies,
                    &mut world.multibody_joints,
                    &ctrl,
                );
            })
            .map_err(gpu_err)
    }

    /// Reads every environment's multibody link states back from the GPU in one
    /// transfer. Returns five float32 numpy arrays with
    /// `num_environments * multibody_links_per_env` rows, environment-major;
    /// links follow the GPU build's traversal order (multibodies, then links,
    /// parent before child), the same order `apply_actuator_controls` drives:
    ///
    /// - `coords (n, 6)`: generalized joint coordinates (only the joint's DOF
    ///   count is meaningful; a revolute joint's angle is `coords[5]`),
    /// - `positions (n, 3)` / `quats (n, 4)`: link world pose (`w, x, y, z`),
    /// - `linvels (n, 3)` / `angvels (n, 3)`: world-space velocities, valid
    ///   after the first simulated step.
    ///
    /// Use `multibody_links_per_env()` to slice a single environment out.
    #[allow(clippy::type_complexity)]
    fn read_multibody_links<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    ) {
        let links = pollster::block_on(self.0.read_multibody_links(viewer.backend()));
        let mut coords = Vec::with_capacity(links.len());
        let mut positions = Vec::with_capacity(links.len());
        let mut quats = Vec::with_capacity(links.len());
        let mut linvels = Vec::with_capacity(links.len());
        let mut angvels = Vec::with_capacity(links.len());
        for ws in &links {
            coords.push(ws.coords.to_vec());
            let (t, q) = (ws.local_to_world.translation, ws.local_to_world.rotation);
            positions.push(vec![t.x, t.y, t.z]);
            quats.push(vec![q.w, q.x, q.y, q.z]);
            let (l, a) = (ws.rb_vels.linear, ws.rb_vels.angular);
            linvels.push(vec![l.x, l.y, l.z]);
            angvels.push(vec![a.x, a.y, a.z]);
        }
        (
            PyArray2::from_vec2(py, &coords).unwrap(),
            PyArray2::from_vec2(py, &positions).unwrap(),
            PyArray2::from_vec2(py, &quats).unwrap(),
            PyArray2::from_vec2(py, &linvels).unwrap(),
            PyArray2::from_vec2(py, &angvels).unwrap(),
        )
    }

    /// Number of link slots per environment, the stride of
    /// `read_multibody_links`.
    fn multibody_links_per_env(&self) -> u32 {
        self.0.multibody_links_per_env()
    }

    // --- robots -------------------------------------------------------------

    /// Loads a URDF robot into environment `env` as one multibody and returns
    /// its [`Robot`] (names, DoF layout, limits, render shapes). Register the
    /// render shapes with `viewer.insert_visual_shape(env, body, shape, pose)`.
    #[pyo3(signature = (env, path, options))]
    fn load_urdf_robot(
        &mut self,
        env: usize,
        path: std::path::PathBuf,
        options: PyRef<UrdfLoaderOptions>,
    ) -> PyResult<Robot> {
        use rapier3d_urdf::{UrdfMultibodyOptions, UrdfRobot};
        let opts = options.to_rapier();
        let (robot, urdf) = UrdfRobot::from_file(&path, opts, None).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to load URDF {}: {e}", path.display()))
        })?;
        if env >= self.0.num_environments() {
            return Err(PyRuntimeError::new_err(format!(
                "environment {env} does not exist"
            )));
        }
        let world = self.0.rbd_world_mut(env);
        let handles = robot.insert_using_multibody_joints(
            &mut world.bodies,
            &mut world.colliders,
            &mut world.multibody_joints,
            UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
        );
        let mut body_names = HashMap::new();
        let mut child_joint_names = HashMap::new();
        for (link, urdf_link) in handles.links.iter().zip(&urdf.links) {
            body_names.insert(link.body, urdf_link.name.clone());
        }
        for (joint, urdf_joint) in handles.joints.iter().zip(&urdf.joints) {
            child_joint_names.insert(joint.link2, urdf_joint.name.clone());
        }
        let mut render_shapes = Vec::new();
        for link in &handles.links {
            for collider in &link.colliders {
                let (shape, local_pose) = match &collider.visual {
                    Some(v) => (v.shape.clone(), v.local_pose),
                    None => (
                        world.colliders[collider.handle].shared_shape().clone(),
                        rp::Pose::IDENTITY,
                    ),
                };
                render_shapes.push((
                    RigidBodyHandle(link.body),
                    SharedShape(shape),
                    Pose(local_pose),
                ));
            }
        }
        let root = handles
            .links
            .first()
            .map(|l| l.body)
            .ok_or_else(|| PyRuntimeError::new_err("URDF has no links"))?;
        build_robot(
            world,
            env,
            root,
            &body_names,
            &child_joint_names,
            render_shapes,
        )
    }

    /// Loads an MJCF robot into environment `env` as one multibody, registers
    /// its visual meshes with `viewer` (environment 0 only) and returns its
    /// [`Robot`]. Unlike `insert_mjcf` this adds no floor and moves no camera.
    /// Joints driven by MJCF position actuators start with those gains as
    /// their PD defaults.
    #[pyo3(signature = (viewer, env, path, register_visuals=true))]
    fn load_mjcf_robot(
        &mut self,
        mut viewer: PyRefMut<NexusViewer>,
        env: usize,
        path: std::path::PathBuf,
        register_visuals: bool,
    ) -> PyResult<Robot> {
        use rapier3d_mjcf::{MjcfLoaderOptions, MjcfMultibodyOptions, MjcfRobot};
        let options = MjcfLoaderOptions {
            skip_plane_geoms: true,
            make_roots_fixed: false,
            create_colliders_from_visual_shapes: false,
            collider_blueprint: rp::ColliderBuilder::default().density(0.0),
            ..Default::default()
        };
        let (robot, _model) = MjcfRobot::from_file(&path, options).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to load MJCF {}: {e}", path.display()))
        })?;
        if env >= self.0.num_environments() {
            return Err(PyRuntimeError::new_err(format!(
                "environment {env} does not exist"
            )));
        }
        let world = self.0.rbd_world_mut(env);
        let handles = robot.clone().insert_using_multibody_joints(
            &mut world.bodies,
            &mut world.colliders,
            &mut world.multibody_joints,
            &mut world.impulse_joints,
            MjcfMultibodyOptions::DISABLE_SELF_CONTACTS,
        );
        // Position servos hold their neutral target from the start.
        let ctrl = vec![0.0; handles.actuators.len()];
        handles.apply_controls_multibody(&mut world.bodies, &mut world.multibody_joints, &ctrl);

        let mut body_names = HashMap::new();
        let mut child_joint_names = HashMap::new();
        let mut root = None;
        for (i, bh) in handles.bodies.iter().enumerate() {
            let Some(bh) = bh else { continue };
            root.get_or_insert(bh.body);
            let name = robot.bodies[i]
                .name
                .clone()
                .unwrap_or_else(|| format!("body_{i}"));
            body_names.insert(bh.body, name);
        }
        for (jh, mj) in handles.joints.iter().zip(&robot.joints) {
            if let Some(name) = &mj.name {
                child_joint_names.insert(jh.link2, name.clone());
            }
        }
        let root = root.ok_or_else(|| PyRuntimeError::new_err("MJCF has no bodies"))?;

        if register_visuals && env == 0 {
            let v = viewer.rust_mut();
            for (i, bh) in handles.bodies.iter().enumerate() {
                let Some(bh) = bh else { continue };
                let mjcf_body = &robot.bodies[i];
                if mjcf_body.visual_meshes.is_empty() {
                    for collider in &bh.colliders {
                        let c = &world.colliders[collider.handle];
                        let local_pose = c
                            .position_wrt_parent()
                            .copied()
                            .unwrap_or(rp::Pose::IDENTITY);
                        v.insert_visual_shape(0, bh.body, c.shared_shape(), local_pose);
                    }
                    continue;
                }
                for vm in &mjcf_body.visual_meshes {
                    let textured = vm.texture.is_some();
                    let color = vm.rgba.unwrap_or(if textured {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [0.7, 0.7, 0.75, 1.0]
                    });
                    let material = vm
                        .material
                        .as_ref()
                        .map(|m| nexus_viewer3d::RenderMaterial {
                            metallic: m.metallic,
                            roughness: m.roughness,
                            reflectance: m.reflectance,
                            emissive: m.emissive,
                        });
                    v.insert_visual_mesh(
                        0,
                        bh.body,
                        &vm.shape,
                        vm.local_pose,
                        color,
                        vm.uvs.as_deref(),
                        vm.normals.as_deref(),
                        vm.texture.as_deref(),
                        material,
                    );
                }
            }
        }
        build_robot(
            world,
            env,
            root,
            &body_names,
            &child_joint_names,
            Vec::new(),
        )
    }

    /// Sets the per-DoF reflected rotor inertia (`armature`, added to the
    /// mass-matrix diagonal) and viscous joint damping of `robot`, one value
    /// per robot DoF each (`None` leaves that quantity unchanged). Both are
    /// read at the next GPU build, so call before `finalize`. MJCF robots
    /// carry theirs from the model; URDF robots start at zero, which leaves a
    /// light arm without the stabilizing inertia its servos assume.
    #[pyo3(signature = (robot, armature=None, damping=None))]
    fn set_robot_joint_dynamics(
        &mut self,
        robot: PyRef<Robot>,
        armature: Option<Vec<f32>>,
        damping: Option<Vec<f32>>,
    ) -> PyResult<()> {
        let robot = robot.clone();
        let n = robot.dof_axes.len();
        for (name, values) in [("armature", &armature), ("damping", &damping)] {
            if let Some(v) = values
                && v.len() != n
            {
                return Err(PyRuntimeError::new_err(format!(
                    "{name} has {} entries for {n} dofs",
                    v.len()
                )));
            }
        }
        let world = self.0.rbd_world_mut(robot.env);
        let rp::PhysicsWorld {
            bodies,
            multibody_joints,
            ..
        } = world;
        let link_id = *multibody_joints
            .rigid_body_link(robot.root)
            .ok_or_else(|| PyRuntimeError::new_err("robot is not a multibody"))?;
        let mb = multibody_joints
            .get_multibody_mut(link_id.multibody)
            .ok_or_else(|| PyRuntimeError::new_err("robot multibody not found"))?;
        let map = assembly_dof_map(mb, bodies);
        if let Some(values) = armature {
            let full = to_assembly(&values, &map);
            let target = mb.armature_mut();
            for (i, v) in full.iter().enumerate() {
                if i < target.len() {
                    target[i] = *v;
                }
            }
        }
        if let Some(values) = damping {
            let full = to_assembly(&values, &map);
            let target = mb.damping_mut();
            for (i, v) in full.iter().enumerate() {
                if i < target.len() {
                    target[i] = *v;
                }
            }
        }
        Ok(())
    }

    /// Current per-DoF `(armature, damping)` of `robot` on the CPU model.
    fn robot_joint_dynamics(&self, robot: PyRef<Robot>) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let world = self.0.rbd_world(robot.env);
        let link_id = world
            .multibody_joints
            .rigid_body_link(robot.root)
            .ok_or_else(|| PyRuntimeError::new_err("robot is not a multibody"))?;
        let mb = world
            .multibody_joints
            .get_multibody(link_id.multibody)
            .ok_or_else(|| PyRuntimeError::new_err("robot multibody not found"))?;
        let map = assembly_dof_map(mb, &world.bodies);
        let pick = |v: &[f32]| -> Vec<f32> {
            map.iter()
                .enumerate()
                .filter_map(|(i, d)| d.map(|_| v.get(i).copied().unwrap_or(0.0)))
                .collect()
        };
        Ok((
            pick(mb.armature().as_slice()),
            pick(mb.damping().as_slice()),
        ))
    }

    /// Generalized coordinates of `robot` as the CPU multibody last saw them
    /// (the authored pose, or the last `set_robot_qpos`). For the simulated
    /// values use `robot_qpos`.
    fn robot_cpu_qpos(&self, robot: PyRef<Robot>) -> Vec<f32> {
        self.0
            .multibody_joint_positions(robot.env, robot.root)
            .unwrap_or_default()
    }

    /// Per-link GPU slots of `robot` (indices into `read_multibody_links` rows,
    /// before the `env * multibody_links_per_env()` offset), in link order.
    fn robot_link_slots(&self, robot: PyRef<Robot>) -> Vec<u32> {
        self.link_slots(&robot)
    }

    /// Reads `robot`'s simulated state back from the GPU: generalized
    /// coordinates `qpos (n_dofs,)`, link positions `(n_links, 3)`, link
    /// quaternions `(n_links, 4)` as `(w, x, y, z)`, and link linear and
    /// angular velocities `(n_links, 3)`.
    #[allow(clippy::type_complexity)]
    fn robot_state<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
        robot: PyRef<Robot>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let links = pollster::block_on(self.0.read_multibody_links(viewer.backend()));
        let per_env = self.0.multibody_links_per_env() as usize;
        let slots = self.link_slots(&robot);
        if slots.len() != robot.link_names.len() {
            return Err(PyRuntimeError::new_err(
                "robot state unavailable: call finalize() first",
            ));
        }
        let mut qpos = Vec::with_capacity(robot.dof_axes.len());
        let mut pos = Vec::with_capacity(slots.len());
        let mut quat = Vec::with_capacity(slots.len());
        let mut linvel = Vec::with_capacity(slots.len());
        let mut angvel = Vec::with_capacity(slots.len());
        for (link_idx, slot) in slots.iter().enumerate() {
            let ws = links
                .get(robot.env * per_env + *slot as usize)
                .ok_or_else(|| PyRuntimeError::new_err("multibody readback too short"))?;
            for (d, axis) in robot.dof_axes.iter().enumerate() {
                if robot.dof_links[d] == link_idx {
                    qpos.push(ws.coords[*axis as usize]);
                }
            }
            let t = ws.local_to_world.translation;
            pos.push(vec![t.x, t.y, t.z]);
            quat.push(to_wxyz(ws.local_to_world.rotation).to_vec());
            let (l, a) = (ws.rb_vels.linear, ws.rb_vels.angular);
            linvel.push(vec![l.x, l.y, l.z]);
            angvel.push(vec![a.x, a.y, a.z]);
        }
        Ok((
            PyArray1::from_vec(py, qpos),
            PyArray2::from_vec2(py, &pos).unwrap(),
            PyArray2::from_vec2(py, &quat).unwrap(),
            PyArray2::from_vec2(py, &linvel).unwrap(),
            PyArray2::from_vec2(py, &angvel).unwrap(),
        ))
    }

    /// Sets `robot`'s generalized coordinates and zeroes its joint velocities
    /// on the GPU (between steps, after `finalize`).
    fn set_robot_qpos(
        &mut self,
        viewer: PyRef<NexusViewer>,
        robot: PyRef<Robot>,
        qpos: Vec<f32>,
    ) -> PyResult<()> {
        self.0
            .set_multibody_joint_positions(viewer.backend(), robot.env, robot.root, &qpos)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Drives `robot`'s DoFs toward `targets` with the robot's per-DoF PD
    /// gains (`robot.kp`, `robot.kv`, `robot.max_force`; force-based motors).
    /// `dofs` selects a subset (default: all, in which case `targets` has one
    /// entry per DoF). Motors of unselected DoFs keep their current target.
    #[pyo3(signature = (viewer, robot, targets, dofs=None))]
    fn set_robot_targets(
        &mut self,
        viewer: PyRef<NexusViewer>,
        robot: PyRef<Robot>,
        targets: Vec<f32>,
        dofs: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let dofs = dofs.unwrap_or_else(|| (0..robot.dof_axes.len()).collect());
        if dofs.len() != targets.len() {
            return Err(PyRuntimeError::new_err(format!(
                "{} targets for {} dofs",
                targets.len(),
                dofs.len()
            )));
        }
        let robot = robot.clone();
        self.0
            .control_multibody_motors_env(viewer.backend(), robot.env, |world| {
                let Some(link_id) = world.multibody_joints.rigid_body_link(robot.root).copied()
                else {
                    return;
                };
                let Some(mb) = world.multibody_joints.get_multibody_mut(link_id.multibody) else {
                    return;
                };
                for (d, target) in dofs.iter().zip(&targets) {
                    let (Some(link_idx), Some(axis)) =
                        (robot.dof_links.get(*d), robot.dof_axes.get(*d))
                    else {
                        continue;
                    };
                    let Some(link) = mb.link_mut(*link_idx) else {
                        continue;
                    };
                    let axis = joint_axis(*axis);
                    link.joint
                        .data
                        .set_motor_model(axis, rp::MotorModel::ForceBased);
                    link.joint
                        .data
                        .set_motor_position(axis, *target, robot.kp[*d], robot.kv[*d]);
                    link.joint
                        .data
                        .set_motor_max_force(axis, robot.max_force[*d]);
                }
            })
            .map_err(gpu_err)
    }

    /// Pose of link `link` of `robot` at coordinates `qpos`, from the CPU
    /// kinematic model (no GPU access): `(position, (w, x, y, z))`. Leaves the
    /// CPU multibody at `qpos`.
    fn robot_forward_kinematics(
        &mut self,
        robot: PyRef<Robot>,
        qpos: Vec<f32>,
        link: &str,
    ) -> PyResult<([f32; 3], [f32; 4])> {
        let link_idx = robot.link_index(link)?;
        let robot = robot.clone();
        let world = self.0.rbd_world_mut_untracked(robot.env);
        let mb = cpu_multibody_at(world, &robot, &qpos)?;
        let pose = mb
            .link(link_idx)
            .map(|l| *l.local_to_world())
            .ok_or_else(|| PyRuntimeError::new_err("link index out of range"))?;
        let t = pose.translation;
        Ok(([t.x, t.y, t.z], to_wxyz(pose.rotation)))
    }

    /// Damped-least-squares inverse kinematics on the CPU model, from
    /// `init_qpos`, for link `link` to reach `target_pos` (of `local_point` in
    /// the link frame) with orientation `target_quat` (`w, x, y, z`).
    /// `constrained_axes` are `[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]`
    /// world-frame error components the solver drives to zero; `dofs`
    /// restricts which DoFs may move (default: all). Coordinates are clamped
    /// to the joint limits after every iteration. Returns `(qpos, error)`
    /// where `error` is `[lin(3), ang(3)]` with unconstrained components zeroed.
    #[pyo3(signature = (robot, link, target_pos, target_quat, init_qpos, local_point=None, constrained_axes=None, dofs=None, max_iters=100, damping=0.05, pos_tol=1.0e-4, rot_tol=1.0e-3))]
    #[allow(clippy::too_many_arguments)]
    fn robot_inverse_kinematics(
        &mut self,
        robot: PyRef<Robot>,
        link: &str,
        target_pos: [f32; 3],
        target_quat: [f32; 4],
        init_qpos: Vec<f32>,
        local_point: Option<[f32; 3]>,
        constrained_axes: Option<[bool; 6]>,
        dofs: Option<Vec<usize>>,
        max_iters: usize,
        damping: f32,
        pos_tol: f32,
        rot_tol: f32,
    ) -> PyResult<(Vec<f32>, [f32; 6])> {
        use rapier3d::dynamics::InverseKinematicsOption;
        let link_idx = robot.link_index(link)?;
        let robot = robot.clone();
        let constrained = constrained_axes.unwrap_or([true; 6]);
        let mut mask = rp::JointAxesMask::empty();
        for (axis, on) in constrained.iter().enumerate() {
            if *on {
                mask |= rp::JointAxesMask::from_bits_truncate(1 << axis);
            }
        }
        let target = pose_from_wxyz(target_pos, target_quat);
        let target = match local_point {
            // Aim the link origin so that `local_point` lands on `target_pos`.
            Some(p) => rp::Pose::from_parts(
                target.translation - target.rotation * glamx::Vec3::from(p),
                target.rotation,
            ),
            None => target,
        };
        let movable: Vec<bool> = {
            let mut m = vec![dofs.is_none(); robot.link_names.len()];
            if let Some(dofs) = &dofs {
                for d in dofs {
                    if let Some(l) = robot.dof_links.get(*d) {
                        m[*l] = true;
                    }
                }
            }
            m
        };
        let options = InverseKinematicsOption {
            damping,
            max_iters: 1,
            constrained_axes: mask,
            epsilon_linear: pos_tol,
            epsilon_angular: rot_tol,
        };

        let world = self.0.rbd_world_mut_untracked(robot.env);
        let root = robot.root;
        cpu_multibody_at(world, &robot, &init_qpos)?;
        let rp::PhysicsWorld {
            bodies,
            multibody_joints,
            ..
        } = world;
        let link_id = *multibody_joints
            .rigid_body_link(root)
            .ok_or_else(|| PyRuntimeError::new_err("robot is not a multibody"))?;
        let mut error = [0.0f32; 6];
        let (map, root_fixed) = {
            let mb = multibody_joints
                .get_multibody(link_id.multibody)
                .unwrap_or_else(|| unreachable!());
            (
                assembly_dof_map(mb, bodies),
                RNexusState::multibody_root_is_fixed(bodies, mb),
            )
        };
        for _ in 0..max_iters {
            let mb = multibody_joints
                .get_multibody_mut(link_id.multibody)
                .unwrap_or_else(|| unreachable!());
            let mut disp = rapier3d::na::DVector::zeros(mb.ndofs());
            mb.inverse_kinematics(
                bodies,
                link_idx,
                &options,
                &target,
                |l| {
                    if l.link_id() == 0 && root_fixed {
                        return false;
                    }
                    movable.get(l.link_id()).copied().unwrap_or(true)
                },
                &mut disp,
            );
            // Clamp to the joint limits: rapier's solver is unaware of them.
            let current = to_assembly(&cpu_qpos(mb, bodies), &map);
            let mut disp_vec: Vec<f32> = disp.as_slice().to_vec();
            for (a, (q, delta)) in current.iter().zip(disp_vec.iter_mut()).enumerate() {
                match map[a] {
                    Some(d) => {
                        let clamped = (q + *delta).clamp(robot.dof_lower[d], robot.dof_upper[d]);
                        *delta = clamped - q;
                    }
                    None => *delta = 0.0,
                }
            }
            mb.apply_displacements(&disp_vec);
            mb.forward_kinematics(bodies, false);
            let pose = *mb
                .link(link_idx)
                .unwrap_or_else(|| unreachable!())
                .local_to_world();
            let lin = target.translation - pose.translation;
            let ang = (target.rotation * pose.rotation.inverse()).to_scaled_axis();
            let raw = [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z];
            for (i, e) in raw.iter().enumerate() {
                error[i] = if constrained[i] { *e } else { 0.0 };
            }
            let lin_err = (error[0] * error[0] + error[1] * error[1] + error[2] * error[2]).sqrt();
            let ang_err = (error[3] * error[3] + error[4] * error[4] + error[5] * error[5]).sqrt();
            if lin_err <= pos_tol && ang_err <= rot_tol {
                break;
            }
        }
        let mb = multibody_joints
            .get_multibody(link_id.multibody)
            .unwrap_or_else(|| unreachable!());
        Ok((cpu_qpos(mb, bodies), error))
    }

    // --- rigid-body state ---------------------------------------------------

    /// GPU pose slot of `handle` in `env`: the row of `read_body_poses` /
    /// `read_body_velocities`. `None` before `finalize`.
    fn body_gpu_index(&self, env: usize, handle: RigidBodyHandle) -> Option<u32> {
        self.0.rigid_body_gpu_index(env, handle.0)
    }

    /// Reads every body's world-origin pose from the GPU: positions `(n, 3)`
    /// and quaternions `(n, 4)` as `(w, x, y, z)`, rows indexed by
    /// `body_gpu_index`.
    fn read_body_poses<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let poses = pollster::block_on(self.0.read_rigid_body_poses(viewer.backend()));
        let mut pos = Vec::with_capacity(poses.len());
        let mut quat = Vec::with_capacity(poses.len());
        for p in &poses {
            let t = p.translation;
            pos.push(vec![t.x, t.y, t.z]);
            quat.push(to_wxyz(p.rotation).to_vec());
        }
        (
            PyArray2::from_vec2(py, &pos).unwrap(),
            PyArray2::from_vec2(py, &quat).unwrap(),
        )
    }

    /// Reads every body's world-space linear `(n, 3)` and angular `(n, 3)`
    /// velocity from the GPU, rows indexed by `body_gpu_index`.
    fn read_body_velocities<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let vels = pollster::block_on(self.0.read_rigid_body_velocities(viewer.backend()));
        let mut lin = Vec::with_capacity(vels.len());
        let mut ang = Vec::with_capacity(vels.len());
        for v in &vels {
            lin.push(vec![v.linear.x, v.linear.y, v.linear.z]);
            ang.push(vec![v.angular.x, v.angular.y, v.angular.z]);
        }
        (
            PyArray2::from_vec2(py, &lin).unwrap(),
            PyArray2::from_vec2(py, &ang).unwrap(),
        )
    }

    /// Teleports free body `handle` of `env` to `pos` / `quat` (`w, x, y, z`)
    /// between steps. Multibody links go through `set_robot_qpos`.
    fn set_body_pose(
        &mut self,
        viewer: PyRef<NexusViewer>,
        env: usize,
        handle: RigidBodyHandle,
        pos: [f32; 3],
        quat: [f32; 4],
    ) -> PyResult<()> {
        self.0
            .set_rigid_body_pose(viewer.backend(), env, handle.0, pose_from_wxyz(pos, quat))
            .map_err(gpu_err)
    }

    /// Sets body `handle`'s world-space linear and angular velocity.
    fn set_body_velocity(
        &mut self,
        viewer: PyRef<NexusViewer>,
        env: usize,
        handle: RigidBodyHandle,
        linvel: [f32; 3],
        angvel: [f32; 3],
    ) -> PyResult<()> {
        self.0
            .set_rigid_body_velocity(
                viewer.backend(),
                env,
                handle.0,
                glamx::Vec3::from(linvel),
                glamx::Vec3::from(angvel),
            )
            .map_err(gpu_err)
    }

    /// Timing and solver statistics of the last `simulate` call as a dict:
    /// `encoding_time_ms` (CPU command encoding), `gpu_total_time_ms` and
    /// `gpu_pass_times` (`{pass label: ms}`) from the GPU timestamp queries
    /// (only populated when a `GpuTimestamps` is passed to `simulate` and
    /// harvested by `viewer.sync`, which lags a frame or two), plus the
    /// constraint-coloring `num_colors` / `coloring_iterations`.
    fn run_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        let stats = &self.0.run_stats;
        let dict = PyDict::new(py);
        dict.set_item("encoding_time_ms", stats.encoding_time_ms())?;
        dict.set_item("gpu_total_time_ms", stats.gpu_total_time_ms)?;
        let passes = PyDict::new(py);
        for (label, ms) in &stats.gpu_pass_times {
            passes.set_item(label, *ms)?;
        }
        dict.set_item("gpu_pass_times", passes)?;
        dict.set_item("num_colors", stats.num_colors)?;
        dict.set_item("coloring_iterations", stats.coloring_iterations)?;
        Ok(dict)
    }

    /// Rigid-body timestep of every environment: `dt` seconds per `simulate`
    /// step, in `substeps` solver substeps. Call before `finalize`.
    fn set_rbd_timestep(&mut self, dt: f32, substeps: u32) {
        self.0.set_rbd_timestep(dt, substeps);
    }

    /// Contact-solver parameters of every environment, applied at the next GPU
    /// build (call before `finalize`). `None` leaves a value unchanged.
    /// `contact_natural_frequency` / `contact_damping_ratio` shape the soft
    /// contact model (higher frequency = stiffer contacts), the `static_*`
    /// pair applies to contacts at rest, `allowed_linear_error` is the
    /// tolerated penetration in length units, `max_corrective_velocity` caps
    /// penetration recovery, `prediction_distance` is the contact detection
    /// margin, `internal_pgs_iterations` the biased-pass PGS iterations per
    /// substep (rigid-body and multibody sweeps alike), and `friction_in_bias_pass` whether friction
    /// rows are solved in every biased PGS iteration instead of only in the
    /// per-substep stabilization sweep (rapier's default, `False`); `True`
    /// gives friction as many iterations as the normal rows, which holds
    /// grasps and resting contacts far more firmly.
    #[pyo3(signature = (contact_natural_frequency=None, contact_damping_ratio=None, static_contact_natural_frequency=None, static_contact_damping_ratio=None, allowed_linear_error=None, max_corrective_velocity=None, prediction_distance=None, internal_pgs_iterations=None, friction_in_bias_pass=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_rbd_solver_params(
        &mut self,
        contact_natural_frequency: Option<f32>,
        contact_damping_ratio: Option<f32>,
        static_contact_natural_frequency: Option<f32>,
        static_contact_damping_ratio: Option<f32>,
        allowed_linear_error: Option<f32>,
        max_corrective_velocity: Option<f32>,
        prediction_distance: Option<f32>,
        internal_pgs_iterations: Option<u32>,
        friction_in_bias_pass: Option<bool>,
    ) {
        for env in 0..self.0.num_environments() {
            let Some(mut params) = self.0.rbd_sim_params(env) else {
                continue;
            };
            if let Some(v) = contact_natural_frequency {
                params.contact_natural_frequency = v;
            }
            if let Some(v) = contact_damping_ratio {
                params.contact_damping_ratio = v;
            }
            if let Some(v) = static_contact_natural_frequency {
                params.static_contact_natural_frequency = v;
            }
            if let Some(v) = static_contact_damping_ratio {
                params.static_contact_damping_ratio = v;
            }
            if let Some(v) = allowed_linear_error {
                params.normalized_allowed_linear_error = v;
            }
            if let Some(v) = max_corrective_velocity {
                params.normalized_max_corrective_velocity = v;
            }
            if let Some(v) = prediction_distance {
                params.normalized_prediction_distance = v;
            }
            if let Some(v) = internal_pgs_iterations {
                params.num_internal_pgs_iterations = v.max(1);
            }
            if let Some(v) = friction_in_bias_pass {
                params.friction_in_bias_pass = v as u32;
            }
            self.0.set_rbd_sim_params(env, params);
        }
    }

    /// The contact-solver parameters of environment 0 as a dict (see
    /// `set_rbd_solver_params`), for attestation.
    fn rbd_solver_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        if let Some(p) = self.0.rbd_sim_params(0) {
            dict.set_item("dt", p.dt)?;
            dict.set_item("substeps", p.num_solver_iterations)?;
            dict.set_item("contact_natural_frequency", p.contact_natural_frequency)?;
            dict.set_item("contact_damping_ratio", p.contact_damping_ratio)?;
            dict.set_item(
                "static_contact_natural_frequency",
                p.static_contact_natural_frequency,
            )?;
            dict.set_item(
                "static_contact_damping_ratio",
                p.static_contact_damping_ratio,
            )?;
            dict.set_item("allowed_linear_error", p.normalized_allowed_linear_error)?;
            dict.set_item(
                "max_corrective_velocity",
                p.normalized_max_corrective_velocity,
            )?;
            dict.set_item("prediction_distance", p.normalized_prediction_distance)?;
            dict.set_item("internal_pgs_iterations", p.num_internal_pgs_iterations)?;
            dict.set_item("friction_in_bias_pass", p.friction_in_bias_pass != 0)?;
        }
        Ok(dict)
    }

    /// Debug readback of the live contact manifolds, one dict per active
    /// manifold: `collider_a` / `collider_b` and `body_a` / `body_b` (GPU
    /// indices), the combined `friction`, `normal_a` and the `points` as
    /// `[x, y, z, dist]` rows, both in collider A's local frame. Blocks on
    /// the GPU; for diagnostics, not for control loops.
    fn debug_contacts<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        use pyo3::types::PyDict;
        let Some(rbd) = self.0.rbd.as_ref() else {
            return Ok(Vec::new());
        };
        let contacts: Vec<nexus3d::rbd::queries::GpuIndexedContact> =
            pollster::block_on(viewer.backend().slow_read_vec(rbd.contacts().buffer()))
                .map_err(gpu_err)?;
        let mut out = Vec::new();
        for c in contacts.iter().filter(|c| c.contact.len > 0) {
            let dict = PyDict::new(py);
            dict.set_item("collider_a", c.colliders.x)?;
            dict.set_item("collider_b", c.colliders.y)?;
            dict.set_item("body_a", c.bodies.x)?;
            dict.set_item("body_b", c.bodies.y)?;
            dict.set_item("friction", c.friction)?;
            let n = c.contact.normal_a;
            dict.set_item("normal_a", [n.x, n.y, n.z])?;
            let points: Vec<[f32; 4]> = (0..c.contact.len as usize)
                .map(|k| {
                    let p = c.contact.points_a[k];
                    [p.pt.x, p.pt.y, p.pt.z, p.dist]
                })
                .collect();
            dict.set_item("points", points)?;
            out.push(dict);
        }
        Ok(out)
    }

    /// Debug readback of the multibody contact constraints as left by the
    /// last step, one dict per active slot: `multibody` and `link` (batch
    /// local), `free_body` (GPU index, `None` for a self-contact), `kind`
    /// (`"normal"` or `"tangent"`), `friction`, the accumulated per-substep
    /// `impulse` and the free-body jacobian direction `dir`. Blocks on the
    /// GPU; for diagnostics only.
    fn debug_multibody_contact_impulses<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        use nexus3d::rbd::shaders::dynamics::{
            MB_CONTACT_KIND_NORMAL, MB_CONTACT_KIND_TANGENT, MultibodyContactConstraint,
            MultibodyInfo,
        };
        use pyo3::types::PyDict;
        let Some(rbd) = self.0.rbd.as_ref() else {
            return Ok(Vec::new());
        };
        let backend = viewer.backend();
        let mb = rbd.multibodies();
        let cons: Vec<MultibodyContactConstraint> =
            pollster::block_on(backend.slow_read_vec(mb.contact_constraints().buffer()))
                .map_err(gpu_err)?;
        let infos: Vec<MultibodyInfo> =
            pollster::block_on(backend.slow_read_vec(mb.multibody_info().buffer()))
                .map_err(gpu_err)?;
        let mut out = Vec::new();
        for info in infos.iter().filter(|i| i.ndofs > 0) {
            let start = info.contact_constraint_start as usize;
            let end = start + info.contact_constraint_count as usize;
            for c in cons.iter().take(end.min(cons.len())).skip(start) {
                let kind = match c.kind {
                    MB_CONTACT_KIND_NORMAL => "normal",
                    MB_CONTACT_KIND_TANGENT => "tangent",
                    _ => continue,
                };
                let dict = PyDict::new(py);
                dict.set_item("multibody", c.multibody_id)?;
                dict.set_item("link", c.link_id)?;
                dict.set_item(
                    "free_body",
                    (c.free_body_id != u32::MAX).then_some(c.free_body_id),
                )?;
                dict.set_item("kind", kind)?;
                dict.set_item("friction", c.friction_coeff)?;
                dict.set_item("impulse", c.impulse)?;
                dict.set_item("dir", [c.lin_jac.x, c.lin_jac.y, c.lin_jac.z])?;
                dict.set_item("free_body_inv_mass", c.free_body_im)?;
                out.push(dict);
            }
        }
        Ok(out)
    }

    /// Debug readback of the rigid-body (two-body) contact constraints as
    /// left by the last step, one dict per active manifold: `body_a` /
    /// `body_b` (GPU indices), `dir_a` (world normal force direction on
    /// body A), the combined `friction`, and per point the accumulated
    /// per-substep `normal_impulse` and `tangent_impulse` pair. Blocks on
    /// the GPU; for diagnostics only.
    fn debug_rigid_contact_impulses<'py>(
        &self,
        py: Python<'py>,
        viewer: PyRef<NexusViewer>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        use nexus3d::rbd::shaders::dynamics::TwoBodyConstraint;
        use pyo3::types::PyDict;
        let Some(rbd) = self.0.rbd.as_ref() else {
            return Ok(Vec::new());
        };
        let cons: Vec<TwoBodyConstraint> = pollster::block_on(
            viewer
                .backend()
                .slow_read_vec(rbd.rigid_contact_constraints().buffer()),
        )
        .map_err(gpu_err)?;
        let mut out = Vec::new();
        for c in cons.iter().filter(|c| c.len > 0) {
            let dict = PyDict::new(py);
            dict.set_item("body_a", c.solver_body_a)?;
            dict.set_item("body_b", c.solver_body_b)?;
            dict.set_item("dir_a", [c.dir_a.x, c.dir_a.y, c.dir_a.z])?;
            dict.set_item("friction", c.limit)?;
            dict.set_item("inv_mass_a", c.im_a.x)?;
            dict.set_item("inv_mass_b", c.im_b.x)?;
            let normal: Vec<f32> = (0..c.len as usize)
                .map(|k| c.elements[k].normal_part.impulse)
                .collect();
            let tangent: Vec<[f32; 2]> = (0..c.len as usize)
                .map(|k| {
                    let t = c.elements[k].tangent_part.impulse;
                    [t.x, t.y]
                })
                .collect();
            dict.set_item("normal_impulse", normal)?;
            dict.set_item("tangent_impulse", tangent)?;
            out.push(dict);
        }
        Ok(out)
    }

    // --- rbd config -------------------------------------------------------

    fn set_rbd_steps_per_frame(&mut self, steps: u32) {
        self.0.set_rbd_steps_per_frame(steps);
    }

    fn set_rbd_gravity(&mut self, viewer: PyRef<NexusViewer>, gravity: Vec3) {
        self.0
            .set_rbd_gravity(viewer.backend(), [gravity.0.x, gravity.0.y, gravity.0.z]);
    }

    fn set_multibody_motor_velocity(
        &mut self,
        viewer: PyRef<NexusViewer>,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) -> PyResult<()> {
        self.0
            .set_multibody_motor_velocity(
                viewer.backend(),
                batch,
                link_id,
                axis.to_rapier(),
                target_vel,
            )
            .map_err(gpu_err)
    }

    // --- mpm --------------------------------------------------------------

    fn set_mpm_params(
        &mut self,
        viewer: PyRef<NexusViewer>,
        params: PyRef<SimulationParams>,
        cell_width: f32,
    ) -> PyResult<()> {
        self.0
            .set_mpm_params(viewer.backend(), params.0, cell_width)
            .map_err(gpu_err)
    }

    fn set_mpm_substeps(&mut self, substeps: u32) {
        self.0.set_mpm_substeps(substeps);
    }

    fn set_mpm_use_cpic(&mut self, enabled: bool) {
        self.0.set_mpm_use_cpic(enabled);
    }

    fn set_mpm_gravity(&mut self, gravity: Vec3) {
        self.0.set_mpm_gravity(gravity.0);
    }

    fn add_particles(
        &mut self,
        viewer: PyRef<NexusViewer>,
        particles: Vec<Particle>,
    ) -> PyResult<NexusParticleChunk> {
        let particles: Vec<_> = particles.into_iter().map(|p| p.0).collect();
        self.0
            .add_particles(viewer.backend(), particles)
            .map(NexusParticleChunk)
            .map_err(gpu_err)
    }

    fn extend_chunk(
        &mut self,
        viewer: PyRef<NexusViewer>,
        chunk: NexusParticleChunk,
        particles: Vec<Particle>,
    ) -> PyResult<()> {
        let particles: Vec<_> = particles.into_iter().map(|p| p.0).collect();
        self.0
            .extend_chunk(viewer.backend(), chunk.0, particles)
            .map_err(gpu_err)
    }

    fn remove_chunk(
        &mut self,
        viewer: PyRef<NexusViewer>,
        chunk: NexusParticleChunk,
    ) -> PyResult<()> {
        self.0
            .remove_chunk(viewer.backend(), chunk.0)
            .map_err(gpu_err)
    }

    // --- lifecycle --------------------------------------------------------

    fn counts(&self) -> NexusCounts {
        let c = self.0.counts();
        NexusCounts {
            num_environments: c.num_environments,
            rigid_bodies: c.rigid_bodies,
            colliders: c.colliders,
            impulse_joints: c.impulse_joints,
            multibodies: c.multibodies,
            multibody_dofs: c.multibody_dofs,
            particles: c.particles,
        }
    }

    /// Uploads the scene to the GPU. Must be called before the first
    /// `simulate`. Blocks on the underlying async GPU work.
    fn finalize(&mut self, viewer: PyRef<NexusViewer>) -> PyResult<()> {
        pollster::block_on(self.0.finalize(viewer.backend())).map_err(gpu_err)
    }
}

/// The GPU compute pipelines (`nexus3d::prelude::NexusPipeline`).
#[pyclass(name = "NexusPipeline", unsendable)]
pub struct NexusPipeline(pub RNexusPipeline);

#[pymethods]
impl NexusPipeline {
    #[new]
    fn new() -> Self {
        NexusPipeline(RNexusPipeline::default())
    }

    /// Compiles all GPU pipelines up-front (RBD + MPM).
    fn preload_pipelines(&mut self, viewer: PyRef<NexusViewer>) -> PyResult<()> {
        self.0
            .preload_pipelines(viewer.backend(), NexusPipelineMask::all())
            .map_err(gpu_err)
    }

    /// Advances the simulation by one frame. Blocks on the async GPU work.
    #[pyo3(signature = (viewer, state, timestamps=None))]
    fn simulate(
        &mut self,
        viewer: PyRef<NexusViewer>,
        mut state: PyRefMut<NexusState>,
        mut timestamps: Option<PyRefMut<GpuTimestamps>>,
    ) -> PyResult<()> {
        let backend = viewer.backend();
        let ts = timestamps.as_deref_mut().map(|t| &mut t.0);
        pollster::block_on(self.0.simulate(backend, &mut state.0, ts)).map_err(gpu_err)
    }
}

/// Generalized coordinates of `mb` in assembly order, without a fixed root's
/// locked coordinates (the GPU build's DoF vector, see `Robot`).
fn cpu_qpos(mb: &rapier3d::dynamics::Multibody, bodies: &rp::RigidBodySet) -> Vec<f32> {
    let root_fixed = RNexusState::multibody_root_is_fixed(bodies, mb);
    let mut out = Vec::with_capacity(mb.ndofs());
    for (i, link) in mb.links().enumerate() {
        if i == 0 && root_fixed {
            continue;
        }
        let coords = link.joint().coords();
        for axis in free_axes(&link.joint().data) {
            out.push(coords[axis as usize]);
        }
    }
    out
}

/// For each entry of rapier's full assembly displacement vector, the robot DoF
/// it maps to (`None` for a fixed root's locked coordinates).
fn assembly_dof_map(
    mb: &rapier3d::dynamics::Multibody,
    bodies: &rp::RigidBodySet,
) -> Vec<Option<usize>> {
    let root_fixed = RNexusState::multibody_root_is_fixed(bodies, mb);
    let mut out = Vec::with_capacity(mb.ndofs());
    let mut next = 0usize;
    for (i, link) in mb.links().enumerate() {
        for _ in 0..link.joint().ndofs() {
            if i == 0 && root_fixed {
                out.push(None);
            } else {
                out.push(Some(next));
                next += 1;
            }
        }
    }
    out
}

/// Expands a robot DoF vector into rapier's full assembly vector (zeros for a
/// fixed root's locked coordinates).
fn to_assembly(values: &[f32], map: &[Option<usize>]) -> Vec<f32> {
    map.iter()
        .map(|d| d.map(|i| values[i]).unwrap_or(0.0))
        .collect()
}

/// Moves `robot`'s CPU multibody to `qpos` (forward kinematics included) and
/// returns it. The CPU model is a scratch kinematic model once the GPU owns the
/// simulation, so this never touches the simulated state.
fn cpu_multibody_at<'a>(
    world: &'a mut rp::PhysicsWorld,
    robot: &Robot,
    qpos: &[f32],
) -> PyResult<&'a rapier3d::dynamics::Multibody> {
    let rp::PhysicsWorld {
        bodies,
        multibody_joints,
        ..
    } = world;
    let link_id = *multibody_joints
        .rigid_body_link(robot.root)
        .ok_or_else(|| PyRuntimeError::new_err("robot is not a multibody"))?;
    let mb = multibody_joints
        .get_multibody_mut(link_id.multibody)
        .ok_or_else(|| PyRuntimeError::new_err("robot multibody not found"))?;
    let current = cpu_qpos(mb, bodies);
    if current.len() != qpos.len() {
        return Err(PyRuntimeError::new_err(format!(
            "qpos has {} entries but the robot has {} DoFs",
            qpos.len(),
            current.len()
        )));
    }
    let map = assembly_dof_map(mb, bodies);
    let delta: Vec<f32> = qpos.iter().zip(&current).map(|(q, c)| q - c).collect();
    mb.apply_displacements(&to_assembly(&delta, &map));
    mb.forward_kinematics(bodies, false);
    Ok(&*mb)
}

impl NexusState {
    /// Per-link GPU slots of `robot`, in link order (empty before `finalize`).
    fn link_slots(&self, robot: &Robot) -> Vec<u32> {
        self.0
            .multibody_link_slots(robot.env, robot.root)
            .map(|(_, _, links)| links.into_iter().map(|(_, slot)| slot).collect())
            .unwrap_or_default()
    }
}
