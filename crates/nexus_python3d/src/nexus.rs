//! Core simulation objects: `NexusState`, `NexusPipeline`, `RbdCoupling`,
//! `GpuTimestamps`, and the various entity handles.

use crate::loaders::{MjcfSceneInfo, UrdfLoaderOptions, UrdfRobotHandles};
use crate::math::{Pose, Vec3};
use crate::mpm::{BoundaryCondition, Particle, SimulationParams};
use crate::rbd::{
    Collider, ImpulseJointHandle, JointArg, JointAxis, MultibodyJointHandle, RigidBody,
    RigidBodyHandle, SharedShape,
};
use crate::viewer::NexusViewer;
use khal::backend::GpuTimestamps as RGpuTimestamps;
use nexus3d::mpm::solver::BoundaryCondition as RBoundaryCondition;
use nexus3d::prelude::{
    NexusPipeline as RNexusPipeline, NexusPipelineMask, NexusState as RNexusState,
    RbdCoupling as RRbdCoupling,
};
use numpy::PyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapier3d::prelude as rp;

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
