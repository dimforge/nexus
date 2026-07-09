//! Core simulation objects: `NexusState`, `NexusPipeline`, `GpuTimestamps`,
//! and the various entity handles.

use crate::loaders::{MjcfSceneInfo, UrdfLoaderOptions, UrdfRobotHandles};
use crate::math::{Pose, Vec3};
use crate::rbd::{
    Collider, ImpulseJointHandle, JointArg, JointAxis, MultibodyJointHandle, RigidBody,
    RigidBodyHandle, SharedShape,
};
use crate::viewer::NexusViewer;
use khal::backend::GpuTimestamps as RGpuTimestamps;
use nexus3d::prelude::{
    NexusPipeline as RNexusPipeline, NexusPipelineMask, NexusState as RNexusState,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapier3d::prelude as rp;

/// Maps a GPU backend error to a Python exception.
fn gpu_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
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
}

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
/// (`nexus3d::prelude::NexusState`).
#[pyclass(name = "NexusState", unsendable)]
pub struct NexusState(pub RNexusState);

#[pymethods]
impl NexusState {
    #[new]
    fn new() -> Self {
        NexusState(RNexusState::default())
    }

    // --- rigid bodies -----------------------------------------------------

    fn insert_rigid_body(
        &mut self,
        body: PyRef<RigidBody>,
        collider: PyRef<Collider>,
    ) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_rigid_body(body.0.clone(), collider.0.clone()))
    }

    fn insert_rigid_body_in(
        &mut self,
        env: usize,
        body: PyRef<RigidBody>,
        collider: PyRef<Collider>,
    ) -> RigidBodyHandle {
        RigidBodyHandle(
            self.0
                .insert_rigid_body_in(env, body.0.clone(), collider.0.clone()),
        )
    }

    fn insert_body(&mut self, body: PyRef<RigidBody>) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_body(body.0.clone()))
    }

    /// Inserts a collider-less body into environment `env`; attach colliders to
    /// it afterwards with `insert_collider_in` (multiple colliders per body).
    fn insert_body_in(&mut self, env: usize, body: PyRef<RigidBody>) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_body_in(env, body.0.clone()))
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
    ) -> PyResult<Vec<RigidBodyHandle>> {
        if bodies.len() != colliders.len() {
            return Err(PyRuntimeError::new_err(
                "bodies and colliders must have the same length",
            ));
        }
        let pairs = bodies.into_iter().zip(colliders).map(|(b, c)| (b.0, c.0));
        self.0
            .add_rigid_bodies(viewer.backend(), pairs)
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

    /// Loads a MuJoCo MJCF scene into environment 0 as multibodies, registering
    /// its render shapes (and a sized floor) with `viewer`. Returns scene info
    /// (suggested camera + whether the scene is Z-up). Call `finalize` after.
    #[pyo3(signature = (viewer, scene_path, render_colliders=false))]
    fn insert_mjcf(
        &mut self,
        viewer: PyRefMut<NexusViewer>,
        scene_path: std::path::PathBuf,
        render_colliders: bool,
    ) -> PyResult<MjcfSceneInfo> {
        crate::loaders::insert_mjcf(&mut self.0, viewer, &scene_path, render_colliders)
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

    /// Compiles all GPU pipelines up-front.
    fn preload_pipelines(&mut self, viewer: PyRef<NexusViewer>) -> PyResult<()> {
        self.0
            .preload_pipelines(viewer.backend(), NexusPipelineMask::all())
            .map_err(gpu_err)
    }

    /// Captures one frame's rigid-body step sequence
    /// (`rbd_steps_per_frame` solver steps) into a CUDA graph, executing it
    /// once. Subsequent `replay_cuda_graph` calls replay the whole sequence
    /// with a single `cuGraphLaunch` — the fast path for capture/eval loops.
    ///
    /// Returns `False` when the backend is not CUDA. Call after the scene is
    /// finalized and a few warmup `simulate` calls (the graph records raw
    /// buffer addresses; buffer growth after capture invalidates it).
    #[cfg(feature = "cuda")]
    fn capture_cuda_graph(
        &mut self,
        viewer: PyRef<NexusViewer>,
        mut state: PyRefMut<NexusState>,
    ) -> PyResult<bool> {
        pollster::block_on(self.0.capture_rbd_graph(viewer.backend(), &mut state.0))
            .map_err(gpu_err)
    }

    /// Replays the captured rigid-body CUDA graph (see `capture_cuda_graph`).
    /// Returns `False` when no graph has been captured.
    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(&mut self) -> PyResult<bool> {
        self.0.replay_rbd_graph().map_err(gpu_err)
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
