//! The windowed viewer (`nexus_viewer3d::NexusViewer`).
//!
//! The Rust builder methods (`with_cpu`, `with_running`, …) consume `self`, so
//! the inner viewer is stored in an `Option` and swapped in place; the builder
//! wrappers return the same Python object for fluent chaining.

use crate::math::Pose;
use crate::math::{Vec3, Vec4};
use crate::nexus::{GpuTimestamps, NexusState};
use crate::rbd::{RigidBodyHandle, SharedShape};
use khal::backend::GpuBackend;
use nexus_viewer3d::NexusViewer as RViewer;
use numpy::{IntoPyArray, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// A windowed viewer that renders the simulation and drives the run loop.
#[pyclass(name = "NexusViewer", unsendable)]
pub struct NexusViewer(Option<RViewer>);

impl Drop for NexusViewer {
    fn drop(&mut self) {
        // kiss3d's window / texture-manager `Drop` reads a thread-local that may
        // already be destroyed when the Python interpreter is tearing down,
        // which aborts the process *after* the user's work is done. The viewer
        // lives for the whole process, so leak it on teardown — the OS reclaims
        // the window and GPU resources on exit — giving a clean shutdown.
        if let Some(inner) = self.0.take() {
            std::mem::forget(inner);
        }
    }
}

impl NexusViewer {
    fn inner(&self) -> &RViewer {
        self.0.as_ref().expect("viewer already consumed")
    }
    fn inner_mut(&mut self) -> &mut RViewer {
        self.0.as_mut().expect("viewer already consumed")
    }
    /// Mutable access to the wrapped viewer (used by the robot loaders).
    pub(crate) fn rust_mut(&mut self) -> &mut RViewer {
        self.inner_mut()
    }
    /// Applies a consuming builder method in place.
    fn map_inplace(&mut self, f: impl FnOnce(RViewer) -> RViewer) {
        let v = self.0.take().expect("viewer already consumed");
        self.0 = Some(f(v));
    }

    /// The active GPU backend (used by `NexusState`/`NexusPipeline`).
    pub fn backend(&self) -> &GpuBackend {
        self.inner().backend()
    }
}

#[pymethods]
impl NexusViewer {
    /// Opens a window and probes the GPU. Blocks on the async setup.
    ///
    /// Must be called on the main thread (required by the OS windowing system).
    #[new]
    fn new() -> Self {
        NexusViewer(Some(pollster::block_on(RViewer::new(Vec::new()))))
    }

    // --- backend selection (fluent) --------------------------------------

    fn with_cpu(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.map_inplace(|v| v.with_cpu());
        slf
    }
    fn with_running(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.map_inplace(|v| v.with_running());
        slf
    }
    #[cfg(feature = "metal")]
    fn with_metal(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.map_inplace(|v| v.with_backend(nexus_viewer3d::BackendType::Metal));
        slf
    }
    #[cfg(feature = "cuda")]
    fn with_cuda(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.map_inplace(|v| v.with_backend(nexus_viewer3d::BackendType::Cuda));
        slf
    }

    fn init_backend(&mut self) {
        self.inner_mut().init_backend();
    }

    // --- camera & lighting ------------------------------------------------

    fn set_camera(&mut self, eye: Vec3, target: Vec3) {
        self.inner_mut().set_camera(eye.0, target.0);
    }
    fn set_up_axis(&mut self, up: Vec3) {
        self.inner_mut().set_up_axis(up.0);
    }
    fn add_directional_light(&mut self, direction: Vec3) {
        self.inner_mut()
            .scene3d_mut()
            .add_directional_light(direction.0);
    }

    // --- shape registration ----------------------------------------------

    fn insert_shape(
        &mut self,
        handle: RigidBodyHandle,
        shape: PyRef<SharedShape>,
        local_pose: Pose,
    ) {
        self.inner_mut()
            .insert_shape(handle.0, &shape.0, local_pose.0);
    }
    fn insert_shape_with_color(
        &mut self,
        handle: RigidBodyHandle,
        shape: PyRef<SharedShape>,
        local_pose: Pose,
        color: Vec4,
    ) {
        self.inner_mut()
            .insert_shape_with_color(handle.0, &shape.0, local_pose.0, color.0);
    }
    #[pyo3(signature = (env, handle, shape, local_pose, color=None))]
    fn insert_shape_in(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: PyRef<SharedShape>,
        local_pose: Pose,
        color: Option<Vec4>,
    ) {
        self.inner_mut()
            .insert_shape_in(env, handle.0, &shape.0, local_pose.0, color.map(|c| c.0));
    }

    /// Registers an instanced "visual" shape (lighter than `insert_shape`; used
    /// for articulated robots loaded from URDF/MJCF).
    fn insert_visual_shape(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: PyRef<SharedShape>,
        local_pose: Pose,
    ) {
        self.inner_mut()
            .insert_visual_shape(env, handle.0, &shape.0, local_pose.0);
    }

    // --- run loop ---------------------------------------------------------

    /// Renders one frame and processes UI/events. Returns `False` when the
    /// window is closed or a demo switch is pending. Blocks on the async render.
    fn render_frame(&mut self) -> bool {
        pollster::block_on(self.inner_mut().render_frame())
    }

    /// Whether the simulation should advance this frame (honors play/pause/step).
    fn simulating(&mut self) -> bool {
        self.inner_mut().simulating()
    }

    /// Returns the last rendered frame as an `(H, W, 3)` `uint8` NumPy array
    /// (row-major, top-to-bottom, RGB), like `mujoco.Renderer.render()`.
    ///
    /// Call once per frame after [`render_frame`][Self::render_frame] to export
    /// frames off-screen (e.g. to encode a video) instead of only presenting to
    /// the window.
    fn render<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let (w, h, rgb) = self.inner_mut().snap_rgb();
        rgb.into_pyarray(py)
            .reshape([h as usize, w as usize, 3])
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
    }

    /// Reads GPU state back into the renderer. Call once per frame after
    /// `simulate`. Blocks on the async readback.
    #[pyo3(signature = (state, timestamps=None))]
    fn sync(
        &mut self,
        mut state: PyRefMut<NexusState>,
        mut timestamps: Option<PyRefMut<GpuTimestamps>>,
    ) -> PyResult<()> {
        let ts = timestamps.as_deref_mut().map(|t| &mut t.0);
        pollster::block_on(self.inner_mut().sync(&mut state.0, ts))
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
    }

    // --- misc -------------------------------------------------------------

    fn clear_scene(&mut self) {
        self.inner_mut().clear_scene();
    }
    fn clear_transition(&mut self) {
        self.inner_mut().clear_transition();
    }
    fn quitting(&self) -> bool {
        self.inner().quitting()
    }
    fn selected_demo(&self) -> usize {
        self.inner().selected_demo()
    }
}
