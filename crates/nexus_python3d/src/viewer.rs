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
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
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

    /// Wraps raw RGB pixels as an `(H, W, 3)` numpy array.
    fn to_array(py: Python<'_>, w: u32, h: u32, rgb: Vec<u8>) -> PyResult<Bound<'_, PyArray3<u8>>> {
        rgb.into_pyarray(py)
            .reshape([h as usize, w as usize, 3])
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
    }
}

#[pymethods]
impl NexusViewer {
    /// Opens a window and probes the GPU. Blocks on the async setup.
    ///
    /// `width`/`height` set the window and render-target resolution
    /// (default 1200x900). Must be called on the main thread (required by the
    /// OS windowing system).
    ///
    /// With `headless=True` no OS window (and no swapchain) is created:
    /// frames render into an off-screen texture, unthrottled by the display's
    /// refresh rate — the fast path for video capture, and the only path on
    /// machines without a display server.
    #[new]
    #[pyo3(signature = (width=1200, height=900, headless=false))]
    fn new(width: u32, height: u32, headless: bool) -> Self {
        let inner = if headless {
            pollster::block_on(RViewer::new_headless_with_size(Vec::new(), width, height))
        } else {
            pollster::block_on(RViewer::new_with_size(Vec::new(), width, height))
        };
        NexusViewer(Some(inner))
    }

    /// Whether presentation is vsync-locked (always `False` headless).
    fn vsync(&self) -> bool {
        self.inner().vsync()
    }

    /// Enables/disables vsync. With vsync off, `render_frame` no longer waits
    /// for the display refresh (~60 Hz), so windowed capture runs as fast as
    /// the GPU allows. No-op on a headless viewer.
    fn set_vsync(&mut self, enabled: bool) {
        self.inner_mut().set_vsync(enabled);
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

    /// Whether `render_frame` draws the built-in egui panel (default `True`).
    /// Disable for clean frame capture with `render`.
    fn set_draw_ui(&mut self, enabled: bool) {
        self.inner_mut().set_draw_ui(enabled);
    }

    /// Renders one path-traced frame with kiss3d's GPU path tracer instead of
    /// the rasterizer. Samples accumulate across calls while the scene is
    /// static; call it several times (or raise `set_raytracer_samples_per_frame`)
    /// before `render` to converge. Returns `False` when the window is closed.
    fn raytrace_frame(&mut self) -> bool {
        pollster::block_on(self.inner_mut().raytrace_frame())
    }

    /// Number of path-tracing samples accumulated per `raytrace_frame` call.
    fn set_raytracer_samples_per_frame(&mut self, samples: u32) {
        self.inner_mut().set_raytracer_samples_per_frame(samples);
    }

    /// Maximum path-tracing bounce depth.
    fn set_raytracer_max_bounces(&mut self, bounces: u32) {
        self.inner_mut().set_raytracer_max_bounces(bounces);
    }

    /// Enables/disables the path tracer's denoiser.
    fn set_raytracer_denoise(&mut self, enabled: bool) {
        self.inner_mut().set_raytracer_denoise(enabled);
    }

    /// Which intersection backend the path tracer uses: `"hardware"` (RT-core
    /// ray queries) or `"software"` (portable compute-shader BVH fallback).
    fn raytracer_backend(&mut self) -> &'static str {
        self.inner_mut().raytracer_backend_name()
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
    fn snap_rgb<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let (w, h, rgb) = self.inner_mut().snap_rgb();
        Self::to_array(py, w, h, rgb)
    }

    /// Pipelined variant of [`snap_rgb`][Self::snap_rgb] for video capture: starts
    /// a non-blocking capture of the frame just rendered and returns the
    /// *previous* frame's pixels (one frame of latency), or `None` on the
    /// first call. Unlike `snap_rgb` this never stalls the GPU pipeline waiting
    /// for the copy. Call [`snap_rgb_flush`][Self::snap_rgb_flush] after the loop
    /// to collect the final frame.
    fn snap_rgb_async<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray3<u8>>>> {
        match self.inner_mut().snap_rgb_async() {
            Some((w, h, rgb)) => Ok(Some(Self::to_array(py, w, h, rgb)?)),
            None => Ok(None),
        }
    }

    /// Completes and returns the capture left in flight by
    /// [`snap_rgb_async`][Self::snap_rgb_async], or `None` when there is none.
    fn snap_rgb_flush<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray3<u8>>>> {
        match self.inner_mut().snap_rgb_flush() {
            Some((w, h, rgb)) => Ok(Some(Self::to_array(py, w, h, rgb)?)),
            None => Ok(None),
        }
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

    /// Reads back environment `env`'s multibody link states from the GPU in one
    /// readback. Returns five float32 numpy arrays, one row per link (in the
    /// GPU build's traversal order — multibodies, then links, parent before
    /// child; the same order `NexusState.apply_actuator_controls` drives):
    ///
    /// - `coords (n, 6)` — generalized joint coordinates (only the joint's DOF
    ///   count is meaningful; a revolute joint's angle is `coords[5]`),
    /// - `positions (n, 3)` / `quats (n, 4)` — link world pose (`w, x, y, z`),
    /// - `linvels (n, 3)` / `angvels (n, 3)` — world-space velocities (valid
    ///   after the first simulated step).
    #[pyo3(signature = (state, env=0))]
    #[allow(clippy::type_complexity)]
    fn read_multibody_links<'py>(
        &mut self,
        py: Python<'py>,
        state: PyRef<NexusState>,
        env: u32,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    ) {
        let links = pollster::block_on(self.inner_mut().read_multibody_links(&state.0, env));
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
