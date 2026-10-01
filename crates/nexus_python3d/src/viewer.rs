//! The windowed viewer (`nexus_viewer3d::NexusViewer`).
//!
//! The Rust builder methods (`with_cpu`, `with_running`, …) consume `self`, so
//! the inner viewer is stored in an `Option` and swapped in place; the builder
//! wrappers return the same Python object for fluent chaining.

use crate::math::Pose;
use crate::math::{Vec3, Vec4};
use crate::nexus::{GpuTimestamps, NexusState};
use crate::rbd::{RigidBodyHandle, SharedShape};
use crate::robot::{pose_from_wxyz, to_wxyz};
use khal::backend::GpuBackend;
use nexus_viewer3d::{NexusViewer as RViewer, VisualTexture};
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

    // --- sensor cameras -----------------------------------------------------

    /// Makes later `insert_shape*` calls register one render node per body
    /// instead of GPU instances. Required for bodies that sensor cameras must
    /// see in their depth and segmentation passes and for
    /// `set_body_segmentation_id`. Call before inserting shapes.
    fn set_sensor_rendering(&mut self, enabled: bool) {
        self.inner_mut().set_sensor_rendering(enabled);
    }

    /// Adds an offscreen sensor camera (`width x height`, vertical field of
    /// view `fov_y_deg` in degrees) and returns its id. It starts at the
    /// identity pose.
    #[pyo3(signature = (width, height, fov_y_deg, znear=0.01, zfar=100.0))]
    fn add_sensor_camera(
        &mut self,
        width: u32,
        height: u32,
        fov_y_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> usize {
        pollster::block_on(self.inner_mut().add_sensor_camera(
            width,
            height,
            fov_y_deg.to_radians(),
            znear,
            zfar,
        ))
    }

    /// Removes sensor camera `id` and frees its GPU resources. The id is not
    /// reused; rendering it afterwards raises. Returns whether a camera was
    /// removed. Each camera holds its own render targets and shadow atlas, so a
    /// process that builds many scenes must release the ones it is done with.
    fn remove_sensor_camera(&mut self, id: usize) -> bool {
        self.inner_mut().remove_sensor_camera(id)
    }

    /// Number of live (not removed) sensor cameras.
    fn num_sensor_cameras(&self) -> usize {
        self.inner().num_sensor_cameras()
    }

    /// Sets sensor camera `id`'s pose: position and `(w, x, y, z)` quaternion
    /// of the OpenGL camera frame (looks down its -Z axis, +Y up).
    fn set_sensor_camera_pose(&mut self, id: usize, pos: [f32; 3], quat: [f32; 4]) {
        self.inner_mut()
            .set_sensor_camera_pose(id, pose_from_wxyz(pos, quat));
    }

    /// Sensor camera `id`'s pose as `(position, (w, x, y, z))`, OpenGL frame.
    fn sensor_camera_pose(&self, id: usize) -> Option<([f32; 3], [f32; 4])> {
        let pose = self.inner().sensor_camera(id)?.pose();
        let t = pose.translation;
        Some(([t.x, t.y, t.z], to_wxyz(pose.rotation)))
    }

    /// Attaches sensor camera `id` to body `handle` of `env` with the mount
    /// pose `pos` / `quat` (`w, x, y, z`, body frame to camera frame). The
    /// camera follows the body at every `sync`.
    fn attach_sensor_camera(
        &mut self,
        id: usize,
        env: u32,
        handle: RigidBodyHandle,
        pos: [f32; 3],
        quat: [f32; 4],
        state: PyRef<NexusState>,
    ) {
        self.inner_mut().attach_sensor_camera(
            id,
            env,
            handle.0,
            pose_from_wxyz(pos, quat),
            &state.0,
        );
    }

    /// Ambient light level of sensor camera `id`'s shaded render.
    fn set_sensor_camera_ambient(&mut self, id: usize, ambient: f32) {
        if let Some(sensor) = self.inner_mut().sensor_camera_mut(id) {
            sensor.set_ambient(ambient);
        }
    }

    /// Ambient light color (RGB) of sensor camera `id`'s shaded render, a tint
    /// on the fill light `set_sensor_camera_ambient` scales.
    fn set_sensor_camera_ambient_color(&mut self, id: usize, rgb: [f32; 3]) {
        if let Some(sensor) = self.inner_mut().sensor_camera_mut(id) {
            sensor.set_ambient_color(rgb);
        }
    }

    /// Background color of sensor camera `id`'s shaded render.
    fn set_sensor_camera_background(&mut self, id: usize, rgba: [f32; 4]) {
        if let Some(sensor) = self.inner_mut().sensor_camera_mut(id) {
            sensor.set_background_color(rgba);
        }
    }

    /// Renders sensor camera `id` and returns `(rgb, depth, segmentation)`:
    /// `rgb` is `(H, W, 3)` uint8, `depth` `(H, W)` float32 linear metric depth
    /// (`0.0` = background), `segmentation` `(H, W)` uint32 per-body ids (`0` =
    /// background). Each is `None` unless requested.
    #[pyo3(signature = (id, rgb=true, depth=false, segmentation=false))]
    #[allow(clippy::type_complexity)]
    fn render_sensor_camera<'py>(
        &mut self,
        py: Python<'py>,
        id: usize,
        rgb: bool,
        depth: bool,
        segmentation: bool,
    ) -> PyResult<(
        Option<Bound<'py, PyArray3<u8>>>,
        Option<Bound<'py, PyArray2<f32>>>,
        Option<Bound<'py, PyArray2<u32>>>,
    )> {
        let (w, h) = self
            .inner()
            .sensor_camera(id)
            .map(|s| s.size())
            .ok_or_else(|| PyRuntimeError::new_err(format!("no sensor camera {id}")))?;
        let (w, h) = (w as usize, h as usize);
        let rgb = if rgb {
            let pixels = pollster::block_on(self.inner_mut().render_sensor_rgb(id))
                .ok_or_else(|| PyRuntimeError::new_err("rgb render failed"))?;
            Some(Self::to_array(py, w as u32, h as u32, pixels)?)
        } else {
            None
        };
        let depth = if depth {
            let values = self
                .inner_mut()
                .render_sensor_depth(id)
                .ok_or_else(|| PyRuntimeError::new_err("depth render failed"))?;
            Some(
                values
                    .into_pyarray(py)
                    .reshape([h, w])
                    .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?,
            )
        } else {
            None
        };
        let segmentation = if segmentation {
            let ids = self
                .inner_mut()
                .render_sensor_segmentation(id)
                .ok_or_else(|| PyRuntimeError::new_err("segmentation render failed"))?;
            Some(
                ids.into_pyarray(py)
                    .reshape([h, w])
                    .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?,
            )
        } else {
            None
        };
        Ok((rgb, depth, segmentation))
    }

    /// Starts a new scene generation: render nodes and sensor cameras created
    /// afterwards belong to it and it becomes the active one, and the previous
    /// scene's nodes leave the graph for good. Use one generation per
    /// `NexusState` sharing this viewer; only the latest can render.
    fn begin_scene(&mut self) -> u32 {
        self.inner_mut().begin_scene()
    }

    /// The active (most recently begun) scene generation.
    fn active_scene(&self) -> u32 {
        self.inner().active_scene()
    }

    /// Tags every render node of body `handle` in `env` with segmentation id
    /// `id` (avoid `0`, the background). Returns the number of nodes tagged;
    /// `0` means the body has no per-body node (see `set_sensor_rendering`).
    fn set_body_segmentation_id(&mut self, env: u32, handle: RigidBodyHandle, id: u32) -> usize {
        self.inner_mut().set_body_segmentation_id(env, handle.0, id)
    }

    /// Sets the base color (RGBA) of every render node of body `handle` in `env`.
    fn set_body_color(&mut self, env: u32, handle: RigidBodyHandle, rgba: [f32; 4]) {
        self.inner_mut().set_body_color(env, handle.0, rgba);
    }

    /// Whether body `handle`'s visual nodes cast shadows (default `True`). A
    /// floor slab that does not cast keeps the shadow map fit to the objects
    /// above it, so hard shadow edges stay crisp.
    fn set_body_casts_shadows(&mut self, env: u32, handle: RigidBodyHandle, casts: bool) {
        self.inner_mut()
            .set_body_casts_shadows(env, handle.0, casts);
    }

    /// Ambient light level of the main window's shaded render.
    fn set_ambient(&mut self, ambient: f32) {
        self.inner_mut().set_ambient(ambient);
    }

    /// MSAA sample count of the sensor cameras' shaded renders, existing and
    /// future ones (`1` disables antialiasing, `4` is the default).
    fn set_sensor_antialiasing(&mut self, samples: u32) {
        self.inner_mut().set_sensor_antialiasing(samples);
    }

    /// Shadow-edge softness of the sensor cameras' shaded renders, existing
    /// and future ones (`1.0` the PCF penumbra, the default; `0.0` hard
    /// edges).
    fn set_sensor_shadow_softness(&mut self, softness: f32) {
        self.inner_mut().set_sensor_shadow_softness(softness);
    }

    /// Shadow-edge softness of the main window's shaded render (`1.0` the
    /// PCF penumbra, the default; `0.0` hard edges).
    fn set_shadow_softness(&mut self, softness: f32) {
        self.inner_mut().set_shadow_softness(softness);
    }

    /// Directional-shadow cascade layout of the sensor cameras' renders,
    /// existing and future ones: the sharpest cascade covers the camera's
    /// first `first_cascade_far_bound` meters (default 3) and shadows stop at
    /// `shadow_distance` meters (default: the camera far plane).
    #[pyo3(signature = (first_cascade_far_bound, shadow_distance=f32::INFINITY))]
    fn set_sensor_shadow_range(&mut self, first_cascade_far_bound: f32, shadow_distance: f32) {
        self.inner_mut()
            .set_sensor_shadow_range(first_cascade_far_bound, shadow_distance);
    }

    /// Shadow map resolution (texels per atlas layer, default 4096) and atlas
    /// layer count (default 4, one directional light's cascades; a point light
    /// needs 6) of the sensor cameras' renders, existing and future ones.
    /// Memory per camera is `resolution² × layers × 8` bytes.
    #[pyo3(signature = (resolution, layers=4))]
    fn set_sensor_shadow_resolution(&mut self, resolution: u32, layers: u32) {
        self.inner_mut()
            .set_sensor_shadow_resolution(resolution, layers);
    }

    /// Shadow map resolution and atlas layer count of the main window's render
    /// (see `set_sensor_shadow_resolution`).
    #[pyo3(signature = (resolution, layers=4))]
    fn set_shadow_resolution(&mut self, resolution: u32, layers: u32) {
        self.inner_mut().set_shadow_resolution(resolution, layers);
    }

    /// Directional-shadow cascade layout of the main window's render (see
    /// `set_sensor_shadow_range`).
    #[pyo3(signature = (first_cascade_far_bound, shadow_distance=f32::INFINITY))]
    fn set_shadow_range(&mut self, first_cascade_far_bound: f32, shadow_distance: f32) {
        self.inner_mut()
            .set_shadow_range(first_cascade_far_bound, shadow_distance);
    }

    /// Registers one render node for body `handle` in `env` drawing `shape`
    /// with base color `rgba`, optional per-vertex UVs (trimesh shapes only)
    /// and an optional encoded image texture (`texture` bytes, PNG/JPEG,
    /// cached under `texture_name`). Unlike `insert_visual_shape`, the node is
    /// never instanced, so sensor cameras see it in every pass.
    #[pyo3(signature = (env, handle, shape, local_pose, rgba, uvs=None, texture=None, texture_name=None))]
    #[allow(clippy::too_many_arguments)]
    fn insert_sensor_shape(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: PyRef<SharedShape>,
        local_pose: Pose,
        rgba: [f32; 4],
        uvs: Option<Vec<[f32; 2]>>,
        texture: Option<Vec<u8>>,
        texture_name: Option<String>,
    ) {
        let name = texture_name.unwrap_or_else(|| format!("sensor-texture-{env}-{:?}", handle.0));
        let tex = match &texture {
            Some(bytes) => VisualTexture::Bytes(bytes, &name),
            None => VisualTexture::None,
        };
        self.inner_mut().insert_visual_mesh_textured(
            env,
            handle.0,
            &shape.0,
            local_pose.0,
            rgba,
            uvs.as_deref(),
            None,
            tex,
            None,
        );
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
