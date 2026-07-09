//! Generic, scene-agnostic rendering/runtime resources.
//!
//! [`NexusViewer`] is the viewer analogue of kiss3d's `Window`: it owns the window,
//! cameras, GPU backends and UI state, but knows nothing about a particular
//! physics scene. Examples build a scene into a [`NexusState`] and register
//! render shapes via [`NexusViewer::insert_shape`], then own the loop:
//!
//! ```ignore
//! let mut viewer = NexusViewer::new(vec![]).await;
//! let mut scene = viewer.set_rbd(state).await;
//! while viewer.render(&mut scene).await {
//!     scene.simulate(&mut viewer).await;
//! }
//! scene.detach(&mut viewer);
//! ```

use glamx::Vec4;
use khal::Shader;
use khal::backend::{
    Backend, GpuBackend as KhalGpuBackend, GpuBackendError, GpuTimestamps, WebGpu,
};
use khal::re_exports::wgpu::{Features, Limits};
use std::time::Duration;

use kiss3d::prelude::Color;
#[cfg(feature = "dim3")]
use kiss3d::renderer::RayTracer;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use kiss3d::window::{NumSamples, Window};

#[cfg(feature = "dim3")]
use kiss3d::camera::{FixedView2d, OrbitCamera3d};
#[cfg(feature = "dim2")]
use kiss3d::camera::{FixedView3d, PanZoomCamera2d};
use nexus::rbd::dynamics::WgRbdPrepRender;
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::RunStats;
use nexus::state::{NexusCounts, NexusState};
use rapier::prelude::{RigidBodyHandle, SharedShape};
// use crate::rbd::{
//     BackendType, RbdScene, RenderContext, SimulationState, setup_physics,
// };
use crate::backend::BackendType;
use crate::graphics::RenderContext;
use crate::{DemoKind, RunState, Transition, UiSections};

/// UI / runtime state that is independent from the GPU/window resources. Kept in
/// its own struct so [`NexusViewer::render_frame`] can split-borrow it from `window`.
pub struct UiState {
    pub run_state: RunState,
    pub run_stats: RunStats,
    pub sync_time: Duration,
    pub ui_sections: UiSections,
    pub backend_type: BackendType,
    pub gpu_init_error: Option<String>,
    /// Names + kinds of all registered demos, used to populate the demo picker.
    pub demos: Vec<(String, DemoKind)>,
    /// Index (into `demos`) of the currently selected demo.
    pub selected_demo: usize,
    /// Pending loop transition requested via the UI (demo switch / quit).
    pub(crate) transition: Option<Transition>,
    /// User-editable per-scene simulation settings, surfaced in the Settings panel.
    pub sim_settings: SimSettings,
    /// Index of the demo [`sim_settings`](Self::sim_settings) was seeded from.
    /// The settings are pulled FROM the scene only when this differs from the
    /// selected demo (i.e. a genuine demo switch) and pushed back TO the scene
    /// otherwise — so restarts and backend switches keep the user's edits.
    pub(crate) settings_demo: Option<usize>,
    /// Which sub-systems the current scene contains (drives which settings show).
    pub(crate) has_rbd: bool,
    /// Current scene entity counts, refreshed every `sync` for the UI.
    pub(crate) counts: NexusCounts,
}

/// Editable simulation settings exposed in the viewer UI. The viewer pulls
/// these from the [`NexusState`] when a demo loads and pushes edits back each
/// frame (see [`NexusViewer::sync`]).
#[derive(Clone)]
pub struct SimSettings {
    /// Rigid-body solver steps advanced per rendered frame.
    pub rbd_steps_per_frame: u32,
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            rbd_steps_per_frame: 1,
        }
    }
}

#[cfg(feature = "dim2")]
pub type SceneNode = SceneNode2d;
#[cfg(feature = "dim3")]
pub type SceneNode = SceneNode3d;

/// Number of frames to render the "compiling shaders" banner before the
/// (blocking) pipeline preload, so the browser actually *presents* it first. On
/// the web, `create_compute_pipeline` stalls the JS thread without yielding, so
/// a banner drawn but not yet composited would never reach the screen before
/// the freeze; rendering a few real frames first forces the paint.
const COMPILE_BANNER_PRESENT_FRAMES: u32 = 10;

pub struct NexusViewer {
    window: Window,
    scene2d: SceneNode2d,
    scene3d: SceneNode3d,
    #[cfg(feature = "dim3")]
    camera3d: OrbitCamera3d,
    /// Camera up axis (default +Y). Re-applied whenever the camera is rebuilt so
    /// it survives [`Self::set_camera`]. Set to `Vec3::Z` for Z-up scenes (MJCF).
    #[cfg(feature = "dim3")]
    up_axis: glamx::Vec3,
    #[cfg(feature = "dim3")]
    camera2d: FixedView2d,
    #[cfg(feature = "dim2")]
    camera3d: FixedView3d,
    #[cfg(feature = "dim2")]
    camera2d: PanZoomCamera2d,
    // TODO: the backends shouldn’t be stored by the viewer.
    webgpu: Option<KhalGpuBackend>,
    /// Whether [`Self::webgpu`] was built on top of kiss3d's wgpu device (shared
    /// device/queue). When true, `sync` can write render data straight into
    /// kiss3d's GPU instance buffers instead of reading back to the CPU.
    webgpu_shared: bool,
    #[cfg(feature = "cuda")]
    cuda: Option<KhalGpuBackend>,
    #[cfg(feature = "metal")]
    metal: Option<KhalGpuBackend>,
    /// CPU backend, stored so [`Self::backend`] can hand out a reference for the
    /// `Cpu`/`Rapier` selections. `None` when compiled without the `cpu` feature.
    cpu: Option<KhalGpuBackend>,
    nexus_render: RenderContext,
    /// GPU kernel that writes rigid-body render data straight into kiss3d's
    /// instance buffers on the shared-device (WebGPU) path. Compiled lazily on
    /// the first direct sync; reused across demos.
    rbd_prep_render: Option<WgRbdPrepRender>,
    /// Backend the cached render-prep shaders/buffers (`rbd_prep_render`, …)
    /// were built for. When the active backend changes, those resources belong
    /// to a different device and must be dropped and rebuilt — otherwise using
    /// them crashes.
    render_resources_backend: Option<BackendType>,
    /// Last GPU pass timings harvested from the non-blocking timestamp readback.
    /// Re-applied to `run_stats` every frame so the profiler UI keeps showing the
    /// latest values between readbacks (which only complete every few frames)
    /// instead of flickering to empty.
    last_gpu_pass_times: Vec<(String, f64)>,
    /// Total GPU time matching [`Self::last_gpu_pass_times`].
    last_gpu_total_time_ms: f64,
    /// Whether [`Self::render_frame`] draws the built-in egui panel. Disable
    /// for clean frame capture ([`Self::snap_rgb`]).
    draw_ui: bool,
    /// Lazily-created path tracer for [`Self::raytrace_frame`]. Kept across
    /// frames so samples keep accumulating while the scene is static.
    #[cfg(feature = "dim3")]
    raytracer: Option<RayTracer>,
    pub ui: UiState,
}

impl NexusViewer {
    /// Creates a viewer, opening the window and probing the WebGPU backend.
    ///
    /// `demos` is the list of `(name, kind)` shown in the demo picker; pass an
    /// empty vec for a standalone single-example viewer (no picker).
    pub async fn new(demos: Vec<(String, DemoKind)>) -> Self {
        Self::new_with_size(demos, 1200, 900).await
    }

    /// Like [`Self::new`] but with an explicit window/render resolution.
    pub async fn new_with_size(demos: Vec<(String, DemoKind)>, width: u32, height: u32) -> Self {
        let window = Window::new_with_setup("nexus demos", width, height, Self::setup()).await;
        Self::with_window(window, demos).await
    }

    /// Like [`Self::new_with_size`] but headless: no OS window, no swapchain,
    /// no vsync — frames render straight into an off-screen texture, which is
    /// what you want for video capture or on a machine with no display server.
    pub async fn new_headless_with_size(
        demos: Vec<(String, DemoKind)>,
        width: u32,
        height: u32,
    ) -> Self {
        let window = Window::new_headless_with_setup(width, height, Self::setup()).await;
        Self::with_window(window, demos).await
    }

    // NOTE: PASSTHROUGH_SHADERS is required for compute shaders with spirv_passthrough on
    //       platforms running vulkan (to work around some naga limitations).
    fn setup() -> kiss3d::window::CanvasSetup {
        kiss3d::window::CanvasSetup {
            required_features: Features::PASSTHROUGH_SHADERS,
            ..Default::default()
        }
    }

    async fn with_window(mut window: Window, demos: Vec<(String, DemoKind)>) -> Self {
        window.set_background_color(Color::new(245.0 / 255.0, 245.0 / 255.0, 236.0 / 255.0, 1.0));
        // Disable MSAA, this puts extra load on the GPU that ends up
        // falsifying the gpu physics timestamps.
        window.set_samples(NumSamples::One);

        #[cfg(feature = "dim2")]
        let (camera2d, camera3d) = {
            let mut sidescroll = PanZoomCamera2d::default();
            sidescroll.look_at(glamx::Vec2::new(0.0, 100.0), 7.5);
            (sidescroll, FixedView3d::default())
        };
        #[cfg(feature = "dim3")]
        let (camera2d, camera3d) = {
            let arc_ball = OrbitCamera3d::new(
                glamx::Vec3::new(-100.0, 100.0, -100.0),
                glamx::Vec3::new(0.0, 40.0, 0.0),
            );
            (FixedView2d::default(), arc_ball)
        };

        let scene3d = SceneNode3d::empty();
        let scene2d = SceneNode2d::empty();

        let mut viewer = Self {
            window,
            scene2d,
            scene3d,
            camera3d,
            #[cfg(feature = "dim3")]
            up_axis: glamx::Vec3::Y,
            camera2d,
            webgpu: None,
            webgpu_shared: false,
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "metal")]
            metal: None,
            cpu: {
                #[cfg(feature = "cpu")]
                {
                    Some(KhalGpuBackend::Cpu)
                }
                #[cfg(not(feature = "cpu"))]
                {
                    None
                }
            },
            nexus_render: RenderContext::new(),
            rbd_prep_render: None,
            render_resources_backend: None,
            last_gpu_pass_times: Vec::new(),
            last_gpu_total_time_ms: 0.0,
            draw_ui: true,
            #[cfg(feature = "dim3")]
            raytracer: None,
            ui: UiState {
                run_state: RunState::Paused,
                run_stats: RunStats::default(),
                sync_time: Duration::default(),
                ui_sections: UiSections {
                    show_examples: true,
                    show_settings: false,
                    show_performance: true,
                },
                backend_type: BackendType::Gpu,
                gpu_init_error: None,
                demos,
                selected_demo: 0,
                transition: None,
                sim_settings: SimSettings::default(),
                settings_demo: None,
                has_rbd: false,
                counts: NexusCounts::default(),
            },
        };

        // Always probe the WebGPU backend at startup so the UI knows whether to
        // offer it as an option, independent of the initial backend choice.
        viewer.webgpu = viewer.init_webgpu().await;
        viewer
    }

    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.ui.backend_type = backend_type;
        self
    }

    pub fn with_cpu(mut self) -> Self {
        self.ui.backend_type = BackendType::Cpu;
        self
    }

    pub fn with_running(mut self) -> Self {
        self.ui.run_state = RunState::Running;
        self
    }

    pub fn with_selected_demo(mut self, idx: usize) -> Self {
        self.ui.selected_demo = idx;
        self
    }

    pub fn selected_demo(&self) -> usize {
        self.ui.selected_demo
    }

    /// Whether the loop should stop entirely (window closed).
    pub fn quitting(&self) -> bool {
        matches!(self.ui.transition, Some(Transition::Quit))
    }

    /// Clears a pending demo-switch transition. Call between two example runs.
    pub fn clear_transition(&mut self) {
        self.ui.transition = None;
    }

    async fn init_webgpu(&mut self) -> Option<KhalGpuBackend> {
        // Prefer sharing kiss3d's wgpu device/queue: `Window::new` (called in
        // `Self::new`, before this) initializes the global kiss3d `Context`, so
        // both the renderer and khal's compute work run on one device. That lets
        // `sync` write render data straight into kiss3d's GPU instance buffers
        // (no GPU→CPU readback), which matters where synchronization is slow
        // (e.g. Firefox/WebGPU). kiss3d requests the adapter's full limits, a
        // superset of the explicit limits the standalone path requested below.
        if kiss3d::context::Context::is_initialized() {
            let ctxt = kiss3d::context::Context::get();
            let wgpu = WebGpu::from_device(
                (*ctxt.instance).clone(),
                (*ctxt.adapter).clone(),
                (*ctxt.device).clone(),
                (*ctxt.queue).clone(),
            );
            self.webgpu_shared = true;
            return Some(KhalGpuBackend::WebGpu(wgpu));
        }

        self.webgpu_shared = false;
        let limits = Limits {
            max_buffer_size: 1 << 30, // Firefox’s limit
            max_storage_buffer_binding_size: 1 << 30,
            max_compute_workgroup_storage_size: 19904,
            ..Default::default()
        };
        match WebGpu::new(Default::default(), limits).await.map(|wgpu| {
            // NOTE: Uncomment this to make debugging easier (all buffers
            //       become host-readable).
            // wgpu.force_buffer_copy_src = true;
            KhalGpuBackend::WebGpu(wgpu)
        }) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                self.ui.gpu_init_error = Some(format!(
                    "GPU backend not available, initialization failed:\n\"{}\"\n",
                    e
                ));
                None
            }
        }
    }

    /// Whether the active backend can write render data directly into kiss3d's
    /// GPU instance buffers (WebGPU sharing kiss3d's device). When false, `sync`
    /// falls back to the GPU→CPU readback path.
    fn direct_render_path(&self) -> bool {
        self.webgpu_shared
            && !self.rt_active()
            && self.ui.backend_type == BackendType::Gpu
            && matches!(self.webgpu, Some(KhalGpuBackend::WebGpu(_)))
    }

    /// Whether the path tracer has been used/configured. The tracer rebuilds its
    /// acceleration structure from the CPU-side instance buffers, so while it is
    /// active `sync` must take the readback path — the zero-readback kernel only
    /// updates kiss3d's GPU instance buffers, which the tracer never reads.
    fn rt_active(&self) -> bool {
        #[cfg(feature = "dim3")]
        {
            self.raytracer.is_some()
        }
        #[cfg(not(feature = "dim3"))]
        {
            false
        }
    }

    #[cfg(feature = "cuda")]
    fn init_cuda(&mut self) -> Option<KhalGpuBackend> {
        match khal::backend::cuda::Cuda::new(0) {
            Ok(cuda) => Some(KhalGpuBackend::Cuda(cuda)),
            Err(e) => {
                self.ui.gpu_init_error = Some(format!(
                    "CUDA backend not available, initialization failed:\n\"{:?}\"\n",
                    e
                ));
                None
            }
        }
    }

    #[cfg(feature = "metal")]
    fn init_metal(&mut self) -> Option<KhalGpuBackend> {
        match khal::backend::metal::Metal::new() {
            Ok(metal) => Some(KhalGpuBackend::Metal(metal)),
            Err(e) => {
                self.ui.gpu_init_error = Some(format!(
                    "Metal backend not available, initialization failed:\n\"{:?}\"\n",
                    e
                ));
                None
            }
        }
    }

    /// Lazily initializes the GPU backend matching the currently selected backend
    /// type if it has not been created yet. WebGPU is always probed at startup, so
    /// only the (synchronous) CUDA/Metal backends are created on demand here.
    fn ensure_backend_initialized(&mut self) {
        match self.ui.backend_type {
            #[cfg(feature = "cuda")]
            BackendType::Cuda if self.cuda.is_none() => {
                self.cuda = self.init_cuda();
            }
            #[cfg(feature = "metal")]
            BackendType::Metal if self.metal.is_none() => {
                self.metal = self.init_metal();
            }
            _ => {}
        }
    }

    /// Returns the active GPU backend for the current backend type, if available.
    pub fn gpu(&self) -> Option<&KhalGpuBackend> {
        match self.ui.backend_type {
            BackendType::Gpu => self.webgpu.as_ref(),
            #[cfg(feature = "cuda")]
            BackendType::Cuda => self.cuda.as_ref(),
            #[cfg(feature = "metal")]
            BackendType::Metal => self.metal.as_ref(),
            _ => None,
        }
    }

    pub fn scene2d_mut(&mut self) -> &mut SceneNode2d {
        &mut self.scene2d
    }

    pub fn scene3d_mut(&mut self) -> &mut SceneNode3d {
        &mut self.scene3d
    }

    pub fn backend(&self) -> &KhalGpuBackend {
        match self.ui.backend_type {
            BackendType::Gpu => self.webgpu.as_ref().unwrap(),
            #[cfg(feature = "cuda")]
            BackendType::Cuda => self.cuda.as_ref().unwrap(),
            #[cfg(feature = "metal")]
            BackendType::Metal => self.metal.as_ref().unwrap(),
            BackendType::Cpu => self
                .cpu
                .as_ref()
                .expect("CPU backend unavailable: compile with the 'cpu' feature"),
            #[allow(unreachable_patterns)]
            _ => panic!("selected backend is not available in this build"),
        }
    }

    /// Whether the WebGPU backend is available (used by the UI backend selector).
    pub fn gpu_available(&self) -> bool {
        self.webgpu.is_some()
    }

    #[cfg(feature = "dim3")]
    pub fn set_camera(&mut self, eye: glamx::Vec3, target: glamx::Vec3) {
        let mut camera = OrbitCamera3d::new(eye, target);
        camera.set_up_axis(self.up_axis);
        self.camera3d = camera;
    }

    /// Sets the camera's up axis (e.g. `Vec3::Z` for Z-up scenes like MJCF
    /// models). Applied immediately and preserved across [`Self::set_camera`],
    /// so callers can pick the world convention instead of rotating their data.
    #[cfg(feature = "dim3")]
    pub fn set_up_axis(&mut self, up: glamx::Vec3) {
        self.up_axis = up;
        self.camera3d.set_up_axis(up);
    }

    #[cfg(feature = "dim2")]
    pub fn set_camera_2d(&mut self, center: glamx::Vec2, zoom: f32) {
        self.camera2d.look_at(center, zoom);
    }

    /// Ensures the GPU backend for the current backend type exists. Call once,
    /// before the demo loop, so [`Self::backend`] is usable by the examples that
    /// drive a [`NexusState`] directly.
    pub fn init_backend(&mut self) {
        self.ensure_backend_initialized();
    }

    /// Registers a render shape for a body in environment 0.
    pub fn insert_shape(&mut self, handle: RigidBodyHandle, shape: &SharedShape, local_pose: Pose) {
        self.insert_shape_in(0, handle, shape, local_pose, None)
    }

    pub fn insert_shape_with_color(
        &mut self,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
        color: Vec4,
    ) {
        self.insert_shape_in(0, handle, shape, local_pose, Some(color))
    }

    /// Registers a render shape for a body in environment `env` (batch).
    pub fn insert_shape_in(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
        color: Option<Vec4>,
    ) {
        self.nexus_render.insert_shape(
            #[cfg(feature = "dim2")]
            &mut self.scene2d,
            #[cfg(feature = "dim3")]
            &mut self.scene3d,
            env,
            handle,
            shape,
            local_pose,
            color,
        )
    }

    /// Registers a render shape with a body-local pose offset (e.g. a URDF
    /// visual mesh whose frame differs from its proxy collider).
    pub fn insert_visual_shape(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
    ) {
        self.nexus_render.insert_shape(
            #[cfg(feature = "dim2")]
            &mut self.scene2d,
            #[cfg(feature = "dim3")]
            &mut self.scene3d,
            env,
            handle,
            shape,
            local_pose,
            None,
        )
    }

    /// Registers a body-attached visual mesh with full material support: base
    /// color, texture (loaded from a file path), per-vertex UVs and normals, and
    /// PBR material parameters. Unlike [`Self::insert_visual_shape`] — which is
    /// instanced and color-only — each mesh becomes its own node, so it can
    /// carry a distinct texture/material. This renders MJCF `<geom>` visuals the
    /// way MuJoCo's own viewer does. Its pose follows the body's world pose every
    /// frame (synced in [`Self::sync`]).
    #[cfg(feature = "dim3")]
    pub fn insert_visual_mesh(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
        color: [f32; 4],
        uvs: Option<&[[f32; 2]]>,
        normals: Option<&[[f32; 3]]>,
        texture: Option<&std::path::Path>,
        material: Option<crate::graphics::RenderMaterial>,
    ) {
        self.nexus_render.insert_visual_mesh(
            &mut self.scene3d,
            env,
            handle,
            shape,
            local_pose,
            color,
            uvs,
            normals,
            texture,
            material,
        );
    }

    async fn sync_timestamps(&mut self, timestamps: Option<&mut GpuTimestamps>) {
        if let Some(timestamps) = timestamps
            && let Some(results) = timestamps.try_take(self.backend())
        {
            if !results.is_empty() {
                let mut aggregated: Vec<(String, f64)> = Vec::new();
                for r in &results {
                    if let Some(existing) =
                        aggregated.iter_mut().find(|(label, _)| label == &r.label)
                    {
                        existing.1 += r.duration_ms;
                    } else {
                        aggregated.push((r.label.clone(), r.duration_ms));
                    }
                }

                self.last_gpu_total_time_ms = aggregated.iter().map(|e| e.1).sum();
                self.last_gpu_pass_times = aggregated;
            }
            // Clear the query set so `simulate` can record the next frame.
            timestamps.reset();
        }
    }

    async fn sync_with_readback(&mut self, state: &mut NexusState) -> Result<(), GpuBackendError> {
        // Rigid bodies: collider poses → instanced shapes.
        if let Some(rbd) = state.rbd.as_ref() {
            let poses = rbd.body_poses();
            let mut cache = vec![Pose::default(); poses.len() as usize];
            let _ = self
                .backend()
                .slow_read_buffer(poses.buffer(), &mut cache)
                .await;
            self.nexus_render.update_instances_from_poses(state, &cache);

            // Body-attached visual meshes follow the body-origin poses, since
            // their local poses are body-relative.
            #[cfg(feature = "dim3")]
            if self.nexus_render.has_visual_nodes() {
                let body_poses = rbd.body_poses();
                let mut body_cache = vec![Pose::default(); body_poses.len() as usize];
                let _ = self
                    .backend()
                    .slow_read_buffer(body_poses.buffer(), &mut body_cache)
                    .await;
                self.nexus_render.update_visual_nodes(state, &body_cache);
            }
        }

        Ok(())
    }
    async fn sync_without_readback(
        &mut self,
        state: &mut NexusState,
    ) -> Result<(), GpuBackendError> {
        // Rigid bodies: collider poses → instanced shapes.
        if let Some(rbd) = state.rbd.as_ref() {
            // Zero-readback path: a compute kernel writes render data straight
            // into kiss3d's GPU instance buffers, reading the live body-pose
            // buffer directly. No GPU→CPU transfer, no `synchronize`.
            let backend = self.backend().clone();
            if self.rbd_prep_render.is_none() {
                self.rbd_prep_render = WgRbdPrepRender::from_backend(&backend).ok();
            }
            if let Some(shader) = self.rbd_prep_render.as_ref() {
                let body_poses = rbd.body_poses();
                let mut enc = backend.begin_encoding();
                let _ = self
                    .nexus_render
                    .update_instances_direct(&backend, state, body_poses, shader, &mut enc);
                // Body-attached visual meshes (e.g. MJCF `<geom>` visuals)
                // follow the body-origin poses; drawn as 1-instance nodes so
                // they too avoid the readback (their local poses are
                // body-relative, composed onto the body pose in the kernel).
                #[cfg(feature = "dim3")]
                if self.nexus_render.has_visual_nodes() {
                    let _ = self
                        .nexus_render
                        .update_visual_nodes_direct(&backend, state, body_poses, shader, &mut enc);
                }
                let _ = backend.submit(enc);
            }
        }

        Ok(())
    }

    /// Reads the latest state from a [`NexusState`] back from the GPU and pushes
    /// it into the viewer-owned render instances (rigid-body collider poses).
    pub async fn sync(
        &mut self,
        state: &mut NexusState,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        // A backend switch leaves the cached render-prep shaders/buffers bound to
        // the previous device; drop them so they're recompiled/reallocated on the
        // current backend (rebuilt lazily below). Without this, using them on the
        // new backend crashes.
        if self.render_resources_backend != Some(self.ui.backend_type) {
            self.invalidate_render_resources();
            self.render_resources_backend = Some(self.ui.backend_type);
        }

        if !self.direct_render_path() {
            // Synchronize so the on-gpu physics runtime doesn’t pollute the
            // sync time measurements.
            self.backend().synchronize()?;
        }

        let t0 = web_time::Instant::now();
        // Settings: seed the UI from the scene only when a different demo is
        // loaded; on a restart / backend switch (same demo) keep the user's
        // current settings and push them back into the freshly-built scene.
        if self.ui.settings_demo != Some(self.ui.selected_demo) {
            self.ui.has_rbd = state.rbd.is_some();
            self.ui.sim_settings.rbd_steps_per_frame = state.rbd_steps_per_frame();
            self.ui.settings_demo = Some(self.ui.selected_demo);
        } else {
            let s = self.ui.sim_settings.clone();
            state.set_rbd_steps_per_frame(s.rbd_steps_per_frame);
        }

        if self.direct_render_path() {
            self.sync_without_readback(state).await?;
        } else {
            self.sync_with_readback(state).await?;
        }

        self.sync_timestamps(timestamps).await;

        // `pipeline.step` overwrites `run_stats` (with empty pass timings) every
        // frame, but the timestamp readback only completes every few frames.
        // Re-apply the last harvested timings so the profiler UI keeps showing
        // them between readbacks instead of flickering to empty.
        if !self.last_gpu_pass_times.is_empty() {
            state.run_stats.gpu_pass_times = self.last_gpu_pass_times.clone();
            state.run_stats.gpu_total_time_ms = self.last_gpu_total_time_ms;
        }

        self.ui.run_stats = state.run_stats.clone();
        self.ui.sync_time = t0.elapsed();
        self.ui.counts = state.counts();
        Ok(())
    }

    /// The backend currently selected in the UI.
    pub fn backend_type(&self) -> BackendType {
        self.ui.backend_type
    }

    /// Renders the "Compiling shaders…" overlay for a few frames and presents
    /// them. Call right before a blocking pipeline compilation so the banner is
    /// on screen during the freeze — and, on the web, actually composited first:
    /// rendering several real frames forces the browser to paint before the
    /// (blocking, non-yielding) `create_compute_pipeline` (see
    /// `COMPILE_BANNER_PRESENT_FRAMES`).
    pub async fn show_compile_banner(&mut self) {
        for _ in 0..COMPILE_BANNER_PRESENT_FRAMES {
            let _ = self
                .window
                .render(
                    Some(&mut self.scene3d),
                    Some(&mut self.scene2d),
                    Some(&mut self.camera3d),
                    Some(&mut self.camera2d),
                    None,
                    None,
                )
                .await;
            let gpu_available = self.webgpu.is_some();
            self.window.draw_ui(|ctx| {
                crate::ui::setup_custom_theme(ctx);
                crate::ui::main_panel(ctx, &mut self.ui, gpu_available);
                crate::ui::compiling_overlay(ctx);
            });
        }
    }

    /// Drops the cached GPU render-prep shaders and their buffers so they're
    /// rebuilt on the next `sync`. Called when the active backend changes, since
    /// those resources belong to the previous backend's device.
    fn invalidate_render_resources(&mut self) {
        self.rbd_prep_render = None;
    }

    /// Tears down the viewer-owned `NexusState` render nodes. A no-op for legacy
    /// `RbdScene` demos (which detach their own nodes). Call between two runs.
    pub fn clear_scene(&mut self) {
        self.scene3d = SceneNode3d::empty();
        self.scene2d = SceneNode2d::empty();
        self.nexus_render.clear();
    }

    /// Whether the simulation should advance this frame, honoring the
    /// run/pause/step UI state. A pending single-step (`Step`) is consumed: this
    /// returns `true` once and then latches the run state back to `Paused`.
    ///
    /// Examples driving a [`NexusState`] gate their `simulate` call on this, the
    /// way the legacy `RbdScene::simulate` did internally.
    pub fn simulating(&mut self) -> bool {
        match self.ui.run_state {
            RunState::Paused => false,
            RunState::Running => true,
            RunState::Step => {
                self.ui.run_state = RunState::Paused;
                true
            }
        }
    }

    /// Renders one frame of the viewer-owned `NexusState` scene and the UI.
    /// Returns `false` when the loop should end (window closed or a new demo
    /// selected). This is the no-scene-argument counterpart of the legacy
    /// scene-argument `render`.
    pub async fn render_frame(&mut self) -> bool {
        let cont = self
            .window
            .render(
                Some(&mut self.scene3d),
                Some(&mut self.scene2d),
                Some(&mut self.camera3d),
                Some(&mut self.camera2d),
                None,
                None,
            )
            .await;

        if !cont {
            self.ui.transition = Some(Transition::Quit);
            return false;
        }

        let gpu_available = self.webgpu.is_some();
        // Disjoint closure capture (edition 2024): the closure borrows `self.ui`
        // and `scene_ui` while `self.window` is the receiver.
        if self.draw_ui {
            self.window.draw_ui(|ctx| {
                crate::ui::setup_custom_theme(ctx);
                crate::ui::main_panel(ctx, &mut self.ui, gpu_available);
            });
        }

        self.ui.transition.is_none()
    }

    /// Captures the last rendered frame as `(width, height, rgb)`, where `rgb`
    /// is row-major, top-to-bottom, 3 bytes (R, G, B) per pixel.
    ///
    /// This is the off-screen counterpart of the on-window presentation done by
    /// [`Self::render_frame`]: call `render_frame` to draw the scene, then this
    /// to read the framebuffer back to the CPU (e.g. to export a video).
    pub fn snap_rgb(&mut self) -> (u32, u32, Vec<u8>) {
        let img = self.window.snap_image();
        (img.width(), img.height(), img.into_raw())
    }

    /// Pipelined frame capture: completes the capture started by the previous
    /// call (returning *that* frame, one frame late) and starts a non-blocking
    /// capture of the frame just rendered. Returns `None` on the first call.
    /// Unlike [`Self::snap_rgb`] this never stalls waiting for the GPU, since
    /// the copy of frame N is collected only after frame N+1 was rendered.
    /// Call [`Self::snap_rgb_flush`] after the loop to collect the last frame.
    pub fn snap_rgb_async(&mut self) -> Option<(u32, u32, Vec<u8>)> {
        let prev = self.snap_rgb_flush();
        self.window.snap_begin();
        prev
    }

    /// Completes and returns a capture left in flight by
    /// [`Self::snap_rgb_async`], or `None` when there is none.
    pub fn snap_rgb_flush(&mut self) -> Option<(u32, u32, Vec<u8>)> {
        let img = self.window.snap_finish()?;
        Some((img.width(), img.height(), img.into_raw()))
    }

    /// Whether presentation is vsync-locked (windowed viewers only; always
    /// `false` headless).
    pub fn vsync(&self) -> bool {
        self.window.vsync()
    }

    /// Enables/disables vsync on the window's swapchain. With vsync off,
    /// [`Self::render_frame`] no longer blocks on the display refresh —
    /// essential when capturing video faster than real time. No-op headless.
    pub fn set_vsync(&mut self, enabled: bool) {
        self.window.set_vsync(enabled);
    }

    /// Whether [`Self::render_frame`] draws the built-in egui panel (default
    /// `true`). Disable for clean frame capture.
    pub fn set_draw_ui(&mut self, enabled: bool) {
        self.draw_ui = enabled;
    }

    /// Renders one path-traced frame of the viewer-owned scene with kiss3d's
    /// GPU path tracer, instead of the rasterizer used by
    /// [`Self::render_frame`].
    ///
    /// The tracer accumulates samples across calls while the scene and camera
    /// are static, and restarts accumulation automatically when either changes.
    /// Call it several times per video frame (or raise
    /// [`Self::set_raytracer_samples_per_frame`]) to converge before grabbing
    /// the image with [`Self::snap_rgb`]. Returns `false` when the loop
    /// should end (window closed).
    #[cfg(feature = "dim3")]
    pub async fn raytrace_frame(&mut self) -> bool {
        let raytracer = self.raytracer.get_or_insert_with(RayTracer::new);
        let cont = self
            .window
            .raytrace_3d(&mut self.scene3d, &mut self.camera3d, raytracer)
            .await;
        if !cont {
            self.ui.transition = Some(Transition::Quit);
        }
        cont
    }

    /// Number of path-tracing samples accumulated per [`Self::raytrace_frame`]
    /// call (default: chosen by kiss3d from the GPU tier).
    #[cfg(feature = "dim3")]
    pub fn set_raytracer_samples_per_frame(&mut self, samples: u32) {
        self.raytracer
            .get_or_insert_with(RayTracer::new)
            .set_samples_per_frame(samples);
    }

    /// Maximum path-tracing bounce depth.
    #[cfg(feature = "dim3")]
    pub fn set_raytracer_max_bounces(&mut self, bounces: u32) {
        self.raytracer
            .get_or_insert_with(RayTracer::new)
            .set_max_bounces(bounces);
    }

    /// Which intersection backend the path tracer uses: hardware ray queries
    /// (RT cores, when the device supports them) or the portable
    /// compute-shader BVH fallback.
    #[cfg(feature = "dim3")]
    pub fn raytracer_backend(&mut self) -> kiss3d::renderer::RayBackend {
        self.raytracer.get_or_insert_with(RayTracer::new).backend()
    }

    /// [`Self::raytracer_backend`] as a string (`"hardware"` / `"software"`),
    /// for consumers that don't depend on kiss3d directly (e.g. the Python
    /// bindings).
    #[cfg(feature = "dim3")]
    pub fn raytracer_backend_name(&mut self) -> &'static str {
        match self.raytracer_backend() {
            kiss3d::renderer::RayBackend::Hardware => "hardware",
            kiss3d::renderer::RayBackend::Software => "software",
        }
    }

    /// Enables/disables the path tracer's denoiser.
    #[cfg(feature = "dim3")]
    pub fn set_raytracer_denoise(&mut self, enabled: bool) {
        self.raytracer
            .get_or_insert_with(RayTracer::new)
            .set_denoise(enabled);
    }

    /// Reads back environment `env`'s multibody link workspaces from the GPU in
    /// one readback: per link, the generalized joint coordinates, accumulated
    /// joint rotation, world pose, and world-space velocity. Links are in the
    /// GPU build's traversal order (multibodies, then links, parent before
    /// child) — the same order `NexusState::control_multibody_motors` targets.
    /// Empty when no multibody state exists.
    ///
    /// Velocities are only meaningful after the first simulated step (the
    /// forward-kinematics pass fills them); coordinates and poses are valid
    /// from `finalize`.
    pub async fn read_multibody_links(
        &mut self,
        state: &NexusState,
        env: u32,
    ) -> Vec<nexus::rbd::shaders::dynamics::MultibodyLinkWorkspace> {
        let Some(rbd) = state.rbd.as_ref() else {
            return Vec::new();
        };
        let mbs = rbd.multibodies();
        let stride = mbs.links_per_batch() as usize;
        if stride == 0 {
            return Vec::new();
        }
        let mut all = bytemuck::zeroed_vec(mbs.links_workspace().len() as usize);
        if self
            .backend()
            .slow_read_buffer(mbs.links_workspace().buffer(), &mut all)
            .await
            .is_err()
        {
            return Vec::new();
        }
        let start = (env as usize * stride).min(all.len());
        let end = (start + stride).min(all.len());
        all[start..end].to_vec()
    }

    /// Draws example-specific egui widgets into the current frame's UI pass.
    ///
    /// Call this once per frame, after [`Self::render_frame`], to overlay a
    /// custom panel or window (e.g. a model picker) on top of the viewer's
    /// built-in UI. Because kiss3d merges every `draw_ui` call of a frame into
    /// the same egui pass, the widgets composite with the main panel and are
    /// presented on the next `render_frame`.
    pub fn draw_custom_ui(&mut self, ui_fn: impl FnOnce(&kiss3d::egui::Context)) {
        self.window.draw_ui(ui_fn);
    }
}
