//! Offscreen sensor cameras.
//!
//! A [`SensorCamera`] renders the viewer's 3D scene on demand into its own
//! [`OffscreenSurface`] (so its resolution is independent of the window), from
//! a [`SensorCamera3d`] whose pose is set explicitly, or follows a rigid body
//! when attached. Besides the shaded RGB image it exposes kiss3d's auxiliary
//! passes: linear metric depth and per-object segmentation ids.
//!
//! Sensors are rendering concerns, so they live in the viewer: the physics
//! state only provides body poses (read back in `NexusViewer::sync`) that the
//! attachment composes with the mount pose.

use glamx::glam::camera::rh::proj::opengl;
use glamx::{Mat4, Pose3, Vec3};
use kiss3d::camera::Camera3d;
use kiss3d::event::WindowEvent;
use kiss3d::prelude::Color;
use kiss3d::scene::SceneNode3d;
use kiss3d::window::{Canvas, NumSamples, OffscreenSurface};
use nexus::rbd::math::Pose;
use rapier::data::Index;

/// A camera with an explicit pose and fixed intrinsics: a vertical field of
/// view, near/far planes, and the aspect ratio of the surface it renders to.
/// It ignores window events; its owner positions it.
///
/// The pose uses the OpenGL camera convention (the camera looks down its local
/// -Z axis with +Y up), the frame `Pose3::look_at_rh` inverts.
pub struct SensorCamera3d {
    pose: Pose3,
    fov_y: f32,
    znear: f32,
    zfar: f32,
    aspect: f32,
    view: Pose3,
    proj: Mat4,
    proj_view: Mat4,
    inv_proj_view: Mat4,
}

impl SensorCamera3d {
    /// A camera at the identity pose. `fov_y` is in radians.
    pub fn new(fov_y: f32, znear: f32, zfar: f32, aspect: f32) -> Self {
        let mut cam = Self {
            pose: Pose3::IDENTITY,
            fov_y,
            znear,
            zfar,
            aspect,
            view: Pose3::IDENTITY,
            proj: Mat4::IDENTITY,
            proj_view: Mat4::IDENTITY,
            inv_proj_view: Mat4::IDENTITY,
        };
        cam.refresh();
        cam
    }

    /// Sets the camera pose (OpenGL convention, see the type docs).
    pub fn set_pose(&mut self, pose: Pose3) {
        self.pose = pose;
        self.refresh();
    }

    /// The camera pose (OpenGL convention).
    pub fn pose(&self) -> Pose3 {
        self.pose
    }

    /// Vertical field of view, in radians.
    pub fn fov_y(&self) -> f32 {
        self.fov_y
    }

    fn refresh(&mut self) {
        self.view = self.pose.inverse();
        self.proj = opengl::perspective(self.fov_y, self.aspect, self.znear, self.zfar);
        self.proj_view = self.proj * self.view.to_mat4();
        self.inv_proj_view = self.proj_view.inverse();
    }
}

impl Camera3d for SensorCamera3d {
    fn handle_event(&mut self, _canvas: &Canvas, _event: &WindowEvent) {}

    fn eye(&self) -> Vec3 {
        self.pose.translation
    }

    fn view_transform(&self) -> Pose3 {
        self.view
    }

    fn transformation(&self) -> Mat4 {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Mat4 {
        self.inv_proj_view
    }

    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    fn update(&mut self, canvas: &Canvas) {
        let (w, h) = canvas.size();
        let aspect = w as f32 / h.max(1) as f32;
        if aspect != self.aspect {
            self.aspect = aspect;
            self.refresh();
        }
    }

    fn view_transform_pair(&self, _pass: usize) -> (Pose3, Mat4) {
        (self.view, self.proj)
    }
}

/// A rigid body a sensor camera follows: `(env, body handle, mount pose)`. The
/// camera pose is `body_pose * mount` after every viewer sync.
#[derive(Copy, Clone, Debug)]
pub struct SensorAttachment {
    pub env: u32,
    pub handle: Index,
    pub local_pose: Pose,
}

/// An offscreen sensor camera: its own render surface plus a fixed-intrinsics
/// camera, optionally attached to a rigid body.
pub struct SensorCamera {
    surface: OffscreenSurface,
    camera: SensorCamera3d,
    width: u32,
    height: u32,
    attachment: Option<SensorAttachment>,
    /// Scene generation the camera belongs to; only the active generation's
    /// attached cameras follow their bodies at sync.
    pub generation: u32,
}

impl SensorCamera {
    /// Creates the offscreen surface (sharing the window's GPU context) and a
    /// camera with the given vertical field of view (radians) and clip planes.
    pub async fn new(width: u32, height: u32, fov_y: f32, znear: f32, zfar: f32) -> Self {
        let surface = OffscreenSurface::new(width, height).await;
        let camera = SensorCamera3d::new(fov_y, znear, zfar, width as f32 / height.max(1) as f32);
        Self {
            surface,
            camera,
            width,
            height,
            attachment: None,
            generation: 0,
        }
    }

    /// Image size `(width, height)` in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Vertical field of view, in radians.
    pub fn fov_y(&self) -> f32 {
        self.camera.fov_y()
    }

    /// Sets the camera pose (OpenGL convention). Overridden at the next sync
    /// while the camera is attached to a body.
    pub fn set_pose(&mut self, pose: Pose) {
        self.camera.set_pose(pose);
    }

    /// The camera pose (OpenGL convention).
    pub fn pose(&self) -> Pose {
        self.camera.pose()
    }

    /// Makes the camera follow a rigid body with a fixed mount pose.
    pub fn attach(&mut self, env: u32, handle: Index, local_pose: Pose) {
        self.attachment = Some(SensorAttachment {
            env,
            handle,
            local_pose,
        });
    }

    /// Stops following a body; the camera keeps its current pose.
    pub fn detach(&mut self) {
        self.attachment = None;
    }

    /// The body this camera follows, if any.
    pub fn attachment(&self) -> Option<SensorAttachment> {
        self.attachment
    }

    /// Ambient light level of the shaded render.
    pub fn set_ambient(&mut self, ambient: f32) {
        self.surface.set_ambient(ambient);
    }

    /// Background color (RGBA) of the shaded render.
    pub fn set_background_color(&mut self, rgba: [f32; 4]) {
        self.surface
            .set_background_color(Color::new(rgba[0], rgba[1], rgba[2], rgba[3]));
    }

    /// MSAA sample count of the shaded render (`1` disables antialiasing;
    /// kiss3d supports 1 and 4, other values round down to 1).
    pub fn set_samples(&mut self, samples: u32) {
        let samples = NumSamples::from_u32(samples).unwrap_or(NumSamples::One);
        self.surface.set_samples(samples);
    }

    /// Shadow-edge softness of the shaded render: `0.0` hard edges, `1.0`
    /// kiss3d's default PCF penumbra.
    pub fn set_shadow_softness(&mut self, softness: f32) {
        self.surface.set_shadow_softness(softness);
    }

    /// Shadow map resolution (texels per atlas layer, square) and the number
    /// of atlas layers to allocate (one directional light needs four).
    pub fn set_shadow_resolution(&mut self, resolution: u32, layers: u32) {
        self.surface.set_shadow_atlas_layers(layers);
        self.surface.set_shadow_resolution(resolution);
    }

    /// Directional-shadow cascade layout of the shaded render: the
    /// highest-resolution cascade covers the camera's first
    /// `first_cascade_far_bound` meters and shadows stop at `shadow_distance`
    /// (`INFINITY` = the camera far plane).
    pub fn set_shadow_range(&mut self, first_cascade_far_bound: f32, shadow_distance: f32) {
        self.surface
            .set_first_cascade_far_bound(first_cascade_far_bound);
        self.surface.set_shadow_distance(shadow_distance);
    }

    /// Renders the shaded scene and returns it as row-major, top-left origin
    /// RGB bytes (`width * height * 3`).
    pub async fn render_rgb(&mut self, scene: &mut SceneNode3d) -> Vec<u8> {
        self.surface.render_3d(scene, &mut self.camera).await;
        self.surface.snap_image().into_raw()
    }

    /// Renders linear eye-space depth in world units, row-major with a top-left
    /// origin; background pixels are `0.0`.
    pub fn render_depth(&mut self, scene: &mut SceneNode3d) -> Vec<f32> {
        self.surface.snap_depth_raw(scene, &mut self.camera)
    }

    /// Renders the per-pixel segmentation id (`0` for background), row-major
    /// with a top-left origin. Ids are the objects' `segmentation_id`s, which
    /// the viewer sets per body.
    pub fn render_segmentation(&mut self, scene: &mut SceneNode3d) -> Vec<u32> {
        self.surface.snap_segmentation(scene, &mut self.camera)
    }
}
