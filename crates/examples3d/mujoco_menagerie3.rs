use khal::backend::GpuTimestamps;
use kiss3d::egui;
use nexus_viewer3d::{NexusViewer, RenderMaterial};
use nexus3d::prelude::{NexusPipeline, NexusState};
use rapier3d::prelude::*;
use nexus3d::rbd::dynamics::convert_joint_motor;
use nexus3d::rbd::shaders::dynamics::JointMotor;
use rapier3d_mjcf::{MjcfLoaderOptions, MjcfMultibodyOptions, MjcfRobot, MjcfRobotHandles};
use std::fs;
use std::path::{Path, PathBuf};

/// Root directory the scene-picker walks. Each robot is expected to live in
/// `<root>/<robot>/scene*.xml`. We recommend cloning
/// `https://github.com/google-deepmind/mujoco_menagerie` next to the nexus
/// checkout (so it ends up at `../mujoco_menagerie` relative to the workspace),
/// which is the default resolved from `CARGO_MANIFEST_DIR`. Override with the
/// `MUJOCO_MENAGERIE_DIR` environment variable.
fn menagerie_root() -> PathBuf {
    if let Ok(dir) = std::env::var("MUJOCO_MENAGERIE_DIR") {
        return PathBuf::from(dir);
    }
    // CARGO_MANIFEST_DIR == <workspace>/crates/examples3d
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../mujoco_menagerie")
}

/// Maximum per-multibody DoF count the nexus GPU solver supports. Mirrors
/// `nexus_rbd_shaders::utils::linalg::MAX_MB_DOFS`; `RbdState::from_rapier`
/// panics on any multibody whose `ndofs()` exceeds it.
const MAX_MB_DOFS: u32 = 64;

/// Cheap pre-flight check: build the model's multibodies *without reading any
/// mesh files* (collider/visual shape creation disabled) and return the largest
/// per-multibody DoF count. `None` if the model fails to parse (those are kept
/// in the list — `load_scene` reports the error gracefully instead of crashing).
/// Used to drop models the GPU solver can't handle from the picker.
fn scene_max_dofs(scene: &Path) -> Option<u32> {
    // Same structural options as the real load (so the DoF count matches), but
    // with every collider/visual shape skipped — only bodies and joints, which
    // determine the multibody DoFs, are needed here.
    let options = MjcfLoaderOptions {
        create_colliders_from_collision_shapes: false,
        create_colliders_from_visual_shapes: false,
        make_roots_fixed: false,
        skip_plane_geoms: true,
        ..MjcfLoaderOptions::default()
    };
    let (robot, _) = MjcfRobot::from_file(scene, options).ok()?;

    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut impulse_joints = ImpulseJointSet::new();
    robot.insert_using_multibody_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        &mut impulse_joints,
        MjcfMultibodyOptions::DISABLE_SELF_CONTACTS,
    );

    Some(
        multibody_joints
            .multibodies()
            .map(|mb| mb.ndofs() as u32)
            .max()
            .unwrap_or(0),
    )
}

/// Walk `root` one level deep and collect any `scene*.xml` found in a
/// sub-directory, sorted by path so the listing is stable.
fn discover_scenes(root: &Path) -> Vec<PathBuf> {
    let mut scenes = Vec::new();
    if let Ok(top) = fs::read_dir(root) {
        for entry in top.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            if let Ok(sub) = fs::read_dir(&dir) {
                for sub_entry in sub.flatten() {
                    let path = sub_entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if name.starts_with("scene") && name.ends_with(".xml") {
                        scenes.push(path);
                    }
                }
            }
        }
    }
    scenes.sort();
    scenes
}

/// Short `<robot>/<file>` label for the picker, e.g. `unitree_a1/scene.xml`.
fn scene_label(path: &Path) -> String {
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("?");
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?.xml");
    format!("{parent}/{name}")
}

/// A body-attached visual mesh awaiting registration with the viewer, with all
/// the data needed to render it with full fidelity (color, texture, UVs,
/// normals, PBR material). Collected during loading and registered once the
/// rapier-world borrow has ended.
struct VisualMeshReg {
    body: RigidBodyHandle,
    shape: SharedShape,
    local_pose: Pose,
    color: [f32; 4],
    uvs: Option<Vec<[f32; 2]>>,
    normals: Option<Vec<[f32; 3]>>,
    texture: Option<PathBuf>,
    material: Option<RenderMaterial>,
}
/// Panel state, mirroring the Example Settings of rapier's `mujoco_menagerie3`.
#[derive(Clone, Copy, PartialEq)]
struct Settings {
    use_multibody: bool,
    render_colliders: bool,
    render_visual_meshes: bool,
    render_visual_primitives: bool,
    disable_collisions: bool,
    enable_controls: bool,
    enable_springs: bool,
    actuator_strength: f32,
    /// Index into the keyframe picker: 0 is "(none)", `i + 1` is keyframe `i`.
    keyframe: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            use_multibody: true,
            render_colliders: false,
            render_visual_meshes: true,
            render_visual_primitives: false,
            disable_collisions: true,
            enable_controls: true,
            enable_springs: true,
            actuator_strength: 1.0,
            keyframe: 0,
        }
    }
}

impl Settings {
    /// With the actuators driving the model, switching keyframe retargets the
    /// servos live; without them nothing tracks the target, so the pose has to
    /// be applied by reloading instead.
    fn keyframe_is_live(&self) -> bool {
        self.use_multibody && self.enable_controls
    }

    /// Whether moving from `self` to `next` requires rebuilding the scene.
    /// Actuator strength is read live every step, and so is the keyframe while
    /// the servos are driving.
    fn needs_reload(&self, next: &Self) -> bool {
        self.use_multibody != next.use_multibody
            || self.render_colliders != next.render_colliders
            || self.render_visual_meshes != next.render_visual_meshes
            || self.render_visual_primitives != next.render_visual_primitives
            || self.disable_collisions != next.disable_collisions
            || self.enable_controls != next.enable_controls
            || self.enable_springs != next.enable_springs
            || (self.keyframe != next.keyframe && !next.keyframe_is_live())
    }
}

/// Everything the per-step actuator drive needs. Multibody path only.
struct Controls {
    handles: MjcfRobotHandles<Option<MultibodyJointHandle>>,
    /// One control vector per keyframe, precomputed at load.
    per_keyframe_ctrl: Vec<Vec<Real>>,
    /// Control vector for "(none)": hold the neutral pose.
    neutral: Vec<Real>,
}

/// A loaded model plus the picker state that depends on it.
struct Loaded {
    state: NexusState,
    controls: Option<Controls>,
    /// "(none)" followed by one entry per declared keyframe.
    keyframe_names: Vec<String>,
}

/// Merge the keyframes from a sibling `keyframes.xml` (next to the scene file)
/// into `robot`, skipping any whose name is already present.
///
/// Menagerie models often keep their keyframes in a standalone file meant to be
/// `<include>`d, which the scene itself does not reference; without this they
/// would never reach the picker.
fn merge_sibling_keyframes(robot: &mut MjcfRobot, scene_path: &Path) {
    let Some(kf_path) = scene_path.parent().map(|d| d.join("keyframes.xml")) else {
        return;
    };
    if !kf_path.exists() {
        return;
    }
    match MjcfRobot::from_file(&kf_path, loader_options()) {
        Ok((kf_robot, _)) => {
            let existing: std::collections::HashSet<String> = robot
                .keyframes
                .iter()
                .filter_map(|k| k.name.clone())
                .collect();
            for k in kf_robot.keyframes {
                if k.name.as_ref().is_none_or(|n| !existing.contains(n)) {
                    robot.keyframes.push(k);
                }
            }
        }
        Err(e) => eprintln!(
            "Failed to load sibling keyframes `{}`: {e}.",
            kf_path.display()
        ),
    }
}

/// Picker entries for a model's keyframes, prefixed with "(none)".
fn keyframe_names(robot: &MjcfRobot) -> Vec<String> {
    let mut names = vec!["(none)".to_string()];
    for (i, k) in robot.keyframes.iter().enumerate() {
        names.push(k.name.clone().unwrap_or_else(|| format!("key {i}")));
    }
    names
}

/// The keyframe a freshly picked model starts on: `home` if it declares one,
/// else its first, else "(none)".
fn default_keyframe(names: &[String]) -> usize {
    names
        .iter()
        .position(|n| n == "home")
        .unwrap_or(if names.len() > 1 { 1 } else { 0 })
}

/// The MJCF loader options shared by the pre-flight DoF check and the real load.
fn loader_options() -> MjcfLoaderOptions {
    MjcfLoaderOptions {
        skip_plane_geoms: true,
        make_roots_fixed: false,
        // Surface visual-only geoms as `MjcfBody::visual_meshes` (forwarded to
        // the viewer below) instead of turning them into colliders.
        create_colliders_from_visual_shapes: false,
        // Density 0: the physical mass comes from the model's `<inertial>` tags.
        collider_blueprint: ColliderBuilder::default().density(0.0),
        ..MjcfLoaderOptions::default()
    }
}

/// Loads a single MuJoCo Menagerie MJCF model into a fresh [`NexusState`] under
/// `settings`, registers its render shapes and a floor with `viewer`, frames the
/// camera on it, and finalizes the state ready for simulation.
///
/// The model is kept in its native Z-up frame (no rotation): the viewer is
/// configured Z-up by the caller and gravity is set to -Z below, so MJCF data is
/// consumed as-authored.
async fn load_scene(
    viewer: &mut NexusViewer,
    scene: &Path,
    settings: &Settings,
) -> anyhow::Result<Loaded> {
    let mut state = NexusState::default();

    // Collected during loading, registered once the world borrow ends.
    let mut visual_meshes: Vec<VisualMeshReg> = Vec::new();
    let mut collider_shapes: Vec<(RigidBodyHandle, SharedShape, Pose, bool)> = Vec::new();
    let mut floor: Option<(Vec3, Vec3)> = None;
    let mut camera: Option<(Vec3, Vec3)> = None;
    let mut controls = None;
    let mut names = vec!["(none)".to_string()];
    let mut gravity = -9.81;

    println!("Loading MJCF scene `{}`.", scene.display());
    match MjcfRobot::from_file(scene, loader_options()) {
        Ok((mut robot, model)) => {
            merge_sibling_keyframes(&mut robot, scene);
            names = keyframe_names(&robot);
            let keyframe = settings
                .keyframe
                .checked_sub(1)
                .and_then(|i| robot.keyframes.get(i))
                .cloned();

            // MJCF gives gravity as a 3-vector, normally (0, 0, -9.81) since the
            // format is Z-up. Keep only the magnitude and lock it to -Z so
            // physics and rendering stay aligned whatever the model declares.
            let g = model.option.gravity;
            let mag = ((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) as f32).sqrt();
            gravity = -mag;

            if settings.disable_collisions {
                for link in &mut robot.bodies {
                    for collider in &mut link.colliders {
                        collider.set_collision_groups(InteractionGroups::new(
                            Group::GROUP_1,
                            Group::GROUP_2,
                            Default::default(),
                        ));
                    }
                }
            }

            let mut mb_options = if settings.disable_collisions {
                MjcfMultibodyOptions::DISABLE_SELF_CONTACTS
            } else {
                MjcfMultibodyOptions::default()
            };
            // `<joint stiffness>` passive springs are integrated implicitly by
            // default; unchecking strips them (e.g. cassie's leg springs).
            if !settings.enable_springs {
                mb_options |= MjcfMultibodyOptions::SKIP_JOINT_SPRINGS;
            }

            let world = state.rbd_world_mut(0);
            // `insert_using_*` consumes the robot, so clone it and keep the
            // original around for its visual meshes and keyframes.
            let body_handles: Vec<Option<RigidBodyHandle>> = if settings.use_multibody {
                let handles = robot.clone().insert_using_multibody_joints(
                    &mut world.bodies,
                    &mut world.colliders,
                    &mut world.multibody_joints,
                    &mut world.impulse_joints,
                    mb_options,
                );
                if let Some(key) = &keyframe {
                    handles.apply_keyframe(
                        &mut world.bodies,
                        &mut world.multibody_joints,
                        &robot,
                        key,
                    );
                }
                let bodies = handles
                    .bodies
                    .iter()
                    .map(|b| b.as_ref().map(|h| h.body))
                    .collect();
                if settings.enable_controls {
                    let per_keyframe_ctrl = robot
                        .keyframes
                        .iter()
                        .map(|k| robot.keyframe_controls(k))
                        .collect();
                    let neutral = vec![0.0; handles.actuators.len()];
                    controls = Some(Controls {
                        handles,
                        per_keyframe_ctrl,
                        neutral,
                    });
                }
                bodies
            } else {
                let handles = robot.clone().insert_using_impulse_joints(
                    &mut world.bodies,
                    &mut world.colliders,
                    &mut world.impulse_joints,
                );
                if let Some(key) = &keyframe {
                    handles.apply_keyframe(&mut world.bodies, &robot, key);
                }
                handles.bodies.iter().map(|b| b.as_ref().map(|h| h.body)).collect()
            };

            // Gather each body's render geometry. Visual meshes carry the
            // authored color / texture / UVs; colliders are the fallback for
            // links that declare none.
            for (i, body) in body_handles.iter().enumerate() {
                let Some(body) = *body else { continue };
                let mjcf_body = &robot.bodies[i];
                let visuals: Vec<_> = mjcf_body
                    .visual_meshes
                    .iter()
                    // "Render visual primitives" keeps the capsules and boxes some
                    // models declare in their visual channel; by default only the
                    // .obj-derived meshes are drawn.
                    .filter(|vm| settings.render_visual_primitives || vm.shape.as_trimesh().is_some())
                    .collect();
                let has_visual = !visuals.is_empty();
                for (handle, _) in world
                    .colliders
                    .iter()
                    .filter(|(_, c)| c.parent() == Some(body))
                    .map(|(h, c)| (h, c))
                    .collect::<Vec<_>>()
                {
                    let c = &world.colliders[handle];
                    let local_pose = c.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
                    collider_shapes.push((body, c.shared_shape().clone(), local_pose, has_visual));
                }
                for vm in visuals {
                    let textured = vm.texture.is_some();
                    let color = vm.rgba.unwrap_or(if textured {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [0.7, 0.7, 0.75, 1.0]
                    });
                    visual_meshes.push(VisualMeshReg {
                        body,
                        shape: vm.shape.clone(),
                        local_pose: vm.local_pose,
                        color,
                        uvs: vm.uvs.clone(),
                        normals: vm.normals.clone(),
                        texture: vm.texture.clone(),
                        material: vm.material.map(|m| RenderMaterial {
                            metallic: m.metallic,
                            roughness: m.roughness,
                            reflectance: m.reflectance,
                            emissive: m.emissive,
                        }),
                    });
                }
            }

            // Bounding box of all colliders, in the native Z-up world frame:
            // drives both the floor placement and the camera framing.
            let mut aabb = Aabb::new_invalid();
            for (_, collider) in world.colliders.iter() {
                aabb.merge(&collider.compute_aabb());
            }
            if aabb.mins.x <= aabb.maxs.x {
                let center = aabb.center();
                let he = aabb.half_extents();
                let footprint = he.x.max(he.y).max(0.5);
                let floor_thick = 0.1;
                floor = Some((
                    Vec3::new(center.x, center.y, center.z - he.z - floor_thick),
                    Vec3::new(footprint * 6.0, footprint * 6.0, floor_thick),
                ));
                let radius = (he.x * he.x + he.y * he.y + he.z * he.z).sqrt().max(0.5);
                let target = Vec3::new(center.x, center.y, center.z);
                camera = Some((target + Vec3::new(radius * 2.2, -radius * 2.2, radius * 1.6), target));
            }
        }
        Err(e) => eprintln!("Failed to load MJCF scene `{}`: {e}.", scene.display()),
    }

    if let Some((center, he)) = floor {
        let body = RigidBodyBuilder::fixed().translation(center).build();
        let collider = ColliderBuilder::cuboid(he.x, he.y, he.z).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);
    }

    if settings.render_colliders || !settings.render_visual_meshes {
        for (body, shape, local_pose, _) in &collider_shapes {
            viewer.insert_visual_shape(0, *body, shape, *local_pose);
        }
    } else {
        for vm in &visual_meshes {
            viewer.insert_visual_mesh(
                0,
                vm.body,
                &vm.shape,
                vm.local_pose,
                vm.color,
                vm.uvs.as_deref(),
                vm.normals.as_deref(),
                vm.texture.as_deref(),
                vm.material,
            );
        }
        // Links with no visual mesh would otherwise be invisible.
        for (body, shape, local_pose, has_visual) in &collider_shapes {
            if !has_visual {
                viewer.insert_visual_shape(0, *body, shape, *local_pose);
            }
        }
    }

    if let Some((eye, target)) = camera {
        viewer.set_camera(eye, target);
    }
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(-1.0, 1.0, -1.0));

    // The impulse-joint path needs a much finer step to stay stable; the
    // multibody path instead raises the PGS iterations per substep. Mirrors the
    // reference example.
    let mut sim_params = nexus3d::rbd::shaders::dynamics::RbdSimParams::default();
    if !settings.use_multibody {
        sim_params.dt = 1.0 / 240.0;
        sim_params.num_solver_iterations = 12;
    }
    state.set_rbd_sim_params(0, sim_params);

    state.finalize(viewer.backend()).await?;
    state.set_rbd_gravity(viewer.backend(), [0.0, 0.0, gravity]);
    if let Some(rbd) = state.rbd.as_mut() {
        if settings.use_multibody {
            rbd.multibodies_mut().set_num_internal_pgs_iterations(4);
        }
        // MuJoCo-style explicit coriolis: a single plain mass matrix, with
        // coriolis / gyroscopic forces applied explicitly on the rhs.
        rbd.set_implicit_coriolis(viewer.backend(), false);
    }
    Ok(Loaded {
        state,
        controls,
        keyframe_names: names,
    })
}

/// Drives the model's actuators toward `ctrl`, scaled by `gain`.
///
/// The motor configuration is baked into the GPU state at finalization, so this
/// runs the MJCF actuator model on the CPU-side joints and then pushes each
/// touched motor across.
fn apply_controls(
    state: &mut NexusState,
    backend: &khal::backend::GpuBackend,
    controls: &Controls,
    ctrl: &[Real],
    gain: Real,
) {
    let mut updates: Vec<(u32, usize, JointMotor)> = Vec::new();
    {
        // Untracked: the rapier sets are only the scratch the MJCF actuator
        // model writes into. Marking them dirty would rebuild the GPU buffers
        // from the authored poses and reset the model every step.
        let world = state.rbd_world_mut_untracked(0);
        controls.handles.apply_controls_multibody_scaled(
            &mut world.bodies,
            &mut world.multibody_joints,
            ctrl,
            gain,
        );
        for ah in &controls.handles.actuators {
            let Some(Some(handle)) = ah.joint else { continue };
            let Some((mb, link_id)) = world.multibody_joints.get(handle) else {
                continue;
            };
            let Some(link) = mb.links().nth(link_id) else { continue };
            // The GPU link id is the body index (see `GpuMultibodySet::set_motor`).
            let body_idx = link.rigid_body_handle().into_raw_parts().0;
            let axes = link.joint().data.motor_axes.bits();
            for axis in 0..6 {
                if axes & (1 << axis) != 0 {
                    updates.push((
                        body_idx,
                        axis,
                        convert_joint_motor(link.joint().data.motors[axis]),
                    ));
                }
            }
        }
    }
    if let Some(rbd) = state.rbd.as_mut() {
        let _ = rbd.multibodies_mut().set_motors(backend, 0, &updates);
    }
}

/// Picks a scene: first runs the cheap DoF pre-check (no mesh I/O); if the model
/// is within the GPU solver's DoF cap it tears down the current scene and loads
/// it. If it exceeds the cap, nothing is loaded and an `Err(message)` is
/// returned for display in the picker.
async fn select_scene(
    viewer: &mut NexusViewer,
    scene: &Path,
    settings: &Settings,
) -> anyhow::Result<Result<Loaded, String>> {
    if let Some(dofs) = scene_max_dofs(scene)
        && dofs > MAX_MB_DOFS
        && settings.use_multibody
    {
        return Ok(Err(format!(
            "{} needs {dofs} DoFs (max {MAX_MB_DOFS}) — not supported by the GPU solver.",
            scene_label(scene)
        )));
    }
    viewer.clear_scene();
    Ok(Ok(load_scene(viewer, scene, settings).await?))
}

/// Loads MuJoCo Menagerie MJCF models and simulates them on the GPU rigid-body
/// pipeline, with a floating egui window carrying the same controls as rapier's
/// `mujoco_menagerie3` example: model picker, render modes, collision / spring /
/// actuator toggles, a keyframe picker and a live actuator-strength slider.
///
/// Scenes are discovered under `MUJOCO_MENAGERIE_DIR` (default:
/// `../mujoco_menagerie` next to the workspace). The initial model is the one
/// matching `MUJOCO_MENAGERIE_SCENE` (default: `unitree_a1`), or the first
/// discovered scene otherwise.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let root = menagerie_root();
    let scenes = discover_scenes(&root);
    let labels: Vec<String> = scenes.iter().map(|p| scene_label(p)).collect();

    if scenes.is_empty() {
        eprintln!(
            "No MuJoCo Menagerie scenes found under `{}`.\n\
             Clone `google-deepmind/mujoco_menagerie` there, or point the\n\
             `MUJOCO_MENAGERIE_DIR` environment variable at your copy.",
            root.display()
        );
    } else {
        println!("Discovered {} MuJoCo Menagerie scene(s).", scenes.len());
    }

    let wanted = std::env::var("MUJOCO_MENAGERIE_SCENE").unwrap_or_else(|_| "unitree_a1".into());
    let mut selected = scenes
        .iter()
        .position(|p| p.to_string_lossy().contains(&wanted))
        .unwrap_or(0);

    // MJCF models are Z-up: orient the viewer's camera accordingly so the model
    // stands upright without rotating its data.
    viewer.set_up_axis(Vec3::Z);

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    let mut error: Option<String> = None;
    let mut settings = Settings::default();
    let mut controls = None;
    let mut keyframe_names = vec!["(none)".to_string()];

    let mut state = match scenes.get(selected) {
        Some(scene) => {
            // A freshly picked model starts on its default keyframe, which is
            // only known once it is loaded: probe the names, then load for real.
            match select_scene(viewer, scene, &settings).await? {
                Ok(loaded) => {
                    settings.keyframe = default_keyframe(&loaded.keyframe_names);
                    keyframe_names = loaded.keyframe_names;
                    if settings.keyframe != 0 {
                        let reloaded = select_scene(viewer, scene, &settings).await?;
                        match reloaded {
                            Ok(l) => {
                                controls = l.controls;
                                l.state
                            }
                            Err(msg) => {
                                error = Some(msg);
                                let mut s = NexusState::default();
                                s.finalize(viewer.backend()).await?;
                                s
                            }
                        }
                    } else {
                        controls = loaded.controls;
                        loaded.state
                    }
                }
                Err(msg) => {
                    eprintln!("{msg}");
                    error = Some(msg);
                    let mut s = NexusState::default();
                    s.finalize(viewer.backend()).await?;
                    s
                }
            }
        }
        None => {
            let mut s = NexusState::default();
            s.finalize(viewer.backend()).await?;
            s
        }
    };

    // Requested through the picker this frame, applied after the UI pass so we
    // never rebuild the scene mid-borrow.
    let mut pending_scene: Option<usize> = None;
    let mut pending_settings: Option<Settings> = None;

    while viewer.render_frame().await {
        {
            let current = selected;
            let labels = &labels;
            let names = &keyframe_names;
            let now = settings;
            let pending_scene = &mut pending_scene;
            let pending_settings = &mut pending_settings;
            let error = error.as_deref();
            let count = labels.len();
            viewer.draw_custom_ui(move |ctx| {
                egui::Window::new("MuJoCo Menagerie")
                    .default_pos([24.0, 220.0])
                    .resizable(true)
                    .show(ctx, |ui| {
                        let mut next = now;
                        ui.checkbox(&mut next.use_multibody, "Use multibody joints");
                        ui.checkbox(&mut next.render_colliders, "Render colliders");
                        ui.checkbox(&mut next.render_visual_meshes, "Render visual meshes");
                        ui.checkbox(
                            &mut next.render_visual_primitives,
                            "Render visual primitives",
                        );
                        ui.checkbox(&mut next.disable_collisions, "Disable collisions");
                        ui.checkbox(&mut next.enable_controls, "Enable joint controls");
                        ui.checkbox(&mut next.enable_springs, "Enable joint springs");
                        ui.add(
                            egui::Slider::new(&mut next.actuator_strength, 0.02..=2.0)
                                .text("Actuator strength"),
                        );
                        ui.horizontal(|ui| {
                            ui.label("Keyframe");
                            // Prev / next step through the model's keyframes,
                            // wrapping at either end, like the scene picker.
                            let n = names.len().max(1);
                            if ui.button("<").clicked() {
                                next.keyframe = (next.keyframe + n - 1) % n;
                            }
                            if ui.button(">").clicked() {
                                next.keyframe = (next.keyframe + 1) % n;
                            }
                            egui::ComboBox::from_id_salt("keyframe")
                                .selected_text(
                                    names
                                        .get(next.keyframe)
                                        .cloned()
                                        .unwrap_or_else(|| "(none)".into()),
                                )
                                .show_ui(ui, |ui| {
                                    for (i, name) in names.iter().enumerate() {
                                        ui.selectable_value(&mut next.keyframe, i, name);
                                    }
                                });
                        });
                        if next != now {
                            *pending_settings = Some(next);
                        }

                        ui.separator();
                        if count > 0 {
                            ui.horizontal(|ui| {
                                if ui.button("<").clicked() {
                                    *pending_scene = Some((current + count - 1) % count);
                                }
                                if ui.button(">").clicked() {
                                    *pending_scene = Some((current + 1) % count);
                                }
                                ui.label(format!("{}/{}", current + 1, count));
                            });
                        }
                        if let Some(msg) = error {
                            ui.colored_label(egui::Color32::RED, msg);
                        }
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .max_height(420.0)
                            .show(ui, |ui| {
                                for (i, label) in labels.iter().enumerate() {
                                    if ui.selectable_label(current == i, label).clicked() {
                                        *pending_scene = Some(i);
                                    }
                                }
                            });
                    });
            });
        }

        // A settings change either retargets the running sim (actuator strength,
        // and the keyframe while the servos are driving) or rebuilds the scene.
        let mut reload = false;
        if let Some(next) = pending_settings.take() {
            reload = settings.needs_reload(&next);
            settings = next;
        }
        // A model change always rebuilds, and resets the keyframe to the new
        // model's default.
        let scene_changed = if let Some(i) = pending_scene.take() {
            if i != selected {
                selected = i;
                settings.keyframe = 0;
                reload = true;
                true
            } else {
                false
            }
        } else {
            false
        };

        if reload && let Some(scene) = scenes.get(selected) {
            match select_scene(viewer, scene, &settings).await? {
                Ok(loaded) => {
                    keyframe_names = loaded.keyframe_names;
                    error = None;
                    if scene_changed {
                        // Only known now that the model is loaded; re-load once
                        // so it actually starts in that pose.
                        let def = default_keyframe(&keyframe_names);
                        if def != settings.keyframe {
                            settings.keyframe = def;
                            match select_scene(viewer, scene, &settings).await? {
                                Ok(l) => {
                                    keyframe_names = l.keyframe_names;
                                    controls = l.controls;
                                    state = l.state;
                                }
                                Err(msg) => {
                                    eprintln!("{msg}");
                                    error = Some(msg);
                                }
                            }
                        } else {
                            controls = loaded.controls;
                            state = loaded.state;
                        }
                    } else {
                        controls = loaded.controls;
                        state = loaded.state;
                    }
                }
                Err(msg) => {
                    eprintln!("{msg}");
                    error = Some(msg);
                }
            }
        }

        if viewer.simulating() {
            if let Some(controls) = controls.as_ref() {
                let ctrl = settings
                    .keyframe
                    .checked_sub(1)
                    .and_then(|i| controls.per_keyframe_ctrl.get(i))
                    .unwrap_or(&controls.neutral);
                apply_controls(
                    &mut state,
                    viewer.backend(),
                    controls,
                    ctrl,
                    settings.actuator_strength,
                );
            }
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    // Restore the default Y-up convention so the next demo (the viewer is reused
    // across demos) isn't left with this demo's Z-up camera.
    viewer.set_up_axis(Vec3::Y);

    Ok(state)
}
