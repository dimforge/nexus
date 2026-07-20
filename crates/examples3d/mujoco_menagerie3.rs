use khal::backend::GpuTimestamps;
use kiss3d::egui;
use nexus_viewer3d::{NexusViewer, RenderMaterial};
use nexus3d::prelude::{NexusPipeline, NexusState};
use rapier3d::prelude::*;
use rapier3d_mjcf::{MjcfLoaderOptions, MjcfMultibodyOptions, MjcfRobot};
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

/// Loads a single MuJoCo Menagerie MJCF model into a fresh [`NexusState`],
/// registers its render shapes and a floor with `viewer`, frames the camera on
/// it, and finalizes the state ready for simulation.
///
/// The model is kept in its native Z-up frame (no rotation): the viewer is
/// configured Z-up by the caller and the rigid-body gravity is set to -Z below,
/// so MJCF data is consumed as-authored.
async fn load_scene(
    viewer: &mut NexusViewer,
    scene: &Path,
    render_colliders: bool,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    // `<geom>` collision shapes get density 0 — the physical mass comes from the
    // model's `<inertial>` tags. Roots stay dynamic so free-based robots fall
    // and land on the floor; set `make_roots_fixed: true` to anchor them.
    let options = MjcfLoaderOptions {
        skip_plane_geoms: true,
        make_roots_fixed: false,
        // Surface visual-only geoms as `MjcfBody::visual_meshes` (forwarded to
        // the viewer below) instead of turning them into colliders.
        create_colliders_from_visual_shapes: false,
        collider_blueprint: ColliderBuilder::default().density(0.0),
        // No `shift`: the model stays in its native MJCF Z-up frame. The viewer
        // is set Z-up and gravity points -Z, so nothing needs rotating.
        ..MjcfLoaderOptions::default()
    };

    // Collected during loading, registered once the world borrow ends. Both the
    // visual meshes (textured/PBR) and every collision collider are gathered so
    // the render mode can be chosen at registration time. Each collider entry is
    // tagged with whether its body has visual meshes, so the visual-mesh mode can
    // still fall back to colliders for links that have none.
    let mut visual_meshes: Vec<VisualMeshReg> = Vec::new();
    let mut collider_shapes: Vec<(RigidBodyHandle, SharedShape, Pose, bool)> = Vec::new();
    // Fixed cuboid floor, sized from the loaded model's bounding box.
    let mut floor: Option<(Vec3, Vec3)> = None;
    // Camera framing for the loaded model: `(eye, target)`.
    let mut camera: Option<(Vec3, Vec3)> = None;

    println!("Loading MJCF scene `{}`.", scene.display());
    match MjcfRobot::from_file(scene, options) {
        Ok((robot, _model)) => {
            // Every `<geom>` collider is kept as-is. Collider-less links need no
            // placeholder: the GPU pipeline now gives every body its own slot.
            let world = state.rbd_world_mut(0);
            // `insert_using_multibody_joints` consumes the robot, so clone
            // it and keep the original around for its visual meshes.
            let handles = robot.clone().insert_using_multibody_joints(
                &mut world.bodies,
                &mut world.colliders,
                &mut world.multibody_joints,
                &mut world.impulse_joints,
                MjcfMultibodyOptions::DISABLE_SELF_CONTACTS,
            );

            // Activate the model's actuators so position-servo robots hold their
            // pose instead of folding under gravity (e.g. anymal_c's
            // `<position kp=100>` joint servos). A zero control vector targets the
            // neutral/rest pose — the same as the rapier testbed's "Enable joint
            // controls" (which applies `ctrl = 0` every frame). The actuator motor
            // config (target + stiffness) is static for a constant `ctrl`, so we
            // configure it once here, before `finalize` bakes the multibody into
            // the GPU state. Without this, `<position>`-actuated models collapse.
            let ctrl = vec![0.0; handles.actuators.len()];
            handles.apply_controls_multibody(&mut world.bodies, &mut world.multibody_joints, &ctrl);

            // Forward each body's visual meshes to the viewer; for bodies
            // without visual meshes, render their collision shapes instead.
            for (i, body_handle) in handles.bodies.iter().enumerate() {
                let Some(body_handle) = body_handle else {
                    continue;
                };
                let mjcf_body = &robot.bodies[i];
                let has_visual = !mjcf_body.visual_meshes.is_empty();
                // Collect every collision collider at its body-local pose (a body
                // can own several now), tagged with whether the body has visuals.
                for collider in &body_handle.colliders {
                    let c = &world.colliders[collider.handle];
                    let local_pose = c.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
                    collider_shapes.push((
                        body_handle.body,
                        c.shared_shape().clone(),
                        local_pose,
                        has_visual,
                    ));
                }
                if has_visual {
                    for vm in &mjcf_body.visual_meshes {
                        // Resolve the base color: geom/material rgba if set,
                        // white behind a texture (so it shows in native colors),
                        // else a neutral grey. Mirrors rapier's testbed.
                        let textured = vm.texture.is_some();
                        let color = vm.rgba.unwrap_or(if textured {
                            [1.0, 1.0, 1.0, 1.0]
                        } else {
                            [0.7, 0.7, 0.75, 1.0]
                        });
                        let material = vm.material.map(|m| RenderMaterial {
                            metallic: m.metallic,
                            roughness: m.roughness,
                            reflectance: m.reflectance,
                            emissive: m.emissive,
                        });
                        visual_meshes.push(VisualMeshReg {
                            body: body_handle.body,
                            shape: vm.shape.clone(),
                            local_pose: vm.local_pose,
                            color,
                            uvs: vm.uvs.clone(),
                            normals: vm.normals.clone(),
                            texture: vm.texture.clone(),
                            material,
                        });
                    }
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

                // A wide, thin floor just below the model (Z is up, so it's thin
                // on Z and sits at the model's lowest Z).
                let floor_thick = 0.1;
                let floor_he = Vec3::new(footprint * 6.0, footprint * 6.0, floor_thick);
                let floor_center = Vec3::new(center.x, center.y, center.z - he.z - floor_thick);
                floor = Some((floor_center, floor_he));

                // Frame the model from a 3/4 view (Z up, so the elevation is +Z).
                let radius = (he.x * he.x + he.y * he.y + he.z * he.z).sqrt().max(0.5);
                let target = Vec3::new(center.x, center.y, center.z);
                let eye = target + Vec3::new(radius * 2.2, -radius * 2.2, radius * 1.6);
                camera = Some((eye, target));
            }
        }
        Err(e) => {
            eprintln!("Failed to load MJCF scene `{}`: {e}.", scene.display());
        }
    }

    // Floor (inserted through `NexusState` so it participates in the GPU sim and
    // gets a render shape registered).
    if let Some((center, he)) = floor {
        let body = RigidBodyBuilder::fixed().translation(center).build();
        let collider = ColliderBuilder::cuboid(he.x, he.y, he.z).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);
    }

    if render_colliders {
        // Collider view: every collision shape (instanced, colored by shape
        // type), at its body-local pose. No visual meshes.
        for (body, shape, local_pose, _) in &collider_shapes {
            viewer.insert_visual_shape(0, *body, shape, *local_pose);
        }
    } else {
        // Visual-mesh view: the authored color/texture/UVs/normals/PBR meshes —
        // rendered the way MuJoCo's own viewer shows them — plus colliders only
        // for links that have no visual mesh (so nothing is invisible).
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

    state.finalize(viewer.backend()).await?;
    // MJCF is Z-up: gravity points along -Z (set after `finalize`, which builds
    // the rigid-body state with the default -Y gravity).
    state.set_rbd_gravity(viewer.backend(), [0.0, 0.0, -9.81]);
    // MuJoCo-style explicit coriolis: the mass matrix / LU / gravity solve
    // runs once per step instead of once per substep. This matches how
    // MuJoCo integrates these models and saves ~25% of the step time.
    if let Some(rbd) = state.rbd.as_mut() {
        rbd.multibodies_mut().set_implicit_coriolis(false);
    }
    Ok(state)
}

/// Picks a scene: first runs the cheap DoF pre-check (no mesh I/O); if the model
/// is within the GPU solver's DoF cap it tears down the current scene and loads
/// it, returning `Ok(state)`. If it exceeds the cap, nothing is loaded and an
/// `Err(message)` is returned for display in the picker.
async fn select_scene(
    viewer: &mut NexusViewer,
    scene: &Path,
    render_colliders: bool,
) -> anyhow::Result<Result<NexusState, String>> {
    if let Some(dofs) = scene_max_dofs(scene)
        && dofs > MAX_MB_DOFS
    {
        return Ok(Err(format!(
            "{} needs {dofs} DoFs (max {MAX_MB_DOFS}) — not supported by the GPU solver.",
            scene_label(scene)
        )));
    }
    viewer.clear_scene();
    let state = load_scene(viewer, scene, render_colliders).await?;
    Ok(Ok(state))
}

/// Loads MuJoCo Menagerie MJCF models and simulates them on the GPU rigid-body
/// pipeline, with a floating egui window to switch between the discovered models
/// at runtime (mirroring the scene picker in rapier's `mujoco_menagerie3`
/// example).
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

    // Pick the initial scene (substring match), defaulting to unitree_a1, and
    // falling back to the first discovered scene otherwise.
    let wanted = std::env::var("MUJOCO_MENAGERIE_SCENE").unwrap_or_else(|_| "unitree_a1".into());
    let mut selected = scenes
        .iter()
        .position(|p| p.to_string_lossy().contains(&wanted))
        .unwrap_or(0);

    // MJCF models are Z-up: orient the viewer's camera accordingly so the model
    // stands upright without rotating its data. Done once; preserved across the
    // per-model `set_camera` calls in `load_scene`.
    viewer.set_up_axis(Vec3::Z);

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);

    // Red message shown in the picker when the highlighted model can't be loaded
    // (currently: too many DoFs for the GPU solver).
    let mut error: Option<String> = None;
    // Render mode: false = textured visual meshes (default), true = the collision
    // shapes. Toggled via the picker checkbox; a change reloads the scene.
    let mut render_colliders = false;

    let mut state = match scenes.get(selected) {
        Some(scene) => match select_scene(viewer, scene, render_colliders).await? {
            Ok(state) => state,
            Err(msg) => {
                eprintln!("{msg}");
                error = Some(msg);
                let mut state = NexusState::default();
                state.finalize(viewer.backend()).await?;
                state
            }
        },
        None => {
            let mut state = NexusState::default();
            state.finalize(viewer.backend()).await?;
            state
        }
    };

    // Model selection requested through the picker this frame, applied after the
    // UI pass so we don't rebuild the scene mid-borrow.
    let mut pending: Option<usize> = None;
    // Render-mode change requested through the picker this frame.
    let mut pending_mode: Option<bool> = None;

    while viewer.render_frame().await {
        // Floating model-picker window (in addition to the viewer's main panel).
        if !labels.is_empty() {
            let current = selected;
            let labels = &labels;
            let pending = &mut pending;
            let error = error.as_deref();
            let count = labels.len();
            let render_colliders_now = render_colliders;
            let pending_mode = &mut pending_mode;
            viewer.draw_custom_ui(move |ctx| {
                egui::Window::new("MuJoCo Menagerie")
                    .default_pos([24.0, 220.0])
                    .resizable(true)
                    .show(ctx, |ui| {
                        // Render-mode toggle: visual meshes (default) vs colliders.
                        let mut rc = render_colliders_now;
                        if ui.checkbox(&mut rc, "Render colliders").changed() {
                            *pending_mode = Some(rc);
                        }
                        ui.separator();
                        // Previous / next buttons cycle through the models,
                        // wrapping around at either end.
                        ui.horizontal(|ui| {
                            if ui.button("<").clicked() {
                                *pending = Some((current + count - 1) % count);
                            }
                            if ui.button(">").clicked() {
                                *pending = Some((current + 1) % count);
                            }
                            ui.label(format!("{}/{}", current + 1, count));
                        });
                        // Red error for an unsupported (e.g. too-many-DoF) model.
                        if let Some(msg) = error {
                            ui.colored_label(egui::Color32::RED, msg);
                        }
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .max_height(420.0)
                            .show(ui, |ui| {
                                for (i, label) in labels.iter().enumerate() {
                                    if ui.selectable_label(current == i, label).clicked() {
                                        *pending = Some(i);
                                    }
                                }
                            });
                    });
            });
        }

        // Apply a model selection: the highlight moves immediately (so prev/next
        // can step past an unsupported model), but the scene is only rebuilt when
        // the model is within the GPU solver's DoF cap — otherwise the current
        // scene stays and the picker shows a red error.
        if let Some(i) = pending.take()
            && i != selected
        {
            selected = i;
            match select_scene(viewer, &scenes[selected], render_colliders).await? {
                Ok(new_state) => {
                    state = new_state;
                    error = None;
                }
                Err(msg) => {
                    eprintln!("{msg}");
                    error = Some(msg);
                }
            }
        }

        // Apply a render-mode toggle: reload the current model with the new mode.
        if let Some(new_mode) = pending_mode.take()
            && new_mode != render_colliders
        {
            render_colliders = new_mode;
            if let Some(scene) = scenes.get(selected) {
                match select_scene(viewer, scene, render_colliders).await? {
                    Ok(new_state) => {
                        state = new_state;
                        error = None;
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        error = Some(msg);
                    }
                }
            }
        }

        if viewer.simulating() {
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
