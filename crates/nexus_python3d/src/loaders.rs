//! URDF / MJCF robot loaders.
//!
//! These wrap the `rapier3d-urdf` / `rapier3d-mjcf` crates. Loading a robot
//! manipulates the underlying rapier `PhysicsWorld` (body/collider/joint sets)
//! directly, which is awkward to expose field-by-field through PyO3, so the
//! load+insert is done in Rust and the bindings hand back just what the caller
//! needs to render and actuate the robot.

use crate::math::Pose;
use crate::rbd::{RigidBodyHandle, SharedShape};
use pyo3::prelude::*;
use rapier3d::prelude as rp;

/// Options controlling how a URDF file is loaded (subset of
/// `rapier3d_urdf::UrdfLoaderOptions`; unspecified fields take rapier defaults).
#[pyclass(name = "UrdfLoaderOptions", from_py_object)]
#[derive(Clone)]
pub struct UrdfLoaderOptions {
    pub create_colliders_from_collision_shapes: bool,
    pub create_colliders_from_visual_shapes: bool,
    pub apply_imported_mass_props: bool,
    pub make_roots_fixed: bool,
    pub enable_joint_collisions: bool,
    pub scale: f32,
    pub shift: Option<Pose>,
}

#[pymethods]
impl UrdfLoaderOptions {
    #[new]
    #[pyo3(signature = (
        create_colliders_from_collision_shapes=true,
        create_colliders_from_visual_shapes=false,
        apply_imported_mass_props=true,
        make_roots_fixed=false,
        enable_joint_collisions=false,
        scale=1.0,
        shift=None,
    ))]
    fn new(
        create_colliders_from_collision_shapes: bool,
        create_colliders_from_visual_shapes: bool,
        apply_imported_mass_props: bool,
        make_roots_fixed: bool,
        enable_joint_collisions: bool,
        scale: f32,
        shift: Option<Pose>,
    ) -> Self {
        Self {
            create_colliders_from_collision_shapes,
            create_colliders_from_visual_shapes,
            apply_imported_mass_props,
            make_roots_fixed,
            enable_joint_collisions,
            scale,
            shift,
        }
    }
}

impl UrdfLoaderOptions {
    pub fn to_rapier(&self) -> rapier3d_urdf::UrdfLoaderOptions {
        rapier3d_urdf::UrdfLoaderOptions {
            create_colliders_from_collision_shapes: self.create_colliders_from_collision_shapes,
            create_colliders_from_visual_shapes: self.create_colliders_from_visual_shapes,
            apply_imported_mass_props: self.apply_imported_mass_props,
            make_roots_fixed: self.make_roots_fixed,
            enable_joint_collisions: self.enable_joint_collisions,
            scale: self.scale,
            shift: self.shift.map(|p| p.0).unwrap_or(rp::Pose::IDENTITY),
            ..Default::default()
        }
    }
}

/// What `NexusState.insert_urdf` returns: the per-collider render shapes (to
/// register with the viewer) and the link count (for per-frame motor control).
#[pyclass(name = "UrdfRobotHandles")]
pub struct UrdfRobotHandles {
    /// `(body_handle, shape, local_pose)` triples to pass to
    /// `viewer.insert_visual_shape(0, body_handle, shape, local_pose)`.
    #[pyo3(get)]
    pub render_shapes: Vec<(RigidBodyHandle, SharedShape, Pose)>,
    #[pyo3(get)]
    pub num_links: u32,
}

/// Info returned by `NexusState.insert_mjcf`. The helper already registers the
/// render shapes, floor, camera and light with the viewer; this just reports
/// whether the scene is Z-up (so the caller can set gravity along -Z).
#[pyclass(name = "MjcfSceneInfo", from_py_object)]
#[derive(Clone, Copy)]
pub struct MjcfSceneInfo {
    /// MJCF scenes are authored Z-up; the caller should set `-Z` gravity.
    #[pyo3(get)]
    pub z_up: bool,
    /// `True` if a model loaded and a camera/floor were framed.
    #[pyo3(get)]
    pub loaded: bool,
}

/// A body-attached visual mesh awaiting registration (full-fidelity render data).
struct VisualMeshReg {
    body: rp::RigidBodyHandle,
    shape: rp::SharedShape,
    local_pose: rp::Pose,
    color: [f32; 4],
    uvs: Option<Vec<[f32; 2]>>,
    normals: Option<Vec<[f32; 3]>>,
    texture: Option<std::path::PathBuf>,
    material: Option<nexus_viewer3d::RenderMaterial>,
}

/// Loads an MJCF scene into environment 0 and registers its render shapes,
/// floor, camera and light with the viewer. Mirrors the Rust `mujoco_menagerie3`
/// example's `load_scene` (minus the runtime model picker). Gravity is left to
/// the caller (set after `finalize`).
pub fn insert_mjcf(
    state: &mut nexus3d::prelude::NexusState,
    mut viewer: PyRefMut<crate::viewer::NexusViewer>,
    scene_path: &std::path::Path,
    render_colliders: bool,
) -> PyResult<MjcfSceneInfo> {
    use pyo3::exceptions::PyRuntimeError;
    use rapier3d::parry::bounding_volume::BoundingVolume; // for `Aabb::merge`
    use rapier3d_mjcf::{MjcfLoaderOptions, MjcfMultibodyOptions, MjcfRobot};

    let options = MjcfLoaderOptions {
        skip_plane_geoms: true,
        make_roots_fixed: false,
        create_colliders_from_visual_shapes: false,
        collider_blueprint: rp::ColliderBuilder::default().density(0.0),
        ..Default::default()
    };

    let mut visual_meshes: Vec<VisualMeshReg> = Vec::new();
    // (body, shape, local_pose, body_has_visual_mesh)
    let mut collider_shapes: Vec<(rp::RigidBodyHandle, rp::SharedShape, rp::Pose, bool)> =
        Vec::new();
    let mut floor: Option<(glamx::Vec3, glamx::Vec3)> = None;
    let mut camera: Option<(glamx::Vec3, glamx::Vec3)> = None;

    match MjcfRobot::from_file(scene_path, options) {
        Ok((robot, _model)) => {
            let world = state.rbd_world_mut(0);
            let handles = robot.clone().insert_using_multibody_joints(
                &mut world.bodies,
                &mut world.colliders,
                &mut world.multibody_joints,
                &mut world.impulse_joints,
                MjcfMultibodyOptions::DISABLE_SELF_CONTACTS,
            );

            // Activate actuators at their neutral pose so position-servo robots
            // hold their stance instead of collapsing under gravity.
            let ctrl = vec![0.0; handles.actuators.len()];
            handles.apply_controls_multibody(&mut world.bodies, &mut world.multibody_joints, &ctrl);

            for (i, body_handle) in handles.bodies.iter().enumerate() {
                let Some(bh) = body_handle else { continue };
                let mjcf_body = &robot.bodies[i];
                let has_visual = !mjcf_body.visual_meshes.is_empty();
                for collider in &bh.colliders {
                    let c = &world.colliders[collider.handle];
                    let local_pose = c
                        .position_wrt_parent()
                        .copied()
                        .unwrap_or(rp::Pose::IDENTITY);
                    collider_shapes.push((
                        bh.body,
                        c.shared_shape().clone(),
                        local_pose,
                        has_visual,
                    ));
                }
                if has_visual {
                    for vm in &mjcf_body.visual_meshes {
                        let textured = vm.texture.is_some();
                        let color = vm.rgba.unwrap_or(if textured {
                            [1.0, 1.0, 1.0, 1.0]
                        } else {
                            [0.7, 0.7, 0.75, 1.0]
                        });
                        let material =
                            vm.material
                                .as_ref()
                                .map(|m| nexus_viewer3d::RenderMaterial {
                                    metallic: m.metallic,
                                    roughness: m.roughness,
                                    reflectance: m.reflectance,
                                    emissive: m.emissive,
                                });
                        visual_meshes.push(VisualMeshReg {
                            body: bh.body,
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

            // Bounding box of all colliders (native Z-up frame) → floor + camera.
            let mut aabb = rp::Aabb::new_invalid();
            for (_, collider) in world.colliders.iter() {
                aabb.merge(&collider.compute_aabb());
            }
            if aabb.mins.x <= aabb.maxs.x {
                let center = aabb.center();
                let he = aabb.half_extents();
                let footprint = he.x.max(he.y).max(0.5);
                let floor_thick = 0.1;
                let floor_he = glamx::Vec3::new(footprint * 6.0, footprint * 6.0, floor_thick);
                let floor_center =
                    glamx::Vec3::new(center.x, center.y, center.z - he.z - floor_thick);
                floor = Some((floor_center, floor_he));

                let radius = (he.x * he.x + he.y * he.y + he.z * he.z).sqrt().max(0.5);
                let target = glamx::Vec3::new(center.x, center.y, center.z);
                let eye = target + glamx::Vec3::new(radius * 2.2, -radius * 2.2, radius * 1.6);
                camera = Some((eye, target));
            }
        }
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "failed to load MJCF {}: {e}",
                scene_path.display()
            )));
        }
    }

    let loaded = camera.is_some();
    let v = viewer.rust_mut();

    // Floor (through `NexusState` so it participates in the GPU sim).
    if let Some((center, he)) = floor {
        let body = rp::RigidBodyBuilder::fixed().translation(center).build();
        let collider = rp::ColliderBuilder::cuboid(he.x, he.y, he.z).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider);
        v.insert_shape(handle, &shape, rp::Pose::IDENTITY);
    }

    if render_colliders {
        for (body, shape, local_pose, _) in &collider_shapes {
            v.insert_visual_shape(0, *body, shape, *local_pose);
        }
    } else {
        for vm in &visual_meshes {
            v.insert_visual_mesh(
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
            if !*has_visual {
                v.insert_visual_shape(0, *body, shape, *local_pose);
            }
        }
    }

    if let Some((eye, target)) = camera {
        v.set_camera(eye, target);
    }
    v.scene3d_mut()
        .add_directional_light(glamx::Vec3::new(-1.0, 1.0, -1.0));

    Ok(MjcfSceneInfo { z_up: true, loaded })
}
