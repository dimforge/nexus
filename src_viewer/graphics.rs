use glamx::{Vec3, Vec4};
use nexus::rbd::math::Pose;

#[cfg(feature = "dim3")]
use kiss3d::color::Color;
#[cfg(feature = "dim3")]
use kiss3d::procedural::IndexBuffer;
#[cfg(feature = "dim2")]
use kiss3d::scene::SceneNode2d;
#[cfg(feature = "dim3")]
use kiss3d::scene::SceneNode3d;
use rapier::data::Index;
use rapier::math::{DIM, Vector};
use rapier::parry::shape::ShapeType;
use rapier::prelude::{RigidBodyHandle, SharedShape};
use std::collections::HashMap;
#[cfg(feature = "dim2")]
use {
    glamx::{Mat2, Vec2},
    kiss3d::resource::GpuMesh2d,
    kiss3d::scene::InstanceData2d,
    std::cell::RefCell,
    std::rc::Rc,
};

use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuBackendError, GpuBufferSliceMut, GpuEncoder};
use nexus::prelude::NexusState;
use nexus::rbd::dynamics::{RbdInstanceDesc, WgRbdPrepRender};
#[cfg(feature = "dim3")]
use std::path::Path;
use vortx::tensor::Tensor;
#[cfg(feature = "dim3")]
use {
    glamx::{Mat3, Vec2},
    kiss3d::scene::InstanceData3d,
};

/// PBR shading parameters for a body-attached visual mesh, applied on top of its
/// base color/texture (see `RenderContext::insert_visual_mesh`). Carries MJCF
/// `<material>` properties (metallic / roughness / specular / emission) through
/// to the renderer; `None` leaves kiss3d's default shading untouched.
#[cfg(feature = "dim3")]
#[derive(Copy, Clone, Debug)]
pub struct RenderMaterial {
    /// Metallic factor in `[0, 1]`.
    pub metallic: f32,
    /// Surface roughness in `[0, 1]`.
    pub roughness: f32,
    /// Dielectric specular reflectance in `[0, 1]` (mapped to the F0 term).
    pub reflectance: f32,
    /// Emissive color added on top of lit shading; `[0, 0, 0]` for none.
    pub emissive: [f32; 3],
}

/// A render-only mesh attached to a rigid body, rendered as its own kiss3d node
/// (so it can carry a per-mesh texture and PBR material, which the instanced
/// path can't). Its pose follows the body's world-origin pose each frame,
/// composed with `local_pose`.
#[cfg(feature = "dim3")]
pub struct VisualNode {
    pub node: SceneNode3d,
    /// Environment (batch) the source body belongs to.
    pub env: u32,
    /// Source rigid-body handle, resolved to a GPU pose slot via `state.rbd2gpu`.
    pub handle: Index,
    /// Mesh pose in the body's local frame.
    pub local_pose: Pose,
    /// Cached GPU pose slot (resolved lazily, like the instanced entries).
    pub pose_index: u32,
    /// Single-entry GPU descriptor for the zero-readback path (the node is drawn
    /// as one compute-written instance). `None` until the first direct sync.
    desc: Option<Tensor<RbdInstanceDesc>>,
    /// Instance-count uniform (always 1) paired with [`Self::desc`].
    count_buf: Option<Tensor<u32>>,
    /// Whether [`Self::desc`] has been built with a resolved pose slot.
    desc_resolved: bool,
}

pub struct InstancedNodeEntry {
    pub pose_index: u32,
    /// Environment (batch) this entry belongs to. Combined with `handle` it
    /// resolves the GPU pose slot through `state.rbd2gpu[env]`.
    pub env: u32,
    pub handle: Index,
    pub color: [f32; 4],
    pub scale: [f32; DIM],
    /// Local pose offset composed with the collider pose at render time. Defaults
    /// to identity; populated when a `VisualShape` override is registered
    /// in `BatchEnvironment::visuals` so that proxy-collider shapes (e.g.
    /// OBBs) can be replaced at the right local frame.
    pub local_pose: Pose,
}

/// Convert polygon vertices to a `Vec<u32>` key for exact matching in batching.
#[cfg(feature = "dim2")]
fn polygon_vertex_key(points: &[Vec2]) -> Vec<u32> {
    let mut key = Vec::with_capacity(points.len() * 2);
    for pt in points {
        key.push(pt.x.to_bits());
        key.push(pt.y.to_bits());
    }
    key
}

/// Convert polyhedron vertices to a `Vec<u32>` key for exact matching in batching.
#[cfg(feature = "dim3")]
fn polyhedron_vertex_key(points: &[Vec3]) -> Vec<u32> {
    let mut key = Vec::with_capacity(points.len() * 3);
    for pt in points {
        key.push(pt.x.to_bits());
        key.push(pt.y.to_bits());
        key.push(pt.z.to_bits());
    }
    key
}

/// Object-level alpha applied to instanced nodes holding translucent instances.
///
/// kiss3d routes a node to the order-independent-transparency (OIT) pass only
/// when its *object* color alpha is `< 1.0` — the per-instance alpha alone isn't
/// enough (the phase split is keyed off the object color). We set it just below
/// 1.0 so the phase flips while the per-instance alpha — which the shader
/// multiplies by this object alpha — stays effectively unchanged. Opaque nodes
/// keep kiss3d's default opaque object color.
#[cfg(feature = "dim3")]
const TRANSPARENT_NODE_ALPHA: f32 = 0.9999;

pub struct InstancedNode {
    #[cfg(feature = "dim2")]
    pub node: SceneNode2d,
    #[cfg(feature = "dim3")]
    pub node: SceneNode3d,
    pub entries: Vec<InstancedNodeEntry>,
    #[cfg(feature = "dim2")]
    pub data: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    pub data: Vec<InstanceData3d>,
    /// GPU per-instance descriptor buffer for the zero-readback render path (one
    /// [`RbdInstanceDesc`] per entry). Rebuilt only when the entry set or a
    /// lazily-resolved pose slot changes; `None` until the first direct sync.
    desc: Option<Tensor<RbdInstanceDesc>>,
    /// Instance-count uniform paired with [`Self::desc`].
    count_buf: Option<Tensor<u32>>,
    /// `entries.len()` that [`Self::desc`] was built for (rebuild trigger).
    desc_len: usize,
    /// Whether every entry had a resolved GPU pose slot when [`Self::desc`] was
    /// last built. False forces a rebuild next frame, since pose slots resolve
    /// lazily over the first few frames.
    desc_resolved: bool,
}

impl InstancedNode {
    /// Wraps a freshly created kiss3d node as an empty instanced node. When
    /// `transparent`, the node is flipped into the OIT pass (see
    /// `TRANSPARENT_NODE_ALPHA`) so its translucent instances blend correctly;
    /// opaque and translucent instances of the same shape must live in separate
    /// nodes because the transparent/opaque phase split is per-node, not
    /// per-instance.
    #[cfg(feature = "dim2")]
    fn new(node: SceneNode2d, _transparent: bool) -> Self {
        // 2D rendering blends per-fragment regardless of node, so the split only
        // keeps the keying symmetric with 3D — no extra styling needed here.
        Self {
            node,
            entries: vec![],
            data: vec![],
            desc: None,
            count_buf: None,
            desc_len: 0,
            desc_resolved: false,
        }
    }

    #[cfg(feature = "dim3")]
    fn new(mut node: SceneNode3d, transparent: bool) -> Self {
        if transparent {
            node.set_color(Color::new(1.0, 1.0, 1.0, TRANSPARENT_NODE_ALPHA));
        }
        Self {
            node,
            entries: vec![],
            data: vec![],
            desc: None,
            count_buf: None,
            desc_len: 0,
            desc_resolved: false,
        }
    }

    /// Rebuilds the GPU descriptor buffer for the zero-readback path if the entry
    /// set or any lazily-resolved pose slot changed since the last build, and
    /// returns the instance count to dispatch. Entries whose GPU pose slot isn't
    /// resolved yet are emitted with zero scale (invisible) and trigger a rebuild
    /// next frame.
    fn ensure_descriptors(
        &mut self,
        backend: &GpuBackend,
        state: &NexusState,
    ) -> Result<u32, GpuBackendError> {
        let n = self.entries.len();
        if self.desc.is_some() && self.desc_len == n && self.desc_resolved {
            return Ok(n as u32);
        }

        let mut descs = Vec::with_capacity(n);
        let mut all_resolved = true;
        for entry in &mut self.entries {
            if entry.pose_index == u32::MAX {
                entry.pose_index = state
                    .rbd2gpu
                    .get(entry.env as usize)
                    .and_then(|env| env.get(entry.handle))
                    .map(|r| r.gpu_id)
                    .unwrap_or(u32::MAX);
            }

            let color = Vec4::new(
                entry.color[0],
                entry.color[1],
                entry.color[2],
                entry.color[3],
            );
            if entry.pose_index == u32::MAX {
                // Not active on the GPU yet: keep the slot but make it invisible
                // (zero scale) and force a rebuild next frame.
                all_resolved = false;
                descs.push(RbdInstanceDesc {
                    color,
                    local_pose: entry.local_pose,
                    scale: Vector::ZERO,
                    pose_index: 0,
                    #[cfg(feature = "dim2")]
                    _pad: 0,
                });
            } else {
                descs.push(RbdInstanceDesc {
                    color,
                    local_pose: entry.local_pose,
                    scale: entry.scale.into(),
                    pose_index: entry.pose_index,
                    #[cfg(feature = "dim2")]
                    _pad: 0,
                });
            }
        }

        self.desc = Some(Tensor::vector(backend, &descs, BufferUsages::STORAGE)?);
        self.count_buf = Some(Tensor::scalar(
            backend,
            n as u32,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )?);
        self.desc_len = n;
        self.desc_resolved = all_resolved;
        Ok(n as u32)
    }
}

pub struct RenderContext {
    /// Instanced nodes for primitive shapes, keyed by shape type **and** whether
    /// the node is translucent. Opaque and translucent instances of the same
    /// shape are kept in separate nodes so each can be routed to the right
    /// (opaque vs OIT) render phase.
    pub shape2instance: HashMap<(ShapeType, bool), usize>,
    /// Instanced nodes for mesh-like shapes (convex hull / trimesh / polyline),
    /// keyed by an exact vertex hash plus the translucency flag (same opaque /
    /// translucent split as [`Self::shape2instance`]) so identical shapes share
    /// one node per phase.
    pub mesh2instance: HashMap<(Vec<u32>, bool), usize>,
    pub instances: Vec<InstancedNode>,
    /// Per-mesh visual nodes (textured / PBR), rendered individually rather than
    /// instanced. Driven by body-origin poses in [`Self::update_visual_nodes`].
    #[cfg(feature = "dim3")]
    pub visual_nodes: Vec<VisualNode>,
}

impl RenderContext {
    pub fn new() -> Self {
        Self {
            shape2instance: HashMap::new(),
            mesh2instance: HashMap::new(),
            instances: Vec::new(),
            #[cfg(feature = "dim3")]
            visual_nodes: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        for instance in &mut self.instances {
            instance.node.detach();
        }
        self.instances.clear();
        self.shape2instance.clear();
        self.mesh2instance.clear();
        #[cfg(feature = "dim3")]
        {
            for visual in &mut self.visual_nodes {
                visual.node.detach();
            }
            self.visual_nodes.clear();
        }
    }

    /// Pushes a render entry for `handle` (in environment `env`) into the
    /// instanced node `instance_id`.
    fn push_entry(
        &mut self,
        instance_id: usize,
        env: u32,
        handle: RigidBodyHandle,
        color: Vec4,
        local_pose: Pose,
        scale: [f32; DIM],
    ) {
        let instanced_node = &mut self.instances[instance_id];
        instanced_node.entries.push(InstancedNodeEntry {
            pose_index: u32::MAX,
            env,
            handle: handle.0,
            color: [color.x, color.y, color.z, color.w],
            local_pose,
            scale,
        });
    }

    pub fn insert_shape(
        &mut self,
        #[cfg(feature = "dim2")] scene_2d: &mut SceneNode2d,
        #[cfg(feature = "dim3")] scene_3d: &mut SceneNode3d,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
        color: Option<Vec4>,
    ) {
        #[cfg(feature = "dim2")]
        let scene = scene_2d;
        #[cfg(feature = "dim3")]
        let scene = scene_3d;

        // Variety shading only applies to default (opaque) colors, so count
        // entries in the opaque node for this shape type.
        let color_id = self
            .shape2instance
            .get(&(shape.shape_type(), false))
            .map(|inst_id| self.instances[*inst_id].entries.len())
            .unwrap_or_default();
        let coeff = (1.0 - 0.15 * (color_id % 5) as f32) / 255.0;
        let rgb = match shape.shape_type() {
            ShapeType::Ball => Vec3::new(55.0, 126.0, 184.0) * coeff,
            ShapeType::Cuboid => Vec3::new(55.0, 126.0, 34.0) * coeff,
            #[cfg(feature = "dim3")]
            ShapeType::Cylinder => Vec3::new(140.0, 86.0, 75.0) * coeff,
            #[cfg(feature = "dim3")]
            ShapeType::Cone => Vec3::new(255.0, 217.0, 47.0) * coeff,
            ShapeType::Capsule => Vec3::new(204.0, 121.0, 167.0) * coeff,
            #[cfg(feature = "dim3")]
            ShapeType::ConvexPolyhedron => Vec3::new(228.0, 26.0, 28.0) * coeff,
            _ => Vec3::new(255.0, 127.0, 0.0) * coeff,
        };
        let color = color.unwrap_or(Vec4::new(rgb.x, rgb.y, rgb.z, 1.0));
        // Translucent instances must go to a dedicated node so they render in the
        // transparent (OIT) pass rather than the opaque one.
        let transparent = color.w < 1.0;

        match shape.shape_type() {
            ShapeType::Ball => {
                let instance_id = *self
                    .shape2instance
                    .entry((ShapeType::Ball, transparent))
                    .or_insert_with(|| {
                        #[cfg(feature = "dim2")]
                        let node = scene.add_circle(0.5);
                        #[cfg(feature = "dim3")]
                        let node = {
                            let lowres_sphere = kiss3d::procedural::sphere(1.0, 10, 10, true);
                            scene.add_render_mesh(lowres_sphere, Vec3::ONE)
                        };
                        self.instances.push(InstancedNode::new(node, transparent));
                        self.instances.len() - 1
                    });
                let ball = shape.as_ball().unwrap();
                self.push_entry(
                    instance_id,
                    env,
                    handle,
                    color,
                    local_pose,
                    [ball.radius * 2.0; DIM],
                );
            }
            ShapeType::Cuboid => {
                let instance_id = *self
                    .shape2instance
                    .entry((ShapeType::Cuboid, transparent))
                    .or_insert_with(|| {
                        #[cfg(feature = "dim2")]
                        let node = scene.add_rectangle(1.0, 1.0);
                        #[cfg(feature = "dim3")]
                        let node = scene.add_cube(1.0, 1.0, 1.0);
                        self.instances.push(InstancedNode::new(node, transparent));
                        self.instances.len() - 1
                    });
                let cuboid = shape.as_cuboid().unwrap();
                let scale = (cuboid.half_extents * 2.0).into();
                self.push_entry(instance_id, env, handle, color, local_pose, scale);
            }
            #[cfg(feature = "dim3")]
            ShapeType::Cylinder => {
                let instance_id = *self
                    .shape2instance
                    .entry((ShapeType::Cylinder, transparent))
                    .or_insert_with(|| {
                        let node = scene.add_cylinder(1.0, 1.0);
                        self.instances.push(InstancedNode::new(node, transparent));
                        self.instances.len() - 1
                    });
                let cyl = shape.as_cylinder().unwrap();
                self.push_entry(
                    instance_id,
                    env,
                    handle,
                    color,
                    local_pose,
                    [cyl.radius, cyl.half_height * 2.0, cyl.radius],
                );
            }
            #[cfg(feature = "dim3")]
            ShapeType::Cone => {
                let instance_id = *self
                    .shape2instance
                    .entry((ShapeType::Cone, transparent))
                    .or_insert_with(|| {
                        let node = scene.add_cone(1.0, 1.0);
                        self.instances.push(InstancedNode::new(node, transparent));
                        self.instances.len() - 1
                    });
                let c = shape.as_cone().unwrap();
                self.push_entry(
                    instance_id,
                    env,
                    handle,
                    color,
                    local_pose,
                    [c.radius, c.half_height * 2.0, c.radius],
                );
            }
            #[cfg(feature = "dim2")]
            ShapeType::Capsule => {
                let instance_id = *self
                    .shape2instance
                    .entry((ShapeType::Capsule, transparent))
                    .or_insert_with(|| {
                        let node = scene.add_capsule(0.5, 1.0);
                        self.instances.push(InstancedNode::new(node, transparent));
                        self.instances.len() - 1
                    });
                let c = shape.as_capsule().unwrap();
                let scale = [c.radius * 2.0, c.segment.length()];
                self.push_entry(instance_id, env, handle, color, local_pose, scale);
            }
            #[cfg(feature = "dim3")]
            ShapeType::Capsule => {
                // Build the exact capsule mesh and render it with no scale.
                // `Capsule::to_trimesh` bakes both the correct spherical caps and
                // the segment orientation (via `canonical_transform`) into the
                // shape-local vertices, so the per-instance deformation only needs
                // to be the body/local rotation+translation. Non-uniformly scaling
                // a unit capsule (the old path) squashed the caps into ellipsoids
                // and ignored the segment axis. Key by radius + segment endpoints
                // so identical capsules share one instanced node.
                let c = shape.as_capsule().unwrap();
                let s = &c.segment;
                let key = (
                    vec![
                        c.radius.to_bits(),
                        s.a.x.to_bits(),
                        s.a.y.to_bits(),
                        s.a.z.to_bits(),
                        s.b.x.to_bits(),
                        s.b.y.to_bits(),
                        s.b.z.to_bits(),
                    ],
                    transparent,
                );
                let instance_id = match self.mesh2instance.get(&key) {
                    Some(id) => *id,
                    None => {
                        let (vtx, idx) = c.to_trimesh(20, 10);
                        let mut render = kiss3d::procedural::RenderMesh::new(
                            vtx,
                            None,
                            None,
                            Some(IndexBuffer::Unified(idx)),
                        );
                        render.recompute_normals();
                        let node = scene.add_render_mesh(render, Vec3::ONE);
                        self.instances.push(InstancedNode::new(node, transparent));
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim2")]
            ShapeType::ConvexPolygon => {
                let poly = shape.as_convex_polygon().unwrap();
                let points: Vec<_> = poly.points().to_vec();
                let key = (polygon_vertex_key(&points), transparent);
                let instance_id = match self.mesh2instance.get(&key) {
                    Some(id) => *id,
                    None => {
                        let node = scene.add_convex_polygon(points, Vec2::ONE);
                        self.instances.push(InstancedNode::new(node, transparent));
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim3")]
            ShapeType::ConvexPolyhedron => {
                let poly = shape.as_convex_polyhedron().unwrap();
                let points: Vec<_> = poly.points().to_vec();
                let key = (polyhedron_vertex_key(&points), transparent);
                let instance_id = match self.mesh2instance.get(&key) {
                    Some(id) => *id,
                    None => {
                        let (vtx, idx) = poly.to_trimesh();
                        let mut render = kiss3d::procedural::RenderMesh::new(
                            vtx,
                            None,
                            None,
                            Some(IndexBuffer::Unified(idx)),
                        );
                        render.replicate_vertices();
                        render.recompute_normals();
                        let node = scene.add_render_mesh(render, Vec3::ONE);
                        self.instances.push(InstancedNode::new(node, transparent));
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim3")]
            ShapeType::TriMesh => {
                let trimesh = shape.as_trimesh().unwrap();
                let vtx: Vec<_> = trimesh.vertices().to_vec();
                let idx: Vec<_> = trimesh.indices().to_vec();
                // Trimeshes are usually unique (a floor, a terrain); key by their
                // full vertex set so re-inserting the same mesh reuses the node.
                let mut vertex_key = Vec::with_capacity(vtx.len() * 3);
                for pt in &vtx {
                    vertex_key.push(pt.x.to_bits());
                    vertex_key.push(pt.y.to_bits());
                    vertex_key.push(pt.z.to_bits());
                }
                let key = (vertex_key, transparent);
                let instance_id = match self.mesh2instance.get(&key) {
                    Some(id) => *id,
                    None => {
                        let mut render = kiss3d::procedural::RenderMesh::new(
                            vtx,
                            None,
                            None,
                            Some(IndexBuffer::Unified(idx)),
                        );
                        render.recompute_normals();
                        let node = scene.add_render_mesh(render, Vec3::ONE);
                        self.instances.push(InstancedNode::new(node, transparent));
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim2")]
            ShapeType::Polyline => {
                let polyline = shape.as_polyline().unwrap();
                let mut vertex_key = Vec::new();
                for v in polyline.vertices() {
                    vertex_key.push(v.x.to_bits());
                    vertex_key.push(v.y.to_bits());
                }
                let key = (vertex_key, transparent);
                let instance_id = match self.mesh2instance.get(&key) {
                    Some(id) => *id,
                    None => {
                        let mut vtx = vec![];
                        let mut idx = vec![];
                        for segment in polyline.segments() {
                            let thickness = 0.2;
                            let center = (segment.a + segment.b) * 0.5;
                            let length = (segment.b - segment.a).length();
                            let scaled_dir = segment.scaled_direction();
                            let angle =
                                scaled_dir.y.atan2(scaled_dir.x) - std::f32::consts::FRAC_PI_2;
                            let cos = angle.cos();
                            let sin = angle.sin();
                            let rot = Mat2::from_cols(Vec2::new(cos, sin), Vec2::new(-sin, cos));
                            let half_w = thickness;
                            let half_h = length / 2.0;
                            let local_vtx = [
                                Vec2::new(-half_w, -half_h),
                                Vec2::new(half_w, -half_h),
                                Vec2::new(half_w, half_h),
                                Vec2::new(-half_w, half_h),
                            ];
                            let base = vtx.len() as u32;
                            for lv in &local_vtx {
                                vtx.push(rot * *lv + center);
                            }
                            idx.push([base, base + 1, base + 2]);
                            idx.push([base, base + 2, base + 3]);
                        }
                        let mesh = GpuMesh2d::new(vtx, idx, None, false);
                        let node = scene.add_mesh(Rc::new(RefCell::new(mesh)), Vec2::ONE);
                        self.instances.push(InstancedNode::new(node, transparent));
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            _ => todo!("unsupported render shape: {:?}", shape.shape_type()),
        }
    }
}

/// Convert a glamx Pose to position and deformation matrix for rendering
#[cfg(feature = "dim2")]
fn pose_to_render_data(pose: &Pose, scale: &[f32; 2]) -> (Vec2, Mat2) {
    let position = pose.translation;
    let cos = pose.rotation.cos();
    let sin = pose.rotation.sin();
    let mut deformation = Mat2::from_cols(Vec2::new(cos, sin), Vec2::new(-sin, cos));
    deformation.x_axis *= scale[0];
    deformation.y_axis *= scale[1];
    (position, deformation)
}

/// Convert a glamx Pose to position and deformation matrix for rendering
#[cfg(feature = "dim3")]
fn pose_to_render_data(pose: &Pose, scale: &[f32; 3]) -> (Vec3, Mat3) {
    let position = pose.translation;
    let deformation = Mat3::from_quat(pose.rotation);
    let deformation = Mat3::from_cols(
        deformation.x_axis * scale[0],
        deformation.y_axis * scale[1],
        deformation.z_axis * scale[2],
    );
    (position, deformation)
}

impl RenderContext {
    /// Update rendering instances from a slice of collider world poses, indexed by
    /// the collider slot stored in each [`InstancedNodeEntry::pose_index`].
    ///
    /// This is the backend-agnostic core of `update_instances`; it is used by the
    /// viewer-owned `NexusState` rendering path, which reads the poses straight from
    /// the GPU buffer instead of going through a `PhysicsBackend`.
    pub fn update_instances_from_poses(&mut self, state: &NexusState, poses: &[Pose]) {
        for instanced_node in &mut self.instances {
            instanced_node.data.clear();

            for entry in &mut instanced_node.entries {
                if entry.pose_index == u32::MAX {
                    entry.pose_index = state
                        .rbd2gpu
                        .get(entry.env as usize)
                        .and_then(|env| env.get(entry.handle))
                        .map(|r| r.gpu_id)
                        .unwrap_or(u32::MAX);
                }

                if entry.pose_index == u32::MAX {
                    continue; // This entry isn’t active on the GPU yet.
                }

                let body_pose = &poses[entry.pose_index as usize];
                let pose = *body_pose * entry.local_pose;
                let (position, deformation) = pose_to_render_data(&pose, &entry.scale);

                #[cfg(feature = "dim2")]
                {
                    instanced_node.data.push(InstanceData2d {
                        position,
                        deformation,
                        color: entry.color,
                        lines_color: None,
                        lines_width: None,
                        points_color: None,
                        points_size: None,
                    });
                }

                #[cfg(feature = "dim3")]
                {
                    instanced_node.data.push(InstanceData3d {
                        position,
                        deformation,
                        color: Color::new(
                            entry.color[0],
                            entry.color[1],
                            entry.color[2],
                            entry.color[3],
                        ),
                        lines_color: None,
                        lines_width: None,
                        points_color: None,
                        points_size: None,
                    });
                }
            }

            instanced_node.node.set_instances(&instanced_node.data);
        }
    }

    /// Zero-readback counterpart of [`Self::update_instances_from_poses`]: writes
    /// each instanced node's render data straight into kiss3d's GPU instance
    /// buffers. Used only when khal shares kiss3d's wgpu device.
    ///
    /// The caller submits `encoder` (on the shared queue) before kiss3d renders.
    pub fn update_instances_direct(
        &mut self,
        backend: &GpuBackend,
        state: &NexusState,
        body_poses: &Tensor<Pose>,
        shader: &WgRbdPrepRender,
        encoder: &mut GpuEncoder,
    ) -> Result<(), GpuBackendError> {
        for node in &mut self.instances {
            let count = node.ensure_descriptors(backend, state)?;
            if count == 0 {
                continue;
            }
            // Fetch fresh each frame: only reallocates when the count grows, and
            // marks the buffers GPU-writable so kiss3d's render-time upload is a
            // no-op and won't clobber the compute writes.
            let bufs = node.node.instance_compute_buffers(count as usize);
            let mut positions = GpuBufferSliceMut::<f32>::from_wgpu(&bufs.positions);
            let mut deformations = GpuBufferSliceMut::<f32>::from_wgpu(&bufs.deformations);
            let mut colors = GpuBufferSliceMut::<f32>::from_wgpu(&bufs.colors);
            let desc = node.desc.as_ref().unwrap();
            let count_buf = node.count_buf.as_ref().unwrap();
            shader.launch(
                encoder,
                &mut positions,
                &mut deformations,
                &mut colors,
                body_poses,
                desc,
                count_buf,
                count,
            )?;
        }
        Ok(())
    }

    /// Registers a body-attached visual mesh rendered as its own node, carrying
    /// the source asset's color, UVs, normals, texture, and PBR material —
    /// everything the instanced path can't express per-mesh. The node's pose
    /// follows the body's world-origin pose (see [`Self::update_visual_nodes`]).
    #[cfg(feature = "dim3")]
    pub fn insert_visual_mesh(
        &mut self,
        scene_3d: &mut SceneNode3d,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
        color: [f32; 4],
        uvs: Option<&[[f32; 2]]>,
        normals: Option<&[[f32; 3]]>,
        texture: Option<&Path>,
        material: Option<RenderMaterial>,
    ) {
        let Some(mut node) = build_visual_node(scene_3d, shape, uvs, normals, texture.is_some())
        else {
            return;
        };

        // Hidden until the first pose sync, so it doesn't flash at the origin
        // for one frame (its transform is only set in `update_visual_nodes`).
        node.set_visible(false);
        node.set_color(Color::new(color[0], color[1], color[2], color[3]));
        // Visual meshes are routinely viewed from both sides and the authored
        // winding isn't guaranteed, so don't cull back faces.
        node.enable_backface_culling_recursive(false);

        if let Some(mat) = material {
            node.set_metallic_recursive(mat.metallic);
            node.set_roughness_recursive(mat.roughness);
            node.set_reflectance(mat.reflectance);
            node.set_emissive_recursive(Color::new(
                mat.emissive[0],
                mat.emissive[1],
                mat.emissive[2],
                1.0,
            ));
        }
        // A translucent base color needs the blend pass; opaque meshes keep the
        // default opaque path.
        if color[3] < 1.0 {
            node.set_alpha_mode(kiss3d::scene::AlphaMode::Blend);
        }
        if let Some(tex_path) = texture {
            // Key the texture cache by the path so the same file uploads once.
            let key = tex_path.to_string_lossy();
            node.set_texture_from_file(tex_path, &key);
        }

        self.visual_nodes.push(VisualNode {
            node,
            env,
            handle: handle.0,
            local_pose,
            pose_index: u32::MAX,
            desc: None,
            count_buf: None,
            desc_resolved: false,
        });
    }

    /// Updates each visual node's transform from the body-origin poses (indexed
    /// by the body's GPU pose slot). Visual-mesh local poses are body-relative,
    /// so they compose with `body_poses` (not the collider world poses the
    /// instanced shapes use).
    #[cfg(feature = "dim3")]
    pub fn update_visual_nodes(&mut self, state: &NexusState, body_poses: &[Pose]) {
        for visual in &mut self.visual_nodes {
            if visual.pose_index == u32::MAX {
                visual.pose_index = state
                    .rbd2gpu
                    .get(visual.env as usize)
                    .and_then(|env| env.get(visual.handle))
                    .map(|r| r.gpu_id)
                    .unwrap_or(u32::MAX);
            }

            if visual.pose_index == u32::MAX {
                continue; // Not active on the GPU yet.
            }

            let pose = body_poses[visual.pose_index as usize] * visual.local_pose;
            visual.node.set_pose(pose);
            visual.node.set_visible(true);
        }
    }

    /// Zero-readback counterpart of [`Self::update_visual_nodes`]: each visual
    /// mesh is drawn as a single compute-written instance, so its world transform
    /// comes from the GPU `body_poses` buffer without a CPU readback. The node's
    /// own transform is left at identity (the instance carries the world pose) so
    /// its per-mesh texture/material — sampled by kiss3d's instanced pipeline —
    /// is preserved; the instance color is white so it doesn't tint the mesh.
    #[cfg(feature = "dim3")]
    pub fn update_visual_nodes_direct(
        &mut self,
        backend: &GpuBackend,
        state: &NexusState,
        body_poses: &Tensor<Pose>,
        shader: &WgRbdPrepRender,
        encoder: &mut GpuEncoder,
    ) -> Result<(), GpuBackendError> {
        for visual in &mut self.visual_nodes {
            if visual.pose_index == u32::MAX {
                visual.pose_index = state
                    .rbd2gpu
                    .get(visual.env as usize)
                    .and_then(|env| env.get(visual.handle))
                    .map(|r| r.gpu_id)
                    .unwrap_or(u32::MAX);
            }
            if visual.pose_index == u32::MAX {
                continue; // Not active on the GPU yet.
            }

            if visual.desc.is_none() || !visual.desc_resolved {
                let desc = RbdInstanceDesc {
                    // The visual mesh already carries real-world-sized
                    // coordinates, so the per-instance deformation must be a pure
                    // rotation: unit scale on every axis. (Zero scale would
                    // collapse every vertex onto the origin → invisible mesh.)
                    color: Vec4::ONE,
                    local_pose: visual.local_pose,
                    scale: Vector::ONE,
                    pose_index: visual.pose_index,
                };
                visual.desc = Some(Tensor::vector(backend, [desc], BufferUsages::STORAGE)?);
                visual.count_buf = Some(Tensor::scalar(
                    backend,
                    1,
                    BufferUsages::STORAGE | BufferUsages::UNIFORM,
                )?);
                visual.desc_resolved = true;
                // The instance carries the world transform; keep the node's own
                // transform at identity so the two don't compound.
                visual.node.set_pose(Pose::IDENTITY);
                visual.node.set_visible(true);
            }

            let bufs = visual.node.instance_compute_buffers(1);
            let mut positions = GpuBufferSliceMut::<f32>::from_wgpu(&bufs.positions);
            let mut deformations = GpuBufferSliceMut::<f32>::from_wgpu(&bufs.deformations);
            let mut colors = GpuBufferSliceMut::<f32>::from_wgpu(&bufs.colors);
            shader.launch(
                encoder,
                &mut positions,
                &mut deformations,
                &mut colors,
                body_poses,
                visual.desc.as_ref().unwrap(),
                visual.count_buf.as_ref().unwrap(),
                1,
            )?;
        }
        Ok(())
    }

    /// Whether any body-attached visual mesh is registered.
    #[cfg(feature = "dim3")]
    pub fn has_visual_nodes(&self) -> bool {
        !self.visual_nodes.is_empty()
    }
}

/// Builds a kiss3d node for a visual mesh: a `RenderMesh` carrying authored
/// UVs/normals for trimeshes, or a matching primitive node otherwise. Returns
/// `None` for shape types we don't render.
#[cfg(feature = "dim3")]
fn build_visual_node(
    scene: &mut SceneNode3d,
    shape: &SharedShape,
    uvs: Option<&[[f32; 2]]>,
    normals: Option<&[[f32; 3]]>,
    has_texture: bool,
) -> Option<SceneNode3d> {
    match shape.shape_type() {
        ShapeType::TriMesh => {
            let trimesh = shape.as_trimesh().unwrap();
            let vtx: Vec<Vec3> = trimesh.vertices().to_vec();
            let idx = trimesh.indices().to_vec();
            let n = vtx.len();

            // UVs are only meaningful with a texture. OBJ stores `v=0` at the
            // bottom of the image while wgpu samples from the top — flip `v`. A
            // UV/normal count that doesn't match the vertex buffer is dropped.
            let uvs_opt: Option<Vec<Vec2>> = if has_texture {
                uvs.filter(|u| u.len() == n)
                    .map(|u| u.iter().map(|uv| Vec2::new(uv[0], 1.0 - uv[1])).collect())
            } else {
                None
            };
            let normals_opt: Option<Vec<Vec3>> = normals
                .filter(|nn| nn.len() == n)
                .map(|nn| nn.iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect());

            let have_normals = normals_opt.is_some();
            let mut mesh = kiss3d::procedural::RenderMesh::new(
                vtx,
                normals_opt,
                uvs_opt,
                Some(IndexBuffer::Unified(idx)),
            );
            if !have_normals {
                // No usable authored normals: split shared vertices and compute
                // flat per-face normals so the mesh still lights.
                mesh.replicate_vertices();
                mesh.recompute_normals();
            }
            Some(scene.add_render_mesh(mesh, Vec3::ONE))
        }
        ShapeType::Ball => Some(scene.add_sphere(shape.as_ball().unwrap().radius)),
        ShapeType::Cuboid => {
            let h = shape.as_cuboid().unwrap().half_extents;
            Some(scene.add_cube(h.x * 2.0, h.y * 2.0, h.z * 2.0))
        }
        ShapeType::Cylinder => {
            let c = shape.as_cylinder().unwrap();
            Some(scene.add_cylinder(c.radius, c.half_height * 2.0))
        }
        ShapeType::Cone => {
            let c = shape.as_cone().unwrap();
            Some(scene.add_cone(c.radius, c.half_height * 2.0))
        }
        ShapeType::Capsule => {
            // `add_capsule` is Y-aligned and ignores the segment axis; build the
            // exact mesh instead (segment orientation baked in by `to_trimesh`).
            let c = shape.as_capsule().unwrap();
            let (vtx, idx) = c.to_trimesh(20, 10);
            let mut mesh = kiss3d::procedural::RenderMesh::new(
                vtx,
                None,
                None,
                Some(IndexBuffer::Unified(idx)),
            );
            mesh.recompute_normals();
            Some(scene.add_render_mesh(mesh, Vec3::ONE))
        }
        ShapeType::ConvexPolyhedron => {
            let (vtx, idx) = shape.as_convex_polyhedron().unwrap().to_trimesh();
            let mut mesh = kiss3d::procedural::RenderMesh::new(
                vtx,
                None,
                None,
                Some(IndexBuffer::Unified(idx)),
            );
            mesh.replicate_vertices();
            mesh.recompute_normals();
            Some(scene.add_render_mesh(mesh, Vec3::ONE))
        }
        other => {
            eprintln!("Skipping unsupported visual shape: {other:?}");
            None
        }
    }
}
