//! A small scene description (boxes, spheres and ball joints), translated into a nexus scene
//! when run.
//!
//! Conventions: Z is up, box sizes are full widths, a zero density makes a body static, and
//! gravity is -10.

use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use rapier3d::glamx::IVec3;
use rapier3d::prelude::*;

/// Collision-group bit shared by every body that isn't jointed to another one.
const FREE_GROUP: u32 = 1 << 31;
/// Number of collision-group bits available to jointed bodies.
const NUM_CLASSES: u32 = 31;

pub struct Rigid {
    /// Full widths along each local axis (a sphere's is `2r` on each side).
    pub size: Vector,
    pub density: f32,
    pub friction: f32,
    pub position: Vector,
    pub rotation: Rotation,
    pub linvel: Vector,
    pub angvel: Vector,
    /// Set for spheres, `None` for boxes.
    pub radius: Option<f32>,
    pub color: Option<Vec4>,
}

/// A ball joint between two body-local anchors.
pub struct BallJoint {
    pub a: usize,
    pub b: usize,
    pub ra: Vector,
    pub rb: Vector,
}

/// Orbit camera framing (azimuth in degrees, elevation in radians).
pub struct Camera {
    pub distance: f32,
    pub target: Vector,
    pub azimuth: f32,
    pub elevation: f32,
}

impl Camera {
    /// A camera with the default azimuth (90°) and elevation (0.35 rad).
    pub fn new(distance: f32, target: [f32; 3]) -> Self {
        Self {
            distance,
            target: Vector::from(target),
            azimuth: 90.0,
            elevation: 0.35,
        }
    }

    pub fn azimuth(mut self, azimuth: f32) -> Self {
        self.azimuth = azimuth;
        self
    }

    pub fn elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation;
        self
    }

    fn eye(&self) -> Vector {
        let az = self.azimuth.to_radians();
        let el = self.elevation;
        self.target
            + self.distance * Vector::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin())
    }
}

pub struct Scene {
    pub bodies: Vec<Rigid>,
    pub joints: Vec<BallJoint>,
    pub camera: Camera,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            bodies: vec![],
            joints: vec![],
            camera,
        }
    }

    /// A box of full widths `size`; static if `density` is zero.
    pub fn rigid(
        &mut self,
        size: [f32; 3],
        density: f32,
        friction: f32,
        position: [f32; 3],
    ) -> usize {
        self.bodies.push(Rigid {
            size: Vector::from(size),
            density,
            friction,
            position: Vector::from(position),
            rotation: Rotation::IDENTITY,
            linvel: Vector::ZERO,
            angvel: Vector::ZERO,
            radius: None,
            color: None,
        });
        self.bodies.len() - 1
    }

    /// A solid sphere of radius `r`; static if `density` is zero.
    pub fn sphere(&mut self, r: f32, density: f32, friction: f32, position: [f32; 3]) -> usize {
        let id = self.rigid([2.0 * r; 3], density, friction, position);
        self.bodies[id].radius = Some(r);
        id
    }

    pub fn set_rotation(&mut self, id: usize, rotation: Rotation) {
        self.bodies[id].rotation = rotation.normalize();
    }

    pub fn set_linvel(&mut self, id: usize, linvel: [f32; 3]) {
        self.bodies[id].linvel = Vector::from(linvel);
    }

    pub fn set_angvel(&mut self, id: usize, angvel: [f32; 3]) {
        self.bodies[id].angvel = Vector::from(angvel);
    }

    /// Sets a body's color from a `0xrrggbb` value.
    pub fn set_color(&mut self, id: usize, rgb: u32) {
        self.bodies[id].color = Some(rgb_color(rgb));
    }

    /// A ball joint between `a` and `b`, which then don't collide with each other.
    pub fn ball_joint(&mut self, a: usize, b: usize, ra: [f32; 3], rb: [f32; 3]) {
        self.joints.push(BallJoint {
            a,
            b,
            ra: Vector::from(ra),
            rb: Vector::from(rb),
        });
    }
}

/// Converts a `0xrrggbb` color to an opaque RGBA color.
pub fn rgb_color(rgb: u32) -> Vec4 {
    let c = |shift: u32| ((rgb >> shift) & 0xff) as f32 / 255.0;
    Vec4::new(c(16), c(8), c(0), 1.0)
}

/// HSL (all in 0..1) to a `0xrrggbb` color, as three.js' `Color.setHSL`.
pub fn hsl(h: f32, s: f32, l: f32) -> u32 {
    let a = s * l.min(1.0 - l);
    let f = |n: f32| {
        let k = (n + h * 12.0) % 12.0;
        let v = l - a * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0);
        (v.clamp(0.0, 1.0) * 255.0).round() as u32
    };
    (f(0.0) << 16) | (f(8.0) << 8) | f(4.0)
}

/// Deterministic pseudo-random numbers in [0, 1) (a linear congruential generator).
pub struct Random(u32);

impl Random {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 as f64 / 4294967296.0) as f32
    }
}

/// Assigns collision groups so that no jointed pair collides.
///
/// Every jointed body gets one of 31 group bits, chosen to differ from its jointed neighbors
/// and their own neighbors, and filters out its neighbors' bits. Unjointed bodies share the
/// last bit and collide with everything. The price is that a jointed body also ignores
/// far-away bodies that happen to share a neighbor's bit.
fn collision_groups(scene: &Scene) -> Vec<InteractionGroups> {
    let n = scene.bodies.len();
    let mut adjacency = vec![vec![]; n];
    for joint in &scene.joints {
        adjacency[joint.a].push(joint.b);
        adjacency[joint.b].push(joint.a);
    }

    const UNASSIGNED: u32 = u32::MAX;
    let mut classes = vec![UNASSIGNED; n];
    let mut next = 0;
    for i in 0..n {
        if adjacency[i].is_empty() {
            continue;
        }
        let mut near = 0u32;
        let mut far = 0u32;
        for &j in &adjacency[i] {
            if classes[j] != UNASSIGNED {
                near |= 1 << classes[j];
            }
            for &k in &adjacency[j] {
                if k != i && classes[k] != UNASSIGNED {
                    far |= 1 << classes[k];
                }
            }
        }
        // Round-robin from `next` spreads the classes evenly.
        let pick = |taken: u32| {
            (0..NUM_CLASSES)
                .map(|k| (next + k) % NUM_CLASSES)
                .find(|c| taken & (1 << c) == 0)
        };
        let class = pick(near | far).or_else(|| pick(near)).unwrap_or(next);
        classes[i] = class;
        next = (class + 1) % NUM_CLASSES;
    }

    (0..n)
        .map(|i| {
            if classes[i] == UNASSIGNED {
                InteractionGroups::new(
                    Group::from_bits_retain(FREE_GROUP),
                    Group::ALL,
                    InteractionTestMode::And,
                )
            } else {
                let excluded = adjacency[i]
                    .iter()
                    .fold(0u32, |acc, &j| acc | (1 << classes[j]));
                InteractionGroups::new(
                    Group::from_bits_retain(1 << classes[i]),
                    Group::from_bits_retain(!excluded),
                    InteractionTestMode::And,
                )
            }
        })
        .collect()
}

/// The number of body pairs whose bounding boxes overlap (within the prediction distance) at
/// the start of `scene`, and the most such pairs of a single body.
fn initial_pairs(scene: &Scene) -> (usize, usize) {
    // The default prediction distance, plus some slack.
    const MARGIN: f32 = 0.025;
    let aabbs: Vec<(Vector, Vector)> = scene
        .bodies
        .iter()
        .map(|b| {
            let half = match b.radius {
                Some(r) => Vector::splat(r),
                None => {
                    let m = Mat3::from_quat(b.rotation);
                    let h = b.size / 2.0;
                    m.x_axis.abs() * h.x + m.y_axis.abs() * h.y + m.z_axis.abs() * h.z
                }
            };
            let half = half + Vector::splat(MARGIN);
            (b.position - half, b.position + half)
        })
        .collect();

    // Grid cells of twice the median body size; larger bodies are tested against every body.
    let mut sizes: Vec<f32> = aabbs
        .iter()
        .map(|(lo, hi)| (*hi - *lo).max_element())
        .collect();
    sizes.sort_by(f32::total_cmp);
    let cell = 2.0 * sizes.get(sizes.len() / 2).copied().unwrap_or(1.0);
    let cell_of = |p: Vector| (p / cell).floor().as_ivec3();
    let overlap = |a: usize, b: usize| {
        let ((lo1, hi1), (lo2, hi2)) = (aabbs[a], aabbs[b]);
        lo1.cmple(hi2).all() && lo2.cmple(hi1).all()
    };
    let solved = |i: usize| scene.bodies[i].density > 0.0;

    let mut grid: std::collections::HashMap<IVec3, Vec<usize>> = Default::default();
    let mut large = vec![];
    for (i, (lo, hi)) in aabbs.iter().enumerate() {
        if (*hi - *lo).max_element() > cell {
            large.push(i);
            continue;
        }
        let (c0, c1) = (cell_of(*lo), cell_of(*hi));
        for x in c0.x..=c1.x {
            for y in c0.y..=c1.y {
                for z in c0.z..=c1.z {
                    grid.entry(IVec3::new(x, y, z)).or_default().push(i);
                }
            }
        }
    }

    let mut degree = vec![0usize; aabbs.len()];
    let mut count = 0;
    let mut add = |a: usize, b: usize, degree: &mut Vec<usize>| {
        if (solved(a) || solved(b)) && overlap(a, b) {
            count += 1;
            degree[a] += 1;
            degree[b] += 1;
        }
    };
    for (key, bodies) in &grid {
        for (k, &a) in bodies.iter().enumerate() {
            for &b in &bodies[k + 1..] {
                // Count a pair once: in the cell holding the low corner of the boxes' overlap.
                let low = aabbs[a].0.max(aabbs[b].0);
                if cell_of(low) == *key {
                    add(a, b, &mut degree);
                }
            }
        }
    }
    for (k, &a) in large.iter().enumerate() {
        for b in 0..aabbs.len() {
            // Large-large pairs once, large-small pairs from the large side.
            if b != a && (!large.contains(&b) || large[..k].contains(&b)) {
                add(a, b, &mut degree);
            }
        }
    }

    // Static bodies aren't colored.
    let max_degree = (0..aabbs.len())
        .filter(|&i| solved(i))
        .map(|i| degree[i])
        .max()
        .unwrap_or(0);
    (count, max_degree)
}

/// A body's render shape and optional color, as built by [`build_state`].
pub struct RenderShape {
    pub handle: RigidBodyHandle,
    pub shape: SharedShape,
    pub color: Option<Vec4>,
}

/// Builds `scene` into a fresh [`NexusState`], returning the render shape of each body.
pub fn build_state(scene: &Scene) -> (NexusState, Vec<RenderShape>) {
    let n = scene.bodies.len();
    // The buffers grow from lagging readbacks: start them with room for the scene's initial
    // contacts (twice, for the debris of an impact) instead of letting early steps overflow.
    let (pairs, max_degree) = initial_pairs(scene);
    let mut capacities = NexusCapacities::default()
        .rbd_bodies((n as u32 + 16).max(65536))
        .rbd_collisions((2 * pairs as u32).max(4096));
    // A body's constraints all need distinct colors.
    capacities.rbd.solver_colors = (max_degree as u32 + 8).max(capacities.rbd.solver_colors);
    let mut state = NexusState::new(capacities);
    let groups = collision_groups(scene);

    let mut shapes = Vec::with_capacity(n);
    for (body, groups) in scene.bodies.iter().zip(groups) {
        let builder = if body.density > 0.0 {
            RigidBodyBuilder::dynamic()
        } else {
            RigidBodyBuilder::fixed()
        };
        let rb = builder
            .can_sleep(false)
            .pose(Pose::from_parts(body.position, body.rotation))
            .linvel(body.linvel)
            .angvel(body.angvel)
            .build();
        let collider = match body.radius {
            Some(r) => ColliderBuilder::ball(r),
            None => {
                ColliderBuilder::cuboid(body.size.x / 2.0, body.size.y / 2.0, body.size.z / 2.0)
            }
        };
        // Frictions mix as sqrt(a * b): multiplying square roots gives exactly that.
        let collider = collider
            .density(body.density.max(0.0))
            .friction(body.friction.sqrt())
            .friction_combine_rule(CoefficientCombineRule::Multiply)
            .collision_groups(groups)
            .build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(rb, collider, RbdCoupling::None);
        shapes.push(RenderShape {
            handle,
            shape,
            color: body.color,
        });
    }

    for joint in &scene.joints {
        let builder = SphericalJointBuilder::new()
            .local_anchor1(joint.ra)
            .local_anchor2(joint.rb);
        state.insert_impulse_joint(shapes[joint.a].handle, shapes[joint.b].handle, builder);
    }

    (state, shapes)
}

/// Gravity of the stress scenes (Z-up).
pub const GRAVITY: [f32; 3] = [0.0, 0.0, -10.0];

/// Builds `scene` into a fresh [`NexusState`] and runs it until the viewer switches demos.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
    scene: Scene,
) -> anyhow::Result<NexusState> {
    let (mut state, shapes) = build_state(&scene);
    for s in &shapes {
        match s.color {
            Some(color) => {
                viewer.insert_shape_with_color(s.handle, &s.shape, Pose::IDENTITY, color)
            }
            None => viewer.insert_shape(s.handle, &s.shape, Pose::IDENTITY),
        }
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, 2.0, -3.0));
    viewer.set_up_axis(Vector::Z);
    viewer.set_camera(scene.camera.eye(), scene.camera.target);

    let result = simulate(viewer, pipeline, &mut state, &mut timestamps).await;
    // The other demos are Y-up.
    viewer.set_up_axis(Vector::Y);
    result.map(|_| state)
}

async fn simulate(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
    state: &mut NexusState,
    timestamps: &mut GpuTimestamps,
) -> anyhow::Result<()> {
    state.finalize(viewer.backend()).await?;
    state.set_rbd_gravity(viewer.backend(), GRAVITY);

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), state, Some(timestamps))
                .await?;
        }
        viewer.sync(state, Some(timestamps)).await?;
    }

    Ok(())
}
