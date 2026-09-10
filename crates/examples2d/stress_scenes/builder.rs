//! A small scene description (boxes and welds), translated into a nexus scene when run.
//!
//! Conventions: box sizes are full widths, positions are `[x, y, angle]`, a zero density makes
//! a body static, and gravity is -10.

use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use rapier2d::glamx::IVec2;
use rapier2d::prelude::*;

/// Collision-group bit shared by every body that isn't excluded from colliding with another one.
const FREE_GROUP: u32 = 1 << 31;
/// Number of collision-group bits available to the other bodies.
const NUM_CLASSES: u32 = 31;

pub struct Rigid {
    /// Full widths along each local axis.
    pub size: Vector,
    pub density: f32,
    pub friction: f32,
    pub position: Vector,
    pub angle: f32,
    pub linvel: Vector,
    pub angvel: f32,
}

/// A weld between two body-local anchors.
pub struct FixedJoint {
    pub a: usize,
    pub b: usize,
    pub ra: Vector,
    pub rb: Vector,
}

/// Camera center and zoom (pixels per meter).
pub struct Camera {
    pub center: Vector,
    pub zoom: f32,
}

impl Camera {
    pub fn new(x: f32, y: f32, zoom: f32) -> Self {
        Self {
            center: Vector::new(x, y),
            zoom,
        }
    }
}

pub struct Scene {
    pub bodies: Vec<Rigid>,
    pub joints: Vec<FixedJoint>,
    /// Body pairs that must not collide (on top of the jointed ones).
    pub ignored: Vec<(usize, usize)>,
    pub camera: Camera,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            bodies: vec![],
            joints: vec![],
            ignored: vec![],
            camera,
        }
    }

    /// A box of full widths `size` at `[x, y, angle]`; static if `density` is zero.
    pub fn rigid(&mut self, size: [f32; 2], density: f32, friction: f32, pos: [f32; 3]) -> usize {
        self.bodies.push(Rigid {
            size: Vector::from(size),
            density,
            friction,
            position: Vector::new(pos[0], pos[1]),
            angle: pos[2],
            linvel: Vector::ZERO,
            angvel: 0.0,
        });
        self.bodies.len() - 1
    }

    /// Sets a body's `[vx, vy, angular velocity]`.
    pub fn set_velocity(&mut self, id: usize, velocity: [f32; 3]) {
        self.bodies[id].linvel = Vector::new(velocity[0], velocity[1]);
        self.bodies[id].angvel = velocity[2];
    }

    pub fn fixed_joint(&mut self, a: usize, b: usize, ra: [f32; 2], rb: [f32; 2]) {
        self.joints.push(FixedJoint {
            a,
            b,
            ra: Vector::from(ra),
            rb: Vector::from(rb),
        });
    }

    pub fn ignore_collision(&mut self, a: usize, b: usize) {
        self.ignored.push((a, b));
    }

    /// Every pair of bodies that must not collide: jointed pairs and explicitly ignored ones.
    fn excluded_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.joints
            .iter()
            .map(|j| (j.a, j.b))
            .chain(self.ignored.iter().copied())
    }
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

/// Assigns collision groups so that no excluded pair collides.
///
/// Every body excluded from colliding with another gets one of 31 group bits, chosen to differ
/// from its excluded neighbors and their own neighbors, and filters out its neighbors' bits.
/// The other bodies share the last bit and collide with everything. The price is that an
/// excluded body also ignores far-away bodies that happen to share a neighbor's bit.
fn collision_groups(scene: &Scene) -> Vec<InteractionGroups> {
    let n = scene.bodies.len();
    let mut adjacency = vec![vec![]; n];
    for (a, b) in scene.excluded_pairs() {
        adjacency[a].push(b);
        adjacency[b].push(a);
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
            let (sin, cos) = b.angle.sin_cos();
            let h = b.size / 2.0;
            let half = Vector::new(
                cos.abs() * h.x + sin.abs() * h.y,
                sin.abs() * h.x + cos.abs() * h.y,
            ) + Vector::splat(MARGIN);
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
    let cell_of = |p: Vector| (p / cell).floor().as_ivec2();
    let overlap = |a: usize, b: usize| {
        let ((lo1, hi1), (lo2, hi2)) = (aabbs[a], aabbs[b]);
        lo1.cmple(hi2).all() && lo2.cmple(hi1).all()
    };
    let solved = |i: usize| scene.bodies[i].density > 0.0;

    let mut grid: std::collections::HashMap<IVec2, Vec<usize>> = Default::default();
    let mut large = vec![];
    for (i, (lo, hi)) in aabbs.iter().enumerate() {
        if (*hi - *lo).max_element() > cell {
            large.push(i);
            continue;
        }
        let (c0, c1) = (cell_of(*lo), cell_of(*hi));
        for x in c0.x..=c1.x {
            for y in c0.y..=c1.y {
                grid.entry(IVec2::new(x, y)).or_default().push(i);
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

/// The gravity of the stress scenes.
pub const GRAVITY: [f32; 3] = [0.0, -10.0, 0.0];

/// Builds `scene` into a fresh [`NexusState`], with each body's handle and shape for rendering.
pub fn build_state(scene: &Scene) -> (NexusState, Vec<(RigidBodyHandle, SharedShape)>) {
    let n = scene.bodies.len();
    // The buffers grow from lagging readbacks: start them with room for the scene's initial
    // contacts (twice, for the bodies coming into contact later) instead of letting early steps
    // overflow.
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
            .translation(body.position)
            .rotation(body.angle)
            .linvel(body.linvel)
            .angvel(body.angvel)
            .build();
        // Frictions mix as sqrt(a * b): multiplying square roots gives exactly that.
        let collider = ColliderBuilder::cuboid(body.size.x / 2.0, body.size.y / 2.0)
            .density(body.density.max(0.0))
            .friction(body.friction.sqrt())
            .friction_combine_rule(CoefficientCombineRule::Multiply)
            .collision_groups(groups)
            .build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(rb, collider, RbdCoupling::None);
        shapes.push((handle, shape));
    }

    for joint in &scene.joints {
        let builder = FixedJointBuilder::new()
            .local_anchor1(joint.ra)
            .local_anchor2(joint.rb);
        state.insert_impulse_joint(shapes[joint.a].0, shapes[joint.b].0, builder);
    }
    (state, shapes)
}

/// Builds `scene` into a fresh [`NexusState`] and runs it until the viewer switches demos.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
    scene: Scene,
) -> anyhow::Result<NexusState> {
    let (mut state, shapes) = build_state(&scene);
    for (handle, shape) in &shapes {
        viewer.insert_shape(*handle, shape, Pose::IDENTITY);
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer.set_camera_2d(scene.camera.center, scene.camera.zoom);
    state.finalize(viewer.backend()).await?;
    state.set_rbd_gravity(viewer.backend(), GRAVITY);

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
