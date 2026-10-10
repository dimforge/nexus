//! Twist contact constraints (central sliding friction and twist friction per manifold) in
//! typed tiles, and their kernels: 21 16-byte solver fields per constraint instead of
//! Coulomb's 29.
use super::contact_tiles::{HeaderTile, TILE_LEN, VecScalar, WarmstartTile, tile_lane};
use super::solver_utils::{ContactPointAccess, ContactSolverData};
use super::twist_friction::{self, TwistAccess, TwistContactPoint};
use super::{
    ContactPoint, RbdSimParams, TwistConstraint as TwoBodyConstraint, TwistFriction, Velocity,
};
use crate::Pose;
use crate::utils::Slice;
use crunchy::unroll;
use khal_std::glamx::{Quat, Vec2, Vec3};
use khal_std::index::MaybeIndexUnchecked;

/// Accumulated central friction impulse, and the coupling term `2 k01` of its effective mass.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwistFrictionRows {
    pub impulse: Vec2,
    pub k01: f32,
    pub _padding: f32,
}

/// [`TILE_LEN`] consecutive twist constraints. The header stores the twist mass next to body
/// B's inertia diagonal, and the twist impulse next to its off-diagonal terms.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TwistTile {
    pub header: HeaderTile,
    /// Offset from the frozen body-A to body-B points, and the friction mass term `k00`.
    pub offset_b: [VecScalar; TILE_LEN],
    /// Friction center relative to body A, and the friction mass term `k11`.
    pub center_a: [VecScalar; TILE_LEN],
    pub friction: [TwistFrictionRows; TILE_LEN],
    /// Body A's frozen frame.
    pub frame_a: [Quat; TILE_LEN],
    /// Body B's frozen frame.
    pub frame_b: [Quat; TILE_LEN],
    /// Each point's accumulated normal impulse.
    pub normal_impulse: [[f32; 4]; TILE_LEN],
    /// Each point's signed distance at the start of the step.
    pub dist: [[f32; 4]; TILE_LEN],
    /// Each point's target normal velocity.
    pub normal_vel: [[f32; 4]; TILE_LEN],
    /// Each point's twist friction radius.
    pub radius: [[f32; 4]; TILE_LEN],
    /// Each point's lever arm from body A's center of mass, and inverse effective normal mass.
    pub arm_a: [[VecScalar; TILE_LEN]; 4],
    pub warmstart: WarmstartTile,
}

impl TwistTile {
    #[inline(always)]
    fn point(&self, lane: usize, k: usize) -> ContactPoint {
        let a = self.arm_a.at(k).read(lane);
        let rb = a.vec + self.offset_b.read(lane).vec;
        ContactPoint {
            r_a: a.vec,
            r_b: rb,
            normal_impulse: self.normal_impulse.at(lane).read(k),
            normal_mass: a.scalar,
            local_pt_a: self.frame_a.read(lane).inverse() * a.vec,
            local_pt_b: self.frame_b.read(lane).inverse() * rb,
            dist: self.dist.at(lane).read(k),
            normal_vel: self.normal_vel.at(lane).read(k),
            ..ContactPoint::default()
        }
    }

    #[inline(always)]
    fn friction(&self, lane: usize) -> TwistFriction {
        let offset_b = self.offset_b.read(lane);
        let center_a = self.center_a.read(lane);
        let friction = self.friction.read(lane);
        TwistFriction {
            center_a: center_a.vec,
            tangent_k: Vec3::new(offset_b.scalar, center_a.scalar, friction.k01),
            tangent_impulse: friction.impulse,
            twist_mass: self.header.inertia_diag_b.read(lane).scalar,
            twist_impulse: self.header.inertia_off_b.read(lane).scalar,
            _padding: [0; 2],
        }
    }

    /// Writes the constraint of `lane`.
    #[inline(always)]
    pub fn write_constraint(&mut self, lane: usize, c: &TwoBodyConstraint) {
        let f = &c.friction;
        self.header
            .write(lane, &c.solver_data(), [f.twist_mass, f.twist_impulse]);
        self.offset_b.write(
            lane,
            VecScalar {
                vec: c.offset_b,
                scalar: f.tangent_k.x,
            },
        );
        self.center_a.write(
            lane,
            VecScalar {
                vec: f.center_a,
                scalar: f.tangent_k.y,
            },
        );
        self.friction.write(
            lane,
            TwistFrictionRows {
                impulse: f.tangent_impulse,
                k01: f.tangent_k.z,
                _padding: 0.0,
            },
        );
        self.frame_a.write(lane, c.frame_a);
        self.frame_b.write(lane, c.frame_b);
        let mut impulses = [0.0; 4];
        let mut distances = [0.0; 4];
        let mut velocities = [0.0; 4];
        let mut radii = [0.0; 4];
        unroll! { for k in 0..4 { if k < c.len as usize {
            let p = c.points.at(k);
            self.arm_a.at_mut(k).write(lane, VecScalar { vec: p.r_a, scalar: p.normal_mass });
            impulses.write(k, p.normal_impulse);
            distances.write(k, p.dist);
            velocities.write(k, p.normal_vel);
            radii.write(k, p.radius);
        } }}
        self.normal_impulse.write(lane, impulses);
        self.dist.write(lane, distances);
        self.normal_vel.write(lane, velocities);
        self.radius.write(lane, radii);
    }

    /// The constraint of `lane`. Its identity fields are left to their defaults: they live in the
    /// contact links.
    #[inline(always)]
    pub fn constraint(&self, lane: usize) -> TwoBodyConstraint {
        let h = self.header.read(lane);
        let mut c = TwoBodyConstraint {
            dir_a: h.dir_a,
            len: h.len,
            tangent_a: h.tangent_a,
            limit: h.limit,
            im_a: h.im_a,
            im_b: h.im_b,
            ii_a: h.ii_a,
            ii_b: h.ii_b,
            solver_body_a: h.solver_body_a,
            solver_body_b: h.solver_body_b,
            vel_slot_a: h.vel_slot_a,
            vel_slot_b: h.vel_slot_b,
            offset_b: self.offset_b.read(lane).vec,
            frame_a: self.frame_a.read(lane),
            frame_b: self.frame_b.read(lane),
            friction: self.friction(lane),
            ..Default::default()
        };
        let impulses = self.normal_impulse.read(lane);
        let distances = self.dist.read(lane);
        let velocities = self.normal_vel.read(lane);
        let radii = self.radius.read(lane);
        unroll! { for k in 0..4 { if k < h.len as usize {
            let a = self.arm_a.at(k).read(lane);
            c.points.write(k, TwistContactPoint {
                r_a: a.vec,
                normal_impulse: impulses.read(k),
                normal_mass: a.scalar,
                dist: distances.read(k),
                normal_vel: velocities.read(k),
                radius: radii.read(k),
            });
        } }}
        c
    }
}

/// The tile type of this model's kernels (see `tile_kernels.rs`).
pub type ContactTile = TwistTile;

#[inline(always)]
fn header(tiles: &[TwistTile], index: usize) -> ContactSolverData {
    let (tile, lane) = tile_lane(index);
    tiles.at(tile).header.read(lane)
}

#[inline(always)]
fn read_constraint(tiles: &[TwistTile], index: usize) -> TwoBodyConstraint {
    let (tile, lane) = tile_lane(index);
    tiles.at(tile).constraint(lane)
}

#[inline(always)]
fn write_constraint(tiles: &mut [TwistTile], index: usize, c: &TwoBodyConstraint) {
    let (tile, lane) = tile_lane(index);
    tiles.at_mut(tile).write_constraint(lane, c);
}

/// The velocity changes of the bodies of `c` applying its accumulated impulses, tagged with
/// their body index (for the warmstart gather).
#[inline(always)]
fn constraint_warmstart(c: &TwoBodyConstraint) -> (Velocity, Velocity) {
    let (mut a, mut b) = (Velocity::default(), Velocity::default());
    c.warmstart_constraint(&mut a, &mut b);
    a.padding1 = c.solver_body_a;
    b.padding1 = c.solver_body_b;
    (a, b)
}

struct TilePoints<'a> {
    tiles: &'a mut [TwistTile],
    index: usize,
}
impl TilePoints<'_> {
    #[inline(always)]
    fn tile(&self) -> (&TwistTile, usize) {
        let (tile, lane) = tile_lane(self.index);
        (self.tiles.at(tile), lane)
    }
}
impl ContactPointAccess for TilePoints<'_> {
    #[inline(always)]
    fn read_point(&self, k: usize) -> ContactPoint {
        let (tile, lane) = self.tile();
        tile.point(lane, k)
    }
    #[inline(always)]
    fn write_normal_impulse(&mut self, k: usize, value: f32) {
        let (tile, lane) = tile_lane(self.index);
        self.tiles
            .at_mut(tile)
            .normal_impulse
            .at_mut(lane)
            .write(k, value);
    }
    #[inline(always)]
    fn write_tangent_impulse(&mut self, _k: usize, _value: Vec2) {}
}
impl TwistAccess for TilePoints<'_> {
    #[inline(always)]
    fn friction(&self) -> TwistFriction {
        let (tile, lane) = self.tile();
        tile.friction(lane)
    }
    #[inline(always)]
    fn offset_b(&self) -> Vec3 {
        let (tile, lane) = self.tile();
        tile.offset_b.read(lane).vec
    }
    #[inline(always)]
    fn frames(&self) -> (Quat, Quat) {
        let (tile, lane) = self.tile();
        (tile.frame_a.read(lane), tile.frame_b.read(lane))
    }
    #[inline(always)]
    fn radius(&self, k: usize) -> f32 {
        let (tile, lane) = self.tile();
        tile.radius.at(lane).read(k)
    }
    #[inline(always)]
    fn write_friction(&mut self, tangent: Vec2, twist: f32) {
        let (tile, lane) = tile_lane(self.index);
        let tile = self.tiles.at_mut(tile);
        tile.friction.at_mut(lane).impulse = tangent;
        tile.header.inertia_off_b.at_mut(lane).scalar = twist;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn solve_constraint(
    h: &ContactSolverData,
    tiles: &mut [TwistTile],
    index: usize,
    poses: &Slice<Pose>,
    params: &RbdSimParams,
    a: &mut Velocity,
    b: &mut Velocity,
    bias: bool,
    friction: bool,
) {
    twist_friction::solve(
        h,
        &mut TilePoints { tiles, index },
        poses,
        params,
        a,
        b,
        bias,
        friction,
    );
}

/// The velocity changes of the bodies of the constraint at `index` applying its accumulated
/// impulses, tagged with their body index (for the warmstart gather).
#[inline(always)]
fn tile_warmstart(
    tiles: &[TwistTile],
    index: usize,
    h: &ContactSolverData,
) -> (Velocity, Velocity) {
    let (mut a, mut b) = (Velocity::default(), Velocity::default());
    let (tile, lane) = tile_lane(index);
    twist_friction::warmstart(
        h,
        &TileConstraint {
            tile: tiles.at(tile),
            lane,
        },
        &mut a,
        &mut b,
    );
    a.padding1 = h.solver_body_a;
    b.padding1 = h.solver_body_b;
    (a, b)
}

/// Read-only access to a constraint in a tile, for the warmstart.
struct TileConstraint<'a> {
    tile: &'a TwistTile,
    lane: usize,
}
impl ContactPointAccess for TileConstraint<'_> {
    #[inline(always)]
    fn read_point(&self, k: usize) -> ContactPoint {
        self.tile.point(self.lane, k)
    }
    #[inline(always)]
    fn write_normal_impulse(&mut self, _k: usize, _value: f32) {}
    #[inline(always)]
    fn write_tangent_impulse(&mut self, _k: usize, _value: Vec2) {}
}
impl TwistAccess for TileConstraint<'_> {
    #[inline(always)]
    fn friction(&self) -> TwistFriction {
        self.tile.friction(self.lane)
    }
    #[inline(always)]
    fn offset_b(&self) -> Vec3 {
        self.tile.offset_b.read(self.lane).vec
    }
    #[inline(always)]
    fn frames(&self) -> (Quat, Quat) {
        (
            self.tile.frame_a.read(self.lane),
            self.tile.frame_b.read(self.lane),
        )
    }
    #[inline(always)]
    fn radius(&self, k: usize) -> f32 {
        self.tile.radius.at(self.lane).read(k)
    }
    #[inline(always)]
    fn write_friction(&mut self, _tangent: Vec2, _twist: f32) {}
}

// The kernels are shared by both friction models, each instantiating them with its own tiles.
#[allow(clippy::duplicate_mod)]
#[path = "tile_kernels.rs"]
mod kernels;
pub use kernels::*;

/// Asserts that solving `initial` from a tile gives `expected` and its body velocities, for
/// constraints at the start and the end of tiles.
#[cfg(all(test, not(target_arch_is_gpu)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_tile_equivalent(
    initial: TwoBodyConstraint,
    poses: &Slice<Pose>,
    params: &RbdSimParams,
    initial_a: Velocity,
    initial_b: Velocity,
    use_bias: bool,
    solve_friction: bool,
    expected: &TwoBodyConstraint,
    expected_a: Velocity,
    expected_b: Velocity,
) {
    for index in [0usize, 31, 32, 63] {
        let mut tiles = vec![TwistTile::default(); 2];
        write_constraint(&mut tiles, index, &initial);
        let h = header(&tiles, index);
        let (mut a, mut b) = (initial_a, initial_b);
        solve_constraint(
            &h,
            &mut tiles,
            index,
            poses,
            params,
            &mut a,
            &mut b,
            use_bias,
            solve_friction,
        );
        let solved = read_constraint(&tiles, index);
        for k in 0..expected.len as usize {
            assert_eq!(
                solved.points[k].normal_impulse,
                expected.points[k].normal_impulse
            );
        }
        assert_eq!(
            solved.friction.tangent_impulse,
            expected.friction.tangent_impulse
        );
        assert_eq!(
            solved.friction.twist_impulse,
            expected.friction.twist_impulse
        );
        assert_eq!(
            (a.linear, a.angular, b.linear, b.angular),
            (
                expected_a.linear,
                expected_a.angular,
                expected_b.linear,
                expected_b.angular
            )
        );
        let (cached_a, cached_b) = tile_warmstart(&tiles, index, &h);
        let (mut warm_a, mut warm_b) = (Velocity::default(), Velocity::default());
        expected.warmstart_constraint(&mut warm_a, &mut warm_b);
        assert_eq!(
            (
                cached_a.linear,
                cached_a.angular,
                cached_b.linear,
                cached_b.angular
            ),
            (warm_a.linear, warm_a.angular, warm_b.linear, warm_b.angular)
        );
    }
}
