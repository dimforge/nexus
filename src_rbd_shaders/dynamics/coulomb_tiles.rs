//! Coulomb contact constraints (friction at each contact point) in typed tiles, and their
//! kernels.
use super::constraint::{ContactPoint, TwoBodyConstraint};
use super::contact_tiles::{HeaderTile, TILE_LEN, WarmstartTile, tile_lane};
use super::solver_utils::{ContactPointAccess, ContactSolverData};
use super::{RbdSimParams, Velocity};
use crate::utils::Slice;
use crate::{Pose, Vector, gcross};
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim2")]
use khal_std::glamx::Vec2;
#[cfg(feature = "dim3")]
use {super::contact_tiles::VecScalar, khal_std::glamx::Vec2};

/// Accumulated friction impulses, and the first two friction effective-mass terms.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct FrictionRows {
    pub impulse: Vec2,
    pub k: Vec2,
}

/// One contact point of the constraints of a [`CoulombTile`].
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct CoulombPointTile {
    /// Lever arm from body A's center of mass, and the accumulated normal impulse.
    pub arm_a: [VecScalar; TILE_LEN],
    /// Lever arm from body B's center of mass, and the inverse effective normal mass.
    pub arm_b: [VecScalar; TILE_LEN],
    /// The point in body A's frame, and the signed distance at the start of the step.
    pub local_a: [VecScalar; TILE_LEN],
    /// The point in body B's frame, and the target normal velocity.
    pub local_b: [VecScalar; TILE_LEN],
    pub friction: [FrictionRows; TILE_LEN],
}

/// [`TILE_LEN`] consecutive Coulomb constraints.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct CoulombTile {
    pub header: HeaderTile,
    /// Each point's friction coupling term (`ContactPoint::tangent_k[2]`).
    pub coupling: [[f32; 4]; TILE_LEN],
    pub points: [CoulombPointTile; 4],
    pub warmstart: WarmstartTile,
}

/// Two vectors of a contact point, one per body.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct VectorPair {
    pub a: Vec2,
    pub b: Vec2,
}

/// The normal row of a contact point.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NormalRow {
    /// Accumulated impulse.
    pub impulse: f32,
    /// Inverse of the effective mass.
    pub mass: f32,
    /// Signed distance at the start of the step.
    pub dist: f32,
    /// Target velocity (restitution).
    pub vel: f32,
}

/// The friction row of a contact point.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TangentRow {
    /// Accumulated impulse.
    pub impulse: f32,
    /// Inverse of the effective mass.
    pub mass: f32,
}

/// One contact point of the constraints of a [`CoulombTile`].
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct CoulombPointTile {
    /// Lever arms from the centers of mass of bodies A and B.
    pub arms: [VectorPair; TILE_LEN],
    /// The point in the frames of bodies A and B.
    pub anchors: [VectorPair; TILE_LEN],
    pub normal: [NormalRow; TILE_LEN],
    pub tangent: [TangentRow; TILE_LEN],
}

/// [`TILE_LEN`] consecutive Coulomb constraints.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct CoulombTile {
    pub header: HeaderTile,
    pub points: [CoulombPointTile; 2],
    pub warmstart: WarmstartTile,
}

impl CoulombTile {
    #[cfg(feature = "dim3")]
    #[inline(always)]
    fn point(&self, lane: usize, k: usize) -> ContactPoint {
        let p = self.points.at(k);
        let arm_a = p.arm_a.read(lane);
        let arm_b = p.arm_b.read(lane);
        let local_a = p.local_a.read(lane);
        let local_b = p.local_b.read(lane);
        let friction = p.friction.read(lane);
        ContactPoint {
            r_a: arm_a.vec,
            normal_impulse: arm_a.scalar,
            r_b: arm_b.vec,
            normal_mass: arm_b.scalar,
            local_pt_a: local_a.vec,
            dist: local_a.scalar,
            local_pt_b: local_b.vec,
            normal_vel: local_b.scalar,
            tangent_impulse: friction.impulse,
            tangent_k: [friction.k.x, friction.k.y, self.coupling.at(lane).read(k)],
            _padding: [0; 3],
        }
    }

    #[cfg(feature = "dim2")]
    #[inline(always)]
    fn point(&self, lane: usize, k: usize) -> ContactPoint {
        let p = self.points.at(k);
        let arms = p.arms.read(lane);
        let anchors = p.anchors.read(lane);
        let normal = p.normal.read(lane);
        let tangent = p.tangent.read(lane);
        ContactPoint {
            r_a: arms.a,
            r_b: arms.b,
            local_pt_a: anchors.a,
            local_pt_b: anchors.b,
            normal_impulse: normal.impulse,
            normal_mass: normal.mass,
            dist: normal.dist,
            normal_vel: normal.vel,
            tangent_impulse: [tangent.impulse],
            tangent_mass: tangent.mass,
            _padding: [0; 2],
        }
    }

    #[cfg(feature = "dim3")]
    #[inline(always)]
    fn write_point(&mut self, lane: usize, k: usize, p: &ContactPoint) -> f32 {
        let dst = self.points.at_mut(k);
        dst.arm_a.write(
            lane,
            VecScalar {
                vec: p.r_a,
                scalar: p.normal_impulse,
            },
        );
        dst.arm_b.write(
            lane,
            VecScalar {
                vec: p.r_b,
                scalar: p.normal_mass,
            },
        );
        dst.local_a.write(
            lane,
            VecScalar {
                vec: p.local_pt_a,
                scalar: p.dist,
            },
        );
        dst.local_b.write(
            lane,
            VecScalar {
                vec: p.local_pt_b,
                scalar: p.normal_vel,
            },
        );
        dst.friction.write(
            lane,
            FrictionRows {
                impulse: p.tangent_impulse,
                k: Vec2::new(p.tangent_k[0], p.tangent_k[1]),
            },
        );
        p.tangent_k[2]
    }

    #[cfg(feature = "dim2")]
    #[inline(always)]
    fn write_point(&mut self, lane: usize, k: usize, p: &ContactPoint) {
        let dst = self.points.at_mut(k);
        dst.arms.write(lane, VectorPair { a: p.r_a, b: p.r_b });
        dst.anchors.write(
            lane,
            VectorPair {
                a: p.local_pt_a,
                b: p.local_pt_b,
            },
        );
        dst.normal.write(
            lane,
            NormalRow {
                impulse: p.normal_impulse,
                mass: p.normal_mass,
                dist: p.dist,
                vel: p.normal_vel,
            },
        );
        dst.tangent.write(
            lane,
            TangentRow {
                impulse: p.tangent_impulse[0],
                mass: p.tangent_mass,
            },
        );
    }

    /// Writes the constraint of `lane`.
    #[inline(always)]
    pub fn write_constraint(&mut self, lane: usize, c: &TwoBodyConstraint) {
        let h = ContactSolverData::from_constraint(c);
        #[cfg(feature = "dim2")]
        {
            self.header.write(lane, &h);
            crate::dynamics::solver_utils::for_contact_point!(k, c.len, {
                self.write_point(lane, k, c.points.at(k));
            });
        }
        #[cfg(feature = "dim3")]
        {
            self.header.write(lane, &h, [0.0; 2]);
            let mut coupling = [0.0; 4];
            crate::dynamics::solver_utils::for_contact_point!(k, c.len, {
                coupling.write(k, self.write_point(lane, k, c.points.at(k)));
            });
            self.coupling.write(lane, coupling);
        }
    }

    /// The constraint of `lane`. Its identity fields are left to their defaults: they live in the
    /// contact links.
    #[inline(always)]
    pub fn constraint(&self, lane: usize) -> TwoBodyConstraint {
        let h = self.header.read(lane);
        let mut c = TwoBodyConstraint {
            dir_a: h.dir_a,
            len: h.len,
            limit: h.limit,
            im_a: h.im_a,
            im_b: h.im_b,
            ii_a: h.ii_a,
            ii_b: h.ii_b,
            solver_body_a: h.solver_body_a,
            solver_body_b: h.solver_body_b,
            vel_slot_a: h.vel_slot_a,
            vel_slot_b: h.vel_slot_b,
            ..Default::default()
        };
        #[cfg(feature = "dim3")]
        {
            c.tangent_a = h.tangent_a;
        }
        crate::dynamics::solver_utils::for_contact_point!(k, h.len, {
            c.points.write(k, self.point(lane, k));
        });
        c
    }
}

/// The tile type of this model's kernels (see `tile_kernels.rs`).
pub type ContactTile = CoulombTile;

#[inline(always)]
fn header(tiles: &[CoulombTile], index: usize) -> ContactSolverData {
    let (tile, lane) = tile_lane(index);
    tiles.at(tile).header.read(lane)
}

#[inline(always)]
fn read_constraint(tiles: &[CoulombTile], index: usize) -> TwoBodyConstraint {
    let (tile, lane) = tile_lane(index);
    tiles.at(tile).constraint(lane)
}

#[inline(always)]
fn write_constraint(tiles: &mut [CoulombTile], index: usize, c: &TwoBodyConstraint) {
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
    tiles: &'a mut [CoulombTile],
    index: usize,
}
impl ContactPointAccess for TilePoints<'_> {
    #[inline(always)]
    fn read_point(&self, k: usize) -> ContactPoint {
        let (tile, lane) = tile_lane(self.index);
        self.tiles.at(tile).point(lane, k)
    }
    #[cfg(feature = "dim3")]
    #[inline(always)]
    fn write_normal_impulse(&mut self, k: usize, value: f32) {
        let (tile, lane) = tile_lane(self.index);
        self.tiles
            .at_mut(tile)
            .points
            .at_mut(k)
            .arm_a
            .at_mut(lane)
            .scalar = value;
    }
    #[cfg(feature = "dim3")]
    #[inline(always)]
    fn write_tangent_impulse(&mut self, k: usize, value: Vec2) {
        let (tile, lane) = tile_lane(self.index);
        let point = self.tiles.at_mut(tile).points.at_mut(k);
        point.friction.at_mut(lane).impulse = value;
    }
    #[cfg(feature = "dim2")]
    #[inline(always)]
    fn write_normal_impulse(&mut self, k: usize, value: f32) {
        let (tile, lane) = tile_lane(self.index);
        self.tiles
            .at_mut(tile)
            .points
            .at_mut(k)
            .normal
            .at_mut(lane)
            .impulse = value;
    }
    #[cfg(feature = "dim2")]
    #[inline(always)]
    fn write_tangent_impulse(&mut self, k: usize, value: [f32; 1]) {
        let (tile, lane) = tile_lane(self.index);
        self.tiles
            .at_mut(tile)
            .points
            .at_mut(k)
            .tangent
            .at_mut(lane)
            .impulse = value[0];
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn solve_constraint(
    h: &ContactSolverData,
    tiles: &mut [CoulombTile],
    index: usize,
    poses: &Slice<Pose>,
    params: &RbdSimParams,
    a: &mut Velocity,
    b: &mut Velocity,
    bias: bool,
    friction: bool,
) {
    h.solve_points(
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
    tiles: &[CoulombTile],
    index: usize,
    h: &ContactSolverData,
) -> (Velocity, Velocity) {
    let (tile, lane) = tile_lane(index);
    let tile = tiles.at(tile);
    let tangents = h.tangents();
    let mut force = Vector::ZERO;
    let mut torque_a = crate::AngVector::default();
    let mut torque_b = crate::AngVector::default();
    crate::dynamics::solver_utils::for_contact_point!(k, h.len as usize, {
        let p = tile.point(lane, k);
        #[cfg(feature = "dim2")]
        let f = h.dir_a * p.normal_impulse + tangents.read(0) * p.tangent_impulse[0];
        #[cfg(feature = "dim3")]
        let f = h.dir_a * p.normal_impulse
            + tangents.read(0) * p.tangent_impulse.x
            + tangents.read(1) * p.tangent_impulse.y;
        force += f;
        torque_a += gcross(p.r_a, f);
        torque_b += gcross(p.r_b, f);
    });
    let mut a = Velocity::default();
    let mut b = Velocity::default();
    a.linear += h.im_a * force;
    a.angular += h.ii_a_mul(torque_a);
    b.linear -= h.im_b * force;
    b.angular -= h.ii_b_mul(torque_b);
    a.padding1 = h.solver_body_a;
    b.padding1 = h.solver_body_b;
    (a, b)
}

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
        let mut tiles = vec![CoulombTile::default(); 2];
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
        crate::dynamics::solver_utils::for_contact_point!(k, expected.len, {
            assert_eq!(
                solved.points[k].normal_impulse,
                expected.points[k].normal_impulse
            );
            assert_eq!(
                solved.points[k].tangent_impulse,
                expected.points[k].tangent_impulse
            );
        });
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

#[cfg(all(test, not(target_arch_is_gpu)))]
#[path = "../tests/contact_constraints.rs"]
mod tests;
