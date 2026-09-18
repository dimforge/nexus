//! Typed tiles of contact constraints, the solver's storage for them.
//!
//! Constraints are sorted by color, so every solver iteration walks a contiguous range of them.
//! A tile holds [`TILE_LEN`] consecutive constraints field by field: each field is an array
//! with one entry per constraint, so neighboring lanes read neighboring memory.

use super::Velocity;
use super::solver_utils::ContactSolverData;
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim2")]
use khal_std::glamx::Vec2;
#[cfg(feature = "dim3")]
use {super::SymInertia, khal_std::glamx::Vec3};

/// Number of neighboring constraints stored together in a tile.
pub const TILE_LEN: usize = 32;
/// First sparse color solved by a single workgroup, with barriers between colors.
pub const TAIL_COLOR: u32 = 24;

/// The tile holding the constraint at `index`, and its lane in it.
#[inline(always)]
pub fn tile_lane(index: usize) -> (usize, usize) {
    (index / TILE_LEN, index % TILE_LEN)
}

/// The velocity changes of the constraints' two bodies applying their accumulated impulses
/// (their warmstart), each tagged with its body index in `padding1`. A constraint's pair is
/// contiguous, for the per-body warmstart gather.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct WarmstartTile {
    pub bodies: [[Velocity; 2]; TILE_LEN],
}

/// A vector and a scalar of a contact point.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct VecScalar {
    pub vec: Vec3,
    pub scalar: f32,
}

/// Contact normal and number of active points.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NormalLen {
    pub normal: Vec3,
    pub len: u32,
}

/// First friction direction and friction coefficient.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TangentLimit {
    pub tangent: Vec3,
    pub limit: f32,
}

/// Inverse mass of a body along each axis, and the body's index.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MassBody {
    pub inv_mass: Vec3,
    pub body: u32,
}

/// Inverse inertia terms, and a velocity slot.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct InertiaSlot {
    pub inertia: Vec3,
    pub vel_slot: u32,
}

/// The constraints' common solver header: normal, friction direction, masses and velocity
/// slots.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct HeaderTile {
    pub normal: [NormalLen; TILE_LEN],
    pub tangent: [TangentLimit; TILE_LEN],
    pub body_a: [MassBody; TILE_LEN],
    pub body_b: [MassBody; TILE_LEN],
    /// Body A's inverse inertia diagonal, and its velocity slot.
    pub inertia_diag_a: [InertiaSlot; TILE_LEN],
    /// Body A's inverse inertia off-diagonal terms, and body B's velocity slot.
    pub inertia_off_a: [InertiaSlot; TILE_LEN],
    /// Body B's inverse inertia diagonal, and a spare scalar.
    pub inertia_diag_b: [VecScalar; TILE_LEN],
    /// Body B's inverse inertia off-diagonal terms, and a spare scalar.
    pub inertia_off_b: [VecScalar; TILE_LEN],
}

#[cfg(feature = "dim3")]
impl HeaderTile {
    #[inline(always)]
    pub(crate) fn read(&self, lane: usize) -> ContactSolverData {
        let normal = self.normal.read(lane);
        let tangent = self.tangent.read(lane);
        let body_a = self.body_a.read(lane);
        let body_b = self.body_b.read(lane);
        let diag_a = self.inertia_diag_a.read(lane);
        let off_a = self.inertia_off_a.read(lane);
        ContactSolverData {
            dir_a: normal.normal,
            len: normal.len,
            tangent_a: tangent.tangent,
            limit: tangent.limit,
            im_a: body_a.inv_mass,
            solver_body_a: body_a.body,
            im_b: body_b.inv_mass,
            solver_body_b: body_b.body,
            ii_a: SymInertia {
                diag: diag_a.inertia,
                off: off_a.inertia,
                _padding0: 0,
                _padding1: 0,
            },
            ii_b: SymInertia {
                diag: self.inertia_diag_b.read(lane).vec,
                off: self.inertia_off_b.read(lane).vec,
                _padding0: 0,
                _padding1: 0,
            },
            vel_slot_a: diag_a.vel_slot,
            vel_slot_b: off_a.vel_slot,
        }
    }

    #[inline(always)]
    pub(crate) fn write(&mut self, lane: usize, h: &ContactSolverData) {
        self.normal.write(
            lane,
            NormalLen {
                normal: h.dir_a,
                len: h.len,
            },
        );
        self.tangent.write(
            lane,
            TangentLimit {
                tangent: h.tangent_a,
                limit: h.limit,
            },
        );
        self.body_a.write(
            lane,
            MassBody {
                inv_mass: h.im_a,
                body: h.solver_body_a,
            },
        );
        self.body_b.write(
            lane,
            MassBody {
                inv_mass: h.im_b,
                body: h.solver_body_b,
            },
        );
        self.inertia_diag_a.write(
            lane,
            InertiaSlot {
                inertia: h.ii_a.diag,
                vel_slot: h.vel_slot_a,
            },
        );
        self.inertia_off_a.write(
            lane,
            InertiaSlot {
                inertia: h.ii_a.off,
                vel_slot: h.vel_slot_b,
            },
        );
        self.inertia_diag_b.write(
            lane,
            VecScalar {
                vec: h.ii_b.diag,
                scalar: 0.0,
            },
        );
        self.inertia_off_b.write(
            lane,
            VecScalar {
                vec: h.ii_b.off,
                scalar: 0.0,
            },
        );
    }
}

/// Contact normal, number of active points and friction coefficient.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NormalLimit {
    pub normal: Vec2,
    pub len: u32,
    pub limit: f32,
}

/// Inverse mass of a body along each axis, its inverse inertia, and its index.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MassBody {
    pub inv_mass: Vec2,
    pub inv_inertia: f32,
    pub body: u32,
}

/// The velocity slots of the two bodies.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct VelSlots {
    pub a: u32,
    pub b: u32,
}

/// The constraints' common solver header: normal, masses and velocity slots.
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct HeaderTile {
    pub normal: [NormalLimit; TILE_LEN],
    pub body_a: [MassBody; TILE_LEN],
    pub body_b: [MassBody; TILE_LEN],
    pub vel_slots: [VelSlots; TILE_LEN],
}

#[cfg(feature = "dim2")]
impl HeaderTile {
    #[inline(always)]
    pub(crate) fn read(&self, lane: usize) -> ContactSolverData {
        let normal = self.normal.read(lane);
        let body_a = self.body_a.read(lane);
        let body_b = self.body_b.read(lane);
        let slots = self.vel_slots.read(lane);
        ContactSolverData {
            dir_a: normal.normal,
            len: normal.len,
            limit: normal.limit,
            im_a: body_a.inv_mass,
            solver_body_a: body_a.body,
            im_b: body_b.inv_mass,
            solver_body_b: body_b.body,
            ii_a: body_a.inv_inertia,
            ii_b: body_b.inv_inertia,
            vel_slot_a: slots.a,
            vel_slot_b: slots.b,
        }
    }

    #[inline(always)]
    pub(crate) fn write(&mut self, lane: usize, h: &ContactSolverData) {
        self.normal.write(
            lane,
            NormalLimit {
                normal: h.dir_a,
                len: h.len,
                limit: h.limit,
            },
        );
        self.body_a.write(
            lane,
            MassBody {
                inv_mass: h.im_a,
                inv_inertia: h.ii_a,
                body: h.solver_body_a,
            },
        );
        self.body_b.write(
            lane,
            MassBody {
                inv_mass: h.im_b,
                inv_inertia: h.ii_b,
                body: h.solver_body_b,
            },
        );
        self.vel_slots.write(
            lane,
            VelSlots {
                a: h.vel_slot_a,
                b: h.vel_slot_b,
            },
        );
    }
}
