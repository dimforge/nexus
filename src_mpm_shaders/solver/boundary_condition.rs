use crate::Vector;

/// Boundary condition type constants.
///
/// Represented as `u32` for GPU compatibility instead of a Rust enum.
pub const BOUNDARY_CONDITION_STICK: u32 = 0;
pub const BOUNDARY_CONDITION_SLIP: u32 = 1;
pub const BOUNDARY_CONDITION_SEPARATE: u32 = 2;
pub const BOUNDARY_CONDITION_NON_REFLECTING: u32 = 3;

/// A boundary condition applied to grid nodes at domain boundaries or collider surfaces.
///
/// The `ty` field should be one of the `BOUNDARY_CONDITION_*` constants.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BoundaryCondition {
    /// The type of boundary condition (see `BOUNDARY_CONDITION_*` constants).
    pub ty: u32,
    /// Friction coefficient. Only meaningful when `ty` is `Separate`.
    pub friction: f32,
    /// Pads the struct to 16 bytes so it can be an element of the `BodyMaterials`
    /// uniform array (std140 requires a 16-byte array stride). Kept on the host
    /// (`bytemuck`) type too so both layouts stay identical.
    pub _pad0: u32,
    pub _pad1: u32,
}

impl BoundaryCondition {
    pub const fn stick() -> BoundaryCondition {
        BoundaryCondition::new(0, 0.0)
    }

    pub const fn slip() -> BoundaryCondition {
        BoundaryCondition::new(1, 0.0)
    }

    pub const fn separate(friction: f32) -> BoundaryCondition {
        BoundaryCondition::new(2, friction)
    }
}

/// Maximum number of collision bodies coupled to the MPM domain (CPIC limit).
pub const MAX_COLLISION_BODIES: usize = 16;

/// Per-body boundary conditions, passed as a **uniform** (read-only, ≤16 bodies)
/// so the MPM kernels consuming it stay within the 8-storage-buffer WebGPU
/// limit. Indexed by body id.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BodyMaterials {
    pub mats: [BoundaryCondition; MAX_COLLISION_BODIES],
}

impl BodyMaterials {
    /// All-zero materials, usable as a `const` placeholder for kernels that
    /// don't use rigid-body coupling (e.g. the non-CPIC P2G path).
    pub const EMPTY: BodyMaterials = BodyMaterials {
        mats: [BoundaryCondition::stick(); MAX_COLLISION_BODIES],
    };
}

impl BoundaryCondition {
    /// Creates a new boundary condition.
    pub const fn new(ty: u32, friction: f32) -> Self {
        Self {
            ty,
            friction,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Projects a velocity according to this boundary condition.
    /// `n` is the boundary normal (pointing inward).
    pub fn project_velocity(&self, vel: Vector, n: Vector) -> Vector {
        if self.ty == BOUNDARY_CONDITION_STICK {
            return Vector::ZERO;
        }

        if self.ty == BOUNDARY_CONDITION_SLIP {
            let normal_vel = vel.dot(n);
            let tangent_vel = vel - n * normal_vel;
            return tangent_vel;
        }

        if self.ty == BOUNDARY_CONDITION_SEPARATE {
            let normal_vel = vel.dot(n);

            if normal_vel < 0.0 {
                let tangent_vel = vel - n * normal_vel;
                let tangent_vel_len = tangent_vel.length();
                let tangent_vel_dir = if tangent_vel_len > 1.0e-8 {
                    tangent_vel / tangent_vel_len
                } else {
                    Vector::ZERO
                };
                let projected_len = tangent_vel_len + self.friction * normal_vel;
                let projected_len = if projected_len > 0.0 {
                    projected_len
                } else {
                    0.0
                };
                return tangent_vel_dir * projected_len;
            } else {
                return vel;
            }
        }

        // BOUNDARY_CONDITION_NON_REFLECTING or unknown: pass through.
        vel
    }
}
