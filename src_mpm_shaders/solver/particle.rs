use crate::grid::grid::AffinityBits;
use crate::{Matrix, PaddedMatrix, UVector, Vector};

/// A particle position in the MPM grid.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Position {
    /// The particle's world-space position.
    pub pt: Vector,
    #[cfg(feature = "dim3")]
    pub padding: u32,
}

impl Position {
    pub fn new(pt: Vector) -> Self {
        Self {
            pt,
            #[cfg(feature = "dim3")]
            padding: 0,
        }
    }
}

/// Contact distance field data for a particle.
///
/// Stores the result of the collision detection between a particle and the
/// nearest rigid collider surface.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch_is_gpu),
    derive(Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct Cdf {
    /// The contact normal direction.
    pub normal: Vector,
    // NOTE: to avoid padding, the location of this field in the struct depends on whether
    //       we are in 2D or 3D.
    /// The signed distance from the particle to the closest collider surface.
    #[cfg(feature = "dim3")]
    pub signed_distance: f32,
    /// The velocity of the rigid body at the closest surface point.
    pub rigid_vel: Vector,
    /// The signed distance from the particle to the closest collider surface.
    #[cfg(feature = "dim2")]
    pub signed_distance: f32,
    /// Affinity bits for CPIC compatibility checks.
    pub affinity: AffinityBits,
}

impl Cdf {
    /// Creates a new zeroed Cdf.
    pub fn zero() -> Self {
        Self {
            normal: Vector::ZERO,
            rigid_vel: Vector::ZERO,
            signed_distance: 0.0,
            affinity: AffinityBits::default(),
        }
    }

    /// Creates a new Cdf with the given values.
    pub fn new(
        normal: Vector,
        rigid_vel: Vector,
        signed_distance: f32,
        affinity: AffinityBits,
    ) -> Self {
        Self {
            normal,
            rigid_vel,
            signed_distance,
            affinity,
        }
    }
}

/// Indices referencing the rigid body element closest to a particle.
///
/// In 2D, this references a segment (edge) by its two vertex indices.
/// In 3D, this references a triangle by its three vertex indices.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct RigidParticleIndices {
    /// The vertex indices of the closest segment (2D) or triangle (3D).
    #[cfg(feature = "dim2")]
    pub segment: UVector,
    /// The vertex indices of the closest segment (2D) or triangle (3D).
    #[cfg(feature = "dim3")]
    pub triangle: UVector,
    /// The collider index this element belongs to.
    pub collider: u32,
    /// SPIR-V padding: UVec2 has align(8) in SPIR-V, so stride must be a multiple of 8.
    #[cfg(feature = "dim2")]
    pub _pad: u32,
}

/// Core kinematic state for APIC particle-grid transfers.
///
/// Contains the fields needed by P2G and G2P kernels: velocity, mass, affine matrix,
/// external forces, and particle status. Separated from deformation and material
/// properties.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Kinematics {
    /// During `particle_update`, this contains the velocity gradient.
    /// After `particle_update`, this contains the affine matrix for APIC transfer.
    pub affine: PaddedMatrix,
    /// The particle's velocity.
    pub velocity: Vector,
    /// Determinant of the velocity gradient. Only the fluid models track it.
    #[cfg(feature = "dim3")]
    pub vel_grad_det: f32,
    /// Additional user-defined force applied to the particle, multiplied by dt.
    /// Reset at each `particle_update` invocation.
    /// Stored as force * dt so that dt is not needed during p2g.
    pub force_dt: Vector,
    /// Determinant of the velocity gradient. Only the fluid models track it.
    #[cfg(feature = "dim2")]
    pub vel_grad_det: f32,
    /// The particle's mass.
    pub mass: f32,
    /// Whether this particle is enabled (non-zero = enabled).
    pub enabled: u32,
    /// Multiplier applied to a collider's friction for this particle.
    ///
    /// Boundary friction is a property of the *pair*, not of the collider alone:
    /// water running over the same floor that sand piles up on should barely feel
    /// it. 1 uses the collider's friction as given, 0 makes the particle slide
    /// freely along the surface while still being stopped from passing through it.
    ///
    /// Only has an effect under CPIC (the default), which resolves the boundary
    /// per particle. Without it the boundary condition is applied to grid nodes,
    /// which hold a blend of every material touching them, and a single node
    /// cannot be frictional for the sand and slippery for the water at once.
    ///
    /// Lives here rather than in `ParticleProperties` because both transfer
    /// kernels already bind the kinematics buffer, and the one that needs it is
    /// at its storage-buffer limit.
    pub boundary_friction: f32,
    /// Alignment padding before the CDF field.
    #[cfg(feature = "dim3")]
    pub _padding: [u32; 2],
    /// Contact distance field data for CPIC rigid body coupling.
    pub cdf: Cdf,
    /// Tail padding so the struct size is a multiple of its alignment.
    #[cfg(feature = "dim2")]
    pub _tail_padding: [u32; 2],
}

/// Static per-particle properties that are read-only on the GPU.
///
/// These fields are set once during particle creation and never modified by any
/// GPU shader.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ParticleProperties {
    /// The particle's initial volume (reference configuration).
    pub init_volume: f32,
    /// The particle's initial radius.
    pub init_radius: f32,
    /// Rayleigh mass-proportional damping coefficient (1/s).
    pub damping: f32,
    /// Phase value for multi-material mixing.
    pub phase: f32,
    /// Whether this particle is fixed in place (non-zero = fixed).
    pub fixed: u32,
    /// Index of the group this particle belongs to. Carries no physics; the
    /// render kernels use it to look up a color in the viewer's group palette.
    pub group_id: u32,
    /// Pad to 32 bytes for GPU cache line alignment.
    pub padding: [u32; 2],
}

/*
 *
 * Grid-related position helper functions.
 *
 */

impl Position {
    /// Returns the position of the grid node closest to the particle.
    ///
    /// This rounds the particle position to the nearest cell center.
    #[inline]
    pub fn closest_grid_pos(&self, cell_width: f32) -> Vector {
        (self.pt / cell_width).round() * cell_width
    }

    /// Returns the position of the "associated" grid node for the particle.
    ///
    /// The associated node is one cell before the closest node in each dimension,
    /// which is the base node for the 3-node (quadratic) B-spline stencil.
    #[inline]
    pub fn associated_grid_pos(&self, cell_width: f32) -> Vector {
        ((self.pt / cell_width).round() - Vector::ONE) * cell_width
    }

    /// Returns the index of the associated cell within its block, offset by one.
    ///
    /// This is what maps a particle to its block in the sparse grid. Blocks are
    /// 8x8 in 2D and 4x4x4 in 3D.
    #[inline]
    pub fn associated_cell_index_in_block_off_by_one(&self, cell_width: f32) -> UVector {
        let assoc_cell = (self.pt / cell_width).round() - Vector::ONE;
        #[cfg(feature = "dim2")]
        let assoc_block = (assoc_cell / 8.0).floor() * 8.0;
        #[cfg(feature = "dim3")]
        let assoc_block = (assoc_cell / 4.0).floor() * 4.0;
        // The result is always non-negative, so the cast to unsigned is safe.
        #[cfg(feature = "dim2")]
        {
            let diff = assoc_cell - assoc_block;
            UVector::new(diff.x as u32, diff.y as u32)
        }
        #[cfg(feature = "dim3")]
        {
            let diff = assoc_cell - assoc_block;
            UVector::new(diff.x as u32, diff.y as u32, diff.z as u32)
        }
    }

    /// Returns the direction vector from the particle to the closest grid node.
    #[inline]
    pub fn dir_to_closest_grid_node(&self, cell_width: f32) -> Vector {
        self.closest_grid_pos(cell_width) - self.pt
    }

    /// Returns the direction vector from the particle to the associated grid node.
    #[inline]
    pub fn dir_to_associated_grid_node(&self, cell_width: f32) -> Vector {
        self.associated_grid_pos(cell_width) - self.pt
    }
}
