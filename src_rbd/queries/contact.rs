//! Contact generation for collision response.
//!
//! GPU-accelerated contact manifold generation between pairs of colliding
//! shapes. A manifold holds up to 2 contact points in 2D / 4 in 3D, a contact
//! normal, and per-point penetration depths.

// Re-export contact types from the shader crate with Gpu prefix for backward compatibility
pub use crate::shaders::queries::{
    ColliderMaterial as GpuColliderMaterial, ContactManifold as GpuContactManifold,
    ContactPoint as GpuContactPoint, IndexedManifold as GpuIndexedContact,
};
