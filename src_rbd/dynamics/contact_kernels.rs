//! The contact constraints of a step, and the host side of their tile kernels.
use crate::shaders::dynamics::contact_tiles::{TILE_LEN, tile_lane};
use crate::shaders::dynamics::coulomb_tiles::CoulombTile;
use crate::shaders::dynamics::*;
use khal::backend::{Backend, GpuBackend, GpuBackendError};
use vortx::tensor::Tensor;

pub(crate) use crate::shaders::dynamics::coulomb_tiles::{
    GpuGatherWarmstartVelocities, GpuHubWarmstartConstraints, GpuPrepareConstraints,
    GpuScaleConstraintImpulses, GpuSolveConstraints, GpuSolveConstraintsBiased,
    GpuSolveConstraintsFused, GpuSolveConstraintsUnbiased, GpuWarmstartConstraints,
    GpuWarmstartConstraintsFused,
};

/// The color-ordered tiles of the contact constraints.
pub type ContactTiles = Tensor<CoulombTile>;

/// The contact constraints of one step: their links in contact order, and their solver data
/// sorted by color.
pub struct ContactConstraints {
    /// The topology and identity of each contact's constraint.
    pub links: Tensor<ContactLink>,
    /// The index of each active constraint in `tiles`.
    pub constraint_indices: Tensor<u32>,
    /// The constraint tiles.
    pub tiles: ContactTiles,
}

impl ContactConstraints {
    /// Uninitialized constraints for `capacity` contact slots.
    pub(crate) fn new(
        backend: &GpuBackend,
        capacity: u32,
        usage: khal::BufferUsages,
    ) -> Result<Self, GpuBackendError> {
        let tiles = capacity.div_ceil(TILE_LEN as u32).max(1);
        let tiles_usage = khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC;
        Ok(Self {
            links: Tensor::vector_uninit(backend, capacity, usage)?,
            constraint_indices: Tensor::vector_uninit(backend, capacity, usage)?,
            tiles: Tensor::vector_uninit(backend, tiles, tiles_usage)?,
        })
    }

    /// Reads the constraints of the first `bound` contact slots, in contact order. This is a
    /// blocking GPU synchronization when driven to completion; keep it outside timed runs.
    pub async fn read_impulses(
        &self,
        backend: &GpuBackend,
        bound: usize,
    ) -> Result<Vec<ContactImpulseSnapshot>, GpuBackendError> {
        let links: Vec<ContactLink> = backend.slow_read_vec(self.links.buffer()).await?;
        let constraint_indices: Vec<u32> = backend
            .slow_read_vec(self.constraint_indices.buffer())
            .await?;
        let tiles: Vec<CoulombTile> = backend.slow_read_vec(self.tiles.buffer()).await?;
        let mut out = Vec::new();
        for (link, index) in links.iter().zip(constraint_indices).take(bound) {
            if link.len == 0 {
                out.push(ContactImpulseSnapshot::inactive(link));
                continue;
            }
            let (tile, lane) = tile_lane(index as usize);
            let c = tiles[tile].constraint(lane);
            let n = c.len as usize;
            #[cfg(feature = "dim3")]
            let tangent = c.points[..n]
                .iter()
                .map(|p| p.tangent_impulse.to_array())
                .collect();
            #[cfg(feature = "dim2")]
            let tangent = c.points[..n].iter().map(|p| p.tangent_impulse).collect();
            out.push(ContactImpulseSnapshot {
                body_a: c.solver_body_a,
                body_b: c.solver_body_b,
                len: c.len,
                dir_a: c.dir_a,
                friction: c.limit,
                inv_mass_a: c.im_a,
                inv_mass_b: c.im_b,
                normal_impulse: c.points[..n].iter().map(|p| p.normal_impulse).collect(),
                tangent_impulse: tangent,
            });
        }
        Ok(out)
    }
}

/// CPU diagnostic view of a contact constraint.
pub struct ContactImpulseSnapshot {
    /// Global solver index of the first body.
    pub body_a: u32,
    /// Global solver index of the second body.
    pub body_b: u32,
    /// Number of active contact points.
    pub len: u32,
    /// World normal force direction on the first body.
    pub dir_a: crate::math::Vector,
    /// Combined friction coefficient.
    pub friction: f32,
    /// Axis-dependent inverse mass of the first body.
    pub inv_mass_a: crate::math::Vector,
    /// Axis-dependent inverse mass of the second body.
    pub inv_mass_b: crate::math::Vector,
    /// Accumulated normal impulse per contact.
    pub normal_impulse: Vec<f32>,
    /// Per-contact sliding impulses.
    pub tangent_impulse: Vec<[f32; SUB_LEN]>,
}

impl ContactImpulseSnapshot {
    /// The snapshot of a gap, inert or multibody-owned slot.
    fn inactive(link: &ContactLink) -> Self {
        Self {
            body_a: link.solver_body_a,
            body_b: link.solver_body_b,
            len: 0,
            dir_a: Default::default(),
            friction: 0.0,
            inv_mass_a: Default::default(),
            inv_mass_b: Default::default(),
            normal_impulse: vec![],
            tangent_impulse: vec![],
        }
    }
}
