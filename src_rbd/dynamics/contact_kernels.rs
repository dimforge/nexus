//! Host dispatch over the contact tiles of the simulation's friction model. Only the selected
//! model's tiles are allocated; kernel selection happens once per dispatch on the CPU.
use crate::math::Pose;
use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::contact_tiles::{TILE_LEN, tile_lane};
use crate::shaders::dynamics::coulomb_tiles::CoulombTile;
#[cfg(feature = "dim3")]
use crate::shaders::dynamics::twist_tiles::TwistTile;
use crate::shaders::dynamics::*;
use crate::shaders::queries::IndexedManifold;
use crate::shaders::utils::BatchIndices;
use khal::backend::{Backend, DispatchGrid, GpuBackend, GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// The color-ordered tiles of the contact constraints, in the layout of the simulation's
/// friction model.
pub enum ContactTiles {
    /// Friction at each contact point.
    Coulomb(Tensor<CoulombTile>),
    #[cfg(feature = "dim3")]
    /// Central sliding friction and twist friction per manifold.
    Simplified(Tensor<TwistTile>),
}

impl ContactTiles {
    /// Uninitialized tiles for `capacity` constraints of `params`' friction model.
    pub(crate) fn new(
        backend: &GpuBackend,
        capacity: u32,
        params: &RbdSimParams,
    ) -> Result<Self, GpuBackendError> {
        let _ = params;
        let tiles = capacity.div_ceil(TILE_LEN as u32).max(1);
        let usage = khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC;
        #[cfg(feature = "dim3")]
        if params.friction_model == FrictionModel::Simplified {
            return Ok(Self::Simplified(Tensor::vector_uninit(
                backend, tiles, usage,
            )?));
        }
        Ok(Self::Coulomb(Tensor::vector_uninit(backend, tiles, usage)?))
    }

    /// The Coulomb tiles, if this simulation uses Coulomb friction.
    pub fn coulomb(&self) -> Option<&Tensor<CoulombTile>> {
        match self {
            Self::Coulomb(x) => Some(x),
            #[cfg(feature = "dim3")]
            _ => None,
        }
    }

    #[cfg(feature = "dim3")]
    /// The twist tiles, if this simulation uses simplified friction.
    pub fn simplified(&self) -> Option<&Tensor<TwistTile>> {
        match self {
            Self::Simplified(x) => Some(x),
            _ => None,
        }
    }
}

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
    /// Uninitialized constraints for `capacity` contact slots of `params`' friction model.
    pub(crate) fn new(
        backend: &GpuBackend,
        capacity: u32,
        usage: khal::BufferUsages,
        params: &RbdSimParams,
    ) -> Result<Self, GpuBackendError> {
        Ok(Self {
            links: Tensor::vector_uninit(backend, capacity, usage)?,
            constraint_indices: Tensor::vector_uninit(backend, capacity, usage)?,
            tiles: ContactTiles::new(backend, capacity, params)?,
        })
    }

    /// Reads the constraints of the first `bound` contact slots, in contact order. This is a
    /// blocking GPU synchronization when driven to completion; keep it outside timed runs.
    pub async fn read_impulses(
        &self,
        backend: &GpuBackend,
        bound: usize,
    ) -> Result<Vec<ContactImpulseSnapshot>, GpuBackendError> {
        self.tiles
            .read_impulses(backend, &self.links, &self.constraint_indices, bound)
            .await
    }
}

// Every tile argument is checked before launch; the old and new tiles always change model
// together when the simulation-wide model is changed.
macro_rules! tile_kernel {
    ($name:ident, $primary:ident, [$($other:ident),*], ($($arg:ident : $ty:ty),* $(,)?)) => {
        pub(crate) struct $name {
            coulomb: crate::shaders::dynamics::coulomb_tiles::$name,
            #[cfg(feature="dim3")]
            simplified: crate::shaders::dynamics::twist_tiles::$name,
        }
        impl $name {
            #[cfg(feature="cpu")]
            pub const __ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU: () = crate::shaders::dynamics::coulomb_tiles::$name::__ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU;
            #[cfg(feature="cpu-parallel")]
            pub const __ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU_PARALLEL: () = crate::shaders::dynamics::coulomb_tiles::$name::__ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CPU_PARALLEL;
            #[cfg(feature="cuda")]
            pub const __ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CUDA: () = crate::shaders::dynamics::coulomb_tiles::$name::__ERROR__SHADER_CRATE_IS_MISSING_FEATURE_NAMED____CUDA;
            pub fn from_dir(backend: &GpuBackend, dir: &include_dir::Dir<'static>) -> Result<Self,GpuBackendError> {
                Ok(Self {
                    coulomb: crate::shaders::dynamics::coulomb_tiles::$name::from_dir(backend,dir)?,
                    #[cfg(feature="dim3")]
                    simplified: crate::shaders::dynamics::twist_tiles::$name::from_dir(backend,dir)?,
                })
            }
            #[allow(clippy::too_many_arguments)]
            pub fn call<'a>(&self, pass: &mut GpuPass, grid: impl Into<DispatchGrid<'a,GpuBackend>>, $($arg:$ty),*) -> Result<(),GpuBackendError> {
                let grid = grid.into();
                match $primary {
                    ContactTiles::Coulomb($primary) => {
                        $(let $other = $other.coulomb().expect("tile models agree");)*
                        self.coulomb.call(pass,grid,$($arg),*)
                    }
                    #[cfg(feature="dim3")]
                    ContactTiles::Simplified($primary) => {
                        $(let $other = $other.simplified().expect("tile models agree");)*
                        self.simplified.call(pass,grid,$($arg),*)
                    }
                }
            }
        }
    };
}

tile_kernel!(GpuPrepareConstraints, tiles, [old_tiles], (
    sorted_links: &Tensor<ContactLink>,
    contacts: &Tensor<IndexedManifold>,
    mprops: &Tensor<WorldMassProperties>,
    solver_poses: &Tensor<Pose>,
    vels: &Tensor<Velocity>,
    states: &Tensor<ContactRecycleState>,
    old_tiles: &ContactTiles,
    tiles: &mut ContactTiles,
    plan: &Tensor<ContactPlan>,
    recycle_offsets: &Tensor<ContactRecycleOffsets>,
));
tile_kernel!(GpuScaleConstraintImpulses, tiles, [], (
    buckets: &Tensor<u32>,
    tiles: &mut ContactTiles,
    ids: &Tensor<BatchIndices>,
    params: &Tensor<RbdSimParams>,
));

/// Declares the wrappers of the solve kernels, which share their arguments.
macro_rules! solve_constraints_kernels {
    ($($name:ident),*) => {$(
        tile_kernel!($name, tiles, [], (
            tiles: &mut ContactTiles,
            solver_vels: &mut Tensor<Velocity>,
            buckets: &Tensor<u32>,
            solver_body_poses: &Tensor<Pose>,
            color: &Tensor<u32>,
            ids: &Tensor<BatchIndices>,
            mode: &Tensor<u32>,
            params: &Tensor<RbdSimParams>,
        ));
    )*};
}
solve_constraints_kernels!(
    GpuSolveConstraints,
    GpuSolveConstraintsBiased,
    GpuSolveConstraintsUnbiased,
    GpuSolveConstraintsUnbiasedCached,
    GpuSolveConstraintsTail,
    GpuSolveConstraintsTailBiased,
    GpuSolveConstraintsTailUnbiased,
    GpuSolveConstraintsTailUnbiasedCached,
    GpuSolveConstraintsFused,
    GpuSolveConstraintsFusedCached
);
tile_kernel!(GpuWarmstartConstraints, tiles, [], (
    tiles: &ContactTiles,
    solver_vels: &mut Tensor<Velocity>,
    buckets: &Tensor<u32>,
    curr_color: &Tensor<u32>,
    ids: &Tensor<BatchIndices>,
));
tile_kernel!(GpuWarmstartConstraintsFused, tiles, [], (
    tiles: &ContactTiles,
    solver_vels: &mut Tensor<Velocity>,
    buckets: &Tensor<u32>,
    num_colors: &Tensor<u32>,
    ids: &Tensor<BatchIndices>,
));
tile_kernel!(GpuGatherWarmstartVelocities, tiles, [], (
    counts: &Tensor<u32>,
    constraint_ids: &Tensor<u32>,
    constraint_indices: &Tensor<u32>,
    tiles: &ContactTiles,
    solver_vels: &mut Tensor<Velocity>,
    hub_first_slot: &Tensor<u32>,
    ids: &Tensor<BatchIndices>,
));
tile_kernel!(GpuHubWarmstartConstraints, tiles, [], (
    solver_vels: &mut Tensor<Velocity>,
    tiles: &ContactTiles,
    hub_slot_constraint: &Tensor<u32>,
    hub_counts: &Tensor<crate::shaders::dynamics::HubCounts>,
    constraint_indices: &Tensor<u32>,
));

/// CPU diagnostic view; tangent impulses are repeated per point for the
/// simplified model, matching Rapier's contact-data convention.
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
    /// Per-contact sliding impulses (central impulse repeated for simplified friction).
    pub tangent_impulse: Vec<[f32; SUB_LEN]>,
    /// Accumulated manifold twist impulse; zero for Coulomb.
    pub twist_impulse: f32,
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
            twist_impulse: 0.0,
        }
    }
}

impl ContactTiles {
    async fn read_impulses(
        &self,
        backend: &GpuBackend,
        links: &Tensor<ContactLink>,
        constraint_indices: &Tensor<u32>,
        bound: usize,
    ) -> Result<Vec<ContactImpulseSnapshot>, GpuBackendError> {
        let links: Vec<ContactLink> = backend.slow_read_vec(links.buffer()).await?;
        let constraint_indices: Vec<u32> =
            backend.slow_read_vec(constraint_indices.buffer()).await?;
        let mut out = Vec::new();
        match self {
            Self::Coulomb(buffer) => {
                let tiles: Vec<CoulombTile> = backend.slow_read_vec(buffer.buffer()).await?;
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
                        twist_impulse: 0.0,
                    });
                }
            }
            #[cfg(feature = "dim3")]
            Self::Simplified(buffer) => {
                let tiles: Vec<TwistTile> = backend.slow_read_vec(buffer.buffer()).await?;
                for (link, index) in links.iter().zip(constraint_indices).take(bound) {
                    if link.len == 0 {
                        out.push(ContactImpulseSnapshot::inactive(link));
                        continue;
                    }
                    let (tile, lane) = tile_lane(index as usize);
                    let c = tiles[tile].constraint(lane);
                    let n = c.len as usize;
                    out.push(ContactImpulseSnapshot {
                        body_a: c.solver_body_a,
                        body_b: c.solver_body_b,
                        len: c.len,
                        dir_a: c.dir_a,
                        friction: c.limit,
                        inv_mass_a: c.im_a,
                        inv_mass_b: c.im_b,
                        normal_impulse: c.points[..n].iter().map(|p| p.normal_impulse).collect(),
                        tangent_impulse: vec![c.friction.tangent_impulse.to_array(); n],
                        twist_impulse: c.friction.twist_impulse,
                    });
                }
            }
        }
        Ok(out)
    }
}
