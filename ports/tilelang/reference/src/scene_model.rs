//! Typed dispatch to native contact kernels; the scene oracle contains no replacement physics.
use nexus_rbd_shaders3d::{
    Pose, broad_phase::ContactPlan, dynamics::*, glamx::UVec3, queries::IndexedManifold,
    utils::BatchIndices,
};
#[allow(clippy::too_many_arguments)]
pub trait Model {
    type Tile: bytemuck::Pod + Default;
    fn prepare(
        gid: UVec3,
        sorted_links: &[ContactLink],
        contacts: &[IndexedManifold],
        mprops: &[WorldMassProperties],
        solver_poses: &[Pose],
        vels: &[Velocity],
        states: &[ContactRecycleState],
        old_tiles: &[Self::Tile],
        tiles: &mut [Self::Tile],
        plan: &ContactPlan,
        recycle_offsets: &ContactRecycleOffsets,
    );
    fn scale(
        gid: UVec3,
        buckets: &[u32],
        tiles: &mut [Self::Tile],
        ids: &BatchIndices,
        params: &RbdSimParams,
    );
    fn warmstart(
        gid: UVec3,
        num_workgroups: UVec3,
        tiles: &[Self::Tile],
        solver_vels: &mut [Velocity],
        buckets: &[u32],
        curr_color: &u32,
        ids: &BatchIndices,
    );
    fn solve(
        gid: UVec3,
        num_workgroups: UVec3,
        tiles: &mut [Self::Tile],
        solver_vels: &mut [Velocity],
        buckets: &[u32],
        solver_body_poses: &[Pose],
        curr_color: &u32,
        ids: &BatchIndices,
        mode: &u32,
        params: &RbdSimParams,
    );
}
pub struct Coulomb;
pub struct Twist;
macro_rules! native_model {
    ($model:ty, $module:ident, $tile:ident) => {
        impl Model for $model {
            type Tile = $module::$tile;
            fn prepare(
                gid: UVec3,
                sorted_links: &[ContactLink],
                contacts: &[IndexedManifold],
                mprops: &[WorldMassProperties],
                solver_poses: &[Pose],
                vels: &[Velocity],
                states: &[ContactRecycleState],
                old_tiles: &[Self::Tile],
                tiles: &mut [Self::Tile],
                plan: &ContactPlan,
                recycle_offsets: &ContactRecycleOffsets,
            ) {
                $module::gpu_prepare_constraints(
                    gid,
                    sorted_links,
                    contacts,
                    mprops,
                    solver_poses,
                    vels,
                    states,
                    old_tiles,
                    tiles,
                    plan,
                    recycle_offsets,
                );
            }
            fn scale(
                gid: UVec3,
                buckets: &[u32],
                tiles: &mut [Self::Tile],
                ids: &BatchIndices,
                params: &RbdSimParams,
            ) {
                $module::gpu_scale_constraint_impulses(gid, buckets, tiles, ids, params);
            }
            fn warmstart(
                gid: UVec3,
                num_workgroups: UVec3,
                tiles: &[Self::Tile],
                solver_vels: &mut [Velocity],
                buckets: &[u32],
                curr_color: &u32,
                ids: &BatchIndices,
            ) {
                $module::gpu_warmstart_constraints(
                    gid,
                    num_workgroups,
                    tiles,
                    solver_vels,
                    buckets,
                    curr_color,
                    ids,
                );
            }
            fn solve(
                gid: UVec3,
                num_workgroups: UVec3,
                tiles: &mut [Self::Tile],
                solver_vels: &mut [Velocity],
                buckets: &[u32],
                solver_body_poses: &[Pose],
                curr_color: &u32,
                ids: &BatchIndices,
                mode: &u32,
                params: &RbdSimParams,
            ) {
                $module::gpu_solve_constraints(
                    gid,
                    num_workgroups,
                    tiles,
                    solver_vels,
                    buckets,
                    solver_body_poses,
                    curr_color,
                    ids,
                    mode,
                    params,
                );
            }
        }
    };
}
native_model!(Coulomb, coulomb_tiles, CoulombTile);
native_model!(Twist, twist_tiles, TwistTile);
