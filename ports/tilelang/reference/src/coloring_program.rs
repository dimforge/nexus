use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan, dynamics::*, glamx::UVec3, utils::BatchIndices,
};
#[allow(clippy::too_many_arguments)]
pub(crate) fn bounded(
    counts: &[u32],
    ids: &[u32],
    links: &[ContactLink],
    colors: &mut [u32],
    colored: &mut [u32],
    pending: &mut [u32],
    flag: &mut u32,
    plan: &ContactPlan,
    body_groups: &[u32],
    batch: &BatchIndices,
    contacts: &[[u32; 3]],
    indirect: &mut [[u32; 3]],
    groups: u32,
    rounds: u32,
    seeded: bool,
) {
    if !seeded {
        for i in 0..plan.bound.max(1) {
            gpu_reset_topo_gc(
                UVec3::new(i, 0, 0),
                colors,
                colored,
                links,
                plan,
                flag,
                pending,
            );
        }
    }
    gpu_clear_completion_flag_topo_gc(UVec3::ZERO, flag, contacts, indirect);
    let grid = UVec3::new(groups, 1, 1);
    for round in 0..rounds.max(2) {
        if round > 0 {
            gpu_reset_completion_flag_topo_gc(UVec3::ZERO, flag, contacts, indirect);
        }
        if indirect[0][0] != 0 {
            for i in 0..groups * 64 {
                gpu_step_graph_coloring_topo_gc(
                    UVec3::new(i, 0, 0),
                    grid,
                    counts,
                    ids,
                    links,
                    colors,
                    colored,
                    flag,
                    plan,
                    body_groups,
                    batch,
                    pending,
                );
            }
            for i in 0..groups * 64 {
                gpu_fix_conflicts_topo_gc(
                    UVec3::new(i, 0, 0),
                    grid,
                    counts,
                    ids,
                    links,
                    colors,
                    colored,
                    flag,
                    plan,
                    body_groups,
                    batch,
                    pending,
                );
            }
        }
    }
    for i in 0..groups * 64 {
        gpu_finalize_topo_gc(UVec3::new(i, 0, 0), grid, colors, colored, plan);
    }
}
