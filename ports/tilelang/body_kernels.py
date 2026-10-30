"""Launch adapters and no-contact composition; physics stays in mirrored modules."""

import tilelang.language as T
from body_validation import dispatch_count
from shaders.dynamics import mprops_update, solver


def make_kernel(layout, case, target):
    kind = case["kind"]
    update_mass = mprops_update.gpu_update_mprops(layout, case)
    cleanup = solver.gpu_solver_cleanup(layout, case)
    init_bodies = solver.gpu_init_solver_bodies(layout, case)
    init_inc = solver.gpu_init_solver_vels_inc(layout, case)
    apply_inc = solver.gpu_apply_solver_vels_inc(layout, case)
    finalize = solver.gpu_solver_finalize(layout, case)
    integrate = solver._integrate_linearized_body(layout, case["count"])
    n = dispatch_count(case)
    storage = case["storage"]
    nsv = case.get("solver_storage", storage) * layout["velocity_words"]
    nv = storage * layout["velocity_words"]
    np = storage * layout["pose_words"]
    nl = storage * layout["local_words"]
    nw = storage * layout["world_words"]
    nh = layout["hub_words"]

    @T.macro
    def one(
        Velocities,
        SolverVelocities,
        Increments,
        Poses,
        SolverPoses,
        LocalMass,
        WorldMass,
        Params,
        Counts,
        FirstSlot,
        HubCounts,
        i,
    ):
        if kind == "update_mprops":
            update_mass(WorldMass, LocalMass, Poses, i)
        if kind == "cleanup" or kind == "free_step":
            cleanup(
                Counts, SolverVelocities, Velocities, WorldMass, FirstSlot, HubCounts, i
            )
        if kind == "init_bodies" or kind == "free_step":
            init_bodies(Poses, LocalMass, SolverPoses, i)
        if kind == "init_increments" or kind == "free_step":
            init_inc(Increments, WorldMass, Params, i)
        if kind == "apply_increments":
            apply_inc(SolverVelocities, Increments, i)
        if kind == "free_step":
            for substep in T.serial(case["substeps"]):
                apply_inc(SolverVelocities, Increments, i)
                integrate(SolverVelocities, SolverPoses, Params, i)
        if kind == "finalize" or kind == "free_step":
            finalize(Velocities, SolverVelocities, Poses, SolverPoses, LocalMass, i)

    if target == "c":

        @T.prim_func
        def kernel(
            Velocities: T.Tensor((nv,), "uint32"),
            SolverVelocities: T.Tensor((nsv,), "uint32"),
            Increments: T.Tensor((nv,), "uint32"),
            Poses: T.Tensor((np,), "uint32"),
            SolverPoses: T.Tensor((np,), "uint32"),
            LocalMass: T.Tensor((nl,), "uint32"),
            WorldMass: T.Tensor((nw,), "uint32"),
            Params: T.Tensor((6,), "float32"),
            Counts: T.Tensor((storage,), "uint32"),
            FirstSlot: T.Tensor((storage,), "uint32"),
            HubCounts: T.Tensor((nh,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(T.max(n, 1)):
                    one(
                        Velocities,
                        SolverVelocities,
                        Increments,
                        Poses,
                        SolverPoses,
                        LocalMass,
                        WorldMass,
                        Params,
                        Counts,
                        FirstSlot,
                        HubCounts,
                        i,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Velocities: A.Tensor((nv,), "uint32"),
            SolverVelocities: A.Tensor((nsv,), "uint32"),
            Increments: A.Tensor((nv,), "uint32"),
            Poses: A.Tensor((np,), "uint32"),
            SolverPoses: A.Tensor((np,), "uint32"),
            LocalMass: A.Tensor((nl,), "uint32"),
            WorldMass: A.Tensor((nw,), "uint32"),
            Params: A.Tensor((6,), "float32"),
            Counts: A.Tensor((storage,), "uint32"),
            FirstSlot: A.Tensor((storage,), "uint32"),
            HubCounts: A.Tensor((nh,), "uint32"),
        ):
            with A.Kernel(max((n + 63) // 64, 1)) as block:
                with A.SimtVF(threads=64):
                    one(
                        Velocities,
                        SolverVelocities,
                        Increments,
                        Poses,
                        SolverPoses,
                        LocalMass,
                        WorldMass,
                        Params,
                        Counts,
                        FirstSlot,
                        HubCounts,
                        block * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Velocities: T.Tensor((nv,), "uint32"),
            SolverVelocities: T.Tensor((nsv,), "uint32"),
            Increments: T.Tensor((nv,), "uint32"),
            Poses: T.Tensor((np,), "uint32"),
            SolverPoses: T.Tensor((np,), "uint32"),
            LocalMass: T.Tensor((nl,), "uint32"),
            WorldMass: T.Tensor((nw,), "uint32"),
            Params: T.Tensor((6,), "float32"),
            Counts: T.Tensor((storage,), "uint32"),
            FirstSlot: T.Tensor((storage,), "uint32"),
            HubCounts: T.Tensor((nh,), "uint32"),
        ):
            with T.Kernel(max((n + 63) // 64, 1), threads=64) as block:
                one(
                    Velocities,
                    SolverVelocities,
                    Increments,
                    Poses,
                    SolverPoses,
                    LocalMass,
                    WorldMass,
                    Params,
                    Counts,
                    FirstSlot,
                    HubCounts,
                    block * 64 + T.get_thread_binding(),
                )

    return kernel
