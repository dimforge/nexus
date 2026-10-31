"""Native color-range launch adapter, consuming original Rust uniform layouts."""

import tilelang.language as T
from shaders.dynamics import tile_kernels as K

BUFFERS = (
    "Tiles",
    "Velocities",
    "Poses",
    "RbdParams",
    "Buckets",
    "BatchIds",
    "Color",
    "Mode",
)


def make_kernel(layout, case, target, operation):
    sizes = case["buffer_words"]
    n0, n1, n2, n3, n4, n5, n6, n7 = [sizes[n] for n in BUFFERS]
    fused = case.get("fused_colors", False)
    if fused and target != "metal":
        raise ValueError("Fused color barriers are currently qualified only for Metal")
    groups = case["batches"] if fused else case["groups"]
    one = K._colored_constraint_body(layout, case, operation)
    if target == "c":

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((n0,), "uint32"),
            Velocities: T.Tensor((n1,), "uint32"),
            Poses: T.Tensor((n2,), "uint32"),
            RbdParams: T.Tensor((n3,), "uint32"),
            Buckets: T.Tensor((n4,), "uint32"),
            BatchIds: T.Tensor((n5,), "uint32"),
            Color: T.Tensor((n6,), "uint32"),
            Mode: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Tiles,
                        Velocities,
                        Poses,
                        RbdParams,
                        Buckets,
                        BatchIds,
                        Color,
                        Mode,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Tiles: A.Tensor((n0,), "uint32"),
            Velocities: A.Tensor((n1,), "uint32"),
            Poses: A.Tensor((n2,), "uint32"),
            RbdParams: A.Tensor((n3,), "uint32"),
            Buckets: A.Tensor((n4,), "uint32"),
            BatchIds: A.Tensor((n5,), "uint32"),
            Color: A.Tensor((n6,), "uint32"),
            Mode: A.Tensor((n7,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Tiles,
                        Velocities,
                        Poses,
                        RbdParams,
                        Buckets,
                        BatchIds,
                        Color,
                        Mode,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((n0,), "uint32"),
            Velocities: T.Tensor((n1,), "uint32"),
            Poses: T.Tensor((n2,), "uint32"),
            RbdParams: T.Tensor((n3,), "uint32"),
            Buckets: T.Tensor((n4,), "uint32"),
            BatchIds: T.Tensor((n5,), "uint32"),
            Color: T.Tensor((n6,), "uint32"),
            Mode: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                if fused:
                    T.import_source(
                        "inline int nexus_color_barrier() { threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); return 0; }\n"
                    )
                one(
                    Tiles,
                    Velocities,
                    Poses,
                    RbdParams,
                    Buckets,
                    BatchIds,
                    Color,
                    Mode,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    fused = case.get("fused_colors", False)
    if fused and (target != "metal" or case["kind"] != "colored_program"):
        raise ValueError("Fused program requires the Metal all-color schedule")
    ops = [(case["kind"], "Color")]
    if case["kind"] == "colored_program":
        ops = [
            (op, f"Color{color}")
            for op in [
                "colored_warmstart",
                "colored_solve",
                "colored_solve",
                "colored_solve",
                "colored_cached",
            ]
            for color in ([3] if fused and op != "colored_cached" else range(1, 4))
        ]
    cache = {}
    plans = []
    for i, (op, color) in enumerate(ops):
        if op not in cache:
            factory = {
                "colored_solve": K.gpu_solve_constraints,
                "colored_warmstart": K.gpu_warmstart_constraints,
                "colored_cached": K.gpu_solve_constraints_unbiased_cached,
            }[op]
            use_fused = fused and op != "colored_cached"
            if use_fused:
                factory = {
                    "colored_solve": K.gpu_solve_constraints_fused,
                    "colored_warmstart": K.gpu_warmstart_constraints_fused,
                }[op]
            cache[op] = factory(
                layout, dict(case, native_color_range=True, fused_colors=use_fused), target
            )
        plans.append(
            {
                "name": f"{op}_{i}",
                "kernel": cache[op],
                "blocks": case["batches"] if fused and op != "colored_cached" else case["groups"],
                "threads": 64,
                "buffer_map": {n: color if n == "Color" else n for n in BUFFERS},
            }
        )
    return plans


def fuse_scene_colors(case, target):
    """Bound serialized work: small average capacity, otherwise retain wide colors.

    This is a scheduling heuristic, not a correctness limit. Unlike native Rust's
    latest pair readback, the portable manifest knows only allocated capacities.
    """
    return target == "metal" and case["contact_capacity"] <= 512 * case["batches"]
