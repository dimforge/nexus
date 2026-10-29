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
    groups = case["groups"]
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
            for color in range(1, 4)
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
            cache[op] = factory(layout, dict(case, native_color_range=True), target)
        plans.append(
            {
                "name": f"{op}_{i}",
                "kernel": cache[op],
                "blocks": case["groups"],
                "threads": 64,
                "buffer_map": {n: color if n == "Color" else n for n in BUFFERS},
            }
        )
    return plans
