"""Small native-uniform and impulse-scaling launch adapters for frame composition."""

import tilelang.language as T
from shaders.dynamics.sim_params import body_coefficients
from shaders.dynamics.tile_kernels import gpu_scale_constraint_impulses


def make_kernel(layout, case, target, kind):
    sizes = case["buffer_words"]
    nr, ng, np, nt, nk, nb = [
        sizes[n]
        for n in ("RbdParams", "Gravity", "Params", "Tiles", "Buckets", "BatchIds")
    ]
    coefficients = body_coefficients(layout)
    scale = gpu_scale_constraint_impulses(layout)
    groups = case["groups"] if kind == "scale_impulses" else 1

    @T.macro
    def one(RbdParams, Gravity, Params, Tiles, Buckets, BatchIds, gid):
        if kind == "body_coefficients":
            if gid == 0:
                coefficients(RbdParams, Gravity, Params)
        else:
            scale(Tiles, Buckets, BatchIds, RbdParams, gid)

    if target == "c":

        @T.prim_func
        def kernel(
            RbdParams: T.Tensor((nr,), "uint32"),
            Gravity: T.Tensor((ng,), "uint32"),
            Params: T.Tensor((np,), "uint32"),
            Tiles: T.Tensor((nt,), "uint32"),
            Buckets: T.Tensor((nk,), "uint32"),
            BatchIds: T.Tensor((nb,), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(RbdParams, Gravity, Params, Tiles, Buckets, BatchIds, gid)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            RbdParams: A.Tensor((nr,), "uint32"),
            Gravity: A.Tensor((ng,), "uint32"),
            Params: A.Tensor((np,), "uint32"),
            Tiles: A.Tensor((nt,), "uint32"),
            Buckets: A.Tensor((nk,), "uint32"),
            BatchIds: A.Tensor((nb,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        RbdParams,
                        Gravity,
                        Params,
                        Tiles,
                        Buckets,
                        BatchIds,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            RbdParams: T.Tensor((nr,), "uint32"),
            Gravity: T.Tensor((ng,), "uint32"),
            Params: T.Tensor((np,), "uint32"),
            Tiles: T.Tensor((nt,), "uint32"),
            Buckets: T.Tensor((nk,), "uint32"),
            BatchIds: T.Tensor((nb,), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    RbdParams,
                    Gravity,
                    Params,
                    Tiles,
                    Buckets,
                    BatchIds,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
