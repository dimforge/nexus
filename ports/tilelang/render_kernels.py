# ruff: noqa: SIM117
"""Native word ABI launch adapter for direct renderer preparation."""

import tilelang.language as T
from shaders.dynamics.prep_render import gpu_rbd_prep_render


def make_kernel(l, c, target):
    one = gpu_rbd_prep_render(l)
    sizes, groups = c["buffer_words"], c["groups"]
    if target == "c":

        @T.prim_func
        def kernel(
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Descriptors: T.Tensor((sizes["Descriptors"],), "uint32"),
            Count: T.Tensor((sizes["Count"],), "uint32"),
            Positions: T.Tensor((sizes["Positions"],), "uint32"),
            Deformations: T.Tensor((sizes["Deformations"],), "uint32"),
            Colors: T.Tensor((sizes["Colors"],), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(groups * 64):
                    one(Poses, Descriptors, Count, Positions, Deformations, Colors, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Descriptors: A.Tensor((sizes["Descriptors"],), "uint32"),
            Count: A.Tensor((sizes["Count"],), "uint32"),
            Positions: A.Tensor((sizes["Positions"],), "uint32"),
            Deformations: A.Tensor((sizes["Deformations"],), "uint32"),
            Colors: A.Tensor((sizes["Colors"],), "uint32"),
        ):
            with A.Kernel(groups) as block:
                with A.SimtVF(threads=64):
                    one(
                        Poses,
                        Descriptors,
                        Count,
                        Positions,
                        Deformations,
                        Colors,
                        block * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Descriptors: T.Tensor((sizes["Descriptors"],), "uint32"),
            Count: T.Tensor((sizes["Count"],), "uint32"),
            Positions: T.Tensor((sizes["Positions"],), "uint32"),
            Deformations: T.Tensor((sizes["Deformations"],), "uint32"),
            Colors: T.Tensor((sizes["Colors"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as block:
                one(
                    Poses,
                    Descriptors,
                    Count,
                    Positions,
                    Deformations,
                    Colors,
                    block * 64 + T.get_thread_binding(),
                )

    return kernel
