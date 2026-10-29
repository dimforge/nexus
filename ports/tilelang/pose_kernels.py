"""Target wrappers for collider pose propagation."""

import tilelang.language as T
from shaders.dynamics.mprops_update import gpu_sync_collider_poses


def make_kernel(layout, case, target):
    nb = case["body_storage"] * layout["pose_words"]
    nc = case["storage"] * layout["pose_words"]
    ns = case["storage"]
    n = case["count"]
    one = gpu_sync_collider_poses(layout, case)
    if target == "c":

        @T.prim_func
        def kernel(
            BodyPoses: T.Tensor((nb,), "uint32"),
            LocalPoses: T.Tensor((nc,), "uint32"),
            WorldPoses: T.Tensor((nc,), "uint32"),
            Parents: T.Tensor((ns,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(n):
                    one(BodyPoses, LocalPoses, WorldPoses, Parents, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            BodyPoses: A.Tensor((nb,), "uint32"),
            LocalPoses: A.Tensor((nc,), "uint32"),
            WorldPoses: A.Tensor((nc,), "uint32"),
            Parents: A.Tensor((ns,), "uint32"),
        ):
            with A.Kernel(max((n + 63) // 64, 1)) as block:
                with A.SimtVF(threads=64):
                    one(
                        BodyPoses,
                        LocalPoses,
                        WorldPoses,
                        Parents,
                        block * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            BodyPoses: T.Tensor((nb,), "uint32"),
            LocalPoses: T.Tensor((nc,), "uint32"),
            WorldPoses: T.Tensor((nc,), "uint32"),
            Parents: T.Tensor((ns,), "uint32"),
        ):
            with T.Kernel(max((n + 63) // 64, 1), threads=64) as block:
                one(
                    BodyPoses,
                    LocalPoses,
                    WorldPoses,
                    Parents,
                    block * 64 + T.get_thread_binding(),
                )

    return kernel
