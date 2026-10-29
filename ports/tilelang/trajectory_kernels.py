# ruff: noqa: SIM117
# TileLang requires distinct kernel and SIMT launch scopes.
"""Record every frame on device; no physics is computed in this diagnostic path."""

import tilelang.language as T


def make_kernel(l, c, target):
    sizes = c["buffer_words"]
    n = c["body_count"]
    pw = l["pose_words"]
    vw = l["velocity_words"]
    frames = c["frames"]
    stride = c["trace_stride"]
    groups = max((n + 63) // 64, 1)

    @T.macro
    def one(Poses, Velocities, Trajectory, FrameIndex, i):
        frame = T.cast(FrameIndex[0], "int32")
        if i < n and 0 <= frame < frames:
            for d in T.serial(pw):
                Trajectory[frame * stride + i * pw + d] = Poses[i * pw + d]
            for d in T.serial(vw):
                Trajectory[frame * stride + n * pw + i * vw + d] = Velocities[
                    i * vw + d
                ]

    if target == "c":

        @T.prim_func
        def kernel(
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Velocities: T.Tensor((sizes["Velocities"],), "uint32"),
            Trajectory: T.Tensor((sizes["Trajectory"],), "uint32"),
            FrameIndex: T.Tensor((sizes["FrameIndex"],), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(n):
                    one(Poses, Velocities, Trajectory, FrameIndex, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Velocities: A.Tensor((sizes["Velocities"],), "uint32"),
            Trajectory: A.Tensor((sizes["Trajectory"],), "uint32"),
            FrameIndex: A.Tensor((sizes["FrameIndex"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Poses,
                        Velocities,
                        Trajectory,
                        FrameIndex,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Velocities: T.Tensor((sizes["Velocities"],), "uint32"),
            Trajectory: T.Tensor((sizes["Trajectory"],), "uint32"),
            FrameIndex: T.Tensor((sizes["FrameIndex"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Poses,
                    Velocities,
                    Trajectory,
                    FrameIndex,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
