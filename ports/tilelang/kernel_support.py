"""TileLang scalar helpers and bit-preserving buffer access; build support."""

import tilelang.language as T


def f32(buffer, index):
    return T.reinterpret(buffer[index], "float32")


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def relative_velocity(direction, torque_a, torque_b, va, vb):
    return (
        direction[0] * va[0]
        + direction[1] * va[1]
        + direction[2] * va[2]
        + (torque_a[0] * va[3] + torque_a[1] * va[4] + torque_a[2] * va[5])
        - (direction[0] * vb[0] + direction[1] * vb[1] + direction[2] * vb[2])
        + (torque_b[0] * vb[3] + torque_b[1] * vb[4] + torque_b[2] * vb[5])
    )


def launch(one, layout, case, target, n):
    nt = case["tiles"] * layout["tile_words"]
    nv = case["bodies"] * layout["velocity_words"]
    np = case["bodies"] * layout["pose_words"]
    if target == "c":

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((nt,), "uint32"),
            Velocities: T.Tensor((nv,), "uint32"),
            Poses: T.Tensor((np,), "uint32"),
            Params: T.Tensor((6,), "float32"),
        ):
            with T.Kernel(1):
                for i in T.serial(n):
                    one(Tiles, Velocities, Poses, Params, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Tiles: A.Tensor((nt,), "uint32"),
            Velocities: A.Tensor((nv,), "uint32"),
            Poses: A.Tensor((np,), "uint32"),
            Params: A.Tensor((6,), "float32"),
        ):
            with A.Kernel(max((n + 63) // 64, 1)) as block:
                with A.SimtVF(threads=64):
                    one(
                        Tiles,
                        Velocities,
                        Poses,
                        Params,
                        block * 64 + A.get_thread_binding(),
                    )
    elif target in ("metal", "cuda"):

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((nt,), "uint32"),
            Velocities: T.Tensor((nv,), "uint32"),
            Poses: T.Tensor((np,), "uint32"),
            Params: T.Tensor((6,), "float32"),
        ):
            with T.Kernel(max((n + 63) // 64, 1), threads=64) as block:
                one(
                    Tiles,
                    Velocities,
                    Poses,
                    Params,
                    block * 64 + T.get_thread_binding(),
                )
    else:
        raise ValueError(f"Unsupported target {target}")
    return kernel
