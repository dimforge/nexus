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
    nv = case.get("solver_storage", case["bodies"]) * layout["velocity_words"]
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


@T.macro
def quat_mul(a, b, result):
    """glam 0.33.12 quaternion product (xyzw), with the Rust expression order."""
    result[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    result[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0]
    result[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3]
    result[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]


@T.macro
def quat_rotate(q, v, result):
    """glam 0.33.12 Quat::mul_vec3; input and output must not alias."""
    scale = q[3] * q[3] - dot(q, q)
    projection = dot(v, q) * 2.0
    result[0] = (v[0] * scale + q[0] * projection) + (q[1] * v[2] - q[2] * v[1]) * (
        q[3] * 2.0
    )
    result[1] = (v[1] * scale + q[1] * projection) + (q[2] * v[0] - q[0] * v[2]) * (
        q[3] * 2.0
    )
    result[2] = (v[2] * scale + q[2] * projection) + (q[0] * v[1] - q[1] * v[0]) * (
        q[3] * 2.0
    )


@T.macro
def quat_to_mat3(q, result):
    """glam 0.33.12 Mat3::from_quat; result indexed [column, row]."""
    x2 = q[0] + q[0]
    y2 = q[1] + q[1]
    z2 = q[2] + q[2]
    xx = q[0] * x2
    xy = q[0] * y2
    xz = q[0] * z2
    yy = q[1] * y2
    yz = q[1] * z2
    zz = q[2] * z2
    wx = q[3] * x2
    wy = q[3] * y2
    wz = q[3] * z2
    result[0, 0] = 1.0 - (yy + zz)
    result[0, 1] = xy + wz
    result[0, 2] = xz - wy
    result[1, 0] = xy - wz
    result[1, 1] = 1.0 - (xx + zz)
    result[1, 2] = yz + wx
    result[2, 0] = xz + wy
    result[2, 1] = yz - wx
    result[2, 2] = 1.0 - (xx + yy)


@T.macro
def cross(a, b, out):
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
