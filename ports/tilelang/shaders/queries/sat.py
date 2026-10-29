"""Counterpart of queries/sat.rs; 3D cuboid separating axes."""

import tilelang.language as T
from kernel_support import dot, quat_rotate


from ..shapes.shape import cuboid_support_point


@T.macro
def cuboid_cuboid_compute_separation_wrt_local_line(
    he1, he2, q12, t12, line, axis, sep
):
    sign = T.if_then_else(dot(t12, line) >= T.float32(0), T.float32(1), T.float32(-1))
    inv = T.alloc_local((4,), "float32")
    neg = T.alloc_local((3,), "float32")
    axis2 = T.alloc_local((3,), "float32")
    p1 = T.alloc_local((3,), "float32")
    p2 = T.alloc_local((3,), "float32")
    rotated = T.alloc_local((3,), "float32")
    delta = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        axis[d] = line[d] * sign
        neg[d] = -axis[d]
        inv[d] = -q12[d]
    inv[3] = q12[3]
    quat_rotate(inv, neg, axis2)
    cuboid_support_point(he1, axis, p1)
    cuboid_support_point(he2, axis2, p2)
    quat_rotate(q12, p2, rotated)
    for d in T.serial(3):
        delta[d] = (rotated[d] + t12[d]) - p1[d]
    sep[0] = dot(delta, axis)


@T.macro
def cuboid_cuboid_find_local_separating_normal_oneway(
    he1, he2, q12, t12, best_dir, best_sep
):
    best_sep[0] = T.float32(-1e18)
    for d in T.serial(3):
        best_dir[d] = T.float32(0)
    inv = T.alloc_local((4,), "float32")
    for d in T.serial(3):
        inv[d] = -q12[d]
    inv[3] = q12[3]
    for axis in T.serial(3):
        sign = T.if_then_else(t12[axis] >= T.float32(0), T.float32(1), T.float32(-1))
        direction = T.alloc_local((3,), "float32")
        direction2 = T.alloc_local((3,), "float32")
        p2 = T.alloc_local((3,), "float32")
        rotated = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            direction[d] = T.if_then_else(d == axis, -sign, T.float32(0))
        quat_rotate(inv, direction, direction2)
        cuboid_support_point(he2, direction2, p2)
        quat_rotate(q12, p2, rotated)
        separation = (rotated[axis] + t12[axis]) * sign - he1[axis]
        if separation > best_sep[0]:
            best_sep[0] = separation
            for d in T.serial(3):
                best_dir[d] = T.if_then_else(d == axis, sign, T.float32(0))


@T.macro
def cuboid_cuboid_find_local_separating_edge_twoway(
    he1, he2, q12, t12, best_dir, best_sep
):
    best_sep[0] = T.float32(-1e18)
    for d in T.serial(3):
        best_dir[d] = T.float32(0)
    for col in T.serial(3):
        unit = T.alloc_local((3,), "float32")
        rotated = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            unit[d] = T.if_then_else(d == col, T.float32(1), T.float32(0))
        quat_rotate(q12, unit, rotated)
        for row in T.serial(3):
            line = T.alloc_local((3,), "float32")
            if row == 0:
                line[0], line[1], line[2] = T.float32(0), -rotated[2], rotated[1]
            elif row == 1:
                line[0], line[1], line[2] = rotated[2], T.float32(0), -rotated[0]
            else:
                line[0], line[1], line[2] = -rotated[1], rotated[0], T.float32(0)
            norm = T.sqrt(dot(line, line))
            if norm > T.float32(1.1920929e-7):
                normalized = T.alloc_local((3,), "float32")
                axis = T.alloc_local((3,), "float32")
                separation = T.alloc_local((1,), "float32")
                for d in T.serial(3):
                    normalized[d] = line[d] / norm
                cuboid_cuboid_compute_separation_wrt_local_line(
                    he1, he2, q12, t12, normalized, axis, separation
                )
                if separation[0] > best_sep[0]:
                    best_sep[0] = separation[0]
                    for d in T.serial(3):
                        best_dir[d] = axis[d]
