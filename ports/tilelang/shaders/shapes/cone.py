"""Counterpart of shapes/cone.rs boundary projection."""

import tilelang.language as T
from kernel_support import cross, dot

from .segment import project_local_point


@T.macro
def project_local_point_on_boundary(height, radius, pt, out, inside):
    radial = T.sqrt(pt[0] * pt[0] + pt[2] * pt[2])
    direction = T.alloc_local((2,), "float32")
    direction[0] = T.float32(1)
    direction[1] = T.float32(0)
    if radial > T.float32(0):
        direction[0] = pt[0] / radial
        direction[1] = pt[2] / radial
    base = T.alloc_local((3,), "float32")
    base[0], base[1], base[2] = pt[0], -height, pt[2]
    inside[0] = False
    if pt[1] < -height and radial <= radius:
        for d in T.serial(3):
            out[d] = base[d]
    else:
        circle = T.alloc_local((3,), "float32")
        circle[0], circle[1], circle[2] = (
            direction[0] * radius,
            -height,
            direction[1] * radius,
        )
        apex = T.alloc_local((3,), "float32")
        apex[0], apex[1], apex[2] = T.float32(0), height, T.float32(0)
        side = T.alloc_local((3,), "float32")
        project_local_point(apex, circle, pt, side)
        axis = T.alloc_local((3,), "float32")
        axis[0], axis[1], axis[2] = T.float32(0), T.float32(-2) * height, T.float32(0)
        seg = T.alloc_local((3,), "float32")
        delta = T.alloc_local((3,), "float32")
        cross1 = T.alloc_local((3,), "float32")
        cross2 = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            seg[d] = circle[d] - apex[d]
            delta[d] = pt[d] - apex[d]
        cross(seg, delta, cross1)
        cross(seg, axis, cross2)
        for d in T.serial(3):
            out[d] = side[d]
        if pt[1] >= -height and pt[1] <= height and dot(cross1, cross2) >= T.float32(0):
            inside[0] = True
            ds = T.alloc_local((3,), "float32")
            db = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                ds[d] = side[d] - pt[d]
                db[d] = base[d] - pt[d]
            if dot(ds, ds) > dot(db, db):
                for d in T.serial(3):
                    out[d] = base[d]


@T.macro
def local_support_point(height, radius, direction, Output):
    length = T.sqrt((direction[0] * direction[0] + 0.0) + direction[2] * direction[2])
    if length == 0.0:
        Output[0] = 0.0
        Output[2] = 0.0
        Output[1] = T.if_then_else(direction[1] >= 0.0, height, -height)
    else:
        factor = radius / length
        Output[0] = direction[0] * factor
        Output[2] = direction[2] * factor
        Output[1] = -height
        projection = (direction[0] * Output[0] + direction[1] * Output[1]) + direction[
            2
        ] * Output[2]
        if projection < direction[1] * height:
            Output[0] = 0.0
            Output[1] = height
            Output[2] = 0.0
