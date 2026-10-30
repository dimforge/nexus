"""Counterpart of shapes/capsule.rs boundary projection."""

import tilelang.language as T
from kernel_support import cross, dot

from .segment import project_local_point


@T.macro
def orthonormal_basis3(v, first, second):
    sign = T.if_then_else(v[2] >= T.float32(0), T.float32(1), T.float32(-1))
    a = T.float32(-1) / (sign + v[2])
    b = v[0] * v[1] * a
    first[0] = T.float32(1) + sign * v[0] * v[0] * a
    first[1] = sign * b
    first[2] = -sign * v[0]
    second[0] = b
    second[1] = sign + v[1] * v[1] * a
    second[2] = -v[1]


@T.macro
def any_orthogonal_vector(v, out):
    second = T.alloc_local((3,), "float32")
    orthonormal_basis3(v, out, second)


@T.macro
def project_local_point_on_boundary(a, b, radius, pt, out, inside):
    axis_point = T.alloc_local((3,), "float32")
    delta = T.alloc_local((3,), "float32")
    project_local_point(a, b, pt, axis_point)
    for d in T.serial(3):
        delta[d] = pt[d] - axis_point[d]
    axis_seg = T.alloc_local((3,), "float32")
    from_a = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        axis_seg[d] = b[d] - a[d]
        from_a[d] = pt[d] - a[d]
    along = dot(axis_seg, from_a)
    axis_len_sq = dot(axis_seg, axis_seg)
    if along > T.float32(0) and along < axis_len_sq:
        first = T.alloc_local((3,), "float32")
        radial = T.alloc_local((3,), "float32")
        cross(axis_seg, from_a, first)
        cross(first, axis_seg, radial)
        for d in T.serial(3):
            delta[d] = radial[d] / axis_len_sq
    distance = T.sqrt(dot(delta, delta))
    if distance > T.float32(0):
        inside[0] = distance <= radius
        for d in T.serial(3):
            out[d] = axis_point[d] + delta[d] * (radius / distance)
    else:
        axis = T.alloc_local((3,), "float32")
        norm = T.alloc_local((3,), "float32")
        direction = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            axis[d] = b[d] - a[d]
        length = T.sqrt(dot(axis, axis))
        if length > T.float32(0):
            for d in T.serial(3):
                norm[d] = axis[d] / length
            any_orthogonal_vector(norm, direction)
        else:
            direction[0], direction[1], direction[2] = (
                T.float32(1),
                T.float32(0),
                T.float32(0),
            )
        for d in T.serial(3):
            out[d] = axis_point[d] + direction[d] * radius
        inside[0] = True


@T.macro
def local_support_point(a, b, radius, direction, Output):
    segment = T.alloc_local((3,), "float32")
    normal = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        segment[d] = b[d] - a[d]
        normal[d] = 0.0
    normal[1] = 1.0
    projection = (
        segment[0] * direction[0]
        + segment[1] * direction[1]
        + segment[2] * direction[2]
    )
    for d in T.serial(3):
        Output[d] = T.if_then_else(projection >= 0.0, b[d], a[d])
    if radius != 0.0:
        length = T.sqrt(
            direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2]
        )
        if length != 0.0:
            for d in T.serial(3):
                normal[d] = direction[d] / length
        for d in T.serial(3):
            Output[d] = Output[d] + normal[d] * radius
