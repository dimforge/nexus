"""Counterpart of shapes/cylinder.rs boundary projection."""

import tilelang.language as T


@T.macro
def project_local_point_on_boundary(height, radius, pt, out, inside):
    radial = T.sqrt(pt[0] * pt[0] + pt[2] * pt[2])
    direction = T.alloc_local((2,), "float32")
    direction[0], direction[1] = T.float32(1), T.float32(0)
    if radial > T.float32(0):
        direction[0], direction[1] = pt[0] / radial, pt[2] / radial
    px, pz = direction[0] * radius, direction[1] * radius
    inside[0] = pt[1] >= -height and pt[1] <= height and radial <= radius
    out[0], out[1], out[2] = px, pt[1], pz
    if inside[0]:
        top, bottom, side = height - pt[1], pt[1] - (-height), radius - radial
        if top < bottom and top < side:
            out[0], out[1], out[2] = pt[0], height, pt[2]
        elif bottom < top and bottom < side:
            out[0], out[1], out[2] = pt[0], -height, pt[2]
    elif pt[1] > height:
        out[1] = height
        if radial <= radius:
            out[0], out[2] = pt[0], pt[2]
    elif pt[1] < -height:
        out[1] = -height
        if radial <= radius:
            out[0], out[2] = pt[0], pt[2]


@T.macro
def local_support_point(height, radius, direction, Output):
    length = T.sqrt((direction[0] * direction[0] + 0.0) + direction[2] * direction[2])
    factor = T.if_then_else(length != 0.0, radius / length, 0.0)
    Output[0] = direction[0] * factor
    Output[2] = direction[2] * factor
    Output[1] = T.if_then_else(direction[1] >= 0.0, height, -height)
