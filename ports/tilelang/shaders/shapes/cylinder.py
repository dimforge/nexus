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


@T.macro
def support_face(half_height, radius, direction, Output, Count):
    planar = T.alloc_local((2,), "float32")
    length = T.sqrt(direction[0] * direction[0] + direction[2] * direction[2])
    if length < T.float32(1.1920929e-7):
        planar[0] = 1.0
        planar[1] = 0.0
    else:
        planar[0] = direction[0] / length
        planar[1] = direction[2] / length
    if T.if_then_else(direction[1] < 0.0, -direction[1], direction[1]) < 0.5:
        for i in T.serial(2):
            Output[i, 0] = planar[0] * radius
            Output[i, 1] = T.if_then_else(i == 0, -half_height, half_height)
            Output[i, 2] = planar[1] * radius
        Count[0] = 2
    else:
        height = T.if_then_else(direction[1] >= 0.0, half_height, -half_height)
        Output[0, 0] = planar[0] * radius
        Output[0, 2] = planar[1] * radius
        Output[1, 0] = -planar[1] * radius
        Output[1, 2] = planar[0] * radius
        Output[2, 0] = -planar[0] * radius
        Output[2, 2] = -planar[1] * radius
        Output[3, 0] = planar[1] * radius
        Output[3, 2] = -planar[0] * radius
        for i in T.serial(4):
            Output[i, 1] = height
        Count[0] = 4
