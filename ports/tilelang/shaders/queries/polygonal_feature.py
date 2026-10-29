"""Counterpart of queries/polygonal_feature.rs; face clipping and manifold reduction."""

import tilelang.language as T
from kernel_support import dot, cross, quat_rotate
from ..utils.basis import orthonormal_basis3


def absolute(value):
    # Used only for magnitude comparisons. Avoid bitcasts of literals, which
    # TileLang C emits as an invalid address-of-rvalue expression.
    return T.if_then_else(value >= T.float32(0), value, -value)


def perp(a, b):
    return a[0] * b[1] - a[1] * b[0]


def relative_eq_scalar(a, b):
    return (absolute(a - b) <= T.float32(1.1920929e-7)) | (
        absolute(a - b) <= T.max(absolute(a), absolute(b)) * T.float32(1.1920929e-7)
    )


@T.macro
def is_outside(poly, count, point, result):
    sign = T.alloc_local((1,), "float32")
    x = poly[0, 0] - poly[count - 1, 0]
    y = poly[0, 1] - poly[count - 1, 1]
    sign[0] = x * (point[1] - poly[count - 1, 1]) - y * (point[0] - poly[count - 1, 0])
    result[0] = False
    for j in T.serial(3):
        if j + 1 < count and not result[0]:
            dx, dy = poly[j + 1, 0] - poly[j, 0], poly[j + 1, 1] - poly[j, 1]
            new_sign = dx * (point[1] - poly[j, 1]) - dy * (point[0] - poly[j, 0])
            if sign[0] == T.float32(0):
                sign[0] = new_sign
            elif sign[0] * new_sign < T.float32(0):
                result[0] = True


@T.macro
def closest_points_line2d(a0, a1, b0, b1, out):
    x1, y1 = a1[0] - a0[0], a1[1] - a0[1]
    x2, y2 = b1[0] - b0[0], b1[1] - b0[1]
    rx, ry = a0[0] - b0[0], a0[1] - b0[1]
    a, e, f = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x2 * rx + y2 * ry
    if a <= T.float32(1.1920929e-7) and e <= T.float32(1.1920929e-7):
        out[0], out[1] = T.float32(0), T.float32(0)
    elif a <= T.float32(1.1920929e-7):
        out[0], out[1] = T.float32(0), f / e
    else:
        c = x1 * rx + y1 * ry
        if e <= T.float32(1.1920929e-7):
            out[0], out[1] = -c / a, T.float32(0)
        else:
            b = x1 * x2 + y1 * y2
            denom = a * e - b * b
            if denom > T.float32(1.1920929e-7):
                s = (b * f - c * e) / denom
                out[0], out[1] = s, (b * s + f) / e
            else:
                out[0], out[1] = T.float32(1e18), T.float32(1e18)


@T.macro
def push_candidate(candidates, count, point, distance):
    if count[0] < 8:
        for d in T.serial(3):
            candidates[count[0], d] = point[d]
        candidates[count[0], 3] = distance
        count[0] += 1


@T.macro
def manifold_reduction(candidates, count, normal, prediction, points, length):
    for i in T.serial(4):
        for d in T.serial(4):
            points[i, d] = T.float32(0)
    length[0] = 0
    if count <= 4:
        for i in T.serial(4):
            if i < count:
                for d in T.serial(4):
                    points[i, d] = candidates[i, d]
        length[0] = count
    else:
        selected = T.alloc_local((4,), "int32")
        for i in T.serial(4):
            selected[i] = 8
        deepest = T.alloc_local((1,), "float32")
        deepest[0] = T.float32(1e18)
        for i in T.serial(8):
            if i < count and candidates[i, 3] < deepest[0]:
                deepest[0] = candidates[i, 3]
                selected[0] = i
        if selected[0] != 8:
            origin = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                origin[d] = candidates[selected[0], d]
            furthest = T.alloc_local((1,), "float32")
            furthest[0] = T.float32(-1e18)
            for i in T.serial(8):
                if i < count:
                    delta = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        delta[d] = candidates[i, d] - origin[d]
                    ds = dot(delta, delta)
                    if (
                        i != selected[0]
                        and candidates[i, 3] <= prediction
                        and ds > furthest[0]
                    ):
                        furthest[0] = ds
                        selected[1] = i
            for d in T.serial(4):
                points[0, d] = candidates[selected[0], d]
            length[0] = 1
            if selected[1] != 8:
                delta = T.alloc_local((3,), "float32")
                for d in T.serial(3):
                    delta[d] = candidates[selected[1], d] - origin[d]
                if (
                    delta[0] != T.float32(0)
                    or delta[1] != T.float32(0)
                    or delta[2] != T.float32(0)
                ):
                    tangent = T.alloc_local((3,), "float32")
                    cross(delta, normal, tangent)
                    min_dot, max_dot = (
                        T.alloc_local((1,), "float32"),
                        T.alloc_local((1,), "float32"),
                    )
                    min_dot[0], max_dot[0] = T.float32(1e18), T.float32(-1e18)
                    for i in T.serial(8):
                        if (
                            i < count
                            and i != selected[0]
                            and i != selected[1]
                            and candidates[i, 3] <= prediction
                        ):
                            dp = T.alloc_local((3,), "float32")
                            for d in T.serial(3):
                                dp[d] = candidates[i, d] - origin[d]
                            v = dot(dp, tangent)
                            if v < min_dot[0]:
                                min_dot[0] = v
                                selected[2] = i
                            if v > max_dot[0]:
                                max_dot[0] = v
                                selected[3] = i
                    for d in T.serial(4):
                        points[1, d] = candidates[selected[1], d]
                    length[0] = 2
                    if selected[2] != 8:
                        for d in T.serial(4):
                            points[2, d] = candidates[selected[2], d]
                        length[0] = 3
                        if selected[2] != selected[3]:
                            for d in T.serial(4):
                                points[3, d] = candidates[selected[3], d]
                            length[0] = 4


@T.macro
def contacts_face_face(
    q12, t12, vertices1, normal, vertices2, num1, num2, prediction, points, length
):
    # Unflipped path used by cuboid_cuboid; generic feature flipping is a later slice.
    basis0, basis1 = T.alloc_local((3,), "float32"), T.alloc_local((3,), "float32")
    orthonormal_basis3(normal, basis0, basis1)
    v2 = T.alloc_local((4, 3), "float32")
    projected1, projected2 = (
        T.alloc_local((4, 2), "float32"),
        T.alloc_local((4, 2), "float32"),
    )
    for i in T.serial(4):
        source, world, v1 = (
            T.alloc_local((3,), "float32"),
            T.alloc_local((3,), "float32"),
            T.alloc_local((3,), "float32"),
        )
        for d in T.serial(3):
            source[d] = vertices2[i, d]
            v1[d] = vertices1[i, d]
        quat_rotate(q12, source, world)
        for d in T.serial(3):
            world[d] = world[d] + t12[d]
            v2[i, d] = world[d]
        projected1[i, 0], projected1[i, 1] = dot(v1, basis0), dot(v1, basis1)
        projected2[i, 0], projected2[i, 1] = dot(world, basis0), dot(world, basis1)
    candidates = T.alloc_local((8, 4), "float32")
    count = T.alloc_local((1,), "int32")
    count[0] = 0
    done = T.alloc_local((1,), "bool")
    done[0] = False
    for side in T.serial(2):
        if (side == 0 and num2 > 2) or (side == 1 and not done[0] and num1 > 2):
            edge0, edge1, plane = (
                T.alloc_local((3,), "float32"),
                T.alloc_local((3,), "float32"),
                T.alloc_local((3,), "float32"),
            )
            for d in T.serial(3):
                edge0[d] = T.if_then_else(
                    side == 0, v2[2, d] - v2[1, d], vertices1[2, d] - vertices1[1, d]
                )
                edge1[d] = T.if_then_else(
                    side == 0, v2[0, d] - v2[1, d], vertices1[0, d] - vertices1[1, d]
                )
            cross(edge0, edge1, plane)
            denom = T.if_then_else(side == 0, dot(plane, normal), -dot(plane, normal))
            if not relative_eq_scalar(denom, T.float32(0)):
                any_outside = T.alloc_local((1,), "bool")
                any_outside[0] = False
                for i in T.serial(4):
                    if i < T.if_then_else(side == 0, num1, num2):
                        projected = T.alloc_local((2,), "float32")
                        outside = T.alloc_local((1,), "bool")
                        delta, point = (
                            T.alloc_local((3,), "float32"),
                            T.alloc_local((3,), "float32"),
                        )
                        for d in T.serial(2):
                            projected[d] = T.if_then_else(
                                side == 0, projected1[i, d], projected2[i, d]
                            )
                        if side == 0:
                            is_outside(projected2, num2, projected, outside)
                        else:
                            is_outside(projected1, num1, projected, outside)
                        any_outside[0] = any_outside[0] or outside[0]
                        for d in T.serial(3):
                            delta[d] = T.if_then_else(
                                side == 0,
                                v2[0, d] - vertices1[i, d],
                                vertices1[0, d] - v2[i, d],
                            )
                        distance = dot(delta, plane) / denom
                        if not outside[0] and distance <= prediction:
                            for d in T.serial(3):
                                point[d] = T.if_then_else(
                                    side == 0,
                                    vertices1[i, d],
                                    v2[i, d] - distance * normal[d],
                                )
                            push_candidate(candidates, count, point, distance)
                done[0] = not any_outside[0]
    for j in T.serial(4):
        if not done[0] and j < num2:
            jnext = T.if_then_else(j + 1 < num2, j + 1, 0)
            for i in T.serial(4):
                if i < num1 and count[0] < 8:
                    inext = T.if_then_else(i + 1 < num1, i + 1, 0)
                    a0, a1, b0, b1 = (
                        T.alloc_local((2,), "float32"),
                        T.alloc_local((2,), "float32"),
                        T.alloc_local((2,), "float32"),
                        T.alloc_local((2,), "float32"),
                    )
                    for d in T.serial(2):
                        a0[d] = projected1[i, d]
                        a1[d] = projected1[inext, d]
                        b0[d] = projected2[j, d]
                        b1[d] = projected2[jnext, d]
                    coords = T.alloc_local((2,), "float32")
                    closest_points_line2d(a0, a1, b0, b1, coords)
                    if (
                        coords[0] > T.float32(0)
                        and coords[0] < T.float32(1)
                        and coords[1] > T.float32(0)
                        and coords[1] < T.float32(1)
                    ):
                        p1, p2, delta = (
                            T.alloc_local((3,), "float32"),
                            T.alloc_local((3,), "float32"),
                            T.alloc_local((3,), "float32"),
                        )
                        for d in T.serial(3):
                            p1[d] = (
                                vertices1[i, d] * (T.float32(1) - coords[0])
                                + vertices1[inext, d] * coords[0]
                            )
                            p2[d] = (
                                v2[j, d] * (T.float32(1) - coords[1])
                                + v2[jnext, d] * coords[1]
                            )
                            delta[d] = p2[d] - p1[d]
                        distance = dot(delta, normal)
                        if distance <= prediction:
                            push_candidate(candidates, count, p1, distance)
    manifold_reduction(candidates, count[0], normal, prediction, points, length)
