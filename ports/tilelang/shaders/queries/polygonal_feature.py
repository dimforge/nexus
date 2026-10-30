"""Counterpart of queries/polygonal_feature.rs; face clipping and manifold reduction."""

import tilelang.language as T
from kernel_support import cross, dot, quat_rotate

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
            # Match Rust's stable 2D determinant and Cramer's rule.
            det = x1 * y2 - y1 * x2
            if det * det > T.float32(1.1920929e-7):
                out[0] = (x2 * ry - y2 * rx) / det
                out[1] = (x1 * ry - y1 * rx) / det
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
    q12,
    t12,
    vertices1,
    normal,
    vertices2,
    num1,
    num2,
    prediction,
    points,
    length,
    flipped,
):
    # Preserve native candidate conversion before manifold reduction.
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
                            p2 = T.alloc_local((3,), "float32")
                            converted = T.alloc_local((3,), "float32")
                            for d in T.serial(3):
                                p2[d] = T.if_then_else(
                                    side == 0,
                                    vertices1[i, d] + distance * normal[d],
                                    v2[i, d],
                                )
                            candidate(q12, t12, point, p2, flipped, converted)
                            push_candidate(candidates, count, converted, distance)
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
                            converted = T.alloc_local((3,), "float32")
                            candidate(q12, t12, p1, p2, flipped, converted)
                            push_candidate(candidates, count, converted, distance)
    manifold_reduction(candidates, count[0], normal, prediction, points, length)


@T.macro
def candidate(q, t, p1, p2, flipped, Output):
    if flipped:
        inverse = T.alloc_local((4,), "float32")
        delta = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            inverse[d] = -q[d]
            delta[d] = p2[d] - t[d]
        inverse[3] = q[3]
        quat_rotate(inverse, delta, Output)
    else:
        for d in T.serial(3):
            Output[d] = p1[d]


@T.macro
def closest_points_segment_segment(a0, a1, b0, b1, Output):
    x1 = a1[0] - a0[0]
    y1 = a1[1] - a0[1]
    x2 = b1[0] - b0[0]
    y2 = b1[1] - b0[1]
    rx = a0[0] - b0[0]
    ry = a0[1] - b0[1]
    aa = x1 * x1 + y1 * y1
    ee = x2 * x2 + y2 * y2
    ff = x2 * rx + y2 * ry
    if (aa <= T.float32(1.1920929e-7)) & (ee <= T.float32(1.1920929e-7)):
        Output[0] = 0.0
        Output[1] = 0.0
    elif aa <= T.float32(1.1920929e-7):
        Output[0] = 0.0
        Output[1] = T.min(T.max(ff / ee, 0.0), 1.0)
    else:
        cc = x1 * rx + y1 * ry
        if ee <= T.float32(1.1920929e-7):
            Output[0] = T.min(T.max(-cc / aa, 0.0), 1.0)
            Output[1] = 0.0
        else:
            bb = x1 * x2 + y1 * y2
            denom = aa * ee - bb * bb
            Output[0] = T.if_then_else(
                denom > T.float32(1.1920929e-7),
                T.min(T.max((bb * ff - cc * ee) / denom, 0.0), 1.0),
                0.0,
            )
            Output[1] = (bb * Output[0] + ff) / ee
            if Output[1] < 0.0:
                Output[1] = 0.0
                Output[0] = T.min(T.max(-cc / aa, 0.0), 1.0)
            elif Output[1] > 1.0:
                Output[1] = 1.0
                Output[0] = T.min(T.max((bb - cc) / aa, 0.0), 1.0)


@T.macro
def clip_segment_segment(a0, a1, b0, b1, Clips, Empty):
    tangent = T.alloc_local((3,), "float32")
    delta0 = T.alloc_local((3,), "float32")
    delta1 = T.alloc_local((3,), "float32")
    c = T.alloc_local((3,), "float32")
    dpoint = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        tangent[d] = a1[d] - a0[d]
        delta0[d] = b0[d] - a0[d]
        delta1[d] = b1[d] - a0[d]
    length1 = dot(tangent, tangent)
    start = dot(delta0, tangent)
    end = dot(delta1, tangent)
    lo = T.min(start, end)
    hi = T.max(start, end)
    for d in T.serial(3):
        c[d] = T.if_then_else(end < start, b1[d], b0[d])
        dpoint[d] = T.if_then_else(end < start, b0[d], b1[d])
    Empty[0] = (lo > length1) | (0.0 > hi)
    for v in T.serial(4):
        for d in T.serial(3):
            Clips[v, d] = 0.0
    if not Empty[0]:
        length2 = hi - lo
        for d in T.serial(3):
            if lo > 0.0:
                Clips[0, d] = a0[d] + tangent[d] * (lo / length1)
                Clips[1, d] = c[d]
            else:
                Clips[0, d] = a0[d]
                Clips[1, d] = c[d] + (dpoint[d] - c[d]) * (-lo / length2)
            if hi < length1:
                Clips[2, d] = a0[d] + tangent[d] * (hi / length1)
                Clips[3, d] = dpoint[d]
            else:
                Clips[2, d] = a1[d]
                Clips[3, d] = c[d] + (dpoint[d] - c[d]) * ((length1 - lo) / length2)


@T.macro
def contacts_edge_edge(q, t, face1, normal, face2, prediction, Points, Count, flipped):
    basis0 = T.alloc_local((3,), "float32")
    basis1 = T.alloc_local((3,), "float32")
    v2 = T.alloc_local((2, 3), "float32")
    p1 = T.alloc_local((2, 2), "float32")
    p2 = T.alloc_local((2, 2), "float32")
    a = T.alloc_local((3,), "float32")
    b = T.alloc_local((3,), "float32")
    world = T.alloc_local((3,), "float32")
    orthonormal_basis3(normal, basis0, basis1)
    Count[0] = 0
    for i in T.serial(4):
        for d in T.serial(4):
            Points[i, d] = 0.0
    for i in T.serial(2):
        for d in T.serial(3):
            a[d] = face1[i, d]
            b[d] = face2[i, d]
        quat_rotate(q, b, world)
        for d in T.serial(3):
            world[d] = world[d] + t[d]
            v2[i, d] = world[d]
        p1[i, 0] = dot(a, basis0)
        p1[i, 1] = dot(a, basis1)
        p2[i, 0] = dot(world, basis0)
        p2[i, 1] = dot(world, basis1)
    t1 = T.alloc_local((2,), "float32")
    t2 = T.alloc_local((2,), "float32")
    crossing = T.alloc_local((1,), "bool")
    for d in T.serial(2):
        t1[d] = p1[1, d] - p1[0, d]
        t2[d] = p2[1, d] - p2[0, d]
    len1 = T.sqrt(t1[0] * t1[0] + t1[1] * t1[1])
    len2 = T.sqrt(t2[0] * t2[0] + t2[1] * t2[1])
    crossing[0] = False
    if (len1 > T.float32(1.1920929e-7)) & (len2 > T.float32(1.1920929e-7)):
        for d in T.serial(2):
            t1[d] = t1[d] / len1
            t2[d] = t2[d] / len2
        crossing[0] = (t1[0] * t2[0] + t1[1] * t2[1]) < T.float32(0.9238795)
    local1 = T.alloc_local((3,), "float32")
    local2 = T.alloc_local((3,), "float32")
    delta = T.alloc_local((3,), "float32")
    converted = T.alloc_local((3,), "float32")
    if crossing[0]:
        a0 = T.alloc_local((2,), "float32")
        a1 = T.alloc_local((2,), "float32")
        b0 = T.alloc_local((2,), "float32")
        b1 = T.alloc_local((2,), "float32")
        bary = T.alloc_local((2,), "float32")
        for d in T.serial(2):
            a0[d] = p1[0, d]
            a1[d] = p1[1, d]
            b0[d] = p2[0, d]
            b1[d] = p2[1, d]
        closest_points_segment_segment(a0, a1, b0, b1, bary)
        for d in T.serial(3):
            local1[d] = face1[0, d] * (1.0 - bary[0]) + face1[1, d] * bary[0]
            local2[d] = v2[0, d] * (1.0 - bary[1]) + v2[1, d] * bary[1]
            delta[d] = local2[d] - local1[d]
        distance = dot(delta, normal)
        if distance <= prediction:
            candidate(q, t, local1, local2, flipped, converted)
            for d in T.serial(3):
                Points[0, d] = converted[d]
            Points[0, 3] = distance
            Count[0] = 1
    else:
        a0 = T.alloc_local((3,), "float32")
        a1 = T.alloc_local((3,), "float32")
        b0 = T.alloc_local((3,), "float32")
        b1 = T.alloc_local((3,), "float32")
        clips = T.alloc_local((4, 3), "float32")
        empty = T.alloc_local((1,), "bool")
        for d in T.serial(3):
            a0[d] = face1[0, d]
            a1[d] = face1[1, d]
            b0[d] = v2[0, d]
            b1[d] = v2[1, d]
        clip_segment_segment(a0, a1, b0, b1, clips, empty)
        if not empty[0]:
            for i in T.serial(2):
                for d in T.serial(3):
                    local1[d] = clips[i * 2, d]
                    local2[d] = clips[i * 2 + 1, d]
                    delta[d] = local2[d] - local1[d]
                distance = dot(delta, normal)
                if distance <= prediction:
                    candidate(q, t, local1, local2, flipped, converted)
                    for d in T.serial(3):
                        Points[Count[0], d] = converted[d]
                    Points[Count[0], 3] = distance
                    Count[0] = Count[0] + 1


@T.macro
def contacts(
    q, t, face1, normal, face2, num1, num2, prediction, Points, Count, flipped
):
    if (num1 == 2) & (num2 == 2):
        contacts_edge_edge(
            q, t, face1, normal, face2, prediction, Points, Count, flipped
        )
    else:
        contacts_face_face(
            q, t, face1, normal, face2, num1, num2, prediction, Points, Count, flipped
        )
