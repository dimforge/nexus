"""Counterpart of queries/gjk/voronoi_simplex3.rs; explicit local state arrays."""

import tilelang.language as T
from kernel_support import cross, dot

from ...shapes.segment import project_local_point_and_get_location as segment
from ...shapes.tetrahedron import project_local_point_and_get_location as tetrahedron
from ...shapes.triangle import project_local_point_and_get_location as triangle
from .cso_point import origin


@T.macro
def init(Point, Vertices, Projection, Dim, PrevVertices, PrevProjection):
    zero = T.alloc_local((3, 3), "float32")
    origin(zero)
    Dim[0] = T.uint32(0)
    Dim[1] = T.uint32(0)
    for i in T.serial(4):
        PrevVertices[i] = T.cast(i, "uint32")
        for field in T.serial(3):
            for d in T.serial(3):
                Vertices[i, field, d] = T.if_then_else(
                    i == 0, Point[field, d], zero[field, d]
                )
    for d in T.serial(3):
        Projection[d] = 0.0
        PrevProjection[d] = 0.0


@T.macro
def add_point(Point, Vertices, Projection, Dim, PrevVertices, PrevProjection):
    accept = T.alloc_local((1,), "bool")
    accept[0] = True
    ab = T.alloc_local((3,), "float32")
    ac = T.alloc_local((3,), "float32")
    ap = T.alloc_local((3,), "float32")
    normal = T.alloc_local((3,), "float32")
    if Dim[0] == 0:
        for d in T.serial(3):
            ab[d] = Vertices[0, 0, d] - Point[0, d]
        if dot(ab, ab) < T.float32(1.0e-6):
            accept[0] = False
    elif Dim[0] == 1:
        for d in T.serial(3):
            ab[d] = Vertices[1, 0, d] - Vertices[0, 0, d]
            ac[d] = Point[0, d] - Vertices[0, 0, d]
        cross(ab, ac, normal)
        if dot(normal, normal) < T.float32(1.0e-6):
            accept[0] = False
    else:
        for d in T.serial(3):
            ab[d] = Vertices[1, 0, d] - Vertices[0, 0, d]
            ac[d] = Vertices[2, 0, d] - Vertices[0, 0, d]
            ap[d] = Point[0, d] - Vertices[0, 0, d]
        cross(ab, ac, normal)
        length = T.sqrt(dot(normal, normal))
        for d in T.serial(3):
            normal[d] = normal[d] / length
        distance = dot(normal, ap)
        if T.if_then_else(distance < 0.0, -distance, distance) < T.float32(1.0e-6):
            accept[0] = False
    # Bounded selects with separate index/vertex loops avoid a pinned Metal
    # local-array optimization regression on rejected duplicate points.
    accepted = accept[0]
    previous_dim = Dim[0]
    Dim[1] = T.Select(accepted, previous_dim, Dim[1])
    for d in T.serial(3):
        PrevProjection[d] = T.Select(accepted, Projection[d], PrevProjection[d])
    for i in T.serial(4):
        PrevVertices[i] = T.Select(accepted, T.cast(i, "uint32"), PrevVertices[i])
    for i in T.serial(4):
        for field in T.serial(3):
            for d in T.serial(3):
                Vertices[i, field, d] = T.Select(
                    accepted & (i == T.cast(previous_dim, "int32") + 1),
                    Point[field, d],
                    Vertices[i, field, d],
                )
    Dim[0] = T.Select(accepted, previous_dim + T.uint32(1), previous_dim)


@T.macro
def swap_vertices(Vertices, PrevVertices, a, b):
    for field in T.serial(3):
        for d in T.serial(3):
            tmp = Vertices[a, field, d]
            Vertices[a, field, d] = Vertices[b, field, d]
            Vertices[b, field, d] = tmp
    old = PrevVertices[a]
    PrevVertices[a] = PrevVertices[b]
    PrevVertices[b] = old


@T.macro
def project_origin_and_reduce(
    Vertices, Projection, Dim, PrevVertices, PrevProjection, Point
):
    a = T.alloc_local((3,), "float32")
    b = T.alloc_local((3,), "float32")
    c = T.alloc_local((3,), "float32")
    d = T.alloc_local((3,), "float32")
    zero = T.alloc_local((3,), "float32")
    values = T.alloc_local((6,), "float32")
    flags = T.alloc_local((3,), "uint32")
    for k in T.serial(3):
        a[k] = Vertices[0, 0, k]
        b[k] = Vertices[1, 0, k]
        c[k] = Vertices[2, 0, k]
        d[k] = Vertices[3, 0, k]
        zero[k] = 0.0
    if Dim[0] == 0:
        Projection[0] = 1.0
        for k in T.serial(3):
            Point[k] = a[k]
    else:
        before = T.cast(Dim[0], "int32")
        if before == 1:
            segment(a, b, zero, True, values, flags)
        elif before == 2:
            triangle(a, b, c, zero, True, values, flags)
        else:
            tetrahedron(a, b, c, d, zero, True, values, flags)
        feature = T.cast(flags[0], "int32")
        index = T.cast(flags[1], "int32")
        if feature == 0:
            swap_vertices(Vertices, PrevVertices, 0, index)
            Projection[0] = 1.0
            Dim[0] = T.uint32(0)
        elif feature == 1:
            if before == 1:
                Projection[0] = values[3]
                Projection[1] = values[4]
            elif before == 2:
                if index == 0:
                    Projection[0] = values[3]
                    Projection[1] = values[4]
                elif index == 1:
                    swap_vertices(Vertices, PrevVertices, 0, 2)
                    Projection[0] = values[4]
                    Projection[1] = values[3]
                else:
                    swap_vertices(Vertices, PrevVertices, 1, 2)
                    Projection[0] = values[3]
                    Projection[1] = values[4]
                Dim[0] = T.uint32(1)
            else:
                if index == 1:
                    swap_vertices(Vertices, PrevVertices, 1, 2)
                elif index == 2:
                    swap_vertices(Vertices, PrevVertices, 1, 3)
                elif index == 3:
                    swap_vertices(Vertices, PrevVertices, 0, 2)
                elif index == 4:
                    swap_vertices(Vertices, PrevVertices, 0, 3)
                elif index == 5:
                    swap_vertices(Vertices, PrevVertices, 0, 2)
                    swap_vertices(Vertices, PrevVertices, 1, 3)
                if (index == 0) | (index == 1) | (index == 2) | (index == 5):
                    Projection[0] = values[3]
                    Projection[1] = values[4]
                elif (index == 3) | (index == 4):
                    Projection[0] = values[4]
                    Projection[1] = values[3]
                Dim[0] = T.uint32(1)
        elif feature == 2:
            if before == 2:
                for k in T.serial(3):
                    Projection[k] = values[3 + k]
                # Match the native perpendicular face projection. Reconstructing
                # from barycentrics can leave a false tangential GJK direction.
                ab = T.alloc_local((3,), "float32")
                ac = T.alloc_local((3,), "float32")
                normal = T.alloc_local((3,), "float32")
                for k in T.serial(3):
                    ab[k] = b[k] - a[k]
                    ac[k] = c[k] - a[k]
                cross(ab, ac, normal)
                norm_sq = dot(normal, normal)
                scale_sq = T.max(dot(a, a), T.max(dot(b, b), dot(c, c)))
                residual_sq = (values[0] * values[0] + values[1] * values[1]) + values[
                    2
                ] * values[2]
                if (norm_sq > 0.0) & (
                    residual_sq <= (scale_sq * T.float32(1e-7)) * T.float32(1e-7)
                ):
                    offset = dot(normal, a) / norm_sq
                    for k in T.serial(3):
                        values[k] = normal[k] * offset
            else:
                if index == 0:
                    for k in T.serial(3):
                        Projection[k] = values[3 + k]
                elif index == 1:
                    for field in T.serial(3):
                        for k in T.serial(3):
                            Vertices[2, field, k] = Vertices[3, field, k]
                    for k in T.serial(3):
                        Projection[k] = values[3 + k]
                elif index == 2:
                    for field in T.serial(3):
                        for k in T.serial(3):
                            Vertices[1, field, k] = Vertices[3, field, k]
                    Projection[0] = values[3]
                    Projection[1] = values[5]
                    Projection[2] = values[4]
                elif index == 3:
                    for field in T.serial(3):
                        for k in T.serial(3):
                            Vertices[0, field, k] = Vertices[3, field, k]
                    Projection[0] = values[5]
                    Projection[1] = values[3]
                    Projection[2] = values[4]
                Dim[0] = T.uint32(2)
        for k in T.serial(3):
            Point[k] = values[k]
