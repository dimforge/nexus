"""Counterpart of queries/contact_pfm_pfm.rs; GJK/EPA and feature manifolds."""

import tilelang.language as T
from kernel_support import cross, dot, f32, quat_rotate

from ..shapes.shape import local_support_point, support_face
from ..shapes.triangle import project_local_point_and_get_location
from .gjk.epa3 import closest_points as epa_closest
from .gjk.gjk import closest_points as gjk_closest
from .gjk.gjk import cso_point_from_shapes
from .gjk.voronoi_simplex3 import init
from .polygonal_feature import contacts


def contact_support_map_support_map(l):
    support = cso_point_from_shapes(l)
    gjk = gjk_closest(l)
    epa = epa_closest(l)

    @T.macro
    def one(Shapes, ia, ib, Vertices, Q, Translation, prediction, Output, Status):
        direction = T.alloc_local((3,), "float32")
        point = T.alloc_local((3, 3), "float32")
        simplex = T.alloc_local((4, 3, 3), "float32")
        projection = T.alloc_local((3,), "float32")
        previous = T.alloc_local((3,), "float32")
        dims = T.alloc_local((2,), "uint32")
        indices = T.alloc_local((4,), "uint32")
        valid = T.alloc_local((1,), "bool")
        square = dot(Translation, Translation)
        if square > T.float32(1e-7) * T.float32(1e-7):
            length = T.sqrt(square)
            for d in T.serial(3):
                direction[d] = Translation[d] / length
        else:
            direction[0] = 1.0
            direction[1] = 0.0
            direction[2] = 0.0
        support(Shapes, ia, ib, Vertices, Q, Translation, direction, point)
        init(point, simplex, projection, dims, indices, previous)
        gjk(
            Shapes,
            ia,
            ib,
            Vertices,
            Q,
            Translation,
            prediction,
            True,
            simplex,
            projection,
            dims,
            indices,
            previous,
            Output,
            Status,
        )
        if Status[0] == 0:
            epa(Shapes, ia, ib, Vertices, Q, Translation, simplex, dims, Output, valid)
            Status[0] = T.if_then_else(valid[0], T.uint32(1), T.uint32(3))
            if not valid[0]:
                Output[2, 0] = 1.0

    return one


@T.macro
def triangle_face_normal_towards(a, b, c, point, Output):
    ab = T.alloc_local((3,), "float32")
    ac = T.alloc_local((3,), "float32")
    normal = T.alloc_local((3,), "float32")
    delta = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        ab[d] = b[d] - a[d]
        ac[d] = c[d] - a[d]
        delta[d] = point[d] - a[d]
        Output[d] = 0.0
    cross(ab, ac, normal)
    length = T.sqrt(dot(normal, normal))
    if length > T.float32(1e-7):
        for d in T.serial(3):
            normal[d] = normal[d] / length
        side = dot(delta, normal)
        if side > T.float32(1e-7):
            for d in T.serial(3):
                Output[d] = normal[d]
        elif side < -T.float32(1e-7):
            for d in T.serial(3):
                Output[d] = -normal[d]


@T.macro
def dedup_manifold_points(Points, Count, prediction):
    eps = T.max(T.float32(0.25) * prediction, T.float32(1e-5))
    square = eps * eps
    length = Count[0]
    out = T.alloc_local((1,), "int32")
    merged = T.alloc_local((1,), "bool")
    p = T.alloc_local((4,), "float32")
    delta = T.alloc_local((3,), "float32")
    out[0] = 0
    for i in T.serial(4):
        if i < length:
            for d in T.serial(4):
                p[d] = Points[i, d]
            merged[0] = False
            for k in T.serial(4):
                if (k < out[0]) & (not merged[0]):
                    for d in T.serial(3):
                        delta[d] = Points[k, d] - p[d]
                    if dot(delta, delta) < square:
                        if p[3] < Points[k, 3]:
                            for d in T.serial(4):
                                Points[k, d] = p[d]
                        merged[0] = True
            if not merged[0]:
                for d in T.serial(4):
                    Points[out[0], d] = p[d]
                out[0] = out[0] + 1
    Count[0] = out[0]


def contact_manifold_pfm_pfm(l):
    query = contact_support_map_support_map(l)
    face = support_face(l)
    support = local_support_point(l)
    sf = l["shape"]

    @T.macro
    def one(
        Shapes,
        ia,
        ib,
        Vertices,
        Indices,
        Q,
        Translation,
        radius1,
        radius2,
        prediction,
        Points,
        Normal,
        Count,
    ):
        total = (prediction + radius1) + radius2
        result = T.alloc_local((3, 3), "float32")
        status = T.alloc_local((1,), "uint32")
        Count[0] = 0
        for d in T.serial(3):
            Normal[d] = 0.0
        for i in T.serial(4):
            for d in T.serial(4):
                Points[i, d] = 0.0
        query(Shapes, ia, ib, Vertices, Q, Translation, total, result, status)
        if status[0] == 1:
            p1 = T.alloc_local((3,), "float32")
            p2 = T.alloc_local((3,), "float32")
            n = T.alloc_local((3,), "float32")
            n2 = T.alloc_local((3,), "float32")
            opposite = T.alloc_local((3,), "float32")
            inverse = T.alloc_local((4,), "float32")
            delta = T.alloc_local((3,), "float32")
            fallback = T.alloc_local((1,), "bool")
            active = T.alloc_local((1,), "bool")
            for d in T.serial(3):
                p1[d] = result[0, d]
                p2[d] = result[1, d]
                n[d] = result[2, d]
                inverse[d] = -Q[d]
            inverse[3] = Q[3]
            fallback[0] = False
            active[0] = True
            if (
                T.cast(Shapes[ia * l["shape_words"] + sf["a"] + 3], "int32")
                == l["shape_types"]["triangle"]
            ):
                a = T.alloc_local((3,), "float32")
                b = T.alloc_local((3,), "float32")
                c = T.alloc_local((3,), "float32")
                face_n = T.alloc_local((3,), "float32")
                axis = T.alloc_local((3,), "float32")
                local = T.alloc_local((3,), "float32")
                for d in T.serial(3):
                    a[d] = f32(Shapes, ia * l["shape_words"] + sf["a"] + d)
                    b[d] = f32(Shapes, ia * l["shape_words"] + sf["b"] + d)
                    c[d] = f32(Shapes, ia * l["shape_words"] + sf["c"] + d)
                triangle_face_normal_towards(a, b, c, Translation, face_n)
                if dot(n, face_n) < 0.0:
                    for d in T.serial(3):
                        opposite[d] = -face_n[d]
                    quat_rotate(inverse, opposite, axis)
                    support(Shapes, ib, Vertices, axis, local)
                    quat_rotate(Q, local, p2)
                    for d in T.serial(3):
                        p2[d] = p2[d] + Translation[d]
                        delta[d] = p2[d] - a[d]
                    distance = dot(delta, face_n)
                    if distance > total:
                        active[0] = False
                    else:
                        for d in T.serial(3):
                            p1[d] = p2[d] - face_n[d] * distance
                            n[d] = face_n[d]
                        fallback[0] = True
            if active[0]:
                for d in T.serial(3):
                    opposite[d] = -n[d]
                quat_rotate(inverse, opposite, n2)
                f1 = T.alloc_local((4, 3), "float32")
                f2 = T.alloc_local((4, 3), "float32")
                c1 = T.alloc_local((1,), "int32")
                c2 = T.alloc_local((1,), "int32")
                face(Shapes, ia, Vertices, Indices, n, f1, c1)
                face(Shapes, ib, Vertices, Indices, n2, f2, c2)
                contacts(
                    Q, Translation, f1, n, f2, c1[0], c2[0], total, Points, Count, False
                )
                append_valid = T.alloc_local((1,), "bool")
                append_valid[0] = True
                if fallback[0] & (Count[0] == 0):
                    ta = T.alloc_local((3,), "float32")
                    tb = T.alloc_local((3,), "float32")
                    tc = T.alloc_local((3,), "float32")
                    values = T.alloc_local((6,), "float32")
                    flags = T.alloc_local((3,), "uint32")
                    gap = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        ta[d] = f32(Shapes, ia * l["shape_words"] + sf["a"] + d)
                        tb[d] = f32(Shapes, ia * l["shape_words"] + sf["b"] + d)
                        tc[d] = f32(Shapes, ia * l["shape_words"] + sf["c"] + d)
                    project_local_point_and_get_location(
                        ta, tb, tc, p1, True, values, flags
                    )
                    for d in T.serial(3):
                        gap[d] = values[d] - p1[d]
                    eps = T.float32(1e-7) * T.float32(100)
                    append_valid[0] = dot(gap, gap) <= eps * eps
                if (
                    append_valid[0]
                    & (Count[0] < 4)
                    & (not (fallback[0] & (Count[0] > 0)))
                ):
                    for d in T.serial(3):
                        delta[d] = p2[d] - p1[d]
                        Points[Count[0], d] = p1[d]
                    Points[Count[0], 3] = dot(delta, n)
                    Count[0] = Count[0] + 1
                if (radius1 != 0.0) | (radius2 != 0.0):
                    for i in T.serial(4):
                        if i < Count[0]:
                            for d in T.serial(3):
                                Points[i, d] = Points[i, d] + n[d] * radius1
                            Points[i, 3] = Points[i, 3] - (radius1 + radius2)
                dedup_manifold_points(Points, Count, prediction)
                if append_valid[0]:
                    for d in T.serial(3):
                        Normal[d] = n[d]
                else:
                    for v in T.serial(4):
                        for d in T.serial(4):
                            Points[v, d] = 0.0
                    for d in T.serial(3):
                        Normal[d] = 0.0
                    Count[0] = 0

    return one
